"""
src/data_loader.py
===================

Data layer cho project sparse+robust portfolio optimization (VN100).

Module này:
1. Tải danh sách mã cổ phiếu thuộc rổ VN100 (qua vnstock.Listing).
2. Tải lịch sử giá đóng cửa (close) 4 năm gần nhất cho từng mã (qua vnstock.Quote),
3. Làm sạch dữ liệu: loại mã thiếu quá nhiều phiên hoặc niêm yết muộn, loại phiên
   cuối tuần/nghi ngày lễ giả (artifact nguồn dữ liệu), forward-fill lỗ hổng
   ngắn, tính simple daily return.
4. Cache kết quả ra parquet/csv trong thư mục data/ để các phase sau (ước lượng
   mu_hat/Sigma, solver, verify, notebook, v.v.) đọc lại mà không cần gọi API
   mỗi lần chạy.

Người đọc mục tiêu là dân TỐI ƯU HOÁ, không nhất thiết chuyên tài chính -- các
giả định tài chính được ghi rõ dưới đây thay vì giấu trong code.

Giả định quan trọng:
- `close` trả về từ source VCI được coi là giá ĐÃ ĐIỀU CHỈNH (adjusted) cho cổ
  tức/chia tách.
- Return dùng ở đây là SIMPLE return = close.pct_change(), KHÔNG phải
  log-return (log-return sẽ khác một chút, không dùng ở phase này).
- Universe là snapshot VN100 HIỆN TẠI (ngày chạy script) áp dụng ngược cho quá
  khứ 4 năm => có survivorship bias (mã đã bị loại khỏi VN100 hoặc hủy niêm
  yết trong 4 năm qua sẽ KHÔNG xuất hiện dù từng thuộc rổ). 
- Ma trận giá được ghép bằng UNION (outer join) các ngày giao dịch của tất cả
  các mã, không phải intersection: nếu 1 mã không có dữ liệu ở 1 ngày (ví dụ
  bị đình chỉ giao dịch), API vnstock thường không trả về row cho ngày đó
  luôn (thay vì trả NaN) -- outer join khiến ngày đó xuất hiện là NaN cho mã
  đó nhưng có giá trị cho các mã khác, đúng với khái niệm "phiên thiếu" mà
  bước làm sạch bên dưới xử lý.

Cache: nếu data/prices.parquet và data/returns.parquet đã tồn tại và
force_refresh=False (mặc định), script đọc thẳng từ cache, KHÔNG gọi mạng.
Đặt biến môi trường FORCE_REFRESH=1 (hoặc gọi `main(force_refresh=True)`) để
ép tải lại từ API.

Chạy: `.venv/bin/python -m src.data_loader`
"""

from __future__ import annotations

import logging
import os
import time
from datetime import date
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - dotenv luôn có trong .venv nhưng phòng hờ
    load_dotenv = None


# --------------------------------------------------------------------------
# Cấu hình chung
# --------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
SYMBOLS_CSV = DATA_DIR / "vn100_symbols.csv"
PRICES_PARQUET = DATA_DIR / "prices.parquet"
RETURNS_PARQUET = DATA_DIR / "returns.parquet"
VN30_INDEX_PARQUET = DATA_DIR / "vn30_index.parquet"
VN30_INDEX_SYMBOL = "VN30"  # chỉ số VN30 (KHÔNG phải rổ 30 mã thành viên), nguồn VCI

VN100_GROUP = "VN100"
LISTING_SOURCE = "VCI"  # Listing chỉ nhận source trong {KBS, VCI, MSN}.
# Quote chỉ nhận source trong {vci, msn, kbs, dnse, binance, fmp, fmarket}.
# TCBS/SSI KHÔNG hợp lệ cho Quote -> KHÔNG dùng (sẽ ném ValueError).
#
# LỖI ĐÃ XÁC NHẬN (Phase 1b, 2026-07-25): MSN từng nằm trong fallback này.
# MSN CHỈ trả về ~365 phiên gần nhất bất kể tham số `start` truyền vào (tức
# là "cắt cụt" lịch sử dài, KHÔNG báo lỗi -- trả về series ngắn một cách âm
# thầm). Với cửa sổ 4 năm (~1000 phiên), điều này khiến clean_prices() gán
# nhãn SAI "listed_late" cho các mã lớn hợp lệ (VD: VIC, VHM, VNM, TCB, VRE,
# VIB, ...) và loại chúng khỏi universe, dù các mã này có lịch sử đầy đủ.
# => MSN bị loại VĨNH VIỄN khỏi fallback cho pipeline lịch sử dài (4 năm).
# Nguyên nhân gốc khiến VCI từng phải fallback là TIMEOUT tạm thời (không
# phải VCI thiếu dữ liệu), nên hướng xử lý đúng là RETRY chính VCI (xem
# `fetch_history_one_symbol`), không phải đổi sang source khác cắt cụt dữ
# liệu. KBS được giữ lại làm fallback cuối cùng vì không quan sát thấy hành
# vi cắt cụt tương tự (nhưng nên verify coverage nếu dùng trong tương lai).
QUOTE_SOURCE_FALLBACK: tuple[str, ...] = ("VCI", "KBS")
INTERVAL = "1D"

DEFAULT_SLEEP_SEC = 1.2  # sleep giữa mỗi request tải giá 1 mã (rate limit)
BACKOFF_SLEEP_SEC = 5.0  # sleep dài hơn nếu cả chuỗi fallback lỗi ở lượt đầu
MAX_BACKOFF_ROUNDS = 1  # số lần thử lại toàn bộ chuỗi source sau backoff

# Retry riêng cho VCI khi bị lỗi/timeout (Phase 1b): timeout của VCI là tạm
# thời, nên đáng để thử lại NHIỀU LẦN với backoff tăng dần TRƯỚC KHI chuyển
# sang source khác (source khác, vd MSN, có thể trả dữ liệu "thành công"
# nhưng bị cắt cụt âm thầm -- xem giải thích ở QUOTE_SOURCE_FALLBACK).
VCI_RETRY_BACKOFFS: tuple[float, ...] = (5.0, 15.0, 30.0)

MAX_MISSING_FRAC = 0.10  # loại mã nếu > 10% phiên thiếu trong cửa sổ
MAX_FFILL_GAP = 3  # forward-fill tối đa 3 phiên liên tiếp

# Lớp phòng vệ thống kê (Phase 7, theo đề xuất user): `dayofweek>=5` chỉ bắt
# được cuối tuần, KHÔNG bắt được ngày lễ VN rơi vào ngày thường (Tết, 30/4,
# 1/5, 2/9, Giỗ Tổ...) -- nếu nguồn dữ liệu lỡ tạo hàng "lặp giá phiên trước"
# cho các ngày đó (cùng loại bug đã gặp với Chủ nhật), nó sẽ để lại đúng dấu
# vết: gần như MỌI mã có return=0.0 CHÍNH XÁC cùng lúc. Ngưỡng 90% dùng để
# phát hiện & loại các phiên đó mà không cần liệt kê thủ công lịch nghỉ lễ.
HOLIDAY_ZERO_RETURN_FRAC = 0.90

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("data_loader")


def _default_date_window(today: date | None = None) -> tuple[str, str]:
    """Trả về (start, end) dạng 'YYYY-MM-DD' cho cửa sổ 4 năm gần nhất.

    end = hôm nay, start = end - 4 năm (dùng relativedelta để trừ đúng theo
    năm/tháng thay vì trừ cứng 4*365 ngày, tránh lệch do năm nhuận).
    """
    today = today or date.today()
    start = today - relativedelta(years=4)
    return start.isoformat(), today.isoformat()


# --------------------------------------------------------------------------
# Bước 1: Universe VN100 (cần mạng)
# --------------------------------------------------------------------------

def load_vn100_symbols() -> list[str]:
    """Tải danh sách mã thuộc rổ VN100 hiện tại qua vnstock.Listing.

    Listing CHỈ chấp nhận source in {KBS, VCI, MSN}. Ta dùng VCI làm mặc định
    vì đây là source ổn định nhất theo brief đã verify thủ công.
    """
    from vnstock import Listing

    logger.info("Đang tải danh sách mã VN100 (Listing source=%s)...", LISTING_SOURCE)
    symbols = Listing(source=LISTING_SOURCE).symbols_by_group(VN100_GROUP)
    symbols = sorted({str(s).strip().upper() for s in symbols if str(s).strip()})
    logger.info("Tải được %d mã trong nhóm %s.", len(symbols), VN100_GROUP)
    return symbols


# --------------------------------------------------------------------------
# Bước 2: Tải lịch sử giá với rate limiting + fallback source (cần mạng)
# --------------------------------------------------------------------------

def fetch_history_one_symbol(
    symbol: str,
    start: str,
    end: str,
    interval: str = INTERVAL,
    sources: Sequence[str] = QUOTE_SOURCE_FALLBACK,
    vci_retry_backoffs: Sequence[float] = VCI_RETRY_BACKOFFS,
) -> tuple[pd.Series | None, str | None]:
    """Tải lịch sử giá close cho 1 mã, thử lần lượt các source trong `sources`.

    Trả về (series close với index=ngày, tên source đã thành công), hoặc
    (None, None) nếu tất cả source đều lỗi.

    Phase 1b fix: khi source là "VCI" và request lỗi/timeout, ta RETRY chính
    VCI tối đa `len(vci_retry_backoffs)` lần (mặc định 3 lần, backoff tăng
    dần 5s/15s/30s) TRƯỚC KHI chuyển sang source kế tiếp trong `sources`. Lý
    do: timeout của VCI thường là tạm thời (rate limit / mạng chập chờn),
    trong khi các source fallback khác (VD: MSN) có thể trả dữ liệu "thành
    công" nhưng bị cắt cụt lịch sử một cách âm thầm (xem comment ở
    QUOTE_SOURCE_FALLBACK) -- ưu tiên retry VCI để tránh dính lỗi cắt cụt đó.

    Hàm này SLEEP bên trong khi retry VCI (backoff), khác với việc chuyển
    sang source khác trong cùng 1 lượt (không sleep, vẫn tính là cùng 1
    "lượt" cho 1 mã). Rate limiting GIỮA CÁC MÃ vẫn được xử lý ở tầng gọi
    (fetch_all_prices).
    """
    from vnstock import Quote

    for source in sources:
        attempts = 1 + (len(vci_retry_backoffs) if source.upper() == "VCI" else 0)
        for attempt in range(attempts):
            try:
                df = Quote(source=source, symbol=symbol).history(
                    start=start, end=end, interval=interval
                )
                if df is None or len(df) == 0:
                    raise ValueError(f"Empty history for {symbol} from {source}")
                df = df.copy()
                # CHUẨN HÓA timestamp về đúng NGÀY (bỏ phần giờ-phút-giây):
                # các source khác nhau trả giờ-trong-ngày khác nhau cho CÙNG 1
                # phiên giao dịch (vd: VCI = 00:00:00, KBS = 07:00:00). Nếu
                # không chuẩn hóa, pd.concat(axis=1) ở fetch_all_prices() coi
                # đây là 2 index KHÁC NHAU cho cùng 1 ngày -> khi có dù chỉ 1
                # mã fallback sang source khác giờ, union index gần như NHÂN
                # ĐÔI và MỌI mã khác bị NaN ~50% một cách giả tạo ở các dòng
                # "giờ lạ" đó (bug đã xác nhận thực tế 2026-07-26, xem
                # task-7-report.md). .dt.normalize() giữ nguyên NGÀY, đặt giờ
                # về 00:00:00 cho MỌI source một cách nhất quán.
                df["time"] = pd.to_datetime(df["time"]).dt.normalize()
                df = df.set_index("time").sort_index()
                # Một số mã có 2 dòng trong cùng 1 ngày sau khi normalize
                # (hiếm, có thể do source trả dữ liệu intraday lẫn vào) -- giữ
                # dòng CUỐI (thường là giá đóng cửa chính thức) để tránh lỗi
                # trùng index tương tự.
                df = df[~df.index.duplicated(keep="last")]
                # Một số source trả dư dữ liệu ngoài [start, end] -> cắt lại cho chuẩn.
                df = df.loc[(df.index >= pd.Timestamp(start)) & (df.index <= pd.Timestamp(end))]
                series = df["close"].astype(float)
                series.name = symbol
                return series, source
            except Exception as exc:  # noqa: BLE001 - cố ý bắt rộng để fallback/retry
                logger.debug("Source %s lỗi cho %s (attempt %d/%d): %s", source, symbol, attempt + 1, attempts, exc)
                if attempt < attempts - 1:
                    backoff = vci_retry_backoffs[attempt]
                    logger.warning(
                        "%s: VCI lỗi, retry sau %.0fs (attempt %d/%d)...",
                        symbol, backoff, attempt + 2, attempts,
                    )
                    time.sleep(backoff)
                continue
    return None, None


def fetch_all_prices(
    symbols: Sequence[str],
    start: str,
    end: str,
    interval: str = INTERVAL,
    sleep_sec: float = DEFAULT_SLEEP_SEC,
    sources: Sequence[str] = QUOTE_SOURCE_FALLBACK,
) -> pd.DataFrame:
    """Tải giá close cho toàn bộ `symbols`, có rate limit + backoff + fallback.

    Với mỗi mã:
      1. Thử lần lượt các source trong `sources` (mặc định VCI -> KBS; MSN đã
         bị loại vì cắt cụt lịch sử ~365 phiên -- xem docstring module).
      2. Nếu TẤT CẢ source lỗi ở lượt đầu, sleep `BACKOFF_SLEEP_SEC` rồi thử
         lại toàn bộ chuỗi source tối đa `MAX_BACKOFF_ROUNDS` lần nữa.
      3. Dù thành công hay thất bại hẳn, sleep `sleep_sec` trước khi sang mã
         kế tiếp (trừ mã cuối) để tránh vượt rate limit free tier (~60
         request/phút).

    Trả về DataFrame giá close: index=date (union các ngày), columns=symbol.
    Mã lỗi hoàn toàn (thử hết source + backoff vẫn lỗi) sẽ KHÔNG có cột.
    """
    n = len(symbols)
    series_list: list[pd.Series] = []
    failed: list[str] = []

    for i, symbol in enumerate(symbols, start=1):
        series, used_source = fetch_history_one_symbol(symbol, start, end, interval, sources)

        rounds_tried = 1
        while series is None and rounds_tried <= MAX_BACKOFF_ROUNDS:
            logger.warning(
                "[%d/%d] %s: tất cả source lỗi, backoff %.1fs rồi thử lại (round %d)...",
                i, n, symbol, BACKOFF_SLEEP_SEC, rounds_tried,
            )
            time.sleep(BACKOFF_SLEEP_SEC)
            series, used_source = fetch_history_one_symbol(symbol, start, end, interval, sources)
            rounds_tried += 1

        if series is not None:
            series_list.append(series)
            logger.info("[%d/%d] %s: ok (source=%s, rows=%d)", i, n, symbol, used_source, len(series))
        else:
            failed.append(symbol)
            logger.warning("[%d/%d] %s: FAILED after all sources", i, n, symbol)

        # Rate limit: sleep giữa MỖI request-mã (kể cả khi lỗi), trừ mã cuối cùng.
        if i < n:
            time.sleep(sleep_sec)

    if failed:
        logger.warning("Có %d/%d mã tải THẤT BẠI hoàn toàn: %s", len(failed), n, failed)

    if not series_list:
        raise RuntimeError(
            "Không tải được dữ liệu cho bất kỳ mã nào -- có thể mạng bị chặn "
            "hoặc toàn bộ source đều lỗi. Kiểm tra .env và kết nối mạng."
        )

    prices = pd.concat(series_list, axis=1).sort_index()
    return prices


# --------------------------------------------------------------------------
# Bước 3: Làm sạch dữ liệu -- LOGIC THUẦN, KHÔNG cần mạng, dễ unit test
# --------------------------------------------------------------------------

def clean_prices(
    prices: pd.DataFrame,
    start: str,
    end: str,
    max_missing_frac: float = MAX_MISSING_FRAC,
    max_ffill_gap: int = MAX_FFILL_GAP,
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Làm sạch ma trận giá thô: loại phiên cuối tuần giả, loại mã thiếu nhiều /
    niêm yết muộn, ffill lỗ hổng ngắn.

    Quy tắc (theo brief Gate 1, + fix phiên cuối tuần phát hiện sau khi bàn với
    user trong notebook):
    - Cắt về đúng cửa sổ [start, end].
    - LOẠI BỎ mọi hàng rơi vào Thứ 7/Chủ nhật (dayofweek >= 5). Đây là bước
      BẮT BUỘC làm TRƯỚC các bước còn lại: sàn HOSE/HNX/UPCOM không có phiên
      cuối tuần, nhưng dữ liệu thô từ vnstock (nguồn VCI) đôi khi trả về một
      hàng "Chủ nhật ma" mỗi tuần -- giá trị trùng CHÍNH XÁC với phiên (thường
      là Thứ 6) liền trước, tức là hàng LẶP LẠI chứ không phải phiên giao dịch
      thật. Nếu không lọc trước, các hàng lặp này sẽ: (a) làm sai `missing_frac`
      và `first_valid_index` bên dưới (đếm nhầm 1 phiên "có dữ liệu" cho ngày
      không hề tồn tại), và (b) tạo ra return=0.0 giả cho MỌI mã cùng lúc ở
      `compute_returns()`, làm méo cả mu_hat (kéo về 0) lẫn Sigma (khuếch đại
      tương quan chéo giả -- mọi mã "đứng yên" đúng cùng 1 ngày). Xem thảo
      luận trong notebook.ipynb, mục "Hạn chế & Kết luận" để biết cách phát
      hiện lỗi này (kiểm tra dayofweek + so khớp giá trùng phiên liền trước).
    - Loại mã có > `max_missing_frac` phiên thiếu (NaN) trong cửa sổ (tính
      TRƯỚC khi forward-fill, để không "che" các mã quá thiếu dữ liệu; tính
      SAU KHI đã loại phiên cuối tuần, nên mẫu số n_sessions phản ánh đúng số
      phiên giao dịch thật).
    - Loại mã niêm yết muộn hơn ngày bắt đầu cửa sổ: không có dữ liệu hợp lệ
      trong `max_ffill_gap` phiên đầu tiên của cửa sổ (dung sai nhỏ để tránh
      false positive do lệch ngày giao dịch giữa các nguồn).
    - Forward-fill lỗ hổng <= `max_ffill_gap` phiên liên tiếp; lỗ hổng dài
      hơn GIỮ NGUYÊN NaN (không fill bừa) -- các phiên còn NaN sau bước này
      sẽ bị loại khỏi TOÀN BỘ universe ở bước tính return (dropna(how="any")
      trong compute_returns), vì 1 mã thiếu ở 1 ngày sẽ làm mất ngày đó cho
      mọi mã khác. Đây là đánh đổi được brief chấp nhận để đảm bảo returns
      cuối cùng không còn NaN nào.

    Trả về (prices_clean, dropped) với dropped: {symbol: lý do bị loại}.
    """
    prices = prices.sort_index()
    prices = prices.loc[(prices.index >= pd.Timestamp(start)) & (prices.index <= pd.Timestamp(end))]

    # Loại phiên cuối tuần giả TRƯỚC mọi bước khác (xem giải thích ở docstring).
    prices = prices.loc[prices.index.dayofweek < 5]

    n_sessions = len(prices)
    dropped: dict[str, str] = {}
    keep_cols: list[str] = []

    cutoff_pos = min(max_ffill_gap, max(n_sessions - 1, 0))
    first_valid_cutoff = prices.index[cutoff_pos] if n_sessions > 0 else None

    for col in prices.columns:
        s = prices[col]
        missing_frac = float(s.isna().mean())
        first_valid = s.first_valid_index()

        listed_late = first_valid is None or (
            first_valid_cutoff is not None and first_valid > first_valid_cutoff
        )
        if listed_late:
            dropped[col] = (
                f"listed_late (first_valid_index={first_valid}, "
                f"window_start={prices.index[0] if n_sessions else None})"
            )
            continue

        if missing_frac > max_missing_frac:
            dropped[col] = f"too_many_missing ({missing_frac:.1%} > {max_missing_frac:.0%})"
            continue

        keep_cols.append(col)

    clean = prices[keep_cols].copy()
    clean = clean.ffill(limit=max_ffill_gap)

    return clean, dropped


def compute_returns(
    prices: pd.DataFrame,
    holiday_zero_frac: float = HOLIDAY_ZERO_RETURN_FRAC,
) -> pd.DataFrame:
    """Tính simple daily return = close.pct_change(), loại bỏ mọi hàng còn NaN.

    Đây LÀ công thức bắt buộc theo brief -- không đổi sang log-return.

    Lưu ý kỹ thuật: pandas mặc định (deprecated) tự forward-fill giá trước khi
    tính pct_change (fill_method='pad'), điều này sẽ ÂM THẦM che mất các lỗ
    hổng dài > max_ffill_gap mà clean_prices() cố tình để lại dưới dạng NaN.
    Ta truyền fill_method=None để tắt hành vi đó -- NaN trong giá phải LAN ra
    thành NaN trong return, rồi mới bị dropna(how="any") loại bỏ ở bước sau
    (đúng như thiết kế: lỗ hổng dài -> coi là thiếu, không tự fill).

    Lớp phòng vệ bổ sung (Phase 7): sau khi tính return thô, loại thêm bất kỳ
    phiên nào có >= `holiday_zero_frac` (mặc định 90%) số mã (trong số mã CÓ
    dữ liệu ngày đó) có return = 0.0 CHÍNH XÁC. Đây là dấu hiệu của hàng dữ
    liệu bị nguồn LẶP LẠI giá phiên trước (artifact, không phải phiên giao
    dịch thật) -- từng gặp với các "Chủ nhật ma" (đã lọc riêng bằng
    dayofweek>=5 ở clean_prices), nhưng ngày lễ VN rơi vào NGÀY THƯỜNG (Tết,
    30/4, 1/5, 2/9, Giỗ Tổ...) không bị dayofweek bắt được. Lớp lọc này tổng
    quát hơn: phát hiện bằng THỐNG KÊ thay vì liệt kê lịch nghỉ lễ thủ công,
    nên tự động bắt được nếu nguồn dữ liệu tái diễn hành vi lặp-giá ở BẤT KỲ
    ngày nào, không riêng gì ngày lễ đã biết trước.
    """
    raw = prices.pct_change(fill_method=None)

    n_valid = raw.notna().sum(axis=1)
    n_zero = (raw == 0.0).sum(axis=1)
    frac_zero = n_zero / n_valid.replace(0, np.nan)
    suspected_holiday = frac_zero.fillna(0.0) >= holiday_zero_frac

    if suspected_holiday.any():
        bad_dates = raw.index[suspected_holiday]
        logger.warning(
            "Loại %d phiên nghi là artifact 'lặp giá' (>=%.0f%% mã return=0.0): %s",
            len(bad_dates), holiday_zero_frac * 100, [d.date() for d in bad_dates],
        )
        raw = raw.loc[~suspected_holiday]

    return raw.dropna(how="any")


# --------------------------------------------------------------------------
# Bước 4: Cache I/O
# --------------------------------------------------------------------------

def _cache_exists() -> bool:
    return PRICES_PARQUET.exists() and RETURNS_PARQUET.exists()


def load_from_cache() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Đọc prices/returns từ parquet cache. KHÔNG gọi mạng."""
    prices = pd.read_parquet(PRICES_PARQUET)
    returns = pd.read_parquet(RETURNS_PARQUET)
    return prices, returns


def save_to_cache(symbols: Sequence[str], prices: pd.DataFrame, returns: pd.DataFrame) -> None:
    """Ghi symbols/prices/returns ra data/ để các lần chạy sau + phase sau dùng lại."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    pd.Series(sorted(symbols), name="symbol").to_csv(SYMBOLS_CSV, index=False)
    prices.to_parquet(PRICES_PARQUET)
    returns.to_parquet(RETURNS_PARQUET)


# --------------------------------------------------------------------------
# Benchmark "mua & giữ chỉ số VN30" (method D trong so sánh 4 phương pháp,
# xem docs/superpowers/plans/... hoặc trao đổi trực tiếp với user) -- tải
# CHÍNH chỉ số VN30 (symbol="VN30", KHÔNG phải rổ 30 mã thành viên), nguồn
# VCI xác nhận hoạt động thủ công 2026-08-10 (24 phiên trả về đúng OHLCV cho
# start='2026-07-01', end='2026-08-01').
# --------------------------------------------------------------------------

def load_vn30_index(force_refresh: bool = False) -> tuple[pd.Series, pd.Series]:
    """Tải + làm sạch giá đóng cửa chỉ số VN30, dùng làm benchmark "mua & giữ
    index" (method D). Tái dùng `clean_prices`/`compute_returns` (đã test kỹ
    trên dữ liệu cổ phiếu VN100) trên DataFrame 1 cột thay vì viết lại logic
    làm sạch riêng -- cùng lớp phòng vệ loại phiên cuối tuần giả và phiên
    "lặp giá" (ngày lễ VN rơi vào ngày thường) áp dụng luôn cho chỉ số, vì đây
    là quirk của NGUỒN DỮ LIỆU (VCI), không riêng gì cổ phiếu lẻ.

    Cache: data/vn30_index.parquet (2 cột: close đã làm sạch, ret = simple
    daily return). Đọc thẳng từ cache nếu tồn tại và force_refresh=False.

    Returns
    -------
    (price, ret) : tuple[pd.Series, pd.Series]
        price : giá đóng cửa VN30 đã làm sạch, index=date.
        ret   : simple daily return (pct_change), index=date -- ngắn hơn
                price đúng 1 phiên đầu tiên (NaN đã bị loại bởi compute_returns).
    """
    if not force_refresh and VN30_INDEX_PARQUET.exists():
        df = pd.read_parquet(VN30_INDEX_PARQUET)
        return df["close"], df["ret"].dropna()

    if load_dotenv is not None:
        load_dotenv(PROJECT_ROOT / ".env")  # nạp VNSTOCK_API_KEY, KHÔNG in ra log

    start, end = _default_date_window()
    logger.info("Đang tải chỉ số %s (nguồn VCI, cửa sổ %s -> %s)...", VN30_INDEX_SYMBOL, start, end)
    series, used_source = fetch_history_one_symbol(VN30_INDEX_SYMBOL, start, end, sources=("VCI",))
    if series is None:
        raise RuntimeError(f"Không tải được chỉ số {VN30_INDEX_SYMBOL} từ VCI.")

    raw = series.to_frame(name=VN30_INDEX_SYMBOL)
    clean, dropped = clean_prices(raw, start, end)
    if VN30_INDEX_SYMBOL not in clean.columns:
        raise RuntimeError(f"clean_prices() loại luôn chỉ số {VN30_INDEX_SYMBOL}: {dropped}")
    ret_df = compute_returns(clean)

    price = clean[VN30_INDEX_SYMBOL]
    ret = ret_df[VN30_INDEX_SYMBOL]

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out = price.to_frame(name="close").join(ret.rename("ret"), how="left")
    out.to_parquet(VN30_INDEX_PARQUET)

    logger.info(
        "Chỉ số %s: %d phiên (%s -> %s), nguồn=%s.",
        VN30_INDEX_SYMBOL, len(price), price.index.min().date(), price.index.max().date(), used_source,
    )
    return price, ret


# --------------------------------------------------------------------------
# Phase 1b: repair_universe() -- tải lại có mục tiêu các mã bị loại sai
# --------------------------------------------------------------------------

def repair_universe(
    missing_symbols: Sequence[str] | None = None,
    sleep_sec: float = 1.5,
    vci_retry_backoffs: Sequence[float] = VCI_RETRY_BACKOFFS,
    max_missing_frac: float = MAX_MISSING_FRAC,
    max_ffill_gap: int = MAX_FFILL_GAP,
    max_bad_allowed: int = 10,
) -> dict:
    """Phase 1b fix: tải lại CÓ MỤC TIÊU các mã đã bị loại sai khỏi universe
    do fallback MSN cắt cụt lịch sử (~365 phiên) ở lần chạy trước.

    KHÔNG tải lại 75 mã full-history đang có trong `data/prices.parquet` --
    CHỈ tải các mã còn thiếu so với universe VN100 hiện tại, và CHỈ từ VCI
    (không dùng MSN/KBS cho bước repair này -- brief yêu cầu VCI ONLY với
    retry kiên nhẫn, vì nguyên nhân gốc là VCI timeout tạm thời chứ không
    phải thiếu dữ liệu).

    Với mỗi mã thiếu: gọi `fetch_history_one_symbol(..., sources=("VCI",))`
    (tự động retry VCI 3 lần theo `vci_retry_backoffs`). Sau khi có series,
    kiểm tra coverage: nếu `series.index.min()` trễ hơn `start + 30 ngày`
    thì coi là "chưa đủ lịch sử" (VCI trả về nhưng bị cắt cụt hoặc mã thật
    sự niêm yết muộn gần sát cửa sổ) -- KHÔNG merge cột đó vào ma trận giá
    (dù vậy `clean_prices()` ở bước sau vẫn là bộ lọc CUỐI CÙNG, với dung
    sai chặt hơn nhiều -- chỉ 3 phiên đầu cửa sổ -- nên mã niêm yết muộn
    thật như DSE/VPL vẫn sẽ bị loại đúng cách kể cả nếu lọt qua bước này).

    Nếu tổng số mã "thất bại hoàn toàn" (không tải được) CỘNG "không đủ
    coverage" vượt quá `max_bad_allowed` (mặc định 10/25 theo brief), hàm
    ném RuntimeError để caller báo BLOCKED -- KHÔNG tự ý fallback sang MSN.

    Ghép cột mới vào `prices` cũ (outer join theo ngày), chạy lại
    `clean_prices` + `compute_returns` trên TOÀN BỘ universe hợp nhất, rồi
    ghi đè cache (`save_to_cache`).

    Trả về dict tổng kết đầy đủ (universe cũ/mới, mã thành công/thất bại/
    insufficient, shape trước/sau, lý do bị loại ở clean_prices) để dùng
    viết report -- không có secret nào bị log/in ra trong hàm này.
    """
    if load_dotenv is not None:
        load_dotenv(PROJECT_ROOT / ".env")  # nạp VNSTOCK_API_KEY, KHÔNG in ra log

    if not _cache_exists():
        raise RuntimeError(
            "repair_universe() cần cache hiện có (data/prices.parquet + "
            "returns.parquet) để giữ nguyên 75 mã full-history -- không tìm thấy."
        )

    old_prices, _old_returns = load_from_cache()
    start, end = _default_date_window()

    if missing_symbols is None:
        all_symbols = load_vn100_symbols()
        missing_symbols = sorted(set(all_symbols) - set(old_prices.columns))
    else:
        missing_symbols = sorted(missing_symbols)

    n = len(missing_symbols)
    logger.info(
        "repair_universe(): %d mã cần tải lại từ VCI ONLY (cửa sổ %s -> %s): %s",
        n, start, end, missing_symbols,
    )

    coverage_cutoff = pd.Timestamp(start) + pd.Timedelta(days=30)

    fetched: dict[str, pd.Series] = {}
    insufficient: dict[str, str] = {}
    fetch_failed: list[str] = []

    for i, symbol in enumerate(missing_symbols, start=1):
        series, used_source = fetch_history_one_symbol(
            symbol, start, end, sources=("VCI",), vci_retry_backoffs=vci_retry_backoffs,
        )
        if series is None:
            fetch_failed.append(symbol)
            logger.warning("[%d/%d] %s: FAILED (VCI only, hết retry)", i, n, symbol)
        else:
            first_valid = series.index.min()
            if pd.isna(first_valid) or first_valid > coverage_cutoff:
                reason = f"first_valid={first_valid} > cutoff={coverage_cutoff.date()} (rows={len(series)})"
                insufficient[symbol] = reason
                logger.warning("[%d/%d] %s: coverage KHÔNG đủ (%s) -- bỏ qua merge", i, n, symbol, reason)
            else:
                fetched[symbol] = series
                logger.info(
                    "[%d/%d] %s: ok (source=%s, rows=%d, first_valid=%s)",
                    i, n, symbol, used_source, len(series), first_valid.date(),
                )

        if i < n:
            time.sleep(sleep_sec)

    n_bad = len(fetch_failed) + len(insufficient)
    if n_bad > max_bad_allowed:
        raise RuntimeError(
            f"BLOCKED: {n_bad}/{n} mã KHÔNG tải được đủ history từ VCI sau retry "
            f"(fetch_failed={fetch_failed}, insufficient_coverage={list(insufficient)}). "
            "KHÔNG fallback sang MSN cho bước repair -- dừng theo brief, cần xem log."
        )

    if fetched:
        new_cols = pd.concat(list(fetched.values()), axis=1)
        combined = pd.concat([old_prices, new_cols], axis=1, join="outer").sort_index()
    else:
        combined = old_prices.copy()

    clean, dropped = clean_prices(
        combined, start, end, max_missing_frac=max_missing_frac, max_ffill_gap=max_ffill_gap,
    )
    returns = compute_returns(clean)

    save_to_cache(list(clean.columns), clean, returns)

    report = {
        "start": start,
        "end": end,
        "old_universe": sorted(old_prices.columns),
        "old_universe_n": old_prices.shape[1],
        "requested_missing": list(missing_symbols),
        "fetched_ok": sorted(fetched.keys()),
        "insufficient_coverage": insufficient,
        "fetch_failed": fetch_failed,
        "combined_raw_shape": combined.shape,
        "clean_shape": clean.shape,
        "returns_shape": returns.shape,
        "dropped_in_clean": dropped,
        "final_universe": sorted(clean.columns),
        "final_universe_n": clean.shape[1],
        "added_vs_old": sorted(set(clean.columns) - set(old_prices.columns)),
    }

    logger.info("=== repair_universe() TỔNG KẾT ===")
    logger.info("Universe cũ: %d mã. Universe mới (sau clean): %d mã.", report["old_universe_n"], report["final_universe_n"])
    logger.info("Mã thêm mới thành công vào universe: %s", report["added_vs_old"])
    logger.info("Mã fetch thất bại hoàn toàn: %s", fetch_failed)
    logger.info("Mã coverage không đủ (bị bỏ qua trước merge): %s", list(insufficient))
    logger.info("Mã bị loại ở clean_prices (toàn universe hợp nhất): %s", dropped)
    logger.info("Shape prices cuối: %s, returns cuối: %s", clean.shape, returns.shape)

    return report


# --------------------------------------------------------------------------
# main()
# --------------------------------------------------------------------------

def main(
    force_refresh: bool | None = None,
    sleep_sec: float = DEFAULT_SLEEP_SEC,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Entry point: đọc cache nếu có, ngược lại tải + làm sạch + cache mới.

    force_refresh: nếu None, lấy từ biến môi trường FORCE_REFRESH (1/true/yes
    -> True). Đặt True để ép tải lại kể cả khi đã có cache.
    """
    if load_dotenv is not None:
        load_dotenv(PROJECT_ROOT / ".env")  # nạp VNSTOCK_API_KEY, KHÔNG in ra log

    if force_refresh is None:
        force_refresh = os.environ.get("FORCE_REFRESH", "").strip().lower() in ("1", "true", "yes")

    t0 = time.monotonic()

    if not force_refresh and _cache_exists():
        prices, returns = load_from_cache()
        elapsed = time.monotonic() - t0
        logger.info(
            "loaded from cache: prices=%s returns=%s trong %.3fs (KHÔNG gọi mạng)",
            prices.shape, returns.shape, elapsed,
        )
        return prices, returns

    if not os.environ.get("VNSTOCK_API_KEY"):
        logger.warning(
            "Không tìm thấy VNSTOCK_API_KEY trong biến môi trường -- kiểm tra "
            "file .env nếu request thất bại (một số source có thể cần key)."
        )

    start, end = _default_date_window()
    logger.info("Cửa sổ dữ liệu: %s -> %s", start, end)

    symbols = load_vn100_symbols()
    raw_prices = fetch_all_prices(symbols, start, end, sleep_sec=sleep_sec)

    clean, dropped = clean_prices(raw_prices, start, end)
    returns = compute_returns(clean)

    # % missing tổng thể của các mã ĐƯỢC GIỮ LẠI, tính TRƯỚC forward-fill.
    overall_missing_frac = raw_prices.reindex(columns=clean.columns).isna().mean().mean()

    logger.info("=== TỔNG KẾT ===")
    logger.info("Số mã tải được (trước làm sạch): %d / %d", raw_prices.shape[1], len(symbols))
    logger.info("Khoảng ngày (sau làm sạch): %s -> %s", clean.index.min(), clean.index.max())
    logger.info("%% missing tổng thể (mã giữ lại, trước ffill): %.2f%%", overall_missing_frac * 100)
    logger.info("Số mã bị loại: %d", len(dropped))
    for sym, reason in dropped.items():
        logger.info("  - %s: %s", sym, reason)
    logger.info("Shape cuối cùng prices (T, N): %s", clean.shape)
    logger.info("Shape cuối cùng returns (T, N): %s", returns.shape)

    save_to_cache(list(clean.columns), clean, returns)

    elapsed = time.monotonic() - t0
    logger.info("Hoàn tất tải + làm sạch + cache trong %.1fs", elapsed)

    return clean, returns


if __name__ == "__main__":
    main()
