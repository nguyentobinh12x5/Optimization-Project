# Task 1 Report — Phase 1: Data Layer (vnstock)

Status: **DONE_WITH_CONCERNS**

## Files đã tạo/sửa

- `src/__init__.py` (mới, rỗng) — cho phép `python -m src.data_loader`.
- `src/data_loader.py` (mới) — module data layer đầy đủ: `load_vn100_symbols`, `fetch_history_one_symbol`, `fetch_all_prices`, `clean_prices`, `compute_returns`, `load_from_cache`, `save_to_cache`, `main`.
- `.gitignore` — thêm dòng `data/` (data không commit).
- `data/vn100_symbols.csv`, `data/prices.parquet`, `data/returns.parquet` — output cache thật, sinh ra từ lần chạy thật (không fabricate).
- Môi trường: cài thêm `pyarrow` và `python-dotenv` vào `.venv` hiện có (KHÔNG tạo venv mới) — cần thiết để ghi parquet và nạp `.env` an toàn; trước đó `.venv` chưa có 2 package này.

## Quyết định / giả định đã đưa ra (ngoài các giá trị verbatim trong brief)

1. **Merge giá bằng UNION (outer join)** các ngày của tất cả mã, không phải intersection thuần. Lý do: nếu 1 mã bị đình chỉ giao dịch 1 ngày, vnstock không trả row cho ngày đó luôn (không trả NaN), nên outer-join là cách duy nhất để "phiên thiếu" của 1 mã hiện ra thành NaN (cần thiết cho rule `> 10% phiên thiếu`). Đã ghi rõ trong docstring module.
2. **`pct_change(fill_method=None)`** thay vì `pct_change()` mặc định. Phát hiện qua unit test cục bộ: pandas mặc định (deprecated) tự forward-fill NaN trước khi tính % thay đổi, việc này ÂM THẦM che mất các lỗ hổng dài > 3 phiên mà `clean_prices()` cố tình để lại làm NaN — vi phạm đúng tinh thần "lỗ hổng dài → coi là thiếu, không fill bừa" của brief. Sửa bằng `fill_method=None` để NaN lan đúng ra return rồi mới bị `dropna(how="any")` loại.
3. **Rule "niêm yết muộn"**: mã bị coi là niêm yết muộn nếu `first_valid_index` của nó nằm sau vị trí phiên thứ `max_ffill_gap` (=3) của cửa sổ, thay vì so đúng bằng ngày đầu cửa sổ — để có dung sai nhỏ tránh false positive do lệch ngày giao dịch giữa các nguồn.
4. Đã verify thủ công (không đoán) chữ ký/cột trả về của `Listing.symbols_by_group` và `Quote.history` trước khi code — khớp đúng với brief.

## Test logic thuần (không cần mạng)

Viết script test riêng (`/private/tmp/.../scratchpad/test_clean.py`, không phải file trong repo) dùng dữ liệu giả lập 5 mã (A: đủ dữ liệu, B: lỗ hổng ngắn 2 phiên, C: thiếu >10% rải rác, D: niêm yết muộn 10 phiên đầu, E: lỗ hổng dài 4 phiên > max_ffill_gap):
- C bị loại đúng lý do `too_many_missing`.
- D bị loại đúng lý do `listed_late`.
- A, B được giữ; B được ffill hết lỗ hổng ngắn.
- E được giữ (missing_frac ở ngưỡng) nhưng còn 1 NaN sau ffill(limit=3) → `returns` giảm đúng số hàng tương ứng, KHÔNG còn NaN nào.
- Toàn bộ assertion PASS. Kết quả này giúp phát hiện bug `pct_change` mặc định ở mục 2 phía trên trước khi chạy thật.

## Output THẬT — Lần chạy 1 (tải từ API)

Lệnh: `set -a && source .env && set +a && time .venv/bin/python -m src.data_loader`
Chạy nền do vượt 120s (rate-limit sleep khiến quá trình mất nhiều thời gian).

Tóm tắt số liệu (log đầy đủ ~340 dòng, trích phần quan trọng):

```
19:26:53 [INFO] === TỔNG KẾT ===
19:26:53 [INFO] Số mã tải được (trước làm sạch): 100 / 100
19:26:53 [INFO] Khoảng ngày (sau làm sạch): 2022-07-25 00:00:00 -> 2026-07-24 00:00:00
19:26:53 [INFO] % missing tổng thể (mã giữ lại, trước ffill): 8.46%
19:26:53 [INFO] Số mã bị loại: 25
19:26:53 [INFO]   - DSE: listed_late (first_valid_index=2024-07-01, window_start=2022-07-25)
19:26:53 [INFO]   - GEE: too_many_missing (10.1% > 10%)
19:26:53 [INFO]   - GVR: listed_late (first_valid_index=2025-02-07, window_start=2022-07-25)
19:26:53 [INFO]   - PNJ: listed_late (first_valid_index=2025-02-07, window_start=2022-07-25)
19:26:53 [INFO]   - POW: listed_late (first_valid_index=2025-02-11, window_start=2022-07-25)
19:26:53 [INFO]   - PVD: listed_late (first_valid_index=2025-02-07, window_start=2022-07-25)
19:26:53 [INFO]   - REE: listed_late (first_valid_index=2024-06-24, window_start=2022-07-25)
19:26:53 [INFO]   - SCS: listed_late (first_valid_index=2025-02-07, window_start=2022-07-25)
19:26:53 [INFO]   - SHB: listed_late (first_valid_index=2025-02-07, window_start=2022-07-25)
19:26:53 [INFO]   - SIP: listed_late (first_valid_index=2025-02-18, window_start=2022-07-25)
19:26:53 [INFO]   - SJS: listed_late (first_valid_index=2025-01-31, window_start=2022-07-25)
19:26:53 [INFO]   - SZC: listed_late (first_valid_index=2025-02-07, window_start=2022-07-25)
19:26:53 [INFO]   - TCB: listed_late (first_valid_index=2025-02-07, window_start=2022-07-25)
19:26:53 [INFO]   - VCG: listed_late (first_valid_index=2025-02-07, window_start=2022-07-25)
19:26:53 [INFO]   - VGC: listed_late (first_valid_index=2025-02-07, window_start=2022-07-25)
19:26:53 [INFO]   - VHC: listed_late (first_valid_index=2025-02-07, window_start=2022-07-25)
19:26:53 [INFO]   - VHM: listed_late (first_valid_index=2025-02-10, window_start=2022-07-25)
19:26:53 [INFO]   - VIB: listed_late (first_valid_index=2025-02-07, window_start=2022-07-25)
19:26:53 [INFO]   - VIC: listed_late (first_valid_index=2025-02-12, window_start=2022-07-25)
19:26:53 [INFO]   - VIX: listed_late (first_valid_index=2025-02-07, window_start=2022-07-25)
19:26:53 [INFO]   - VNM: listed_late (first_valid_index=2025-02-07, window_start=2022-07-25)
19:26:53 [INFO]   - VPL: listed_late (first_valid_index=2025-05-13, window_start=2022-07-25)
19:26:53 [INFO]   - VRE: listed_late (first_valid_index=2025-02-07, window_start=2022-07-25)
19:26:53 [INFO]   - VSC: listed_late (first_valid_index=2025-02-07, window_start=2022-07-25)
19:26:53 [INFO]   - VTP: listed_late (first_valid_index=2025-02-07, window_start=2022-07-25)
19:26:53 [INFO] Shape cuối cùng prices (T, N): (1090, 75)
19:26:53 [INFO] Shape cuối cùng returns (T, N): (1069, 75)
19:26:56 [INFO] Hoàn tất tải + làm sạch + cache trong 3116.1s
```

Phân bổ nguồn dữ liệu thành công (đếm từ log, không mã nào fail hoàn toàn):
- VCI: 77/100 mã (đủ ~998 phiên, tức đủ ~4 năm).
- MSN (fallback): 23/100 mã (chỉ ~365 phiên).
- KBS: 0/100 (không cần dùng tới).
- FAILED hoàn toàn: 0/100.

Verify file cache bằng code:
```
prices shape (1090, 75), index monotonic increasing: True, dtype: datetime64[ns]
prices index min/max: 2022-07-25 -> 2026-07-24
returns shape (1069, 75), any NaN: False
```

## Output THẬT — Lần chạy 2 (đọc cache)

```
19:28:10 [INFO] loaded from cache: prices=(1090, 75) returns=(1069, 75) trong 0.074s (KHÔNG gọi mạng)
.venv/bin/python -m src.data_loader  0.52s user 0.10s system 97% cpu 0.638 total
```
Thời gian đọc cache nội bộ: 0.074s (< 1s theo yêu cầu). Tổng thời gian process (bao gồm khởi động Python + import vnstock): 0.638s. Không có log request mạng nào (không có dòng `Listing`/`Quote`/`API request`).

## Acceptance criteria — soát lại

1. ✅ Lần đầu in đủ: số mã tải được, khoảng ngày, % missing, số mã bị loại + lý do, shape cuối.
2. ✅ Universe sạch = 75 mã ≥ 70 (nhưng biên độ an toàn khá mỏng — xem Concerns).
3. ✅ `prices.parquet`/`returns.parquet` tạo được; `returns` không NaN; `prices.index` datetime tăng dần.
4. ✅ Lần 2 đọc cache, 0.074s < 1s, không gọi mạng.
5. ✅ Code có docstring/comment tiếng Việt giải thích rate-limit + các bước làm sạch, viết cho dân tối ưu hoá.

## Concerns (QUAN TRỌNG — đọc trước khi dùng ở phase sau)

1. **Thời gian chạy thật lâu hơn "vài phút" nhiều: ~52 phút (3116s)**, không phải do lỗi code mà do source VCI (`trading.vietcap.com.vn`) bị timeout (đọc 30s) rất thường xuyên trong suốt phiên chạy (không phải chỉ lúc đầu) — cơ chế fallback + backoff hoạt động ĐÚNG như thiết kế (không mã nào fail hẳn), nhưng mỗi lần fallback tốn thêm ~30-90s chờ timeout/backoff trước khi chuyển sang MSN. Đây là tình trạng mạng/server thật tại thời điểm chạy, không giả lập.

2. **Phát hiện quan trọng nhất: MSN (fallback source) chỉ trả về ~365 phiên gần nhất bất kể `start` truyền vào là gì**, đã verify độc lập bằng 1 request riêng: `Quote(source='MSN', symbol='TCB').history(start='2022-07-25', end='2026-07-25')` trả về đúng 365 dòng, từ 2025-02-07 đến 2026-07-24 (tức luôn là ~1 năm gần nhất, không phải từ ngày niêm yết). Hệ quả: 23 mã phải fallback qua MSN (do VCI timeout) chỉ có ~365 phiên, khiến `clean_prices()` gắn nhãn `listed_late` cho chúng — **nhưng phần lớn trong số này KHÔNG thực sự niêm yết muộn**, mà là các blue-chip lâu đời của VN30/VN100 (TCB, VHM, VIC, VNM, VRE, VIB, VCG, VGC, VHC, VIX, REE, SHB, POW, PVD, PNJ, GVR, SCS, SIP, SJS, SZC, VSC, VTP, VPL — 23/25 mã bị loại rơi vào nhóm này, chỉ GEE và DSE bị loại vì lý do hợp lệ thật sự: GEE thiếu >10% phiên rải rác, DSE thực sự niêm yết 2024).
   - Nói cách khác: nhãn "listed_late" trong log/report cho 23 mã này là **artifact của giới hạn dữ liệu nguồn MSN**, không phải sự thật tài chính. Universe 75 mã hiện tại **thiếu nhiều mã vốn hóa lớn quan trọng** (VIC, VHM, VNM, VCB-group banks như TCB, VIB...) — điều này có thể ảnh hưởng đáng kể đến tính đại diện của universe cho các phase ước lượng μ̂/Σ và tối ưu hoá sau này.
   - Đã KHÔNG chạy lại toàn bộ pipeline để khắc phục (chi phí thời gian ~52 phút/lần, kết quả không chắc cải thiện vì VCI có vẻ không ổn định trong suốt phiên). Đề xuất fix cho lần chạy sau (không nằm trong scope code đã nộp): thêm retry lại CHÍNH source VCI (không chuyển ngay sang MSN) 1-2 lần với backoff dài hơn trước khi chấp nhận fallback; hoặc validate số phiên fallback trả về, nếu quá ít so với kỳ vọng (`rows < 90% * expected_sessions`) thì thử lại VCI thêm lần nữa thay vì chấp nhận luôn.
   - Nếu phase sau (ước lượng μ̂/Σ) cần universe đại diện tốt hơn, khuyến nghị **chạy lại `FORCE_REFRESH=1 python -m src.data_loader` vào lúc VCI ổn định hơn** (ví dụ giờ thấp điểm) trước khi dùng dữ liệu cho các bước quan trọng.

3. `VNSTOCK_API_KEY` trong `.env` không rõ có thực sự được vnstock free-tier endpoints (Listing/Quote) sử dụng hay không — code chỉ nạp qua dotenv và cảnh báo nếu thiếu, không ép buộc; các request vẫn chạy được bình thường không cần key (đúng như brief mô tả free tier).

4. Giả định "`close` của VCI là giá đã điều chỉnh" (theo yêu cầu brief) chưa được verify thủ công với 1 mã có sự kiện chia tách/cổ tức thực tế trong 4 năm — nằm ngoài phạm vi thời gian cho phép của task này, ghi lại như giả định chưa kiểm chứng 100%.

## Tóm tắt 1 dòng cho controller

Chạy thật thành công: 100/100 mã VN100 tải được (77 qua VCI, 23 fallback MSN), 75/100 mã sạch sau lọc (≥70 mã, đạt), shape prices=(1090,75) returns=(1069,75) không NaN, mất 3116s (~52 phút, lâu do VCI timeout liên tục); cache lần 2 = 0.074s không gọi mạng — đạt yêu cầu; NHƯNG phát hiện 23/25 mã bị loại là do MSN giới hạn ~365 phiên (artifact nguồn dữ liệu) chứ không thực sự niêm yết muộn, làm universe thiếu nhiều blue-chip lớn (VIC, VHM, VNM, TCB...).
