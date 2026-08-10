"""
src/backtest.py
=================
Walk-forward backtest out-of-sample cho project sparse+robust portfolio
optimization (VN100). Xem thiết kế đầy đủ ở
docs/superpowers/specs/2026-07-26-walk-forward-backtest-design.md.

SOLVER DÙNG TRONG MODULE NÀY: CVXPY (`src.cvxpy_check.cvxpy_solve` /
`cvxpy_solve_long_only`), KHÔNG PHẢI solver proximal-subgradient tự viết
(`src.prox_solver.solve` / `solve_long_only`) -- quyết định có chủ đích
(trao đổi với user 2026-08-10): các hàm walk-forward ở đây gọi grid search
hàng chục tổ hợp tham số MỖI kỳ rebalance x hàng chục kỳ, nên tốc độ per-
solve quan trọng hơn việc "tự viết" (vốn chỉ là yêu cầu cho phần core report
§1-6, KHÔNG áp dụng cho phần backtest so sánh A-F này). CVXPY (CLARABEL) đo
được nhanh hơn ~4x so với solver tay trên cùng bài toán 98 mã (0.03s vs
0.14s/lần giải) -- với lưới 60 tổ hợp x 25 kỳ x 2 lần giải/kỳ (validation +
refit) của `walk_forward_backtest_long_short_full`, chênh lệch này là
~13 phút (solver tay) so với ~3 phút (CVXPY). `src/prox_solver.py` VẪN LÀ
solver chính thức của core report (§1-6, in-sample, joint-prox, cross-verify
CVXPY) -- module này không thay thế nó, chỉ dùng CVXPY cho riêng phần
backtest walk-forward.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

ANNUALIZATION_FACTOR = 252

# Ngưỡng coi std ngày là "bằng 0" (return hằng số). pandas.Series.std trên
# chuỗi hằng số không luôn cho đúng 0.0 tuyệt đối (sai số làm tròn floating
# point cỡ 1e-19 trong thuật toán tổng bình phương độ lệch) -> so sánh
# `> 0` thuần không đủ an toàn, cần ngưỡng epsilon.
STD_ZERO_EPS = 1e-12


def performance_metrics(daily_returns: pd.Series, rf: float = 0.0) -> dict:
    """Tính các chỉ số hiệu suất chuẩn từ chuỗi return NGÀY của 1 danh mục.

    rf: lãi suất phi rủi ro NĂM (annualized), mặc định 0 theo thiết kế đã
    chốt. Sharpe = (mean(excess_daily)) / std(daily) * sqrt(252); nếu
    std(daily) xấp xỉ 0 (dưới ngưỡng STD_ZERO_EPS; return hằng số, vd
    chuỗi giả lập trong test), Sharpe KHÔNG xác định -> trả NaN thay vì
    chia cho một số cực nhỏ (pandas.Series.std trên chuỗi hằng số có thể
    ra ~1e-19 do sai số làm tròn, không đúng 0.0 tuyệt đối).

    max_drawdown: mức sụt giảm lớn nhất từ đỉnh giá trị danh mục tích luỹ
    (giá trị ÂM hoặc 0, không phải giá trị dương).

    CẢNH BÁO "VỠ NỢ" trong annualized_return: nếu `cumulative_return <= -1`
    (mất sạch vốn hoặc hơn -- có thể xảy ra với danh mục long-short đòn bẩy
    cao, vd CVXPY tìm nghiệm tối ưu chính xác với gross exposure rất lớn
    trên 1 cửa sổ validation ngắn), công thức CAGR chuẩn
    `(1+cumulative_return)**(1/n_years)` nhận cơ số <= 0 với số mũ phân số
    -> ra SỐ PHỨC trong Python, không phải lỗi làm tròn mà lỗi toán học thật
    (căn bậc lẻ/phân số của số âm không xác định trên trục thực). Xử lý:
    chặn annualized_return ở đúng -1.0 (-100%, cùng quy ước "vỡ nợ" với
    `_simulate_period`) thay vì để `float()` ném TypeError khi ép số phức.
    """
    daily_returns = daily_returns.dropna()
    n = len(daily_returns)
    cumulative_return = float((1 + daily_returns).prod() - 1)

    n_years = n / ANNUALIZATION_FACTOR
    if n_years > 0:
        base = 1 + cumulative_return
        annualized_return = -1.0 if base <= 0 else float(base ** (1 / n_years) - 1)
    else:
        annualized_return = float("nan")

    daily_std = float(daily_returns.std(ddof=1)) if n > 1 else 0.0
    annualized_vol = daily_std * np.sqrt(ANNUALIZATION_FACTOR)

    if daily_std > STD_ZERO_EPS:
        rf_daily = rf / ANNUALIZATION_FACTOR
        sharpe = float(
            (daily_returns.mean() - rf_daily) / daily_std * np.sqrt(ANNUALIZATION_FACTOR)
        )
    else:
        sharpe = float("nan")

    equity_curve = (1 + daily_returns).cumprod()
    running_max = equity_curve.cummax()
    drawdown = equity_curve / running_max - 1
    max_drawdown = float(drawdown.min()) if n > 0 else float("nan")

    return {
        "cumulative_return": cumulative_return,
        "annualized_return": annualized_return,
        "annualized_vol": annualized_vol,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "n_days": n,
    }


def _build_rebalance_windows(
    index: pd.DatetimeIndex, lookback_months: int, validation_months: int,
) -> list[dict]:
    """Với mỗi tháng lịch (period 'M') đủ `lookback_months` tháng dữ liệu
    phía trước, trả về 1 dict mô tả kỳ rebalance:
      - rebalance_date: ngày giao dịch ĐẦU TIÊN của tháng target (Timestamp)
      - e_mask, v_mask, target_mask: boolean array cùng độ dài `index`

    E (estimation) = (lookback_months - validation_months) tháng đầu của cửa
    sổ lookback_months tháng ngay trước tháng target. Vwin (validation) =
    validation_months tháng cuối của cửa sổ đó. Cả E và Vwin đều nằm HOÀN
    TOÀN trước rebalance_date -- đây là bất biến chống look-ahead, có test
    riêng khẳng định (`test_build_windows_no_look_ahead`): mọi tháng trong
    window (E union Vwin) đứng TRƯỚC target_month trong danh sách tháng lịch
    sắp xếp, nên mọi ngày giao dịch của E/Vwin nghiêm ngặt < rebalance_date
    (ngày đầu tiên của target_month).
    """
    periods = index.to_period("M")
    months = sorted(periods.unique())
    windows: list[dict] = []

    for i in range(lookback_months, len(months)):
        target_month = months[i]
        window_months = months[i - lookback_months : i]
        n_e = lookback_months - validation_months
        e_months = set(window_months[:n_e])
        v_months = set(window_months[n_e:])

        target_mask = periods == target_month
        if not target_mask.any():
            continue

        windows.append({
            "rebalance_date": index[target_mask][0],
            "e_mask": periods.isin(e_months),
            "v_mask": periods.isin(v_months),
            "target_mask": target_mask,
        })

    return windows


def _simulate_period(
    w_start: np.ndarray,
    period_returns: pd.DataFrame,
    prev_weights_drifted: np.ndarray | None,
    fee: float,
) -> tuple[pd.Series, float, float, np.ndarray, bool]:
    """Mô phỏng 1 kỳ nắm giữ (thường 1 tháng) với trọng số CỐ ĐỊNH lúc đầu kỳ
    `w_start`, để trọng số TRÔI theo giá từng ngày (mua-và-giữ, không giao
    dịch giữa kỳ). Turnover tính so với trọng số ĐÃ TRÔI cuối kỳ TRƯỚC
    (`prev_weights_drifted`), vì đó là trạng thái danh mục THỰC TẾ ngay
    trước khi rebalance -- không phải trọng số MỤC TIÊU của kỳ trước.

    Trả về (daily_net_returns, turnover, cost, w_end_drifted). Phí `cost`
    bị trừ vào return của NGÀY ĐẦU TIÊN trong kỳ (ngày rebalance). Dùng
    chung cho walk-forward VÀ baseline equal-weight (Task 5).

    CẢNH BÁO "VỠ NỢ" (bug fix): công thức trôi "w = w*(1+r); w = w/w.sum()"
    giả định giá trị danh mục sau mỗi ngày LUÔN DƯƠNG (đúng cho long-only,
    vì mọi w_i>=0 và return từng mã > -100% -> tổng luôn >0). Với danh mục
    long-short đòn bẩy cao (`walk_forward_backtest_long_short(_full)`,
    gross exposure = sum(|w|) có thể vượt xa 1, không có trần), 1 ngày xấu
    có thể khiến port_ret <= -100% -- nếu KHÔNG xử lý riêng, w.sum() sau khi
    nhân sẽ <=0, chia cho số đó sẽ LẬT DẤU toàn bộ trọng số một cách vô
    nghĩa (đã tái tạo bằng số cụ thể, xem thảo luận với user 2026-08-10).
    Xử lý: khi phát hiện port_ret <= -100% trong 1 ngày, coi đó là "vỡ nợ"
    (mất toàn bộ vốn) -- CHẶN return ngày đó ở đúng -100% (không cho âm hơn,
    quy ước chuẩn cho equity curve compound để (1+r).cumprod() không xuống
    dưới 0), đặt w về toàn 0 (không còn vị thế), các ngày CÒN LẠI trong kỳ
    trả return=0% (coi như "tiền mặt" tới kỳ rebalance sau). Phát cảnh báo
    rõ ràng (`warnings.warn`) thay vì âm thầm tiếp tục -- caller có thể bắt
    qua cột `wiped_out` trả về để biết kỳ nào bị ảnh hưởng.

    Parameters
    ----------
    w_start : np.ndarray, shape (N,)
        Trọng số MỤC TIÊU ngay sau khi rebalance đầu kỳ.
    period_returns : pd.DataFrame, shape (T_period, N)
        Simple daily returns của kỳ nắm giữ, cùng thứ tự cột với w_start.
    prev_weights_drifted : np.ndarray | None, shape (N,)
        Trọng số ĐÃ TRÔI cuối kỳ trước; None nếu đây là kỳ đầu tiên (turnover
        toàn phần, tương đương coi danh mục trước đó toàn 0).
    fee : float
        Phí giao dịch tỉ lệ (vd 0.002 = 0.20%) nhân với turnover.

    Returns
    -------
    (daily_net, turnover, cost, w_end, wiped_out) : tuple
        daily_net : pd.Series, cùng index với period_returns. Nếu vỡ nợ,
            ngày vỡ nợ = -1.0 CHÍNH XÁC, các ngày sau đó (nếu còn) = 0.0.
        turnover : float, = sum(|w_start - prev_weights_drifted|).
        cost : float, = fee * turnover.
        w_end : np.ndarray, shape (N,), trọng số đã trôi cuối kỳ (sum=1,
            hoặc toàn 0 nếu vỡ nợ trong kỳ).
        wiped_out : bool, True nếu kỳ này có ít nhất 1 ngày vỡ nợ.
    """
    prev = np.zeros_like(w_start) if prev_weights_drifted is None else prev_weights_drifted
    turnover = float(np.abs(w_start - prev).sum())
    cost = fee * turnover

    w = w_start.copy()
    daily_net = []
    wiped_out = False
    for _, day_ret in period_returns.iterrows():
        if wiped_out:
            daily_net.append(0.0)
            continue

        port_ret = float(w @ day_ret.values)
        if port_ret <= -1.0:
            warnings.warn(
                f"_simulate_period: danh mục 'vỡ nợ' (port_ret={port_ret:.4f} <= -100%, "
                f"gross exposure ban đầu={np.abs(w_start).sum():.3f}) -- return ngày này "
                "bị chặn ở -100%, các ngày còn lại trong kỳ coi như return=0 (không thể "
                "'hồi phục' từ vỡ nợ). Xem docstring _simulate_period mục 'CẢNH BÁO VỠ NỢ'.",
                RuntimeWarning,
                stacklevel=2,
            )
            daily_net.append(-1.0)
            w = np.zeros_like(w)
            wiped_out = True
            continue

        daily_net.append(port_ret)
        w = w * (1.0 + day_ret.values)
        w = w / w.sum()

    daily_net = pd.Series(daily_net, index=period_returns.index)
    daily_net.iloc[0] -= cost

    return daily_net, turnover, cost, w, wiped_out


@dataclass
class BacktestResult:
    """Kết quả trả về bởi `walk_forward_backtest`.

    daily_returns : pd.Series
        Return NGÀY net-of-fee của toàn bộ giai đoạn out-of-sample, nối các
        kỳ rebalance liên tiếp lại, sắp xếp theo thời gian.
    rebalance_log : pd.DataFrame
        1 dòng / kỳ rebalance, index = rebalance_date, cột: kappa, gamma,
        turnover, cost, n_active, val_sharpe (Sharpe trên Vwin ứng với tham
        số được chọn).
    weights : pd.DataFrame
        1 dòng / kỳ rebalance, index = rebalance_date, cột = tên tài sản
        (khớp `returns.columns`), giá trị = trọng số MỤC TIÊU triển khai
        ngay sau rebalance (trước khi trôi theo giá trong kỳ).
    """

    daily_returns: pd.Series
    rebalance_log: pd.DataFrame
    weights: pd.DataFrame


DEFAULT_PARAM_GRID = [
    (kappa, gamma)
    for kappa in (0.0, 0.5, 1.0, 2.0)
    for gamma in (1.0, 5.0, 10.0)
]

DEFAULT_PARAM_GRID_LONG_SHORT = [
    (lam, gamma)
    for lam in (0.0, 0.001, 0.005, 0.01, 0.02)
    for gamma in (1.0, 5.0, 10.0)
]


def walk_forward_backtest(
    returns: pd.DataFrame,
    *,
    lookback_months: int = 24,
    validation_months: int = 6,
    param_grid: list[tuple[float, float]] | None = None,
    shrinkage: str | None = "lw",
    fee: float = 0.002,
    rf: float = 0.0,
    selection_metric: str = "sharpe",
    max_weight: float | None = None,
) -> BacktestResult:
    """Walk-forward backtest out-of-sample, long-only, tự chọn (kappa,gamma)
    mỗi kỳ qua validation trên Vwin -- tiêu chí chọn là `selection_metric`
    ("sharpe": Sharpe ratio, MẶC ĐỊNH -- đồng bộ với
    `walk_forward_backtest_long_short(_full)` vốn luôn chọn theo Sharpe;
    "return": cumulative return thô, giữ lại làm tuỳ chọn lịch sử). Xem
    thiết kế đầy đủ trong
    docs/superpowers/specs/2026-07-26-walk-forward-backtest-design.md.

    LƯU Ý PHƯƠNG PHÁP LUẬN (đổi lại mặc định về "sharpe" theo yêu cầu của
    user -- để CẢ 6 phương pháp so sánh trong report dùng CHUNG một tiêu chí
    chọn tham số, không lệch nhau giữa nhánh long-only và long-short):
    bản trước có một giai đoạn đổi mặc định sang "return" thô (không điều
    chỉnh rủi ro) -- dễ chọn trúng cấu hình "may mắn" ngắn hạn trên
    validation 6 tháng hơn Sharpe (vốn cân bằng return với biến động), như
    minh hoạ trong `test_walk_forward_backtest_selects_max_return_when_requested`.
    Tuỳ chọn "return" vẫn giữ nguyên qua `selection_metric="return"` cho ai
    cần tái lập kết quả cũ, nhưng KHÔNG còn là mặc định.

    SOLVER: `cvxpy_solve_long_only` (CVXPY/CLARABEL) -- xem LƯU Ý ở đầu
    module, KHÔNG PHẢI `src.prox_solver.solve_long_only` (solver tay chỉ
    dùng cho core report §1-6).

    Với mỗi kỳ rebalance hàng tháng (xem `_build_rebalance_windows`):
    1. Ước lượng (mu_hat, Sigma, Sigma_sqrt) trên cửa sổ E (estimation).
    2. Với MỖI (kappa, gamma) trong param_grid: giải `cvxpy_solve_long_only`
       trên E, đánh giá `selection_metric` của danh mục thu được trên Vwin
       (validation) -- KHÔNG dùng lại tham số đã fit trên Vwin để giải lại,
       chỉ dùng Vwin để CHỌN tham số tốt nhất.
    3. Chọn (kappa*, gamma*) có `selection_metric` validation cao nhất.
    4. Ước lượng lại (mu, Sigma, Sigma_sqrt) trên TOÀN BỘ E union Vwin (dùng
       hết dữ liệu trước rebalance_date để có ước lượng tốt nhất), giải lại
       `cvxpy_solve_long_only` với (kappa*, gamma*) -- đây là trọng số THẬT
       SỰ triển khai cho tháng target.
    5. Mô phỏng return của tháng target bằng `_simulate_period`, dùng trọng
       số đã trôi cuối kỳ TRƯỚC làm baseline turnover (không phải trọng số
       mục tiêu của kỳ trước).

    KHÔNG LOOK-AHEAD: với mỗi kỳ, estimation (E) và validation (Vwin) chỉ
    dùng dữ liệu có ngày < ngày rebalance (đảm bảo bởi _build_rebalance_windows,
    có test cấu trúc riêng `test_build_windows_no_look_ahead`). Trọng số triển
    khai ước lượng trên E union Vwin, KHÔNG dùng dữ liệu của tháng target.

    Parameters
    ----------
    returns : pd.DataFrame, shape (T, N)
        Simple daily returns, index = date (tăng dần), columns = symbol.
    lookback_months : int, default 24
        Tổng số tháng lịch sử dùng cho mỗi kỳ (E + Vwin).
    validation_months : int, default 6
        Số tháng cuối của cửa sổ lookback dùng làm Vwin (phần đầu còn lại là E).
    param_grid : list[tuple[float, float]] | None, default None
        Danh sách (kappa, gamma) cần thử mỗi kỳ; None -> DEFAULT_PARAM_GRID
        (kappa in {0,0.5,1,2} x gamma in {1,5,10} = 12 tổ hợp).
    shrinkage : str | None, default "lw"
        Truyền thẳng vào `estimate_all` (Ledoit-Wolf bật xuyên suốt theo
        thiết kế đã chốt).
    fee : float, default 0.002
        Phí giao dịch tỉ lệ trên turnover mỗi kỳ (xem `_simulate_period`).
    rf : float, default 0.0
        Lãi suất phi rủi ro NĂM -- dùng để tính Sharpe LOG (`val_sharpe`)
        luôn luôn, và còn dùng để CHỌN tham số nếu `selection_metric="sharpe"`.
    selection_metric : {"sharpe", "return"}, default "sharpe"
        Tiêu chí chọn (kappa,gamma) trên Vwin mỗi kỳ. "sharpe" (MẶC ĐỊNH) =
        Sharpe ratio (rf=`rf`) -- đồng bộ với nhánh long-short
        (`walk_forward_backtest_long_short(_full)`), vốn luôn chọn theo
        Sharpe, để so sánh công bằng giữa mọi phương pháp trong report.
        "return" = cumulative return thô, giữ lại làm tuỳ chọn lịch sử (xem
        LƯU Ý PHƯƠNG PHÁP LUẬN ở trên). `rebalance_log` luôn có cả 2 cột
        `val_return` và `val_sharpe` bất kể chọn tiêu chí nào (cột kia chỉ
        mang tính LOG).
    max_weight : float | None, default None
        Trần trọng số MỖI mã, truyền thẳng vào `cvxpy_solve_long_only` (xem
        docstring hàm đó) -- None = không ràng buộc (hành vi gốc). Đây là
        ràng buộc LỒI thật, không phải heuristic: max_weight=0.20 đảm bảo
        TOÁN HỌC luôn có >= 5 mã active (vì mỗi mã đóng góp tối đa 20% vào
        tổng=1), áp dụng cho CẢ bước chọn tham số (Vwin) lẫn bước deploy,
        để validation phản ánh đúng hiệu suất của chính danh mục sẽ triển
        khai (không chọn tham số dựa trên 1 danh mục rồi deploy 1 danh mục
        khác về mặt ràng buộc).

    Returns
    -------
    BacktestResult
    """
    from src.cvxpy_check import cvxpy_solve_long_only
    from src.estimators import estimate_all

    if param_grid is None:
        param_grid = DEFAULT_PARAM_GRID
    if selection_metric not in ("return", "sharpe"):
        raise ValueError(f'selection_metric phải là "return" hoặc "sharpe", nhận: {selection_metric!r}')

    windows = _build_rebalance_windows(returns.index, lookback_months, validation_months)
    if not windows:
        raise ValueError(
            f"Không đủ dữ liệu cho lookback_months={lookback_months} "
            f"(có {returns.index.to_period('M').nunique()} tháng)."
        )

    all_daily: list[pd.Series] = []
    log_rows: list[dict] = []
    weight_rows: dict = {}
    prev_drifted: np.ndarray | None = None

    for win in windows:
        E = returns.loc[win["e_mask"]]
        Vwin = returns.loc[win["v_mask"]]
        target = returns.loc[win["target_mask"]]

        best_score = -np.inf
        best_return = float("nan")
        best_sharpe_at_best = float("nan")
        best_params = param_grid[0]
        mu_e, sigma_e, sqrt_e = estimate_all(E, shrinkage=shrinkage)
        for kappa, gamma in param_grid:
            w_val, _ = cvxpy_solve_long_only(mu_e, sigma_e, sqrt_e, kappa, gamma, max_weight)
            val_daily = pd.Series(Vwin.values @ w_val, index=Vwin.index)
            metrics = performance_metrics(val_daily, rf=rf)
            val_return = metrics["cumulative_return"]
            val_sharpe = metrics["sharpe"]
            score = val_return if selection_metric == "return" else val_sharpe
            if np.isnan(score):
                score = -np.inf
            if score > best_score:
                best_score = score
                best_return = val_return
                best_sharpe_at_best = val_sharpe
                best_params = (kappa, gamma)

        full_window = pd.concat([E, Vwin])
        mu_f, sigma_f, sqrt_f = estimate_all(full_window, shrinkage=shrinkage)
        w_t, _ = cvxpy_solve_long_only(mu_f, sigma_f, sqrt_f, *best_params, max_weight)

        daily_net, turnover, cost, w_end, wiped_out = _simulate_period(w_t, target, prev_drifted, fee)
        prev_drifted = w_end
        all_daily.append(daily_net)

        log_rows.append({
            "date": win["rebalance_date"], "kappa": best_params[0], "gamma": best_params[1],
            "turnover": turnover, "cost": cost,
            "n_active": int(np.sum(w_t > 1e-4)),
            "val_return": best_return, "val_sharpe": best_sharpe_at_best,
            "wiped_out": wiped_out,
        })
        weight_rows[win["rebalance_date"]] = w_t

    daily_returns = pd.concat(all_daily).sort_index()
    rebalance_log = pd.DataFrame(log_rows).set_index("date")
    weights = pd.DataFrame.from_dict(weight_rows, orient="index", columns=returns.columns)

    return BacktestResult(daily_returns=daily_returns, rebalance_log=rebalance_log, weights=weights)


def walk_forward_backtest_long_short(
    returns: pd.DataFrame,
    *,
    kappa: float = 0.0,
    lookback_months: int = 24,
    validation_months: int = 6,
    param_grid: list[tuple[float, float]] | None = None,
    shrinkage: str | None = "lw",
    fee: float = 0.002,
    rf: float = 0.0,
) -> BacktestResult:
    """Walk-forward backtest OOS trên nhánh LONG-SHORT (`cvxpy_solve` --
    xem LƯU Ý SOLVER ở đầu module, cho phép bán khống -- KHÔNG có ràng buộc
    `w>=0`, chỉ `1ᵀw=1`), tự chọn (lam, gamma) mỗi kỳ qua Sharpe validation,
    với `kappa` CỐ ĐỊNH (mặc định 0.0 -- "chỉ sparse, không robust", theo
    yêu cầu so sánh với method A-D).

    KHÁC với `walk_forward_backtest` (nhánh long-only, chọn kappa/gamma,
    KHÔNG có lambda vì L1 vô nghĩa dưới ràng buộc long-only -- xem docstring
    module `prox_solver.py`): ở đây lambda MỚI thật sự có tác dụng (điều
    khiển sparsity qua soft-threshold + joint prox trong `solve()`), còn
    kappa cố định (không nằm trong `param_grid`) thay vì được chọn tự động.

    BUG ĐÃ SỬA (2026-08-10, phát hiện khi n_active=1 MỌI kỳ một cách bất
    thường, kể cả khi lam=0 được chọn -- không có cách nào L1=0 lại tự ép
    về 1 mã): bước deploy cuối gọi `cvxpy_solve(mu_f, sigma_f, sqrt_f,
    kappa, *best_params)` với `best_params=(lam, gamma)`, trong khi
    `cvxpy_solve` cần thứ tự `(kappa, gamma, lam)` -- unpack `*best_params`
    theo thứ tự vị trí khiến gamma THẬT nhận giá trị lam (rất nhỏ, tối đa
    0.02 trong grid) còn lam THẬT nhận giá trị gamma (1-10, lớn hơn MỌI lam
    từng thử tới 50-500 lần) -- risk aversion gần như = 0 + phạt L1 khổng lồ
    ép danh mục về đúng 1 mã (mã có mu cao nhất) mỗi kỳ, bất kể lam/gamma
    log ra là gì. Vòng lặp CHỌN tham số (bên trên) không bị ảnh hưởng (dùng
    tên biến tường minh `kappa, gamma, lam`, không unpack tuple) -- chỉ bước
    deploy cuối bị lỗi, nên (lam,gamma) log trong `rebalance_log` là ĐÚNG
    tham số được CHỌN, chỉ SAI ở chỗ không phải tham số THỰC SỰ được dùng để
    giải nghiệm triển khai. Đã sửa bằng cách unpack tường minh
    `lam_star, gamma_star = best_params` trước khi gọi. Xem
    `test_walk_forward_backtest_long_short_deploys_with_correct_param_order`.

    CẢNH BÁO MÔ HÌNH HOÁ (khác biệt quan trọng so với nhánh long-only): `_simulate_period`
    dùng công thức "trôi theo giá rồi chuẩn hoá lại tổng=1" (`w = w*(1+r); w
    = w/w.sum()`) -- ĐÚNG cho tỉ trọng giá trị của vị thế long, nhưng chỉ là
    XẤP XỈ cho vị thế short (không mô hình phí vay mượn cổ phiếu/margin
    call/lãi suất margin thật). Đây là hạn chế kế thừa từ chính công thức
    Markowitz gốc `w ∈ R^N` không ràng buộc dấu (xem README/notebook), không
    phải lỗi riêng của hàm này. Turnover cũng KHÔNG có trần (solve() không
    giới hạn `‖w‖₁` ngoài phạt mềm qua lambda) -- xem cột `gross_exposure`
    trong `rebalance_log` để theo dõi mức đòn bẩy thực tế mỗi kỳ.

    Parameters
    ----------
    returns : pd.DataFrame, shape (T, N)
        Simple daily returns, index = date (tăng dần), columns = symbol.
    kappa : float, default 0.0
        Hệ số robust CỐ ĐỊNH xuyên suốt (không nằm trong param_grid) -- 0.0
        nghĩa là tắt hẳn robust term, chỉ còn mean-variance + sparsity.
    lookback_months, validation_months : int, default 24, 6
        Xem `walk_forward_backtest`.
    param_grid : list[tuple[float, float]] | None, default None
        Danh sách (lam, gamma) cần thử mỗi kỳ; None -> `DEFAULT_PARAM_GRID_LONG_SHORT`
        (lam in {0, 0.001, 0.005, 0.01, 0.02} x gamma in {1,5,10} = 15 tổ hợp).
    shrinkage, fee, rf : xem `walk_forward_backtest`.

    Returns
    -------
    BacktestResult
        `rebalance_log` có cột `lam`/`gamma` (thay vì `kappa`/`gamma`) +
        thêm cột `gross_exposure` (= `sum(|w|)`, >1 nếu có short) so với
        `walk_forward_backtest`.
    """
    from src.cvxpy_check import cvxpy_solve
    from src.estimators import estimate_all

    if param_grid is None:
        param_grid = DEFAULT_PARAM_GRID_LONG_SHORT

    windows = _build_rebalance_windows(returns.index, lookback_months, validation_months)
    if not windows:
        raise ValueError(
            f"Không đủ dữ liệu cho lookback_months={lookback_months} "
            f"(có {returns.index.to_period('M').nunique()} tháng)."
        )

    all_daily: list[pd.Series] = []
    log_rows: list[dict] = []
    weight_rows: dict = {}
    prev_drifted: np.ndarray | None = None

    for win in windows:
        E = returns.loc[win["e_mask"]]
        Vwin = returns.loc[win["v_mask"]]
        target = returns.loc[win["target_mask"]]

        best_sharpe = -np.inf
        best_params = param_grid[0]
        mu_e, sigma_e, sqrt_e = estimate_all(E, shrinkage=shrinkage)
        for lam, gamma in param_grid:
            w_val, _ = cvxpy_solve(mu_e, sigma_e, sqrt_e, kappa, gamma, lam)
            val_daily = pd.Series(Vwin.values @ w_val, index=Vwin.index)
            sharpe = performance_metrics(val_daily, rf=rf)["sharpe"]
            if np.isnan(sharpe):
                sharpe = -np.inf
            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_params = (lam, gamma)

        full_window = pd.concat([E, Vwin])
        mu_f, sigma_f, sqrt_f = estimate_all(full_window, shrinkage=shrinkage)
        lam_star, gamma_star = best_params
        w_t, _ = cvxpy_solve(mu_f, sigma_f, sqrt_f, kappa, gamma_star, lam_star)

        daily_net, turnover, cost, w_end, wiped_out = _simulate_period(w_t, target, prev_drifted, fee)
        prev_drifted = w_end
        all_daily.append(daily_net)

        log_rows.append({
            "date": win["rebalance_date"], "lam": best_params[0], "gamma": best_params[1],
            "turnover": turnover, "cost": cost,
            "n_active": int(np.sum(np.abs(w_t) > 1e-4)), "val_sharpe": best_sharpe,
            "gross_exposure": float(np.sum(np.abs(w_t))),
            "wiped_out": wiped_out,
        })
        weight_rows[win["rebalance_date"]] = w_t

    daily_returns = pd.concat(all_daily).sort_index()
    rebalance_log = pd.DataFrame(log_rows).set_index("date")
    weights = pd.DataFrame.from_dict(weight_rows, orient="index", columns=returns.columns)

    return BacktestResult(daily_returns=daily_returns, rebalance_log=rebalance_log, weights=weights)


DEFAULT_PARAM_GRID_LONG_SHORT_FULL = [
    (kappa, lam, gamma)
    for kappa in (0.0, 0.5, 1.0, 2.0)
    for lam in (0.0, 0.001, 0.005, 0.01, 0.02)
    for gamma in (1.0, 5.0, 10.0)
]


def walk_forward_backtest_long_short_full(
    returns: pd.DataFrame,
    *,
    lookback_months: int = 24,
    validation_months: int = 6,
    param_grid: list[tuple[float, float, float]] | None = None,
    shrinkage: str | None = "lw",
    fee: float = 0.002,
    rf: float = 0.0,
) -> BacktestResult:
    """Walk-forward OOS trên nhánh LONG-SHORT với ĐẦY ĐỦ phương trình gốc
    (`cvxpy_solve` -- xem LƯU Ý SOLVER ở đầu module, w không ràng buộc dấu
    -- cho bán khống), tự chọn CẢ BA (kappa, lam, gamma) mỗi kỳ qua Sharpe
    validation.

    KHÁC với `walk_forward_backtest_long_short`: ở đó `kappa` bị CỐ ĐỊNH
    (không nằm trong param_grid, mặc định 0.0 -- "chỉ sparse, không robust",
    dùng cho method E trong bảng so sánh). Ở đây `kappa` nằm TRONG param_grid
    cùng `lam`/`gamma` -- tức là robust term VÀ sparsity term VÀ risk term
    đều được chọn tự động mỗi kỳ, đúng công thức đầy đủ:

        min_w  -mu^T w + kappa*||Sigma^(1/2) w||_2 + gamma*w^T Sigma w
               + lam*||w||_1
        s.t.   1^T w = 1

    Cùng cảnh báo mô hình hoá như `walk_forward_backtest_long_short` (drift
    "trôi rồi chuẩn hoá lại tổng=1" là xấp xỉ cho vị thế short, không mô
    hình phí vay mượn cổ phiếu/margin; turnover không có trần -- theo dõi
    qua cột `gross_exposure`).

    Parameters
    ----------
    returns : pd.DataFrame, shape (T, N)
        Simple daily returns, index = date (tăng dần), columns = symbol.
    lookback_months, validation_months : int, default 24, 6
        Xem `walk_forward_backtest`.
    param_grid : list[tuple[float, float, float]] | None, default None
        Danh sách (kappa, lam, gamma) cần thử mỗi kỳ; None ->
        `DEFAULT_PARAM_GRID_LONG_SHORT_FULL` (kappa in {0,0.5,1,2} x
        lam in {0,0.001,0.005,0.01,0.02} x gamma in {1,5,10} = 60 tổ hợp
        -- lớn hơn 5x so với lưới 12 tổ hợp của nhánh long-only, cân nhắc
        truyền param_grid nhỏ hơn nếu chạy trên toàn bộ universe 98 mã).
    shrinkage, fee, rf : xem `walk_forward_backtest`.

    Returns
    -------
    BacktestResult
        `rebalance_log` có cột `kappa`/`lam`/`gamma` (cả ba, khác với
        `walk_forward_backtest_long_short` chỉ có `lam`/`gamma`) +
        `gross_exposure` (= `sum(|w|)`, >1 nếu có short).
    """
    from src.cvxpy_check import cvxpy_solve
    from src.estimators import estimate_all

    if param_grid is None:
        param_grid = DEFAULT_PARAM_GRID_LONG_SHORT_FULL

    windows = _build_rebalance_windows(returns.index, lookback_months, validation_months)
    if not windows:
        raise ValueError(
            f"Không đủ dữ liệu cho lookback_months={lookback_months} "
            f"(có {returns.index.to_period('M').nunique()} tháng)."
        )

    all_daily: list[pd.Series] = []
    log_rows: list[dict] = []
    weight_rows: dict = {}
    prev_drifted: np.ndarray | None = None

    for win in windows:
        E = returns.loc[win["e_mask"]]
        Vwin = returns.loc[win["v_mask"]]
        target = returns.loc[win["target_mask"]]

        best_sharpe = -np.inf
        best_params = param_grid[0]
        mu_e, sigma_e, sqrt_e = estimate_all(E, shrinkage=shrinkage)
        for kappa, lam, gamma in param_grid:
            w_val, _ = cvxpy_solve(mu_e, sigma_e, sqrt_e, kappa, gamma, lam)
            val_daily = pd.Series(Vwin.values @ w_val, index=Vwin.index)
            sharpe = performance_metrics(val_daily, rf=rf)["sharpe"]
            if np.isnan(sharpe):
                sharpe = -np.inf
            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_params = (kappa, lam, gamma)

        full_window = pd.concat([E, Vwin])
        mu_f, sigma_f, sqrt_f = estimate_all(full_window, shrinkage=shrinkage)
        kappa_star, lam_star, gamma_star = best_params
        w_t, _ = cvxpy_solve(mu_f, sigma_f, sqrt_f, kappa_star, gamma_star, lam_star)

        daily_net, turnover, cost, w_end, wiped_out = _simulate_period(w_t, target, prev_drifted, fee)
        prev_drifted = w_end
        all_daily.append(daily_net)

        log_rows.append({
            "date": win["rebalance_date"],
            "kappa": kappa_star, "lam": lam_star, "gamma": gamma_star,
            "turnover": turnover, "cost": cost,
            "n_active": int(np.sum(np.abs(w_t) > 1e-4)), "val_sharpe": best_sharpe,
            "gross_exposure": float(np.sum(np.abs(w_t))),
            "wiped_out": wiped_out,
        })
        weight_rows[win["rebalance_date"]] = w_t

    daily_returns = pd.concat(all_daily).sort_index()
    rebalance_log = pd.DataFrame(log_rows).set_index("date")
    weights = pd.DataFrame.from_dict(weight_rows, orient="index", columns=returns.columns)

    return BacktestResult(daily_returns=daily_returns, rebalance_log=rebalance_log, weights=weights)


def index_buy_and_hold_backtest(
    index_returns: pd.Series,
    oos_index: pd.DatetimeIndex,
    fee: float = 0.002,
) -> pd.Series:
    """Benchmark "mua & giữ" chỉ số VN30 (method D), trên CÙNG giai đoạn OOS
    như `walk_forward_backtest`/`equal_weight_backtest`, để so sánh công bằng
    4 phương pháp trên đúng 1 khoảng thời gian.

    KHÔNG rebalance (mua 1 lần ở đầu giai đoạn OOS, giữ nguyên tới cuối) --
    không có estimation/turnover/weights vì đây là chỉ số thị trường có sẵn,
    không phải danh mục tự xây từ universe VN100 (nên KHÔNG trả về
    `BacktestResult` như 2 hàm kia, chỉ trả `daily_returns`). Phí giao dịch
    (nếu > 0) chỉ trừ 1 LẦN vào ngày đầu tiên của giai đoạn OOS (mua), khác
    với 2 method kia (trừ phí MỖI kỳ rebalance hàng tháng).

    Parameters
    ----------
    index_returns : pd.Series
        Simple daily return của chỉ số VN30 (cột "ret" trả về bởi
        `src.data_loader.load_vn30_index`), index=date.
    oos_index : pd.DatetimeIndex
        Danh sách ngày OOS cần so sánh -- thường lấy từ
        `walk_forward_backtest(...).daily_returns.index` để đảm bảo đúng
        cùng giai đoạn với 3 method kia.
    fee : float, default 0.002
        Phí giao dịch tỉ lệ, trừ 1 lần vào return ngày đầu tiên.

    Returns
    -------
    pd.Series
        Daily net return, index = intersection(oos_index, index_returns.index).
        Cảnh báo (warnings.warn) nếu có ngày OOS thiếu dữ liệu VN30 tương ứng
        -- lịch giao dịch chỉ số và cổ phiếu lẻ hiếm khi lệch nhau nhưng
        không đảm bảo tuyệt đối trùng khớp.
    """
    common = oos_index.intersection(index_returns.index)
    missing = oos_index.difference(index_returns.index)
    if len(missing) > 0:
        warnings.warn(
            f"index_buy_and_hold_backtest: {len(missing)}/{len(oos_index)} ngày OOS "
            f"không có dữ liệu VN30 tương ứng, bị bỏ qua: "
            f"{[d.date() for d in missing[:5]]}{'...' if len(missing) > 5 else ''}",
            stacklevel=2,
        )

    daily_net = index_returns.loc[common].sort_index().copy()
    if len(daily_net) > 0 and fee > 0:
        daily_net.iloc[0] -= fee

    return daily_net


def vn100_buy_and_hold_backtest(
    returns: pd.DataFrame,
    oos_index: pd.DatetimeIndex,
    fee: float = 0.002,
) -> pd.Series:
    """Benchmark "mua & giữ" chỉ số VN100 (method D, thay cho VN30 external
    index) -- mua equal-weight CHÍNH 98 mã trong `returns` (CÙNG universe với
    A/B/C, không phải rổ VN30 khác thành phần) một lần duy nhất ở đầu giai
    đoạn OOS, giữ nguyên tới cuối, không rebalance.

    Khác `index_buy_and_hold_backtest` (dùng giá chỉ số VN30 có sẵn từ nguồn
    ngoài) -- hàm này tự dựng "chỉ số" bằng cách trôi theo giá 98 mã trong
    chính universe đang nghiên cứu, nên so sánh với A/B/C/E/F công bằng hơn
    (cùng 98 mã, chỉ khác cách phân bổ trọng số/tần suất rebalance), thay vì
    so với 1 rổ 30 mã vốn hoá lớn khác hẳn thành phần.

    Parameters
    ----------
    returns : pd.DataFrame, shape (T, N)
        Simple daily returns, index = date (tăng dần), columns = symbol --
        CÙNG DataFrame truyền cho `walk_forward_backtest`/`equal_weight_backtest`.
    oos_index : pd.DatetimeIndex
        Danh sách ngày OOS cần so sánh -- lấy từ
        `walk_forward_backtest(...).daily_returns.index` để đảm bảo đúng
        cùng giai đoạn với các method kia.
    fee : float, default 0.002
        Phí giao dịch tỉ lệ, trừ 1 LẦN vào ngày đầu tiên (mua), khác
        `equal_weight_backtest` (trừ phí MỖI kỳ rebalance hàng tháng).

    Returns
    -------
    pd.Series
        Daily net return, index = oos_index (nội suy trực tiếp từ
        `returns.loc[oos_index]`, không cần intersection vì `returns` là
        nguồn dữ liệu gốc của chính oos_index).
    """
    n = returns.shape[1]
    w_uniform = np.full(n, 1.0 / n)
    period_returns = returns.loc[oos_index]

    daily_net, _, _, _, wiped_out = _simulate_period(
        w_uniform, period_returns, prev_weights_drifted=None, fee=fee,
    )
    if wiped_out:
        warnings.warn(
            "vn100_buy_and_hold_backtest: danh mục 'vỡ nợ' trong giai đoạn OOS "
            "-- xem cảnh báo chi tiết từ _simulate_period.",
            RuntimeWarning,
            stacklevel=2,
        )
    return daily_net


def equal_weight_backtest(
    returns: pd.DataFrame,
    *,
    lookback_months: int = 24,
    validation_months: int = 6,
    fee: float = 0.002,
) -> BacktestResult:
    """Benchmark equal-weight 1/N trên CÙNG danh sách ngày rebalance như
    `walk_forward_backtest` (dùng cùng `lookback_months`/`validation_months`
    CHỈ để sinh windows giống hệt, đảm bảo so sánh công bằng trên cùng giai
    đoạn OOS -- benchmark không ước lượng gì (không dùng E/Vwin), trọng số
    mục tiêu mỗi kỳ luôn `1/N` với N = số tài sản trong `returns`.

    Parameters
    ----------
    returns : pd.DataFrame, shape (T, N)
        Simple daily returns, index = date (tăng dần), columns = symbol.
    lookback_months : int, default 24
        Chỉ dùng để tạo cùng danh sách ngày rebalance với walk-forward
        (xem `_build_rebalance_windows`); benchmark không cần E/Vwin.
    validation_months : int, default 6
        Tương tự -- chỉ ảnh hưởng danh sách ngày rebalance, không dùng để
        ước lượng gì ở đây.
    fee : float, default 0.002
        Phí giao dịch tỉ lệ trên turnover mỗi kỳ (xem `_simulate_period`).

    Returns
    -------
    BacktestResult
    """
    windows = _build_rebalance_windows(returns.index, lookback_months, validation_months)
    if not windows:
        raise ValueError(f"Không đủ dữ liệu cho lookback_months={lookback_months}.")

    n = returns.shape[1]
    w_uniform = np.full(n, 1.0 / n)

    all_daily: list[pd.Series] = []
    log_rows: list[dict] = []
    weight_rows: dict = {}
    prev_drifted: np.ndarray | None = None

    for win in windows:
        target = returns.loc[win["target_mask"]]
        daily_net, turnover, cost, w_end, wiped_out = _simulate_period(
            w_uniform, target, prev_drifted, fee,
        )
        prev_drifted = w_end
        all_daily.append(daily_net)
        log_rows.append({
            "date": win["rebalance_date"], "turnover": turnover, "cost": cost,
            "n_active": n, "wiped_out": wiped_out,
        })
        weight_rows[win["rebalance_date"]] = w_uniform

    daily_returns = pd.concat(all_daily).sort_index()
    rebalance_log = pd.DataFrame(log_rows).set_index("date")
    weights = pd.DataFrame.from_dict(weight_rows, orient="index", columns=returns.columns)

    return BacktestResult(daily_returns=daily_returns, rebalance_log=rebalance_log, weights=weights)
