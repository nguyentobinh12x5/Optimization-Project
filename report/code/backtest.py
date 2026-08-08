"""
src/backtest.py
=================
Walk-forward backtest out-of-sample cho project sparse+robust portfolio
optimization (VN100). Xem thiết kế đầy đủ ở
docs/superpowers/specs/2026-07-26-walk-forward-backtest-design.md.
"""

from __future__ import annotations

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
    """
    daily_returns = daily_returns.dropna()
    n = len(daily_returns)
    cumulative_return = float((1 + daily_returns).prod() - 1)

    n_years = n / ANNUALIZATION_FACTOR
    if n_years > 0:
        annualized_return = float((1 + cumulative_return) ** (1 / n_years) - 1)
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
) -> tuple[pd.Series, float, float, np.ndarray]:
    """Mô phỏng 1 kỳ nắm giữ (thường 1 tháng) với trọng số CỐ ĐỊNH lúc đầu kỳ
    `w_start`, để trọng số TRÔI theo giá từng ngày (mua-và-giữ, không giao
    dịch giữa kỳ). Turnover tính so với trọng số ĐÃ TRÔI cuối kỳ TRƯỚC
    (`prev_weights_drifted`), vì đó là trạng thái danh mục THỰC TẾ ngay
    trước khi rebalance -- không phải trọng số MỤC TIÊU của kỳ trước.

    Trả về (daily_net_returns, turnover, cost, w_end_drifted). Phí `cost`
    bị trừ vào return của NGÀY ĐẦU TIÊN trong kỳ (ngày rebalance). Dùng
    chung cho walk-forward VÀ baseline equal-weight (Task 5).

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
    (daily_net, turnover, cost, w_end) : tuple
        daily_net : pd.Series, cùng index với period_returns.
        turnover : float, = sum(|w_start - prev_weights_drifted|).
        cost : float, = fee * turnover.
        w_end : np.ndarray, shape (N,), trọng số đã trôi cuối kỳ (sum=1).
    """
    prev = np.zeros_like(w_start) if prev_weights_drifted is None else prev_weights_drifted
    turnover = float(np.abs(w_start - prev).sum())
    cost = fee * turnover

    w = w_start.copy()
    daily_net = []
    for _, day_ret in period_returns.iterrows():
        port_ret = float(w @ day_ret.values)
        daily_net.append(port_ret)
        w = w * (1.0 + day_ret.values)
        w = w / w.sum()

    daily_net = pd.Series(daily_net, index=period_returns.index)
    daily_net.iloc[0] -= cost

    return daily_net, turnover, cost, w


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


def walk_forward_backtest(
    returns: pd.DataFrame,
    *,
    lookback_months: int = 24,
    validation_months: int = 6,
    param_grid: list[tuple[float, float]] | None = None,
    shrinkage: str | None = "lw",
    fee: float = 0.002,
    rf: float = 0.0,
) -> BacktestResult:
    """Walk-forward backtest out-of-sample, long-only, tự chọn (kappa,gamma)
    mỗi kỳ qua Sharpe validation. Xem thiết kế đầy đủ trong
    docs/superpowers/specs/2026-07-26-walk-forward-backtest-design.md.

    Với mỗi kỳ rebalance hàng tháng (xem `_build_rebalance_windows`):
    1. Ước lượng (mu_hat, Sigma, Sigma_sqrt) trên cửa sổ E (estimation).
    2. Với MỖI (kappa, gamma) trong param_grid: giải `solve_long_only` trên
       E, đánh giá Sharpe (rf=rf) của danh mục thu được trên Vwin
       (validation) -- KHÔNG dùng lại tham số đã fit trên Vwin để giải lại,
       chỉ dùng Vwin để CHỌN tham số tốt nhất.
    3. Chọn (kappa*, gamma*) có Sharpe validation cao nhất.
    4. Ước lượng lại (mu, Sigma, Sigma_sqrt) trên TOÀN BỘ E union Vwin (dùng
       hết dữ liệu trước rebalance_date để có ước lượng tốt nhất), giải lại
       `solve_long_only` với (kappa*, gamma*) -- đây là trọng số THẬT SỰ
       triển khai cho tháng target.
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
        Lãi suất phi rủi ro NĂM dùng khi tính Sharpe validation (xem
        `performance_metrics`).

    Returns
    -------
    BacktestResult
    """
    from src.estimators import estimate_all
    from src.prox_solver import solve_long_only

    if param_grid is None:
        param_grid = DEFAULT_PARAM_GRID

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
        for kappa, gamma in param_grid:
            w_val = solve_long_only(mu_e, sigma_e, sqrt_e, kappa, gamma).w
            val_daily = pd.Series(Vwin.values @ w_val, index=Vwin.index)
            sharpe = performance_metrics(val_daily, rf=rf)["sharpe"]
            if np.isnan(sharpe):
                sharpe = -np.inf
            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_params = (kappa, gamma)

        full_window = pd.concat([E, Vwin])
        mu_f, sigma_f, sqrt_f = estimate_all(full_window, shrinkage=shrinkage)
        result = solve_long_only(mu_f, sigma_f, sqrt_f, *best_params)
        w_t = result.w

        daily_net, turnover, cost, w_end = _simulate_period(w_t, target, prev_drifted, fee)
        prev_drifted = w_end
        all_daily.append(daily_net)

        log_rows.append({
            "date": win["rebalance_date"], "kappa": best_params[0], "gamma": best_params[1],
            "turnover": turnover, "cost": cost,
            "n_active": int(np.sum(w_t > 1e-4)), "val_sharpe": best_sharpe,
        })
        weight_rows[win["rebalance_date"]] = w_t

    daily_returns = pd.concat(all_daily).sort_index()
    rebalance_log = pd.DataFrame(log_rows).set_index("date")
    weights = pd.DataFrame.from_dict(weight_rows, orient="index", columns=returns.columns)

    return BacktestResult(daily_returns=daily_returns, rebalance_log=rebalance_log, weights=weights)


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
        daily_net, turnover, cost, w_end = _simulate_period(
            w_uniform, target, prev_drifted, fee,
        )
        prev_drifted = w_end
        all_daily.append(daily_net)
        log_rows.append({
            "date": win["rebalance_date"], "turnover": turnover, "cost": cost,
            "n_active": n,
        })
        weight_rows[win["rebalance_date"]] = w_uniform

    daily_returns = pd.concat(all_daily).sort_index()
    rebalance_log = pd.DataFrame(log_rows).set_index("date")
    weights = pd.DataFrame.from_dict(weight_rows, orient="index", columns=returns.columns)

    return BacktestResult(daily_returns=daily_returns, rebalance_log=rebalance_log, weights=weights)
