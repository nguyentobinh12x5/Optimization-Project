"""
src/backtest.py
=================
Out-of-sample walk-forward backtest for the sparse+robust portfolio
optimization project (VN100). See the full design in
docs/superpowers/specs/2026-07-26-walk-forward-backtest-design.md.

SOLVER USED IN THIS MODULE: CVXPY (`src.cvxpy_check.cvxpy_solve` /
`cvxpy_solve_long_only`), NOT the hand-written proximal-subgradient solver
(`src.prox_solver.solve` / `solve_long_only`) -- a deliberate decision
(discussed with the user on 2026-08-10): the walk-forward functions here
call grid search over dozens of parameter combinations PER rebalance
period x dozens of periods, so per-solve speed matters more than "hand-
written" (which was only required for the core report §1-6, NOT applicable
to this A-F comparison backtest section). CVXPY (CLARABEL) measured ~4x
faster than the hand-written solver on the same 98-asset problem (0.03s vs
0.14s/solve) -- with the grid of 60 combos x 25 periods x 2 solves/period
(validation + refit) of `walk_forward_backtest_long_short_full`, this gap
amounts to ~13 minutes (hand-written solver) vs ~3 minutes (CVXPY).
`src/prox_solver.py` REMAINS the official solver for the core report
(§1-6, in-sample, joint-prox, cross-verified against CVXPY) -- this module
does not replace it, it just uses CVXPY for the walk-forward backtest
section specifically.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

ANNUALIZATION_FACTOR = 252

# Threshold for treating the daily std as "zero" (constant return).
# pandas.Series.std on a constant series does not always yield exactly
# 0.0 (floating-point rounding error on the order of 1e-19 in the sum-of-
# squared-deviations algorithm) -> a plain `> 0` comparison is not safe
# enough, an epsilon threshold is needed.
STD_ZERO_EPS = 1e-12


def performance_metrics(daily_returns: pd.Series, rf: float = 0.0) -> dict:
    """Compute standard performance metrics from the DAILY return series of
    1 portfolio.

    rf: ANNUAL (annualized) risk-free rate, defaults to 0 per the finalized
    design. Sharpe = (mean(excess_daily)) / std(daily) * sqrt(252); if
    std(daily) is approximately 0 (below the STD_ZERO_EPS threshold;
    constant return, e.g. a synthetic series in a test), Sharpe is
    UNDEFINED -> return NaN instead of dividing by a near-zero number
    (pandas.Series.std on a constant series can come out to ~1e-19 due to
    rounding error, not exactly 0.0).

    max_drawdown: the largest decline from the peak of the cumulative
    portfolio value (a NEGATIVE value or 0, not a positive value).

    "WIPE-OUT" WARNING in annualized_return: if `cumulative_return <= -1`
    (total loss of capital or worse -- possible for a highly-leveraged
    long-short portfolio, e.g. CVXPY finding an exact optimum with very
    large gross exposure over a short validation window), the standard CAGR
    formula `(1+cumulative_return)**(1/n_years)` receives a base <= 0 with
    a fractional exponent -> yields a COMPLEX NUMBER in Python, which is
    not a rounding bug but a genuine mathematical issue (an odd/fractional
    root of a negative number is undefined on the real axis). Handling:
    clamp annualized_return at exactly -1.0 (-100%, the same "wipe-out"
    convention as `_simulate_period`) instead of letting `float()` raise a
    TypeError when casting a complex number.
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
    """For each calendar month (period 'M') that has at least `lookback_months`
    months of data preceding it, return 1 dict describing a rebalance period:
      - rebalance_date: the FIRST trading day of the target month (Timestamp)
      - e_mask, v_mask, target_mask: boolean arrays with the same length as
        `index`

    E (estimation) = the first (lookback_months - validation_months) months
    of the lookback_months-month window immediately preceding the target
    month. Vwin (validation) = the last validation_months months of that
    window. Both E and Vwin lie ENTIRELY before rebalance_date -- this is
    the anti-look-ahead invariant, confirmed by a dedicated test
    (`test_build_windows_no_look_ahead`): every month in the window (E
    union Vwin) precedes target_month in the sorted list of calendar
    months, so every trading day of E/Vwin is strictly < rebalance_date
    (the first day of target_month).
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
    """Simulate 1 holding period (usually 1 month) with weights FIXED at the
    start of the period `w_start`, letting the weights DRIFT with price
    day by day (buy-and-hold, no trading within the period). Turnover is
    computed relative to the DRIFTED weights at the end of the PREVIOUS
    period (`prev_weights_drifted`), since that is the ACTUAL portfolio
    state right before the rebalance -- not the previous period's TARGET
    weights.

    Returns (daily_net_returns, turnover, cost, w_end_drifted). The fee
    `cost` is deducted from the FIRST DAY's return of the period (the
    rebalance day). Shared by walk-forward AND the equal-weight baseline
    (Task 5).

    "WIPE-OUT" WARNING (bug fix): the drift formula "w = w*(1+r); w =
    w/w.sum()" assumes the portfolio value after each day is ALWAYS
    POSITIVE (true for long-only, since every w_i>=0 and each asset's
    return > -100% -> the sum is always >0). For a highly-leveraged
    long-short portfolio (`walk_forward_backtest_long_short(_full)`,
    gross exposure = sum(|w|) can far exceed 1, with no cap), 1 bad day
    can drive port_ret <= -100% -- if NOT handled specially, w.sum() after
    multiplying would be <=0, and dividing by that would FLIP THE SIGN of
    every weight in a meaningless way (reproduced with concrete numbers,
    see the discussion with the user on 2026-08-10). Handling: when a day
    with port_ret <= -100% is detected, treat it as a "wipe-out" (total
    loss of capital) -- CLAMP that day's return at exactly -100% (never
    more negative, the standard convention for a compounding equity curve
    so that (1+r).cumprod() never goes below 0), set w to all zeros (no
    remaining position), and the REMAINING days of the period return 0%
    (treated as "cash" until the next rebalance). Issue a clear warning
    (`warnings.warn`) instead of silently continuing -- the caller can
    check the returned `wiped_out` column to know which periods were
    affected.

    Parameters
    ----------
    w_start : np.ndarray, shape (N,)
        TARGET weights right after the start-of-period rebalance.
    period_returns : pd.DataFrame, shape (T_period, N)
        Simple daily returns for the holding period, same column order as
        w_start.
    prev_weights_drifted : np.ndarray | None, shape (N,)
        DRIFTED weights at the end of the previous period; None if this is
        the first period (full turnover, equivalent to treating the
        previous portfolio as all zeros).
    fee : float
        Proportional transaction fee (e.g. 0.002 = 0.20%) multiplied by
        turnover.

    Returns
    -------
    (daily_net, turnover, cost, w_end, wiped_out) : tuple
        daily_net : pd.Series, same index as period_returns. If wiped out,
            the wipe-out day = EXACTLY -1.0, and any subsequent days = 0.0.
        turnover : float, = sum(|w_start - prev_weights_drifted|).
        cost : float, = fee * turnover.
        w_end : np.ndarray, shape (N,), drifted weights at end of period
            (sum=1, or all zeros if wiped out during the period).
        wiped_out : bool, True if this period had at least 1 wipe-out day.
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
                f"_simulate_period: portfolio 'wiped out' (port_ret={port_ret:.4f} <= -100%, "
                f"initial gross exposure={np.abs(w_start).sum():.3f}) -- this day's return "
                "is clamped at -100%, remaining days in the period are treated as return=0 "
                "(cannot 'recover' from a wipe-out). See the _simulate_period docstring section "
                "'WIPE-OUT WARNING'.",
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
    """Result returned by `walk_forward_backtest`.

    daily_returns : pd.Series
        Net-of-fee DAILY returns for the entire out-of-sample period,
        concatenating consecutive rebalance periods, sorted by time.
    rebalance_log : pd.DataFrame
        1 row / rebalance period, index = rebalance_date, columns: kappa,
        gamma, turnover, cost, n_active, val_sharpe (Sharpe on Vwin for the
        selected parameters).
    weights : pd.DataFrame
        1 row / rebalance period, index = rebalance_date, columns = asset
        names (matching `returns.columns`), values = TARGET weights
        deployed right after the rebalance (before drifting with price
        during the period).
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
    """Out-of-sample, long-only walk-forward backtest, automatically
    selecting (kappa,gamma) each period via validation on Vwin -- the
    selection criterion is `selection_metric` ("sharpe": Sharpe ratio,
    DEFAULT -- consistent with `walk_forward_backtest_long_short(_full)`
    which always selects by Sharpe; "return": raw cumulative return,
    kept as a legacy option). See the full design in
    docs/superpowers/specs/2026-07-26-walk-forward-backtest-design.md.

    METHODOLOGICAL NOTE (default changed back to "sharpe" at the user's
    request -- so that ALL 6 methods compared in the report use the SAME
    parameter-selection criterion, without a mismatch between the
    long-only and long-short branches): an earlier version had a phase
    where the default was changed to raw "return" (not risk-adjusted) --
    this is more prone to picking a short-term "lucky" configuration on
    the 6-month validation window than Sharpe (which balances return
    against volatility), as illustrated in
    `test_walk_forward_backtest_selects_max_return_when_requested`. The
    "return" option is still available via `selection_metric="return"`
    for anyone needing to reproduce the old results, but it is NO LONGER
    the default.

    SOLVER: `cvxpy_solve_long_only` (CVXPY/CLARABEL) -- see the NOTE at
    the top of the module, NOT `src.prox_solver.solve_long_only` (the
    hand-written solver, used only for the core report §1-6).

    For each monthly rebalance period (see `_build_rebalance_windows`):
    1. Estimate (mu_hat, Sigma, Sigma_sqrt) on window E (estimation).
    2. For EACH (kappa, gamma) in param_grid: solve `cvxpy_solve_long_only`
       on E, evaluate the `selection_metric` of the resulting portfolio on
       Vwin (validation) -- do NOT reuse parameters fit on Vwin to solve
       again, Vwin is used ONLY to SELECT the best parameters.
    3. Select the (kappa*, gamma*) with the highest validation
       `selection_metric`.
    4. Re-estimate (mu, Sigma, Sigma_sqrt) on the FULL E union Vwin
       (using all data before rebalance_date for the best estimate),
       re-solve `cvxpy_solve_long_only` with (kappa*, gamma*) -- these
       are the weights ACTUALLY deployed for the target month.
    5. Simulate the target month's return with `_simulate_period`, using
       the previous period's drifted end-of-period weights as the
       turnover baseline (not the previous period's target weights).

    NO LOOK-AHEAD: for each period, estimation (E) and validation (Vwin)
    only use data with dates < the rebalance date (guaranteed by
    _build_rebalance_windows, with its own dedicated structural test
    `test_build_windows_no_look_ahead`). The deployed weights are
    estimated on E union Vwin, WITHOUT using data from the target month.

    Parameters
    ----------
    returns : pd.DataFrame, shape (T, N)
        Simple daily returns, index = date (ascending), columns = symbol.
    lookback_months : int, default 24
        Total number of months of history used per period (E + Vwin).
    validation_months : int, default 6
        Number of months at the end of the lookback window used as Vwin
        (the remaining leading portion is E).
    param_grid : list[tuple[float, float]] | None, default None
        List of (kappa, gamma) to try each period; None -> DEFAULT_PARAM_GRID
        (kappa in {0,0.5,1,2} x gamma in {1,5,10} = 12 combinations).
    shrinkage : str | None, default "lw"
        Passed straight through to `estimate_all` (Ledoit-Wolf enabled
        throughout per the finalized design).
    fee : float, default 0.002
        Proportional transaction fee on turnover each period (see
        `_simulate_period`).
    rf : float, default 0.0
        ANNUAL risk-free rate -- used to always compute the LOGGED Sharpe
        (`val_sharpe`), and also used to SELECT parameters if
        `selection_metric="sharpe"`.
    selection_metric : {"sharpe", "return"}, default "sharpe"
        Criterion for selecting (kappa,gamma) on Vwin each period. "sharpe"
        (DEFAULT) = Sharpe ratio (rf=`rf`) -- consistent with the
        long-short branch (`walk_forward_backtest_long_short(_full)`),
        which always selects by Sharpe, for a fair comparison across all
        methods in the report. "return" = raw cumulative return, kept as
        a legacy option (see the METHODOLOGICAL NOTE above).
        `rebalance_log` always has both the `val_return` and `val_sharpe`
        columns regardless of which criterion is selected (the other
        column is purely for LOGGING).
    max_weight : float | None, default None
        Per-asset weight cap, passed straight through to
        `cvxpy_solve_long_only` (see that function's docstring) -- None =
        unconstrained (original behavior). This is a genuine CONVEX
        constraint, not a heuristic: max_weight=0.20 guarantees
        MATHEMATICALLY at least 5 active assets (since each asset
        contributes at most 20% to the total=1), applied to BOTH the
        parameter-selection step (Vwin) and the deploy step, so that
        validation reflects the true performance of the very portfolio
        that will be deployed (not selecting parameters based on one
        portfolio then deploying a different one in terms of
        constraints).

    Returns
    -------
    BacktestResult
    """
    from src.cvxpy_check import cvxpy_solve_long_only
    from src.estimators import estimate_all

    if param_grid is None:
        param_grid = DEFAULT_PARAM_GRID
    if selection_metric not in ("return", "sharpe"):
        raise ValueError(f'selection_metric must be "return" or "sharpe", got: {selection_metric!r}')

    windows = _build_rebalance_windows(returns.index, lookback_months, validation_months)
    if not windows:
        raise ValueError(
            f"Not enough data for lookback_months={lookback_months} "
            f"(have {returns.index.to_period('M').nunique()} months)."
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
    """OOS walk-forward backtest on the LONG-SHORT branch (`cvxpy_solve` --
    see the SOLVER NOTE at the top of the module, allows short-selling --
    NO `w>=0` constraint, only `1ᵀw=1`), automatically selecting (lam,
    gamma) each period via Sharpe validation, with `kappa` FIXED (default
    0.0 -- "sparsity only, no robustness", per the requirement to compare
    against methods A-D).

    DIFFERENT from `walk_forward_backtest` (the long-only branch, which
    selects kappa/gamma, with NO lambda since L1 is meaningless under the
    long-only constraint -- see the `prox_solver.py` module docstring):
    here lambda ACTUALLY has an effect (controlling sparsity via
    soft-threshold + joint prox in `solve()`), while kappa is fixed (not
    in `param_grid`) instead of being auto-selected.

    BUG FIXED (2026-08-10, discovered when n_active=1 EVERY period in an
    abnormal way, even when lam=0 was selected -- there is no way L1=0
    would force the solution down to 1 asset): the final deploy step
    called `cvxpy_solve(mu_f, sigma_f, sqrt_f, kappa, *best_params)` with
    `best_params=(lam, gamma)`, while `cvxpy_solve` expects the order
    `(kappa, gamma, lam)` -- unpacking `*best_params` positionally caused
    the REAL gamma to receive the lam value (very small, at most 0.02 in
    the grid) and the REAL lam to receive the gamma value (1-10, 50-500x
    larger than every lam ever tried) -- risk aversion nearly = 0 plus a
    huge L1 penalty forced the portfolio down to exactly 1 asset (the one
    with the highest mu) every period, regardless of what (lam,gamma) were
    logged. The parameter-SELECTION loop (above) was unaffected (uses
    explicit variable names `kappa, gamma, lam`, not tuple unpacking) --
    only the final deploy step had the bug, so the (lam,gamma) logged in
    `rebalance_log` are the CORRECT selected parameters, just WRONG in
    that they were not the actual parameters used to solve the deployed
    solution. Fixed by explicitly unpacking
    `lam_star, gamma_star = best_params` before the call. See
    `test_walk_forward_backtest_long_short_deploys_with_correct_param_order`.

    MODELING CAVEAT (an important difference from the long-only branch):
    `_simulate_period` uses the "drift with price then renormalize to
    sum=1" formula (`w = w*(1+r); w = w/w.sum()`) -- CORRECT for the value
    weighting of long positions, but only an APPROXIMATION for short
    positions (it does not model actual stock-borrow fees/margin
    calls/margin interest). This is a limitation inherited from the
    original Markowitz formulation itself `w ∈ R^N` with no sign
    constraint (see the README/notebook), not a bug specific to this
    function. Turnover also has NO cap (solve() does not bound `‖w‖₁`
    beyond the soft penalty via lambda) -- see the `gross_exposure` column
    in `rebalance_log` to track the actual leverage level each period.

    Parameters
    ----------
    returns : pd.DataFrame, shape (T, N)
        Simple daily returns, index = date (ascending), columns = symbol.
    kappa : float, default 0.0
        FIXED robustness coefficient throughout (not in param_grid) -- 0.0
        means the robust term is fully disabled, leaving only
        mean-variance + sparsity.
    lookback_months, validation_months : int, default 24, 6
        See `walk_forward_backtest`.
    param_grid : list[tuple[float, float]] | None, default None
        List of (lam, gamma) to try each period; None ->
        `DEFAULT_PARAM_GRID_LONG_SHORT` (lam in {0, 0.001, 0.005, 0.01,
        0.02} x gamma in {1,5,10} = 15 combinations).
    shrinkage, fee, rf : see `walk_forward_backtest`.

    Returns
    -------
    BacktestResult
        `rebalance_log` has `lam`/`gamma` columns (instead of
        `kappa`/`gamma`) + an extra `gross_exposure` column
        (= `sum(|w|)`, >1 if there is any short position) compared to
        `walk_forward_backtest`.
    """
    from src.cvxpy_check import cvxpy_solve
    from src.estimators import estimate_all

    if param_grid is None:
        param_grid = DEFAULT_PARAM_GRID_LONG_SHORT

    windows = _build_rebalance_windows(returns.index, lookback_months, validation_months)
    if not windows:
        raise ValueError(
            f"Not enough data for lookback_months={lookback_months} "
            f"(have {returns.index.to_period('M').nunique()} months)."
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
    """OOS walk-forward on the LONG-SHORT branch with the FULL original
    equation (`cvxpy_solve` -- see the SOLVER NOTE at the top of the
    module, w has no sign constraint -- allows short-selling),
    automatically selecting ALL THREE (kappa, lam, gamma) each period via
    Sharpe validation.

    DIFFERENT from `walk_forward_backtest_long_short`: there, `kappa` is
    FIXED (not in param_grid, default 0.0 -- "sparsity only, no
    robustness", used for method E in the comparison table). Here `kappa`
    IS IN param_grid alongside `lam`/`gamma` -- meaning the robust term
    AND the sparsity term AND the risk term are all auto-selected each
    period, matching the full equation exactly:

        min_w  -mu^T w + kappa*||Sigma^(1/2) w||_2 + gamma*w^T Sigma w
               + lam*||w||_1
        s.t.   1^T w = 1

    Same modeling caveats as `walk_forward_backtest_long_short` (the
    "drift then renormalize to sum=1" is an approximation for short
    positions, does not model stock-borrow fees/margin; turnover has no
    cap -- track it via the `gross_exposure` column).

    Parameters
    ----------
    returns : pd.DataFrame, shape (T, N)
        Simple daily returns, index = date (ascending), columns = symbol.
    lookback_months, validation_months : int, default 24, 6
        See `walk_forward_backtest`.
    param_grid : list[tuple[float, float, float]] | None, default None
        List of (kappa, lam, gamma) to try each period; None ->
        `DEFAULT_PARAM_GRID_LONG_SHORT_FULL` (kappa in {0,0.5,1,2} x
        lam in {0,0.001,0.005,0.01,0.02} x gamma in {1,5,10} = 60
        combinations -- more than 5x larger than the 12-combination grid
        of the long-only branch, consider passing a smaller param_grid
        when running on the full 98-asset universe).
    shrinkage, fee, rf : see `walk_forward_backtest`.

    Returns
    -------
    BacktestResult
        `rebalance_log` has `kappa`/`lam`/`gamma` columns (all three,
        unlike `walk_forward_backtest_long_short` which only has
        `lam`/`gamma`) + `gross_exposure` (= `sum(|w|)`, >1 if there is
        any short position).
    """
    from src.cvxpy_check import cvxpy_solve
    from src.estimators import estimate_all

    if param_grid is None:
        param_grid = DEFAULT_PARAM_GRID_LONG_SHORT_FULL

    windows = _build_rebalance_windows(returns.index, lookback_months, validation_months)
    if not windows:
        raise ValueError(
            f"Not enough data for lookback_months={lookback_months} "
            f"(have {returns.index.to_period('M').nunique()} months)."
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
    """"Buy & hold" benchmark for the VN30 index (method D), over the SAME
    OOS period as `walk_forward_backtest`/`equal_weight_backtest`, for a
    fair comparison of the 4 methods over exactly the same time span.

    NO rebalancing (bought once at the start of the OOS period, held
    unchanged to the end) -- no estimation/turnover/weights since this is
    a ready-made market index, not a portfolio built from scratch from the
    VN100 universe (so it does NOT return a `BacktestResult` like the
    other 2 functions, only `daily_returns`). The transaction fee (if > 0)
    is deducted only ONCE on the first day of the OOS period (the
    purchase), unlike the other 2 methods (which deduct fees EVERY monthly
    rebalance period).

    Parameters
    ----------
    index_returns : pd.Series
        Simple daily return of the VN30 index (the "ret" column returned
        by `src.data_loader.load_vn30_index`), index=date.
    oos_index : pd.DatetimeIndex
        The list of OOS dates to compare against -- usually taken from
        `walk_forward_backtest(...).daily_returns.index` to ensure it
        matches exactly the same period as the other 3 methods.
    fee : float, default 0.002
        Proportional transaction fee, deducted once on the first day's
        return.

    Returns
    -------
    pd.Series
        Daily net return, index = intersection(oos_index, index_returns.index).
        Warns (warnings.warn) if any OOS day is missing corresponding VN30
        data -- the index and individual-stock trading calendars rarely
        diverge but an exact match is not absolutely guaranteed.
    """
    common = oos_index.intersection(index_returns.index)
    missing = oos_index.difference(index_returns.index)
    if len(missing) > 0:
        warnings.warn(
            f"index_buy_and_hold_backtest: {len(missing)}/{len(oos_index)} OOS days "
            f"have no corresponding VN30 data, skipped: "
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
    """"Buy & hold" benchmark for the VN100 index (method D, replacing the
    external VN30 index) -- buys equal-weight the very 98 assets in
    `returns` (SAME universe as A/B/C, not a differently-composed VN30
    basket) once at the start of the OOS period, held unchanged to the
    end, no rebalancing.

    Unlike `index_buy_and_hold_backtest` (which uses ready-made VN30 index
    prices from an external source) -- this function builds its own
    "index" by drifting with the prices of the very 98 assets in the
    universe under study, making it a fairer comparison against
    A/B/C/E/F (same 98 assets, differing only in weight allocation /
    rebalance frequency), rather than comparing against a 30-asset
    large-cap basket with an entirely different composition.

    Parameters
    ----------
    returns : pd.DataFrame, shape (T, N)
        Simple daily returns, index = date (ascending), columns = symbol --
        the SAME DataFrame passed to
        `walk_forward_backtest`/`equal_weight_backtest`.
    oos_index : pd.DatetimeIndex
        The list of OOS dates to compare against -- taken from
        `walk_forward_backtest(...).daily_returns.index` to ensure it
        matches exactly the same period as the other methods.
    fee : float, default 0.002
        Proportional transaction fee, deducted ONCE on the first day
        (the purchase), unlike `equal_weight_backtest` (which deducts
        fees EVERY monthly rebalance period).

    Returns
    -------
    pd.Series
        Daily net return, index = oos_index (sliced directly from
        `returns.loc[oos_index]`, no intersection needed since `returns`
        is the original data source for oos_index itself).
    """
    n = returns.shape[1]
    w_uniform = np.full(n, 1.0 / n)
    period_returns = returns.loc[oos_index]

    daily_net, _, _, _, wiped_out = _simulate_period(
        w_uniform, period_returns, prev_weights_drifted=None, fee=fee,
    )
    if wiped_out:
        warnings.warn(
            "vn100_buy_and_hold_backtest: portfolio 'wiped out' during the OOS period "
            "-- see the detailed warning from _simulate_period.",
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
    """Equal-weight 1/N benchmark on the SAME list of rebalance dates as
    `walk_forward_backtest` (uses the same `lookback_months`/
    `validation_months` ONLY to generate identical windows, ensuring a
    fair comparison over the same OOS period -- the benchmark estimates
    nothing (does not use E/Vwin), the target weight each period is
    always `1/N` with N = the number of assets in `returns`.

    Parameters
    ----------
    returns : pd.DataFrame, shape (T, N)
        Simple daily returns, index = date (ascending), columns = symbol.
    lookback_months : int, default 24
        Used only to generate the same list of rebalance dates as
        walk-forward (see `_build_rebalance_windows`); the benchmark does
        not need E/Vwin.
    validation_months : int, default 6
        Same as above -- only affects the list of rebalance dates, not
        used for any estimation here.
    fee : float, default 0.002
        Proportional transaction fee on turnover each period (see
        `_simulate_period`).

    Returns
    -------
    BacktestResult
    """
    windows = _build_rebalance_windows(returns.index, lookback_months, validation_months)
    if not windows:
        raise ValueError(f"Not enough data for lookback_months={lookback_months}.")

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
