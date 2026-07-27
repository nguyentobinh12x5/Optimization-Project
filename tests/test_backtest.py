import numpy as np
import pandas as pd
from src.backtest import performance_metrics
from src.backtest import _build_rebalance_windows


def test_metrics_constant_return():
    r = 0.001
    n = 252
    daily = pd.Series([r] * n)
    m = performance_metrics(daily, rf=0.0)
    expected_cum = (1 + r) ** n - 1
    assert abs(m["cumulative_return"] - expected_cum) < 1e-9
    assert abs(m["annualized_vol"] - 0.0) < 1e-9
    assert np.isnan(m["sharpe"])  # vol=0 -> sharpe không xác định
    assert abs(m["max_drawdown"] - 0.0) < 1e-9
    assert m["n_days"] == n


def test_metrics_known_drawdown_sequence():
    # Giá trị danh mục: 1.0 -> 1.10 -> 1.045 -> 0.9928 (giảm) -> 1.20 (hồi phục)
    daily = pd.Series([0.10, -0.05, -0.05, 0.209])
    m = performance_metrics(daily, rf=0.0)
    curve = (1 + daily).cumprod()
    running_max = curve.cummax()
    expected_dd = float((curve / running_max - 1).min())
    assert abs(m["max_drawdown"] - expected_dd) < 1e-9
    assert m["max_drawdown"] < 0  # phải âm (giảm giá trị)


def test_metrics_sharpe_sign():
    rng = np.random.default_rng(0)
    daily_pos = pd.Series(rng.normal(0.001, 0.01, 500))
    daily_neg = pd.Series(rng.normal(-0.001, 0.01, 500))
    assert performance_metrics(daily_pos)["sharpe"] > 0
    assert performance_metrics(daily_neg)["sharpe"] < 0


def _fake_daily_index(n_months: int) -> pd.DatetimeIndex:
    # ~21 phiên/tháng, đơn giản hoá bằng business day range qua nhiều tháng
    return pd.bdate_range("2020-01-01", periods=n_months * 21, freq="B")


def test_build_windows_no_look_ahead():
    index = _fake_daily_index(30)  # 30 tháng dữ liệu giả lập
    windows = _build_rebalance_windows(index, lookback_months=24, validation_months=6)
    assert len(windows) > 0
    for w in windows:
        e_dates = index[w["e_mask"]]
        v_dates = index[w["v_mask"]]
        assert e_dates.max() < w["rebalance_date"]
        assert v_dates.max() < w["rebalance_date"]
        assert e_dates.max() < v_dates.min()  # E hoàn toàn trước Vwin


def test_build_windows_count_matches_lookback():
    index = _fake_daily_index(30)
    windows = _build_rebalance_windows(index, lookback_months=24, validation_months=6)
    # Có dữ liệu 30 tháng, cần 24 tháng lookback -> tối đa ~6 kỳ rebalance hợp lệ
    assert 1 <= len(windows) <= 7


def test_simulate_period_first_period_full_turnover():
    from src.backtest import _simulate_period
    w_start = np.array([0.5, 0.5])
    period_returns = pd.DataFrame(
        {"A": [0.01, 0.02], "B": [-0.01, 0.0]},
        index=pd.bdate_range("2024-01-02", periods=2),
    )
    daily_net, turnover, cost, w_end = _simulate_period(
        w_start, period_returns, prev_weights_drifted=None, fee=0.002,
    )
    assert turnover == 1.0  # kỳ đầu, prev=None -> mua toàn bộ
    assert abs(cost - 0.002 * 1.0) < 1e-12
    assert abs(w_end.sum() - 1.0) < 1e-9
    assert len(daily_net) == 2
    # ngày đầu đã trừ cost
    gross_day0 = w_start @ period_returns.iloc[0].values
    assert abs(daily_net.iloc[0] - (gross_day0 - cost)) < 1e-12


def test_simulate_period_turnover_against_prev_drifted():
    from src.backtest import _simulate_period
    w_start = np.array([0.6, 0.4])
    prev = np.array([0.5, 0.5])
    period_returns = pd.DataFrame(
        {"A": [0.0], "B": [0.0]}, index=pd.bdate_range("2024-02-01", periods=1),
    )
    _, turnover, cost, _ = _simulate_period(w_start, period_returns, prev, fee=0.01)
    assert abs(turnover - np.abs(w_start - prev).sum()) < 1e-12
    assert abs(cost - 0.01 * turnover) < 1e-12


def test_walk_forward_backtest_runs_and_is_feasible():
    from src.backtest import walk_forward_backtest

    rng = np.random.default_rng(42)
    n_days = 21 * 40  # 40 tháng giả lập, đủ cho lookback 24 + vài kỳ OOS
    index = pd.bdate_range("2020-01-01", periods=n_days)
    N = 6
    returns = pd.DataFrame(
        rng.normal(0.0003, 0.01, size=(n_days, N)), index=index,
        columns=[f"S{i}" for i in range(N)],
    )

    result = walk_forward_backtest(
        returns, lookback_months=24, validation_months=6,
        param_grid=[(0.0, 1.0), (1.0, 5.0)], shrinkage="lw", fee=0.002,
    )

    assert not result.daily_returns.isna().any()
    assert len(result.rebalance_log) > 0
    assert {"kappa", "gamma", "turnover", "cost", "n_active", "val_sharpe"}.issubset(
        result.rebalance_log.columns
    )
    for _, w_row in result.weights.iterrows():
        assert (w_row.values >= -1e-6).all()
        assert abs(w_row.values.sum() - 1.0) < 1e-6
    # turnover kỳ đầu = 1.0 (mua từ đầu)
    assert abs(result.rebalance_log.iloc[0]["turnover"] - 1.0) < 1e-6


def test_equal_weight_backtest_matches_uniform_weights():
    from src.backtest import equal_weight_backtest, walk_forward_backtest

    rng = np.random.default_rng(7)
    n_days = 21 * 40
    index = pd.bdate_range("2020-01-01", periods=n_days)
    N = 5
    returns = pd.DataFrame(
        rng.normal(0.0002, 0.01, size=(n_days, N)), index=index,
        columns=[f"S{i}" for i in range(N)],
    )

    bench = equal_weight_backtest(returns, lookback_months=24, validation_months=6, fee=0.002)

    assert not bench.daily_returns.isna().any()
    for _, w_row in bench.weights.iterrows():
        assert np.allclose(w_row.values, np.full(N, 1.0 / N), atol=1e-9)

    # Cùng số kỳ rebalance với walk-forward trên cùng data/tham số cửa sổ
    wf = walk_forward_backtest(
        returns, lookback_months=24, validation_months=6,
        param_grid=[(0.0, 1.0)],
    )
    assert list(bench.rebalance_log.index) == list(wf.rebalance_log.index)
