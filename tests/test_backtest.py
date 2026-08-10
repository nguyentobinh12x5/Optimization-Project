import numpy as np
import pandas as pd
import pytest
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


def test_metrics_wipeout_does_not_raise_and_floors_annualized_return():
    """cumulative_return <= -1 (mất sạch vốn hoặc hơn, có thể xảy ra với
    danh mục long-short đòn bẩy cao) khiến (1+cumulative_return)**(1/n_years)
    nhận cơ số <= 0 -- PHẢI trả về -1.0 (chặn ở -100%), KHÔNG được ném lỗi
    (TypeError ép số phức sang float) như trước khi sửa.
    """
    # 1 ngày giảm đúng -100% -> cumulative_return <= -1 (mọi ngày sau vô nghĩa).
    daily_full_wipeout = pd.Series([-1.0, 0.3, -0.2])
    m = performance_metrics(daily_full_wipeout, rf=0.0)
    assert m["cumulative_return"] <= -1.0 + 1e-12
    assert m["annualized_return"] == -1.0
    assert isinstance(m["annualized_return"], float)  # không phải complex


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
    daily_net, turnover, cost, w_end, wiped_out = _simulate_period(
        w_start, period_returns, prev_weights_drifted=None, fee=0.002,
    )
    assert turnover == 1.0  # kỳ đầu, prev=None -> mua toàn bộ
    assert abs(cost - 0.002 * 1.0) < 1e-12
    assert abs(w_end.sum() - 1.0) < 1e-9
    assert len(daily_net) == 2
    assert wiped_out is False
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
    _, turnover, cost, _, _ = _simulate_period(w_start, period_returns, prev, fee=0.01)
    assert abs(turnover - np.abs(w_start - prev).sum()) < 1e-12
    assert abs(cost - 0.01 * turnover) < 1e-12


def test_simulate_period_wipeout_caps_return_and_zeros_remaining_days():
    """Danh mục đòn bẩy cao gặp 1 ngày lỗ >=100% -- PHẢI chặn ở đúng -100%
    (không lật dấu trọng số vô nghĩa như bug cũ), các ngày sau trong kỳ
    return=0, và phải phát cảnh báo RuntimeWarning."""
    from src.backtest import _simulate_period

    w_start = np.array([5.0, -4.0])  # gross exposure 9 -- long-short đòn bẩy cao
    period_returns = pd.DataFrame(
        {"A": [-0.5, 0.10, 0.20], "B": [0.6, -0.05, 0.05]},
        index=pd.bdate_range("2024-01-02", periods=3),
    )
    # port_ret ngày 1 = 5.0*(-0.5) + (-4.0)*0.6 = -2.5 - 2.4 = -4.9 <= -1.0 -> vỡ nợ

    with pytest.warns(RuntimeWarning, match="vỡ nợ"):
        daily_net, turnover, cost, w_end, wiped_out = _simulate_period(
            w_start, period_returns, prev_weights_drifted=None, fee=0.0,
        )

    assert wiped_out is True
    assert daily_net.iloc[0] == -1.0  # chặn đúng -100%, không phải -4.9
    assert daily_net.iloc[1] == 0.0  # ngày sau vỡ nợ -> return=0 (không lật dấu)
    assert daily_net.iloc[2] == 0.0
    assert np.allclose(w_end, 0.0)  # không còn vị thế nào sau khi vỡ nợ


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
    assert {
        "kappa", "gamma", "turnover", "cost", "n_active", "val_return", "val_sharpe",
    }.issubset(result.rebalance_log.columns)
    for _, w_row in result.weights.iterrows():
        assert (w_row.values >= -1e-6).all()
        assert abs(w_row.values.sum() - 1.0) < 1e-6
    # turnover kỳ đầu = 1.0 (mua từ đầu)
    assert abs(result.rebalance_log.iloc[0]["turnover"] - 1.0) < 1e-6


def test_walk_forward_backtest_max_weight_guarantees_minimum_active_count():
    """max_weight=0.20 (20%) phải đảm bảo TOÁN HỌC >= 5 mã active mỗi kỳ
    (vì mỗi mã đóng góp tối đa 20% vào tổng=1) -- kiểm tra trên universe đủ
    lớn (N=20) để không bị chặn bởi chính N (nếu N<5 thì test này vô nghĩa).
    Đồng thời xác nhận KHÔNG mã nào vượt quá max_weight (trong sai số solver).
    """
    from src.backtest import walk_forward_backtest

    rng = np.random.default_rng(99)
    n_days = 21 * 40
    index = pd.bdate_range("2020-01-01", periods=n_days)
    N = 20
    returns = pd.DataFrame(
        rng.normal(0.0003, 0.01, size=(n_days, N)), index=index,
        columns=[f"S{i}" for i in range(N)],
    )

    result = walk_forward_backtest(
        returns, lookback_months=24, validation_months=6,
        param_grid=[(0.0, 1.0), (1.0, 5.0)], shrinkage="lw", fee=0.002,
        max_weight=0.20,
    )

    for _, w_row in result.weights.iterrows():
        assert (w_row.values <= 0.20 + 1e-6).all()  # không mã nào vượt trần
        n_active = int((w_row.values > 1e-4).sum())
        assert n_active >= 5  # 5 * 20% = 100%, không thể ít hơn 5 mã
    assert result.rebalance_log["n_active"].min() >= 5


def test_walk_forward_backtest_selects_max_return_when_requested(monkeypatch):
    """Với selection_metric="return" (KHÔNG còn là mặc định, xem
    `test_walk_forward_backtest_selects_max_sharpe_by_default`) -- dựng 2
    "danh mục" đối lập: kappa=0.0 -> 100% tài sản A (return ngày nhỏ đều
    đặn, Sharpe validation cao nhưng cumulative return thấp), kappa=1.0 ->
    100% tài sản B (flat suốt kỳ trừ 1 ngày tăng vọt bên trong Vwin ->
    cumulative return validation cao hơn hẳn A, nhưng Sharpe rất thấp vì 1
    ngày biến động cực lớn). walk_forward_backtest phải chọn kappa=1.0.
    """
    import src.cvxpy_check as cvxpy_check
    from src.backtest import _build_rebalance_windows, walk_forward_backtest

    lookback_months, validation_months = 24, 6
    n_days = 21 * 40
    index = pd.bdate_range("2020-01-01", periods=n_days)
    windows = _build_rebalance_windows(index, lookback_months, validation_months)
    v_dates = index[windows[0]["v_mask"]]

    rng = np.random.default_rng(3)
    returns = pd.DataFrame(
        {
            "A": 0.0003 + rng.normal(0.0, 1e-6, n_days),
            "B": rng.normal(0.0, 1e-6, n_days),
        },
        index=index,
    )
    spike_day = v_dates[len(v_dates) // 2]
    returns.loc[spike_day, "B"] = 0.5  # 1 ngày tăng vọt bên trong Vwin của kỳ đầu

    def fake_cvxpy_solve_long_only(mu, sigma, sigma_sqrt, kappa, gamma, max_weight=None, **kwargs):
        w = np.array([1.0, 0.0]) if kappa == 0.0 else np.array([0.0, 1.0])
        return w, 0.0

    monkeypatch.setattr(cvxpy_check, "cvxpy_solve_long_only", fake_cvxpy_solve_long_only)

    result = walk_forward_backtest(
        returns, lookback_months=lookback_months, validation_months=validation_months,
        param_grid=[(0.0, 1.0), (1.0, 1.0)], shrinkage=None, fee=0.0,
        selection_metric="return",
    )

    first_row = result.rebalance_log.iloc[0]
    assert first_row["kappa"] == 1.0  # chọn B (return cao) chứ không phải A (Sharpe cao)
    assert first_row["val_return"] > 0.4  # phần lớn từ ngày tăng vọt 0.5


def test_walk_forward_backtest_selects_max_sharpe_by_default(monkeypatch):
    """MẶC ĐỊNH (không truyền selection_metric) giờ là "sharpe" -- đảo ngược
    của test trên: cùng kịch bản A (đều đặn, Sharpe cao) vs B (1 ngày tăng
    vọt, Sharpe thấp) thì walk_forward_backtest phải chọn kappa=0.0 (A),
    KHÔNG phải kappa=1.0 (B), NGAY CẢ KHI không truyền selection_metric --
    xác nhận default đã đổi thật, không chỉ tuỳ chọn tường minh mới hoạt
    động đúng.
    """
    import src.cvxpy_check as cvxpy_check
    from src.backtest import _build_rebalance_windows, walk_forward_backtest

    lookback_months, validation_months = 24, 6
    n_days = 21 * 40
    index = pd.bdate_range("2020-01-01", periods=n_days)
    windows = _build_rebalance_windows(index, lookback_months, validation_months)
    v_dates = index[windows[0]["v_mask"]]

    rng = np.random.default_rng(3)
    returns = pd.DataFrame(
        {
            "A": 0.0003 + rng.normal(0.0, 1e-6, n_days),
            "B": rng.normal(0.0, 1e-6, n_days),
        },
        index=index,
    )
    spike_day = v_dates[len(v_dates) // 2]
    returns.loc[spike_day, "B"] = 0.5

    def fake_cvxpy_solve_long_only(mu, sigma, sigma_sqrt, kappa, gamma, max_weight=None, **kwargs):
        w = np.array([1.0, 0.0]) if kappa == 0.0 else np.array([0.0, 1.0])
        return w, 0.0

    monkeypatch.setattr(cvxpy_check, "cvxpy_solve_long_only", fake_cvxpy_solve_long_only)

    result = walk_forward_backtest(
        returns, lookback_months=lookback_months, validation_months=validation_months,
        param_grid=[(0.0, 1.0), (1.0, 1.0)], shrinkage=None, fee=0.0,
        # KHÔNG truyền selection_metric -- kiểm tra đúng giá trị MẶC ĐỊNH.
    )

    first_row = result.rebalance_log.iloc[0]
    assert first_row["kappa"] == 0.0  # chọn A (Sharpe cao) chứ không phải B (return cao)


def test_walk_forward_backtest_rejects_invalid_selection_metric():
    from src.backtest import walk_forward_backtest

    rng = np.random.default_rng(1)
    n_days = 21 * 40
    index = pd.bdate_range("2020-01-01", periods=n_days)
    returns = pd.DataFrame(
        rng.normal(0.0003, 0.01, size=(n_days, 3)), index=index, columns=["A", "B", "C"],
    )
    with pytest.raises(ValueError):
        walk_forward_backtest(returns, selection_metric="bogus")


def test_walk_forward_backtest_long_short_runs_and_is_feasible():
    from src.backtest import walk_forward_backtest_long_short

    rng = np.random.default_rng(11)
    n_days = 21 * 40
    index = pd.bdate_range("2020-01-01", periods=n_days)
    N = 6
    returns = pd.DataFrame(
        rng.normal(0.0003, 0.01, size=(n_days, N)), index=index,
        columns=[f"S{i}" for i in range(N)],
    )

    result = walk_forward_backtest_long_short(
        returns, lookback_months=24, validation_months=6,
        param_grid=[(0.0, 1.0), (0.01, 5.0)], shrinkage="lw", fee=0.002,
    )

    assert not result.daily_returns.isna().any()
    assert len(result.rebalance_log) > 0
    assert {"lam", "gamma", "turnover", "cost", "n_active", "val_sharpe", "gross_exposure"}.issubset(
        result.rebalance_log.columns
    )
    # Budget constraint (sum=1) vẫn giữ dù cho phép short (không còn w>=0).
    for _, w_row in result.weights.iterrows():
        assert abs(w_row.values.sum() - 1.0) < 1e-6
    # gross_exposure = sum(|w|) phải >= 1 (bằng 1 chỉ khi mọi w cùng dấu, tức không short).
    assert (result.rebalance_log["gross_exposure"] >= 1.0 - 1e-6).all()


def test_walk_forward_backtest_long_short_deploys_with_correct_param_order(monkeypatch):
    """Regression cho bug đã sửa: bước deploy cuối từng gọi
    `cvxpy_solve(..., kappa, *best_params)` với `best_params=(lam, gamma)`
    trong khi `cvxpy_solve` cần thứ tự `(kappa, gamma, lam)` -- unpack theo
    VỊ TRÍ khiến gamma/lam bị hoán đổi (xem "BUG ĐÃ SỬA" trong docstring
    `walk_forward_backtest_long_short`). Test ghi lại MỌI lời gọi
    `cvxpy_solve` (cả bước chọn tham số lẫn bước deploy) và xác nhận
    (gamma, lam) truyền vào ĐÚNG với tổ hợp đã chọn, dùng lam != gamma rõ
    rệt (0.3 vs 7.0) để chắc chắn phát hiện được nếu hoán đổi tái diễn.
    """
    import src.cvxpy_check as cvxpy_check
    from src.backtest import walk_forward_backtest_long_short

    rng = np.random.default_rng(7)
    n_days = 21 * 40
    index = pd.bdate_range("2020-01-01", periods=n_days)
    N = 4
    returns = pd.DataFrame(
        rng.normal(0.0003, 0.01, size=(n_days, N)), index=index,
        columns=[f"S{i}" for i in range(N)],
    )

    calls: list[tuple[float, float, float]] = []

    def fake_cvxpy_solve(mu, sigma, sigma_sqrt, kappa, gamma, lam):
        calls.append((kappa, gamma, lam))
        n = mu.shape[0]
        return np.full(n, 1.0 / n), 0.0

    monkeypatch.setattr(cvxpy_check, "cvxpy_solve", fake_cvxpy_solve)

    LAM, GAMMA = 0.3, 7.0  # cố ý khác xa nhau để hoán đổi (nếu có) lộ rõ.
    walk_forward_backtest_long_short(
        returns, kappa=1.5, lookback_months=24, validation_months=6,
        param_grid=[(LAM, GAMMA)], shrinkage=None, fee=0.0,
    )

    assert len(calls) > 0
    # Chỉ 1 tổ hợp trong grid -> CẢ vòng chọn tham số LẪN bước deploy đều
    # phải gọi cvxpy_solve với đúng (kappa=1.5, gamma=7.0, lam=0.3).
    for kappa_call, gamma_call, lam_call in calls:
        assert kappa_call == 1.5
        assert gamma_call == GAMMA  # phải là 7.0 -- nếu bug tái diễn sẽ là 0.3
        assert lam_call == LAM  # phải là 0.3 -- nếu bug tái diễn sẽ là 7.0


def test_walk_forward_backtest_long_short_full_runs_and_is_feasible():
    from src.backtest import walk_forward_backtest_long_short_full

    rng = np.random.default_rng(23)
    n_days = 21 * 40
    index = pd.bdate_range("2020-01-01", periods=n_days)
    N = 6
    returns = pd.DataFrame(
        rng.normal(0.0003, 0.01, size=(n_days, N)), index=index,
        columns=[f"S{i}" for i in range(N)],
    )

    result = walk_forward_backtest_long_short_full(
        returns, lookback_months=24, validation_months=6,
        param_grid=[(0.0, 0.0, 1.0), (1.0, 0.01, 5.0)], shrinkage="lw", fee=0.002,
    )

    assert not result.daily_returns.isna().any()
    assert len(result.rebalance_log) > 0
    assert {
        "kappa", "lam", "gamma", "turnover", "cost", "n_active", "val_sharpe", "gross_exposure",
    }.issubset(result.rebalance_log.columns)
    # Budget constraint (sum=1) vẫn giữ dù cho phép short.
    for _, w_row in result.weights.iterrows():
        assert abs(w_row.values.sum() - 1.0) < 1e-6
    # gross_exposure = sum(|w|) phải >= 1 (bằng 1 chỉ khi mọi w cùng dấu, tức không short).
    assert (result.rebalance_log["gross_exposure"] >= 1.0 - 1e-6).all()
    # kappa được chọn phải luôn nằm trong param_grid truyền vào.
    assert result.rebalance_log["kappa"].isin([0.0, 1.0]).all()


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


def test_index_buy_and_hold_backtest_aligns_and_applies_fee_once():
    from src.backtest import index_buy_and_hold_backtest

    oos_index = pd.bdate_range("2024-01-02", periods=5)
    # index_returns có nhiều ngày hơn oos_index (trước và sau) -- phải bị cắt đúng theo oos_index
    wide_index = pd.bdate_range("2023-12-20", periods=20)
    index_returns = pd.Series(0.001, index=wide_index)

    out = index_buy_and_hold_backtest(index_returns, oos_index, fee=0.002)

    assert list(out.index) == list(oos_index)
    assert abs(out.iloc[0] - (0.001 - 0.002)) < 1e-12  # phí trừ đúng 1 lần, ngày đầu
    assert np.allclose(out.iloc[1:].values, 0.001)  # các ngày sau không bị trừ phí


def test_index_buy_and_hold_backtest_warns_on_missing_dates():
    from src.backtest import index_buy_and_hold_backtest

    oos_index = pd.bdate_range("2024-01-02", periods=5)
    # index_returns thiếu hẳn ngày cuối của oos_index
    short_index = oos_index[:-1]
    index_returns = pd.Series(0.001, index=short_index)

    with pytest.warns(UserWarning, match="ngày OOS"):
        out = index_buy_and_hold_backtest(index_returns, oos_index, fee=0.0)

    assert len(out) == 4
    assert oos_index[-1] not in out.index
