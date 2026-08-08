# Task Brief — Walk-Forward Backtest: Task 4 (core engine — task lớn nhất)

Đây là brief trích từ plan đầy đủ: `docs/superpowers/plans/2026-07-26-walk-forward-backtest.md`.

## Global Constraints (áp dụng cho toàn bộ plan, đọc kỹ)

- Spec đầy đủ: `docs/superpowers/specs/2026-07-26-walk-forward-backtest-design.md`.
- Solver long-short hiện tại (`solve()`, `portfolio_objective()`, `_prox_l1_simplex_eq`, `_robust_subgrad`) GIỮ NGUYÊN KHÔNG SỬA — đã CVXPY-verify, các phase trước phụ thuộc vào nó.
- Long-only: ràng buộc `w≥0, Σw=1`. Vì `‖w‖₁=Σw=1` là hằng số dưới ràng buộc này, **λ vô tác dụng** → KHÔNG đưa λ vào đường long-only.
- Rolling window: `lookback_months=24` (`E`=18 tháng đầu, `Vwin`=6 tháng cuối), rebalance **hàng tháng** (ngày giao dịch đầu tiên mỗi tháng).
- Param grid mặc định: κ ∈ {0, 0.5, 1, 2} × γ ∈ {1, 5, 10} = 12 tổ hợp. Chọn theo **Sharpe (rf=0)** trên `Vwin`.
- Shrinkage `"lw"` (Ledoit-Wolf) BẬT xuyên suốt backtest (cả lúc chọn tham số lẫn giải nghiệm cuối).
- Phí giao dịch mặc định `fee=0.002` (0.20%) × turnover mỗi kỳ; kỳ đầu turnover=1.
- KHÔNG look-ahead: mọi ước lượng/validation của kỳ `t` chỉ dùng dữ liệu có ngày < ngày rebalance `t`. Phải có test cấu trúc khẳng định điều này.
- Môi trường: `.venv/bin/python` (Python 3.14). KHÔNG scipy/sklearn — mọi thuật toán (kể cả simplex projection) viết bằng numpy thuần. Không pip install gì thêm (cvxpy đã có).
- Data thật: `data/returns.parquet` (965×98, đã sạch — xem `src/data_loader.py`).
- Project KHÔNG dùng git — KHÔNG chạy `git init`/`git add`/`git commit`. Ghi nhận hoàn tất từng task bằng cách append 1 dòng vào `.sdd/progress.md` (đúng quy ước ledger đã dùng xuyên suốt các phase trước của project).
- Style code/docstring: tiếng Việt/Anh nhất quán như các module hiện có, người đọc là dân tối ưu hoá.

---

### Task 4: Walk-forward backtest engine (core)

**Files:**
- Modify: `src/backtest.py` (thêm vào file đã tạo ở Task 3)
- Modify: `tests/test_backtest.py`

**Interfaces:**
- Consumes: `solve_long_only` (Task 1, `src/prox_solver.py`), `estimate_all` (`src/estimators.py`), `performance_metrics` (Task 3).
- Produces (dùng ở Task 5, 6, 7):
  - `BacktestResult` (dataclass): `daily_returns: pd.Series`, `rebalance_log: pd.DataFrame`, `weights: pd.DataFrame`.
  - `_build_rebalance_windows(index, lookback_months, validation_months) -> list[dict]` (internal, nhưng test trực tiếp để verify no-look-ahead).
  - `_simulate_period(w_start, period_returns, prev_weights_drifted, fee) -> tuple[pd.Series, float, float, np.ndarray]` — trả (daily_net_returns, turnover, cost, w_end_drifted). Dùng chung cho walk-forward VÀ equal-weight (Task 5).
  - `walk_forward_backtest(returns, *, lookback_months=24, validation_months=6, param_grid=None, shrinkage="lw", fee=0.002, rf=0.0) -> BacktestResult`.

- [ ] **Step 1: Viết test cho `_build_rebalance_windows` (no-look-ahead, thuần logic)**

```python
import pandas as pd
from src.backtest import _build_rebalance_windows


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
```

- [ ] **Step 2: Chạy test, xác nhận FAIL**

Run: `.venv/bin/python -m pytest tests/test_backtest.py -k build_windows -v`
Expected: FAIL.

- [ ] **Step 3: Implement `_build_rebalance_windows`**

```python
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
    riêng khẳng định.
    """
    periods = index.to_period("M")
    months = sorted(periods.unique())
    windows: list[dict] = []

    for i in range(lookback_months, len(months)):
        target_month = months[i]
        window_months = window_months = months[i - lookback_months : i]
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
```

(Sửa lỗi gõ trùng `window_months = window_months = ...` thành `window_months = ...` khi viết thật -- giữ lại 1 lần gán.)

- [ ] **Step 4: Chạy test, xác nhận PASS**

Run: `.venv/bin/python -m pytest tests/test_backtest.py -k build_windows -v`
Expected: PASS.

- [ ] **Step 5: Viết test cho `_simulate_period`**

```python
import numpy as np


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
```

- [ ] **Step 6: Chạy test, xác nhận FAIL, rồi implement `_simulate_period`**

```python
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
    bị trừ vào return của NGÀY ĐẦU TIÊN trong kỳ (ngày rebalance).
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
```

- [ ] **Step 7: Chạy test, xác nhận PASS**

Run: `.venv/bin/python -m pytest tests/test_backtest.py -v`
Expected: PASS toàn bộ (metrics + build_windows + simulate_period).

- [ ] **Step 8: Viết test cho `walk_forward_backtest` (chạy trên data giả lập nhỏ, KHÔNG cần data thật)**

```python
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
```

- [ ] **Step 9: Chạy test, xác nhận FAIL**

Run: `.venv/bin/python -m pytest tests/test_backtest.py -k walk_forward -v`
Expected: FAIL.

- [ ] **Step 10: Implement `BacktestResult` (dataclass) và `walk_forward_backtest`**

```python
from dataclasses import dataclass


@dataclass
class BacktestResult:
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

    KHÔNG LOOK-AHEAD: với mỗi kỳ, estimation (E) và validation (Vwin) chỉ
    dùng dữ liệu có ngày < ngày rebalance (đảm bảo bởi _build_rebalance_windows,
    có test cấu trúc riêng). Trọng số triển khai ước lượng trên E union Vwin,
    KHÔNG dùng dữ liệu của tháng target.
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
```

- [ ] **Step 11: Chạy test, xác nhận PASS**

Run: `.venv/bin/python -m pytest tests/test_backtest.py -v`
Expected: PASS toàn bộ.

- [ ] **Step 12: Chạy thử trên DATA THẬT (smoke test, không phải test chính thức) để đo thời gian**

```bash
.venv/bin/python -c "
import time, pandas as pd
from src.backtest import walk_forward_backtest
returns = pd.read_parquet('data/returns.parquet')
t0 = time.monotonic()
result = walk_forward_backtest(returns)
print('elapsed:', time.monotonic() - t0, 's')
print('n periods:', len(result.rebalance_log))
print(result.rebalance_log)
"
```

Ghi lại thời gian chạy thật và số kỳ rebalance vào report. Nếu > 5 phút, cân nhắc giảm `max_iter` mặc định của `solve_long_only` khi gọi trong vòng lặp validation (không bắt buộc, chỉ nếu thật sự chậm).

- [ ] **Step 13: Ghi nhận hoàn tất task**

Project này KHÔNG dùng git (xem Global Constraints) — thay vì `git commit`, append 1 dòng vào `.sdd/progress.md` theo đúng quy ước ledger project đã dùng xuyên suốt các phase trước:
`Task <tên task>: complete (feat: implement walk-forward backtest engine)`

---

