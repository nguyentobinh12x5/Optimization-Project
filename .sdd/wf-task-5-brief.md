# Task Brief — Walk-Forward Backtest: Task 5

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

### Task 5: Equal-weight benchmark

**Files:**
- Modify: `src/backtest.py`
- Modify: `tests/test_backtest.py`

**Interfaces:**
- Consumes: `_build_rebalance_windows`, `_simulate_period`, `BacktestResult` (Task 4).
- Produces: `equal_weight_backtest(returns, *, lookback_months=24, validation_months=6, fee=0.002) -> BacktestResult` (dùng `lookback_months`/`validation_months` CHỈ để tạo cùng danh sách ngày rebalance như walk-forward, để so sánh công bằng trên cùng giai đoạn OOS -- benchmark KHÔNG cần ước lượng gì, trọng số luôn `1/N`).

- [ ] **Step 1: Viết test**

```python
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
```

- [ ] **Step 2: Chạy test, xác nhận FAIL, rồi implement**

```python
def equal_weight_backtest(
    returns: pd.DataFrame,
    *,
    lookback_months: int = 24,
    validation_months: int = 6,
    fee: float = 0.002,
) -> BacktestResult:
    """Benchmark equal-weight 1/N trên CÙNG danh sách ngày rebalance như
    walk_forward_backtest (dùng cùng lookback_months/validation_months chỉ
    để sinh windows -- benchmark không ước lượng gì, không cần Vwin/E)."""
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
```

- [ ] **Step 3: Chạy test, xác nhận PASS**

Run: `.venv/bin/python -m pytest tests/test_backtest.py -v`
Expected: PASS toàn bộ.

- [ ] **Step 4: Ghi nhận hoàn tất task**

Project này KHÔNG dùng git (xem Global Constraints) — thay vì `git commit`, append 1 dòng vào `.sdd/progress.md` theo đúng quy ước ledger project đã dùng xuyên suốt các phase trước:
`Task <tên task>: complete (feat: add equal-weight benchmark backtest)`

---

