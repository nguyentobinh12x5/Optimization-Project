# Task Brief — Walk-Forward Backtest: Task 2

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

### Task 2: Verify solver long-only bằng CVXPY

**Files:**
- Modify: `src/cvxpy_check.py`
- Modify: `tests/test_cvxpy_check.py`

**Interfaces:**
- Consumes: `solve_long_only`, `portfolio_objective_long_only` từ `src/prox_solver.py` (Task 1); `estimate_all` từ `src/estimators.py`.
- Produces: `cvxpy_solve_long_only(mu, sigma, sigma_sqrt, kappa, gamma) -> tuple[np.ndarray, float]`; `compare_long_only(mu, sigma, sigma_sqrt, param_grid, *, active_thresh=1e-4) -> pd.DataFrame` (dùng ở Task 4 để double-check nếu cần, nhưng mục đích chính là test ở đây).

- [ ] **Step 1: Đọc `cvxpy_solve`/`compare` hiện có trong `src/cvxpy_check.py`** để tái dùng đúng pattern (cách gọi `cp.psd_wrap`, `cp.quad_form`, solver CLARABEL).

- [ ] **Step 2: Viết test so sánh long-only vs CVXPY**

```python
def test_long_only_matches_cvxpy_on_real_data():
    import pandas as pd
    from src.estimators import estimate_all
    from src.cvxpy_check import cvxpy_solve_long_only, compare_long_only
    from src.prox_solver import solve_long_only

    returns = pd.read_parquet("data/returns.parquet")
    mu, sigma, sigma_sqrt = estimate_all(returns, shrinkage="lw")

    param_grid = [(0.0, 1.0), (0.5, 5.0), (1.0, 5.0), (2.0, 10.0)]
    df = compare_long_only(mu, sigma, sigma_sqrt, param_grid)

    assert (df["relgap"].abs() < 0.005).all(), df[["kappa", "gamma", "relgap"]]
    assert (df["w_hand"].apply(lambda w: (w >= -1e-6).all())).all()
    assert (df["w_hand"].apply(lambda w: abs(w.sum() - 1.0) < 1e-6)).all()
```

(Điều chỉnh tên cột DataFrame cho khớp với thiết kế thật của `compare_long_only` bạn viết ở Step 3 -- giữ tên cột `relgap` nhất quán với `compare()` cũ trong cùng file.)

- [ ] **Step 3: Chạy test, xác nhận FAIL**

Run: `.venv/bin/python -m pytest tests/test_cvxpy_check.py -k long_only -v`
Expected: FAIL (hàm chưa tồn tại).

- [ ] **Step 4: Implement `cvxpy_solve_long_only` và `compare_long_only`**

```python
def cvxpy_solve_long_only(
    mu: np.ndarray, sigma: np.ndarray, sigma_sqrt: np.ndarray,
    kappa: float, gamma: float,
) -> tuple[np.ndarray, float]:
    """CVXPY ground-truth cho bài toán long-only (w>=0, sum(w)=1), KHÔNG có
    term lambda*||w||_1 (xem prox_solver.portfolio_objective_long_only)."""
    n = len(mu)
    w = cp.Variable(n)
    obj = (
        -mu @ w
        + kappa * cp.norm(sigma_sqrt @ w, 2)
        + gamma * cp.quad_form(w, cp.psd_wrap(sigma))
    )
    prob = cp.Problem(cp.Minimize(obj), [w >= 0, cp.sum(w) == 1])
    prob.solve(solver=cp.CLARABEL)
    return w.value, prob.value


def compare_long_only(
    mu: np.ndarray, sigma: np.ndarray, sigma_sqrt: np.ndarray,
    param_grid: list[tuple[float, float]], *, active_thresh: float = 1e-4,
) -> pd.DataFrame:
    """So sánh solve_long_only (tay) vs cvxpy_solve_long_only (ground-truth)
    trên nhiều bộ (kappa, gamma). Cột trả về tương tự compare() (long-short)
    nhưng không có cột lambda."""
    from src.prox_solver import solve_long_only, portfolio_objective_long_only

    rows = []
    for kappa, gamma in param_grid:
        w_cvx, f_cvx = cvxpy_solve_long_only(mu, sigma, sigma_sqrt, kappa, gamma)
        result = solve_long_only(mu, sigma, sigma_sqrt, kappa, gamma)
        f_hand = result.best_obj
        relgap = (f_hand - f_cvx) / max(abs(f_cvx), 1e-12)
        active_hand = set(np.where(np.abs(result.w) > active_thresh)[0])
        active_cvx = set(np.where(np.abs(w_cvx) > active_thresh)[0])
        jaccard = len(active_hand & active_cvx) / max(len(active_hand | active_cvx), 1)
        rows.append({
            "kappa": kappa, "gamma": gamma,
            "f_hand": f_hand, "f_cvx": f_cvx, "relgap": relgap,
            "active_hand": len(active_hand), "active_cvx": len(active_cvx),
            "jaccard": jaccard, "linf": float(np.max(np.abs(result.w - w_cvx))),
            "n_iter": result.n_iter, "converged": result.converged,
            "w_hand": result.w, "w_cvx": w_cvx,
        })
    return pd.DataFrame(rows)
```

Kiểm tra `import cvxpy as cp` đã có ở đầu file (chắc chắn có, vì `cvxpy_solve` cũ dùng nó).

- [ ] **Step 5: Chạy test thật trên data thật, xác nhận PASS**

Run: `.venv/bin/python -m pytest tests/test_cvxpy_check.py -v`
Expected: PASS toàn bộ. Ghi lại relgap thật quan sát được vào report (kỳ vọng < 0.5%, tương tự đường long-short).

- [ ] **Step 6: Ghi nhận hoàn tất task**

Project này KHÔNG dùng git (xem Global Constraints) — thay vì `git commit`, append 1 dòng vào `.sdd/progress.md` theo đúng quy ước ledger project đã dùng xuyên suốt các phase trước:
`Task <tên task>: complete (feat: verify long-only solver against CVXPY)`

---

