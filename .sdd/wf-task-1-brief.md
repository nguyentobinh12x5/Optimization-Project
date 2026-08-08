# Task Brief — Walk-Forward Backtest: Task 1

Đây là brief trích từ plan đầy đủ: `docs/superpowers/plans/2026-07-26-walk-forward-backtest.md`
(đọc file plan đó nếu cần thêm ngữ cảnh, nhưng brief này đã đủ để thực hiện task).

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

### Task 1: Simplex projection + solver long-only

**Files:**
- Modify: `src/prox_solver.py` (thêm hàm mới, KHÔNG sửa hàm cũ)
- Test: `tests/test_prox_solver.py` (thêm test mới vào file hiện có)

**Interfaces:**
- Consumes: `_robust_subgrad(w, sigma, sigma_sqrt, kappa)` và cách tính smooth-subgradient đã có sẵn trong file (đọc code hiện tại của `solve()` để tái dùng đúng công thức `g = -mu + 2*gamma*sigma@w + robust_subgrad`).
- Produces (dùng ở Task 2, 4):
  - `simplex_projection(v: np.ndarray) -> np.ndarray` — chiếu Euclid lên `{w>=0, sum(w)=1}`.
  - `portfolio_objective_long_only(w: np.ndarray, mu: np.ndarray, sigma: np.ndarray, sigma_sqrt: np.ndarray, kappa: float, gamma: float) -> float` — objective KHÔNG có term `lambda*||w||_1`.
  - `solve_long_only(mu, sigma, sigma_sqrt, kappa, gamma, *, max_iter=5000, alpha0=10.0, tol=1e-8, patience=100, w0=None) -> SolveResult` — dùng lại `SolveResult` (dataclass) đã có trong file.

- [ ] **Step 1: Viết test cho `simplex_projection`**

```python
def test_simplex_projection_symmetric_case():
    v = np.array([0.5, 0.5, 0.5])
    w = simplex_projection(v)
    assert np.allclose(w, np.array([1 / 3, 1 / 3, 1 / 3]), atol=1e-10)
    assert abs(w.sum() - 1.0) < 1e-10
    assert (w >= -1e-12).all()


def test_simplex_projection_dominant_component():
    v = np.array([2.0, 0.2, 0.2])
    w = simplex_projection(v)
    assert np.allclose(w, np.array([1.0, 0.0, 0.0]), atol=1e-10)


def test_simplex_projection_already_feasible_is_fixed_point():
    v = np.array([0.2, 0.3, 0.5])  # đã w>=0, sum=1
    w = simplex_projection(v)
    assert np.allclose(w, v, atol=1e-10)


def test_simplex_projection_random_always_feasible():
    rng = np.random.default_rng(1)
    for _ in range(20):
        v = rng.normal(0, 3, size=10)
        w = simplex_projection(v)
        assert (w >= -1e-9).all()
        assert abs(w.sum() - 1.0) < 1e-8
```

- [ ] **Step 2: Chạy test, xác nhận FAIL (chưa có hàm)**

Run: `.venv/bin/python -m pytest tests/test_prox_solver.py -k simplex_projection -v`
Expected: FAIL với `ImportError`/`NameError` (hàm chưa tồn tại).

- [ ] **Step 3: Implement `simplex_projection` (thuật toán Duchi et al. 2008)**

Thêm vào `src/prox_solver.py`:

```python
def simplex_projection(v: np.ndarray) -> np.ndarray:
    """Chiếu Euclid vector v lên probability simplex {w : w>=0, sum(w)=1}.

    Thuật toán Duchi, Shalev-Shwartz, Singer, Chandra (2008) "Efficient
    Projections onto the l1-Ball for Learning in High Dimensions", O(N log N):
    sort v giảm dần thành u; tìm rho = chỉ số lớn nhất j sao cho
    u_j - (cumsum(u)_j - 1)/j > 0; theta = (cumsum(u)_rho - 1)/rho;
    w = max(v - theta, 0). Kết quả LUÔN thoả w>=0 và sum(w)=1 (tới sai số
    số học), bất kể v là gì -- không cần v đã "gần" simplex.
    """
    n = v.shape[0]
    u = np.sort(v)[::-1]
    cumsum_u = np.cumsum(u)
    j = np.arange(1, n + 1)
    cond = u - (cumsum_u - 1) / j > 0
    rho = np.nonzero(cond)[0][-1]  # chỉ số 0-based của j lớn nhất thoả cond
    theta = (cumsum_u[rho] - 1) / (rho + 1)
    return np.maximum(v - theta, 0.0)
```

- [ ] **Step 4: Chạy test simplex_projection, xác nhận PASS**

Run: `.venv/bin/python -m pytest tests/test_prox_solver.py -k simplex_projection -v`
Expected: PASS toàn bộ 4 test.

- [ ] **Step 5: Viết test cho `solve_long_only` (đáp án đóng: Sigma=I, mu=0 → uniform)**

```python
def test_solve_long_only_uniform_when_isotropic():
    """min w^T w s.t. w>=0, sum(w)=1 (mu=0, kappa=0, Sigma=I) có nghiệm đóng
    w = 1/N (do tính đối xứng -- đây là bài toán min-variance isotropic
    kinh điển, nghiệm là phân bổ đều)."""
    N = 6
    mu = np.zeros(N)
    sigma = np.eye(N)
    sigma_sqrt = np.eye(N)
    result = solve_long_only(mu, sigma, sigma_sqrt, kappa=0.0, gamma=1.0)
    assert np.allclose(result.w, np.full(N, 1.0 / N), atol=1e-3)
    assert abs(result.w.sum() - 1.0) < 1e-9
    assert (result.w >= -1e-9).all()


def test_solve_long_only_feasible_general_case():
    rng = np.random.default_rng(2)
    N = 8
    A = rng.normal(0, 1, size=(N, N))
    sigma = A @ A.T / N + 0.1 * np.eye(N)
    sigma_sqrt = matrix_sqrt_psd(sigma)
    mu = rng.normal(0, 0.001, size=N)
    result = solve_long_only(mu, sigma, sigma_sqrt, kappa=1.0, gamma=5.0)
    assert (result.w >= -1e-8).all()
    assert abs(result.w.sum() - 1.0) < 1e-6
    assert not np.isnan(result.obj_history).any()


def test_solve_long_only_returns_best_not_last():
    rng = np.random.default_rng(3)
    N = 5
    A = rng.normal(0, 1, size=(N, N))
    sigma = A @ A.T / N + 0.1 * np.eye(N)
    sigma_sqrt = matrix_sqrt_psd(sigma)
    mu = rng.normal(0, 0.001, size=N)
    result = solve_long_only(mu, sigma, sigma_sqrt, kappa=0.5, gamma=2.0)
    f_w = portfolio_objective_long_only(result.w, mu, sigma, sigma_sqrt, 0.5, 2.0)
    assert abs(f_w - result.best_obj) < 1e-6
```

(Import `matrix_sqrt_psd` từ `src.estimators` ở đầu file test nếu chưa có.)

- [ ] **Step 6: Chạy test, xác nhận FAIL**

Run: `.venv/bin/python -m pytest tests/test_prox_solver.py -k solve_long_only -v`
Expected: FAIL (hàm chưa tồn tại).

- [ ] **Step 7: Implement `portfolio_objective_long_only` và `solve_long_only`**

Đọc kỹ `solve()` hiện có trong `src/prox_solver.py` trước khi viết, để tái dùng ĐÚNG công thức subgradient (`-mu + 2*gamma*sigma@w + robust term qua _robust_subgrad`) và cấu trúc `SolveResult`/best-iterate/điều kiện dừng. Thêm:

```python
def portfolio_objective_long_only(
    w: np.ndarray, mu: np.ndarray, sigma: np.ndarray, sigma_sqrt: np.ndarray,
    kappa: float, gamma: float,
) -> float:
    """Objective long-only: giống portfolio_objective() nhưng KHÔNG có term
    lambda*||w||_1 -- dưới ràng buộc w>=0, sum(w)=1 thì ||w||_1 = sum(w) = 1
    là hằng số, phạt L1 không còn tác dụng điều khiển nghiệm nên bị loại
    (xem docs/superpowers/specs/2026-07-26-walk-forward-backtest-design.md
    mục 2)."""
    robust = kappa * np.linalg.norm(sigma_sqrt @ w, 2)
    risk = gamma * float(w @ sigma @ w)
    ret = -float(mu @ w)
    return ret + robust + risk


def solve_long_only(
    mu: np.ndarray, sigma: np.ndarray, sigma_sqrt: np.ndarray,
    kappa: float, gamma: float, *,
    max_iter: int = 5000, alpha0: float = 10.0, tol: float = 1e-8,
    patience: int = 100, w0: np.ndarray | None = None,
) -> SolveResult:
    """Projected subgradient trên probability simplex (long-only).

    Mỗi vòng: g = -mu + 2*gamma*sigma@w + robust_subgrad(w) (CÙNG công thức
    subgradient trơn+robust như solve() long-short); z = w - alpha_k*g;
    w_{k+1} = simplex_projection(z). KHÔNG có bước prox L1 (lambda bị loại,
    xem portfolio_objective_long_only). Trả best-iterate (subgradient không
    đơn điệu, giống solve()).
    """
    n = len(mu)
    w = np.full(n, 1.0 / n) if w0 is None else w0.copy()
    obj_history = np.empty(max_iter)
    best_w = w.copy()
    best_obj = portfolio_objective_long_only(w, mu, sigma, sigma_sqrt, kappa, gamma)
    obj_history[0] = best_obj
    stall_count = 0
    n_iter = 1

    for k in range(1, max_iter):
        alpha_k = alpha0 / np.sqrt(k + 1)
        robust_term = _robust_subgrad(w, sigma, sigma_sqrt, kappa)
        g = -mu + 2 * gamma * (sigma @ w) + robust_term
        z = w - alpha_k * g
        w = simplex_projection(z)

        f_w = portfolio_objective_long_only(w, mu, sigma, sigma_sqrt, kappa, gamma)
        obj_history[k] = f_w
        n_iter = k + 1

        if f_w < best_obj - tol * max(abs(best_obj), 1.0):
            best_obj = f_w
            best_w = w.copy()
            stall_count = 0
        else:
            stall_count += 1

        if stall_count >= patience:
            break

    converged = stall_count >= patience
    return SolveResult(
        w=best_w, obj_history=obj_history[:n_iter], best_obj=best_obj,
        n_iter=n_iter, converged=converged,
    )
```

QUAN TRỌNG: kiểm tra chữ ký thật của `_robust_subgrad` trong file (tham số, thứ tự) và `SolveResult` (tên field chính xác) trước khi copy đoạn trên -- sửa lại cho khớp 100% với những gì ĐÃ CÓ trong `solve()`, đừng đoán.

- [ ] **Step 8: Chạy lại toàn bộ test file, xác nhận PASS**

Run: `.venv/bin/python -m pytest tests/test_prox_solver.py -v`
Expected: PASS toàn bộ (test cũ + test mới).

- [ ] **Step 9: Ghi nhận hoàn tất task**

Project này KHÔNG dùng git (xem Global Constraints) — thay vì `git commit`, append 1 dòng vào `.sdd/progress.md` theo đúng quy ước ledger project đã dùng xuyên suốt các phase trước:
`Task <tên task>: complete (feat: add simplex projection and long-only solver)`
(Nếu project chưa phải git repo, bỏ qua bước git — ghi rõ trong report thay vì lỗi.)

---

