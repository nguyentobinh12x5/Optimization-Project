# Task Brief — Phase 3: Proximal-Subgradient Solver (viết tay)

Đây là YÊU CẦU của bạn. Đọc trước tiên. Thiết kế đã được user duyệt (Gate 2) — KHÔNG đổi cách tiếp cận.

## Mục tiêu
Viết `src/prox_solver.py`: giải bài toán bằng proximal-subgradient TỰ VIẾT (numpy thuần, KHÔNG dùng cvxpy/scipy.optimize hay solver có sẵn). Kèm test `tests/test_prox_solver.py`.

## Bài toán
```
min_w  f(w) = -μ̂ᵀw + κ‖Σ^½ w‖₂ + γ wᵀΣw + λ‖w‖₁
s.t.   1ᵀw = 1
```
w có thể âm (cho phép bán khống — KHÔNG ràng buộc w≥0).

## Môi trường
- `.venv/bin/python` (Python 3.14). Có numpy/pandas/pytest. KHÔNG scipy/sklearn/cvxpy — đừng import, đừng pip install.
- Consume Phase 2: `from src.estimators import estimate_all`. Input thật: `data/returns.parquet` (1068×97).
- μ̂, Σ, Σ^½ đều là numpy array; thứ tự asset = `list(returns.columns)`.

## Thuật toán (đã duyệt — cài đặt ĐÚNG như sau)
Mỗi vòng lặp k (bắt đầu w₀, mặc định w₀ = 1/N vector đều, thỏa 1ᵀw=1):
1. Subgradient phần trơn + robust:
   `v = -μ̂ + 2γ·Σ·wₖ + robust_subgrad`
   với `u = Σ^½·wₖ`; nếu `‖u‖₂ > eps` (eps ví dụ 1e-12): `robust_subgrad = κ·(Σ·wₖ)/‖u‖₂`; nếu `‖u‖₂ ≤ eps`: `robust_subgrad = 0` (chọn subgradient 0 trong tập dưới vi phân).
   LƯU Ý: gradient của `‖Σ^½w‖₂` theo w là `Σ^½·(Σ^½w)/‖Σ^½w‖ = Σw/‖Σ^½w‖` (vì Σ^½ đối xứng). Dùng đúng `Σw`, KHÔNG phải `Σ^½w`.
2. Bước xuống: `z = wₖ - αₖ·v`
3. Prox L1 (soft-threshold): `w_half = sign(z)·maximum(|z| - αₖ·λ, 0)`
4. Chiếu hyperplane {1ᵀw=1}: `wₖ₊₁ = w_half + (1 - sum(w_half))/N · 1`

Step size: `αₖ = α0 / sqrt(k+1)` (giảm dần). α0 là tham số; PHẢI tinh chỉnh trên data thật (xem Lưu ý scale) và ghi giá trị đã chọn + lý do trong report/docstring.

Theo dõi & dừng:
- Tính `f(wₖ)` (đầy đủ, w feasible vì đã chiếu) mỗi vòng, lưu `obj_history`.
- Giữ `best_w`, `best_obj` = w có f nhỏ nhất từ trước tới nay (subgradient KHÔNG đơn điệu → PHẢI trả best-iterate, không phải iterate cuối).
- Dừng khi: đạt `max_iter` (mặc định 5000) HOẶC thay đổi tương đối của best_obj < `tol` (mặc định 1e-8) trong `patience` (mặc định 100) vòng liên tiếp.

## Lưu ý scale (quan trọng cho tinh chỉnh α0)
Daily returns rất nhỏ: μ̂ ~ 1e-3, Σ ~ 1e-4. Nên gradient phần trơn ~ 1e-3, trong khi λ‖w‖₁ với λ=0.1 có thể ÁP ĐẢO các term khác. Các term của f có scale rất khác nhau — điều này bình thường, nhưng khiến α0 cần đủ lớn để tiến triển. Hãy thử vài giá trị α0 (ví dụ 0.01, 0.1, 1, 10) trên 1 bộ tham số thật và chọn cái hội tụ ổn định, ghi lại.

## Interface (Phase 4 CVXPY verify sẽ consume — giữ đúng tên/chữ ký)
```python
def portfolio_objective(w, mu, sigma, sigma_sqrt, kappa, gamma, lam) -> float
    # trả f(w) đầy đủ (không gồm ràng buộc)

@dataclass  # hoặc namedtuple/dict rõ ràng
class SolveResult:
    w: np.ndarray          # (N,) nghiệm tốt nhất (best-iterate)
    obj_history: np.ndarray # (n_iter,) f(wₖ) mỗi vòng
    best_obj: float
    n_iter: int
    converged: bool

def solve(mu, sigma, sigma_sqrt, kappa, gamma, lam,
          *, max_iter=5000, alpha0=<chọn>, tol=1e-8, patience=100,
          w0=None) -> SolveResult
```

## Test bắt buộc (`tests/test_prox_solver.py`, pytest, KHÔNG cần mạng, seed cố định)
1. **test_closed_form_meanvar**: N=2, κ=0, λ=0, γ tùy (vd 1.0). Khi đó bài toán là QP có ràng buộc affine, nghiệm đóng qua hệ KKT:
   `[[2γΣ, 1],[1ᵀ, 0]] · [w; η] = [μ̂; 1]`. Giải hệ tuyến tính này bằng `np.linalg.solve` để lấy w tham chiếu. Chạy `solve(...)` với cùng dữ liệu (dùng Σ nhỏ PD tự tạo, μ̂ tùy) và khẳng định `‖w_solve - w_ref‖∞ < 1e-4`. (Cho max_iter đủ lớn, vd 20000.)
2. **test_sum_to_one**: với bộ tham số bất kỳ (κ,γ,λ>0), nghiệm trả về thỏa `abs(sum(w) - 1) < 1e-9`.
3. **test_sparsity_increases_with_lambda**: giữ κ,γ cố định, tăng λ (vd 1e-4, 1e-2, 1e-1) → số phần tử `|wᵢ| > 1e-4` KHÔNG tăng (giảm hoặc bằng); ít nhất khẳng định count(λ lớn) < count(λ nhỏ) cho một cặp đủ tách biệt.
4. **test_best_obj_monotone**: `best_obj` theo vòng (chạy lại tích lũy min của obj_history) là dãy KHÔNG tăng.
5. **test_returns_best_not_last**: tạo trường hợp obj_history có iterate cuối xấu hơn best → khẳng định `portfolio_objective(result.w,...) == result.best_obj` (≈, atol nhỏ) và ≤ f(iterate cuối).
6. **test_robust_subgrad_zero_safe**: chọn w sao cho Σ^½w ≈ 0 (vd w=0 vector — nhưng w=0 vi phạm sum=1; thay vào đó test rằng hàm subgradient nội bộ KHÔNG chia cho 0 / KHÔNG NaN khi ‖Σ^½w‖ nhỏ). Có thể test qua một helper nội bộ hoặc chạy solve với κ>0 và khẳng định không có NaN trong obj_history.

Dùng dữ liệu giả lập nhỏ có seed (`np.random.default_rng(0)`), Σ tạo từ `A@A.T + εI` để PD. KHÔNG phụ thuộc data thật trong test.

## Acceptance criteria (verify thật, dán vào report)
1. `.venv/bin/python -m pytest tests/test_prox_solver.py -v` PASS toàn bộ (dán output thật).
2. Chạy solve trên DATA THẬT: `mu,sigma,sqrt = estimate_all(pd.read_parquet('data/returns.parquet'))`, chạy `solve` với ít nhất 2 bộ tham số (vd (κ=1,γ=5,λ=0.01) và (κ=0,γ=5,λ=0.1)); dán vào report: n_iter, converged, best_obj, sum(w), số tài sản active (|wᵢ|>1e-4), và xác nhận obj_history không NaN, best_obj giảm dần. In α0 đã chọn.
3. Docstring/comment giải thích công thức subgradient (đặc biệt chỗ Σw/‖Σ^½w‖), caveat "prox-then-project là heuristic, verify bằng CVXPY ở Phase 4", và lý do trả best-iterate. Người đọc dân tối ưu.

## Report vào `.sdd/task-3-report.md`
Đầy đủ: file tạo, output pytest thật, kết quả solve trên data thật (2 bộ tham số), α0 chọn + lý do, concerns. Trả về controller: status, danh sách file, 1 dòng tóm tắt (pytest X/X, solve hội tụ chưa trên data thật), concerns.
