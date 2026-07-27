# Task Brief — Phase 4: CVXPY Cross-Verification

Đọc trước tiên. Cách tiếp cận đã được controller chứng minh hoạt động (solver tay khớp CVXPY gap ~0, active-set Jaccard=1.0). Nhiệm vụ: đóng gói thành module sạch + bảng so sánh.

## Mục tiêu
Viết `src/cvxpy_check.py`: giải CÙNG bài toán bằng CVXPY (làm ground-truth) và so sánh với solver tay Phase 3 trên nhiều bộ tham số. CVXPY CHỈ dùng ở module này (tách khỏi solver chính).

## Môi trường
- `.venv/bin/python`. `cvxpy 1.9.2` ĐÃ cài, có solver CLARABEL/SCS/OSQP. numpy/pandas có sẵn.
- Consume: `from src.estimators import estimate_all`, `from src.prox_solver import solve, portfolio_objective`.
- Data thật: `data/returns.parquet` (1068×97).

## Bài toán (khớp CHÍNH XÁC solver tay)
```
min_w  -μ̂ᵀw + κ‖Σ^½w‖₂ + γ wᵀΣw + λ‖w‖₁   s.t. 1ᵀw = 1
```
CVXPY formulation (đã verify chạy được):
```python
w = cp.Variable(N)
obj = (-mu @ w + kappa*cp.norm(sigma_sqrt @ w, 2)
       + gamma*cp.quad_form(w, cp.psd_wrap(sigma)) + lam*cp.norm1(w))
prob = cp.Problem(cp.Minimize(obj), [cp.sum(w) == 1])
prob.solve(solver=cp.CLARABEL)
```
- `cp.psd_wrap(sigma)` cần thiết vì Σ có thể có eigenvalue âm cực nhỏ do sai số số học (CLARABEL nếu không sẽ than phiền không PSD). Ghi comment giải thích.
- Dùng CLARABEL (mặc định tốt cho SOCP+QP này); nếu CLARABEL lỗi trên 1 bộ tham số, fallback SCS và ghi chú.

## Interface
```python
def cvxpy_solve(mu, sigma, sigma_sqrt, kappa, gamma, lam) -> tuple[np.ndarray, float]
    # trả (w_optimal, objective_value)

def compare(mu, sigma, sigma_sqrt, param_grid, *, active_thresh=1e-4) -> pd.DataFrame
    # param_grid: list các dict/tuple (kappa,gamma,lam). Với mỗi bộ: chạy solve() tay và cvxpy_solve(),
    # tính: f_hand, f_cvx, relgap=(f_hand-f_cvx)/|f_cvx|, active_hand, active_cvx, jaccard active-set,
    #        linf = ‖w_hand - w_cvx‖∞, n_iter, converged.
    # trả DataFrame 1 hàng / bộ tham số.
```

## Bộ tham số so sánh (≥5, bắt buộc gồm các bộ này + tự thêm cho đủ đa dạng)
- (κ=1, γ=5, λ=0.01)
- (κ=0, γ=5, λ=0.1)      # κ=0: tắt robust term
- (κ=1, γ=5, λ=0.001)    # λ nhỏ: ít thưa
- (κ=2, γ=10, λ=0.05)
- (κ=0.5, γ=1, λ=0.02)
- (κ=1, γ=5, λ=0.0)      # λ=0: không thưa, kiểm tra term khác

## Acceptance criteria (verify thật, dán bảng vào report)
1. Bảng so sánh in ra cho ≥5 bộ tham số với đầy đủ cột trên.
2. **relgap objective < 0.5%** cho MỌI bộ (kỳ vọng ~0). Nếu bộ nào vượt → điều tra (dùng tăng max_iter cho solver tay), KHÔNG chỉnh ngưỡng cho qua.
3. Active-set Jaccard cao (kỳ vọng ~1.0 cho các bộ có thưa); `‖Δw‖∞` nhỏ (báo cáo số thật).
4. Module chạy được: `.venv/bin/python -m src.cvxpy_check` in bảng đẹp (dùng pandas to_string hoặc tabulate thủ công).
5. Docstring giải thích: vì sao dùng CVXPY làm ground-truth, ý nghĩa psd_wrap, và rằng đây là bước KIỂM CHỨNG chứ không thay solver tay.

## Lưu ý
- Có thể có `RuntimeWarning` benign từ Apple Accelerate BLAS (đã biết ở Phase 2/3) — không phải lỗi, có thể filter khi in bảng nhưng ĐỪNG che lỗi thật.
- KHÔNG cần test pytest riêng cho phase này (đây là script kiểm chứng), nhưng nếu tách hàm thuần thì thêm 1-2 test nhẹ cũng tốt (không bắt buộc).

## Report vào `.sdd/task-4-report.md`
Đầy đủ: file tạo, bảng so sánh THẬT ≥5 bộ, kết luận (solver tay có khớp CVXPY trong ngưỡng không). Trả về controller: status, danh sách file, 1 dòng tóm tắt (max relgap qua các bộ, active-set khớp chưa), concerns.
