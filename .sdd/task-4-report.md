# Task 4 Report — CVXPY Cross-Verification

## File tạo

- `src/cvxpy_check.py` — module chính: `cvxpy_solve(mu, sigma, sigma_sqrt, kappa, gamma, lam) -> (w, obj)`, `compare(mu, sigma, sigma_sqrt, param_grid, *, active_thresh=1e-4, max_iter=5000) -> pd.DataFrame`, hằng số `DEFAULT_PARAM_GRID` (6 bộ tham số bắt buộc theo brief), entry point `_main()` chạy được qua `python -m src.cvxpy_check`. `cvxpy` CHỈ import ở module này — `src/prox_solver.py` không đụng đến (đã grep xác nhận, không có match).
- `tests/test_cvxpy_check.py` — 2 test nhẹ (không bắt buộc theo brief, thêm cho chắc): verify `cvxpy_solve` trả w feasible + obj khớp `portfolio_objective`, verify `compare()` trả đúng cột yêu cầu trên bài toán nhỏ giả lập (N=6).

## Formulation CVXPY (khớp chính xác solver tay)

```python
w = cp.Variable(n)
obj = (-mu @ w + kappa * cp.norm(sigma_sqrt @ w, 2)
       + gamma * cp.quad_form(w, cp.psd_wrap(sigma)) + lam * cp.norm1(w))
prob = cp.Problem(cp.Minimize(obj), [cp.sum(w) == 1])
prob.solve(solver=cp.CLARABEL)  # fallback cp.SCS nếu SolverError, có warning
```

`cp.psd_wrap(sigma)` dùng vì Sigma ước lượng từ data thật chỉ PSD tới sai số số học (eigenvalue âm cực nhỏ do rounding) — `cp.quad_form` mặc định kiểm tra PSD nghiêm ngặt và sẽ ném `DCPError` nếu không wrap. Giải thích đầy đủ + lý do dùng CVXPY làm ground truth nằm trong docstring module (dài, không lặp lại ở đây).

Giá trị `obj` trả về từ `cvxpy_solve` được tính lại bằng `portfolio_objective` (Phase 3) tại `w.value` — không dùng `prob.value` nội bộ của CVXPY — để đảm bảo so sánh hai con số tính từ đúng CÙNG MỘT công thức.

## Bảng so sánh THẬT (data/returns.parquet, 1068×97, `estimate_all` không shrinkage, solver tay `alpha0=ALPHA0_DEFAULT=10.0`, `max_iter=5000` mặc định)

Lệnh: `.venv/bin/python -m src.cvxpy_check`

| kappa | gamma | lam | f_hand | f_cvx | relgap | active_hand | active_cvx | jaccard | ‖Δw‖∞ | n_iter | converged |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1.0 | 5.0 | 0.01 | 1.508118e-02 | 1.508118e-02 | 2.22e-07 | 16 | 16 | 1.0000 | 6.284e-04 | 563 | True |
| 0.0 | 5.0 | 0.1 | 9.925261e-02 | 9.925192e-02 | 6.92e-06 | 9 | 9 | 1.0000 | 1.696e-02 | 2186 | True |
| 1.0 | 5.0 | 0.001 | 5.885283e-03 | 5.885281e-03 | 2.68e-07 | 38 | 38 | 1.0000 | 4.367e-04 | 660 | True |
| 2.0 | 10.0 | 0.05 | 6.027137e-02 | 6.027137e-02 | -8.81e-10 | 16 | 16 | 1.0000 | 1.049e-04 | 232 | True |
| 0.5 | 1.0 | 0.02 | 2.240925e-02 | 2.240921e-02 | 1.96e-06 | 16 | 16 | 1.0000 | 3.189e-03 | 1371 | True |
| 1.0 | 5.0 | 0.0 | 4.362267e-03 | 4.362265e-03 | 5.85e-07 | 97 | 96 | 0.9897 | 4.325e-04 | 828 | True |

- **max |relgap| = 0.000692%** (bộ B, kappa=0, gamma=5, lam=0.1) — dưới ngưỡng 0.5% yêu cầu ~720 lần.
- 6 bộ tham số đều `converged=True` trong ≤5000 vòng (không cần tăng `max_iter`).
- Bộ cuối (lam=0) không thưa theo thiết kế (không có L1 penalty) — jaccard=0.9897 (96/97 vs 97/97, lệch đúng 1 toạ độ nằm sát ngưỡng `active_thresh=1e-4`) là kết quả HỢP LÝ, không phải lỗi: ở lam=0 không có cơ chế đẩy toạ độ về 0 chính xác, một toạ độ dao động rất nhỏ quanh ngưỡng threshold là nhiễu hội tụ bậc nhất bình thường, không ảnh hưởng kết luận.
- 5 bộ còn lại (đều có lam>0, tức có sparsity penalty) đạt **Jaccard = 1.0 chính xác**.

## Kết luận

Solver tay (Phase 3, proximal-subgradient + joint prox chính xác qua bisection) KHỚP CVXPY (CLARABEL) trong ngưỡng yêu cầu trên cả 6/6 bộ tham số: relgap tối đa 0.000692% (<<0.5%), active-set khớp chính xác (Jaccard=1.0) ở mọi bộ có sparsity, và gần khớp (0.9897) ở bộ không sparsity (lam=0, đúng như kỳ vọng lý thuyết). Không có bộ nào cần tăng `max_iter` để đạt ngưỡng — toàn bộ 6 bộ đã `converged=True` với cấu hình mặc định của `solve()`. Không phát hiện fallback SCS (CLARABEL solve thành công `optimal` cho mọi bộ).

## Ghi chú

- `RuntimeWarning` benign từ Apple Accelerate BLAS (đã biết từ Phase 2/3) được filter khi in bảng (chỉ message cụ thể `"encountered in matmul"`), các warning khác (vd fallback solver, `optimal_inaccurate`) KHÔNG bị che.
- Runtime thực đo: ~3.4s cho toàn bộ 6 bộ tham số (solve tay + CVXPY) trên máy dev, không phải vấn đề ở quy mô N=97.
- Test suite đầy đủ (`tests/`): 18 passed (8 estimators + 8 prox_solver + 2 cvxpy_check mới).

## Concerns

1. Bộ tham số lam=0 (không sparsity) cho Jaccard=0.9897 chứ không phải 1.0 tuyệt đối — đây là nhiễu hội tụ bậc nhất bình thường quanh ngưỡng `active_thresh`, không phải bug; nếu Phase 5/6 cần active-set chính xác tuyệt đối ở lam=0, nên dùng ngưỡng active_thresh lớn hơn hoặc coi lam=0 là trường hợp "dense" không cần active-set matching chặt.
2. `compare()` gọi `solve()` với `max_iter=5000` mặc định (khớp `ALPHA0_DEFAULT` đã tune ở Phase 3); nếu Phase 5/6 dùng bộ tham số có scale rất khác 6 bộ đã test ở đây, nên chạy lại `compare()` với `max_iter` lớn hơn trước khi tin tưởng relgap thấp.
3. Đã xác nhận `src/prox_solver.py` không import `cvxpy` (grep sạch) — ràng buộc tách biệt của brief được giữ nguyên.

## Trạng thái

DONE. 6/6 bộ tham số đạt acceptance criteria (relgap <0.5%, Jaccard cao/=1.0, converged=True).
