# Report — wf-Task 2: Verify solver long-only bằng CVXPY

## Status: COMPLETE

## Files changed
- `src/cvxpy_check.py` — thêm `cvxpy_solve_long_only(mu, sigma, sigma_sqrt, kappa, gamma) -> tuple[np.ndarray, float]`
  và `compare_long_only(mu, sigma, sigma_sqrt, param_grid, *, active_thresh=1e-4, max_iter=5000) -> pd.DataFrame`.
  Tái dùng đúng pattern của `cvxpy_solve`/`compare` sẵn có (`cp.psd_wrap(sigma)`, `cp.quad_form`,
  solver CLARABEL với fallback SCS + cảnh báo `optimal_inaccurate`). Formulation KHÔNG có term
  `lam*||w||_1` (đúng thiết kế `portfolio_objective_long_only` của Task 1 — dưới ràng buộc long-only
  `||w||_1 = sum(w) = 1` là hằng số). `compare_long_only` trả thêm 2 cột `w_hand`, `w_cvx` (nghiệm đầy
  đủ) để test kiểm tra feasibility trực tiếp trên DataFrame; không có cột `lam`.
- `tests/test_cvxpy_check.py` — thêm 2 test:
  - `test_long_only_matches_cvxpy_on_real_data`: chạy trên `data/returns.parquet` (965×98) thật,
    `estimate_all(returns, shrinkage="lw")`, param_grid `[(0,1),(0.5,5),(1,5),(2,10)]`, assert
    `relgap.abs() < 0.005`, `w_hand >= -1e-6`, `sum(w_hand) ~= 1`.
  - `test_cvxpy_solve_long_only_feasible_and_matches_objective`: test nhẹ trên bài toán giả lập nhỏ
    (N=6, giống `_make_problem` sẵn có) kiểm tra feasibility + obj khớp `portfolio_objective_long_only`.

## Quy trình TDD đã thực hiện
1. Đọc kỹ chữ ký thật của `solve_long_only`, `portfolio_objective_long_only`, `SolveResult` trong
   `src/prox_solver.py` (không đoán từ brief) trước khi viết test/implementation.
2. Viết test trước → chạy `pytest tests/test_cvxpy_check.py -k long_only -v` → xác nhận FAIL
   (`ImportError: cannot import name 'compare_long_only'`).
3. Implement `cvxpy_solve_long_only` + `compare_long_only` trong `src/cvxpy_check.py`.
4. Chạy lại → PASS. Chạy toàn bộ suite `pytest -q` → 31 passed (29 cũ + 2 mới), không phá test nào.

## Số liệu THẬT quan sát được (data/returns.parquet, 965×98, shrinkage="lw")

param_grid = [(kappa=0, gamma=1), (0.5, 5), (1, 5), (2, 10)]:

| kappa | gamma | f_hand      | f_cvx       | relgap        | active_hand | active_cvx | jaccard | linf     | n_iter | converged |
|-------|-------|-------------|-------------|---------------|-------------|------------|---------|----------|--------|-----------|
| 0.0   | 1.0   | -1.956424e-3| -1.958106e-3| +0.0859%      | 3           | 3          | 1.0     | 4.535e-2 | 5000   | False     |
| 0.5   | 5.0   | 2.691100e-3 | 2.691095e-3 | +0.000157%    | 19          | 19         | 1.0     | 9.248e-4 | 1463   | True      |
| 1.0   | 5.0   | 5.443769e-3 | 5.443769e-3 | +0.0000148%   | 18          | 18         | 1.0     | 3.214e-4 | 546    | True      |
| 2.0   | 10.0  | 1.100735e-2 | 1.100735e-2 | -0.0000033%   | 17          | 17         | 1.0     | 4.387e-5 | 237    | True      |

Tất cả 4 bộ: `relgap.abs() < 0.5%` (max quan sát được ~0.086%, ở bộ kappa=0/gamma=1 — bộ duy nhất
không converge trong 5000 vòng, robust term=0 nên landscape "phẳng" hơn khiến subgradient hội tụ
chậm, nhưng vẫn đạt gap rất nhỏ). `jaccard=1.0` ở cả 4 bộ (active-set khớp tuyệt đối với CVXPY).
`w_hand` feasible: mọi toạ độ >= -1e-6 (không âm tới sai số số học), `sum(w_hand)` khớp 1.0 tới 1e-6.
Không có NaN/Inf trong bất kỳ `w_hand` nào (đã kiểm tra riêng).

Ghi chú: có ~51 `RuntimeWarning` (divide-by-zero/overflow/invalid trong `matmul`) xuất hiện khi chạy
— đã xác minh đây là warning benign đã biết từ Phase 2/3 (Apple Accelerate BLAS trên `sqrt_sigma`
tại `src/estimators.py:237` cùng vài chỗ nội bộ tương tự trong `prox_solver`/`cvxpy`'s matmul), KHÔNG
phải divergence thật — đã kiểm tra trực tiếp: không có NaN/Inf trong bất kỳ `w_hand` nào, và các
số liệu relgap/jaccard ở bảng trên đều hợp lý, hội tụ tốt (trừ bộ kappa=0 chưa converge trong
max_iter mặc định nhưng vẫn đạt gap nhỏ). Cùng loại warning này đã xuất hiện trong test suite cũ
(`test_prox_solver.py`) từ trước, không phải lỗi mới do task này gây ra.

## pytest tổng
`.venv/bin/python -m pytest -q` → **31 passed** (27 cũ liên quan prox_solver/cvxpy_check/data_loader/estimators
+ 4 test trong `test_cvxpy_check.py`, trong đó 2 mới của task này).

## Concerns
- Bộ `(kappa=0, gamma=1)` không `converged=True` trong `max_iter=5000` mặc định của `solve_long_only`
  (dừng do chạm max_iter, không phải do patience). Gap vẫn nhỏ (0.086% < 0.5%) nên không chặn task,
  nhưng nếu backtest thật (Task 4+) dùng grid có `kappa=0` cần lưu ý có thể cần tăng `max_iter` nếu
  muốn `converged=True` chặt chẽ hơn — không phải bug, chỉ là đặc tính landscape phẳng hơn khi robust
  term=0 (đã ghi lại trong report để Task sau tham khảo).
- Không có concern nào khác về tính đúng đắn: active-set khớp tuyệt đối (jaccard=1.0) ở mọi bộ tham số
  test trên data thật.
