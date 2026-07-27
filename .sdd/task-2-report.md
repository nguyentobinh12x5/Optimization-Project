# Task 2 Report — Estimators (μ̂, Σ, Σ^(1/2))

## File tạo

- `src/estimators.py` — module chính: `estimate_mu`, `estimate_sigma`, `matrix_sqrt_psd`, `estimate_all`, plus helper nội bộ `_mean_var_target`, `_ledoit_wolf_delta`, và `_main()` (smoke test khi chạy `python -m src.estimators`).
- `tests/test_estimators.py` — 8 test (7 bắt buộc theo brief + 1 bonus `test_estimate_all_shapes_and_order`), dữ liệu giả lập `rng = np.random.default_rng(0)`, N=5 (cột A..E), T=200, không phụ thuộc data thật.
- Thư mục `tests/` mới được tạo (chưa tồn tại trước task này).

## Thiết kế / quyết định kỹ thuật

1. **estimate_mu**: `returns.mean(axis=0).to_numpy(dtype=float64)`.
2. **estimate_sigma**: mặc định `returns.cov()` (pandas, ddof=1). Shrinkage:
   - `None`: sample covariance thuần.
   - `"lw"`/`"ledoit-wolf"`: tự cài Ledoit-Wolf (2004) bằng numpy thuần, target F = mean(diag(S))·I (constant-variance-identity target, đơn giản hóa so với bản constant-correlation đầy đủ trong bài báo gốc — ghi rõ trong docstring). Công thức: `pi_hat = (1/T) Σ_t ||x_t x_t^T - S||_F^2`, `gamma_hat = ||S-F||_F^2`, `delta = clip(pi_hat/(T·gamma_hat), 0, 1)`.
   - `float` trong [0,1]: dùng trực tiếp làm δ, cùng target F.
   - Luôn ép đối xứng `Σ = (Σ+Σ.T)/2` trước khi trả.
3. **matrix_sqrt_psd**: `np.linalg.eigh` trên bản ép đối xứng của input, clip eigenvalue âm về 0, dựng lại `V·diag(sqrt(λ_clip))·V.T`, ép đối xứng kết quả. KHÔNG dùng Cholesky (lý do giải thích chi tiết trong docstring module: Σ có thể chỉ PSD, Cholesky đòi PD nghiêm ngặt → fail trên input suy biến).
4. **estimate_all**: gọi gộp 3 hàm trên, trả tuple `(mu, sigma, sigma_sqrt)`, thứ tự asset = `list(returns.columns)`.

## Output pytest THẬT

Lệnh: `.venv/bin/python -m pytest tests/test_estimators.py -v`

```
============================= test session starts ==============================
platform darwin -- Python 3.14.5, pytest-9.1.1, pluggy-1.6.0 -- /Users/nguyentobinh12gmail.com/Documents/Optimization Project/.venv/bin/python
cachedir: .pytest_cache
rootdir: /Users/nguyentobinh12gmail.com/Documents/Optimization Project
collecting ... collected 8 items

tests/test_estimators.py::test_mu_shape_and_value PASSED                 [ 12%]
tests/test_estimators.py::test_sigma_symmetric PASSED                    [ 25%]
tests/test_estimators.py::test_sigma_psd PASSED                          [ 37%]
tests/test_estimators.py::test_sqrt_reconstructs PASSED                  [ 50%]
tests/test_estimators.py::test_sqrt_symmetric PASSED                     [ 62%]
tests/test_estimators.py::test_shrinkage_between PASSED                  [ 75%]
tests/test_estimators.py::test_sqrt_on_psd_singular PASSED               [ 87%]
tests/test_estimators.py::test_estimate_all_shapes_and_order PASSED      [100%]

============================== 8 passed in 0.67s ===============================
```

8/8 PASS (7 test bắt buộc theo brief + 1 test bonus kiểm tra shape/order của `estimate_all`).

## Chạy trên data thật (`data/returns.parquet`)

Lệnh: `.venv/bin/python -m src.estimators` (gọi `estimate_all(pd.read_parquet("data/returns.parquet"))`, shrinkage=None mặc định).

```
returns shape: (1069, 75)
mu shape: (75,)
sigma shape: (75, 75)
sigma_sqrt shape: (75, 75)
||Sigma_sqrt^2 - Sigma||_F / ||Sigma||_F = 1.232e-15
sigma_sqrt symmetric: True
sigma symmetric: True
```

Kiểm tra bổ sung (không có trong acceptance criteria nhưng chạy để xác nhận):
- `np.any(np.isnan(sigma_sqrt))` = False, `np.any(np.isnan(sigma))` = False.
- Eigenvalue của Σ trên data thật: min ≈ 3.99e-05, max ≈ 1.62e-02 (PSD, well-conditioned, condition number ≈ 407 — không suy biến nặng, không cần shrinkage để Σ^(1/2) hoạt động đúng, dù shrinkage vẫn dùng được nếu Phase 3 cần).
- Test nhanh `estimate_sigma(returns, shrinkage="lw")` trên data thật: chạy thành công (~0.5s), kết quả đối xứng, min eigenvalue ≈ 4.52e-05.

## Concern kỹ thuật (không chặn, nhưng cần biết)

Khi chạy `matrix_sqrt_psd` / `estimate_all` trên **data thật** (75×75), numpy phát ra `RuntimeWarning: divide by zero / overflow / invalid value encountered in matmul` tại các dòng `eigvecs @ diag(...) @ eigvecs.T` và `sigma_sqrt @ sigma_sqrt`. Đã điều tra: đây là hiện tượng đã biết của backend BLAS **Accelerate** trên Apple Silicon (macOS ARM) — `numpy.__config__.show()` xác nhận numpy trong `.venv` build với `blas: accelerate`. Accelerate được biết là bật cờ FPE (floating-point exception) giả trong một số phép `matmul` dù kết quả số học vẫn ĐÚNG. Đã verify:
- Không có NaN trong `sigma` hay `sigma_sqrt`.
- Sai số tái tạo `‖Σ^(1/2)² − Σ‖_F/‖Σ‖_F = 1.232e-15` — cực nhỏ, đúng như kỳ vọng.
- Test giả lập trên dữ liệu N=5 KHÔNG phát cảnh báo này (chỉ xuất hiện với ma trận 75×75 thật) → củng cố giả thuyết đây là quirk kích thước-liên-quan-BLAS-backend, không phải lỗi logic.

Không sửa gì thêm vì đây là warning vô hại của môi trường (Accelerate BLAS), không phải lỗi trong thuật toán; nêu ra để Phase 3/4 không bất ngờ nếu thấy warning tương tự khi gọi `estimate_all` trên ma trận lớn hơn N=5. Nếu muốn im lặng warning này, có thể bọc bằng `with np.errstate(...)` nhưng KHÔNG làm vì brief không yêu cầu và việc che warning có thể che luôn cảnh báo thật trong tương lai.

## Trạng thái

DONE.
