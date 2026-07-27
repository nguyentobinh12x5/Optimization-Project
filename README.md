# Sparse + Robust Portfolio Optimization — VN100

Project tối ưu hoá danh mục đầu tư cho rổ **VN100** (98 mã cổ phiếu Việt Nam sau làm
sạch dữ liệu, 4 năm daily return) theo công thức **sparse + robust mean-variance**:

```
min_w  -μ̂ᵀw + κ‖Σ^(1/2)w‖₂ + γ·wᵀΣw + λ‖w‖₁   s.t.  1ᵀw = 1
```

kết hợp 4 mục tiêu trong cùng một bài toán lồi: tối đa hoá return kỳ vọng, phạt rủi ro
kiểu robust (norm bậc 1 của `Σ^(1/2)w`) và kiểu Markowitz cổ điển (toàn phương), và
khuyến khích danh mục **thưa** (ít mã nắm giữ) qua phạt L1 — cho phép bán khống (không có
ràng buộc `w ≥ 0`). Solver chính là một **proximal-subgradient method tự viết bằng numpy
thuần** (không dùng cvxpy/scipy.optimize/sklearn), với bước prox **chính xác** của
`L1 + ràng buộc ngân sách` giải bằng bisection (soft-threshold + tìm nhân tử Lagrange `ν`).
Kết quả được **kiểm chứng chéo** bằng CVXPY (solver nội điểm CLARABEL) như một "ground
truth" độc lập.

Ngoài phần in-sample trên, project còn có một **walk-forward backtest ngoài mẫu**
(module `src/backtest.py`): rolling window 24 tháng (18 tháng ước lượng + 6 tháng
validation), rebalance hàng tháng, long-only (`w≥0, Σw=1`), tự động chọn lại `(κ,γ)` mỗi
kỳ qua Sharpe validation, có phí giao dịch và so sánh với benchmark equal-weight 1/N — xem
mục "Walk-forward backtest" bên dưới.

Xem `notebook.ipynb` cho câu chuyện đầy đủ end-to-end (dữ liệu → ước lượng → thuật toán →
kết quả → verify → backtest OOS → kết luận), và thư mục `.sdd/` cho nhật ký thiết kế/báo
cáo từng phase.

## Cấu trúc thư mục

```
.
├── src/                    # Logic chính (mọi import của notebook đều từ đây)
│   ├── data_loader.py      # Tải + làm sạch dữ liệu VN100 qua vnstock, cache ra data/
│   ├── estimators.py       # Ước lượng μ̂, Σ, Σ^(1/2) (eigh + clip, Ledoit-Wolf tự viết)
│   ├── prox_solver.py      # Solver proximal-subgradient tự viết (numpy thuần)
│   ├── cvxpy_check.py      # Kiểm chứng chéo bằng CVXPY (chỉ nơi DUY NHẤT import cvxpy)
│   ├── backtest.py         # Walk-forward OOS backtest + benchmark equal-weight 1/N
│   └── viz.py              # 9 hàm vẽ hình (fig1..fig9), lưu ra figures/
├── tests/                  # pytest cho estimators / prox_solver / cvxpy_check / backtest
├── data/                   # Cache parquet/csv (returns.parquet, prices.parquet, symbols) — gitignored
├── figures/                # 9 PNG đã sinh sẵn (fig1_data_overview.png .. fig9_selected_params.png)
├── notebook.ipynb          # Deliverable end-to-end: import từ src/, chạy sạch từ đầu tới cuối
├── .sdd/                   # Task brief + report từng phase (nhật ký thiết kế)
└── .env                    # VNSTOCK_API_KEY (không commit — đã có trong .gitignore)
```

## Yêu cầu môi trường

- Python **3.10+** (đã test trên 3.14.5 trong `.venv/`).
- Package chính: `numpy`, `pandas`, `pyarrow` (đọc/ghi parquet), `matplotlib`, `cvxpy`,
  `pytest`, `vnstock`, `python-dotenv`, `python-dateutil`, cùng jupyter stack
  (`ipykernel`, `nbconvert`, `nbformat`) để build/chạy notebook.
- Đã có sẵn `requirements.txt` (pin phiên bản đã kiểm thử) ở project root — cách cài
  khuyến nghị là `pip install -r requirements.txt`.

## Cách chạy

```bash
# 1. Tạo & kích hoạt virtualenv
python3 -m venv .venv
source .venv/bin/activate   # hoặc .venv/bin/python cho từng lệnh không cần activate

# 2. Cài dependencies (khuyến nghị)
pip install -r requirements.txt
# hoặc cài trực tiếp:
# pip install numpy pandas pyarrow matplotlib cvxpy pytest vnstock \
#             python-dotenv python-dateutil ipykernel nbconvert nbformat

# 3. (Chỉ cần nếu chưa có data/*.parquet) Tải + làm sạch dữ liệu VN100
#    Cần VNSTOCK_API_KEY trong file .env ở project root (không commit file này).
#    Lần tải ĐẦU TIÊN mất khá lâu (hàng chục phút) do rate-limit free tier của vnstock
#    (~1.2s sleep giữa mỗi request cho từng mã trong 97+ mã) — CHỈ cần chạy 1 lần,
#    kết quả được cache ra data/prices.parquet + data/returns.parquet, các lần sau
#    (kể cả notebook) đọc thẳng từ cache, KHÔNG gọi mạng nữa.
python -m src.data_loader

# 4. Chạy test suite
pytest tests/ -v

# 5. Mở / chạy notebook
#    Đăng ký kernel trỏ đúng venv này (chỉ cần làm 1 lần):
python -m ipykernel install --user --name optproj-venv --display-name "Python (Optimization Project venv)"
#    Chạy lại toàn bộ notebook từ đầu (không cần mở Jupyter UI):
python -m nbconvert --to notebook --execute notebook.ipynb --output notebook.ipynb
#    Hoặc mở tương tác: jupyter notebook notebook.ipynb  (chọn kernel "optproj-venv")
```

### Lưu ý môi trường

- **`.env` / API key**: `src/data_loader.py` đọc `VNSTOCK_API_KEY` từ file `.env` ở
  project root qua `python-dotenv` (không log ra key). Nếu `data/*.parquet` đã tồn tại,
  `main()` đọc thẳng từ cache và **không cần** `.env`/mạng — chỉ cần khi tải lại từ đầu
  hoặc gọi `repair_universe()`.
- **SSL / proxy công ty**: nếu máy chạy sau một Cloudflare Zero Trust Gateway (hoặc proxy
  TLS-inspection tương tự do chính sách quản lý thiết bị của tổ chức), lần gọi mạng tới
  `vnstock` có thể lỗi `SSL: CERTIFICATE_VERIFY_FAILED — self-signed certificate in
  certificate chain`. Đây KHÔNG phải lỗi code — cần export root CA của gateway đó từ
  System Keychain (macOS: `security find-certificate`), gộp vào bundle CA của `certifi`,
  rồi set biến môi trường `REQUESTS_CA_BUNDLE`/`SSL_CERT_FILE` trỏ tới bundle gộp đó
  TRƯỚC khi chạy `python -m src.data_loader`. Không cần sửa gì trong `src/data_loader.py`
  (đây là cấu hình máy/mạng, không phải logic ứng dụng). Xem chi tiết đầy đủ ở
  `.sdd/task-1b-report.md`.

## Kết quả `pytest tests/ -v` (chạy thật, sau khi thêm walk-forward backtest)

```
============================= test session starts ==============================
platform darwin -- Python 3.14.5, pytest-9.1.1, pluggy-1.6.0 -- .venv/bin/python
cachedir: .pytest_cache
rootdir: /Users/nguyentobinh12gmail.com/Documents/Optimization Project
collecting ... collected 40 items

tests/test_backtest.py::test_metrics_constant_return PASSED              [  2%]
tests/test_backtest.py::test_metrics_known_drawdown_sequence PASSED      [  5%]
tests/test_backtest.py::test_metrics_sharpe_sign PASSED                  [  7%]
tests/test_backtest.py::test_build_windows_no_look_ahead PASSED          [ 10%]
tests/test_backtest.py::test_build_windows_count_matches_lookback PASSED [ 12%]
tests/test_backtest.py::test_simulate_period_first_period_full_turnover PASSED [ 15%]
tests/test_backtest.py::test_simulate_period_turnover_against_prev_drifted PASSED [ 17%]
tests/test_backtest.py::test_walk_forward_backtest_runs_and_is_feasible PASSED [ 20%]
tests/test_backtest.py::test_equal_weight_backtest_matches_uniform_weights PASSED [ 22%]
tests/test_cvxpy_check.py::test_cvxpy_solve_feasible_and_matches_objective PASSED [ 25%]
tests/test_cvxpy_check.py::test_compare_columns_and_small_relgap PASSED  [ 27%]
tests/test_cvxpy_check.py::test_long_only_matches_cvxpy_on_real_data PASSED [ 30%]
tests/test_cvxpy_check.py::test_cvxpy_solve_long_only_feasible_and_matches_objective PASSED [ 32%]
tests/test_data_loader.py::test_weekend_rows_dropped_by_clean_prices PASSED [ 35%]
tests/test_data_loader.py::test_compute_returns_drops_suspected_holiday_row PASSED [ 37%]
tests/test_data_loader.py::test_compute_returns_keeps_genuine_flat_day_for_minority PASSED [ 40%]
tests/test_data_loader.py::test_compute_returns_no_nan_after_filtering PASSED [ 42%]
tests/test_estimators.py::test_mu_shape_and_value PASSED                 [ 45%]
tests/test_estimators.py::test_sigma_symmetric PASSED                    [ 47%]
tests/test_estimators.py::test_sigma_psd PASSED                          [ 50%]
tests/test_estimators.py::test_sqrt_reconstructs PASSED                  [ 52%]
tests/test_estimators.py::test_sqrt_symmetric PASSED                     [ 55%]
tests/test_estimators.py::test_shrinkage_between PASSED                  [ 57%]
tests/test_estimators.py::test_sqrt_on_psd_singular PASSED               [ 60%]
tests/test_estimators.py::test_estimate_all_shapes_and_order PASSED      [ 62%]
tests/test_prox_solver.py::test_closed_form_meanvar PASSED               [ 65%]
tests/test_prox_solver.py::test_sum_to_one PASSED                        [ 67%]
tests/test_prox_solver.py::test_sparsity_increases_with_lambda PASSED    [ 70%]
tests/test_prox_solver.py::test_prox_l1_simplex_eq_exact_zero_and_sum_one PASSED [ 72%]
tests/test_prox_solver.py::test_prox_l1_simplex_eq_reduces_to_projection_when_t_zero PASSED [ 75%]
tests/test_prox_solver.py::test_best_obj_monotone PASSED                 [ 77%]
tests/test_prox_solver.py::test_returns_best_not_last PASSED             [ 80%]
tests/test_prox_solver.py::test_robust_subgrad_zero_safe PASSED          [ 82%]
tests/test_prox_solver.py::test_simplex_projection_symmetric_case PASSED [ 85%]
tests/test_prox_solver.py::test_simplex_projection_dominant_component PASSED [ 87%]
tests/test_prox_solver.py::test_simplex_projection_already_feasible_is_fixed_point PASSED [ 90%]
tests/test_prox_solver.py::test_simplex_projection_random_always_feasible PASSED [ 92%]
tests/test_prox_solver.py::test_solve_long_only_uniform_when_isotropic PASSED [ 95%]
tests/test_prox_solver.py::test_solve_long_only_feasible_general_case PASSED [ 97%]
tests/test_prox_solver.py::test_solve_long_only_returns_best_not_last PASSED [100%]

======================= 40 passed, 27 warnings in 11.99s =======================
```

Các `RuntimeWarning` (divide-by-zero/overflow/invalid trong `matmul`) là noise đã biết của
Apple Accelerate BLAS khi nhân ma trận gần suy biến hoặc trong các test cố ý dựng trường hợp
biên (`kappa=0`, `Sigma^(1/2)w≈0`) — không phải lỗi, không ảnh hưởng kết quả pass (đã verify
không có NaN/Inf lọt vào kết quả cuối).

## Notebook

`notebook.ipynb` chạy sạch end-to-end bằng:

```bash
.venv/bin/python -m nbconvert --to notebook --execute notebook.ipynb --output notebook.ipynb
```

Kết quả lần chạy thật gần nhất: 28 cell (15 code + 13 markdown), **0 lỗi**, cả **9 hình**
(`fig1`..`fig9`, gồm 3 hình walk-forward backtest mới) được tái sinh trực tiếp trong
notebook (gọi hàm trong `src/viz.py`, không đọc PNG tĩnh) và hiển thị inline, tổng thời
gian thực thi ~50 giây (không gọi mạng — chỉ đọc `data/*.parquet` cache; phần lớn thời gian
này là walk-forward backtest tự chạy lại grid-search 12 tổ hợp tham số × 25 kỳ rebalance).
Notebook không copy-paste logic — mọi tính toán import trực tiếp từ `src/data_loader`,
`src/estimators`, `src/prox_solver`, `src/cvxpy_check`, `src/backtest`, `src/viz`.
