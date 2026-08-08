# wf-Task 4 Report — Walk-forward backtest engine (core)

## Tóm tắt

Đã thêm vào `src/backtest.py` (không tạo file mới, không xoá `performance_metrics` đã có
từ Task 3):
- `_build_rebalance_windows(index, lookback_months, validation_months) -> list[dict]`
- `_simulate_period(w_start, period_returns, prev_weights_drifted, fee) -> tuple[pd.Series, float, float, np.ndarray]`
- `BacktestResult` (dataclass: `daily_returns`, `rebalance_log`, `weights`)
- `DEFAULT_PARAM_GRID` (kappa in {0,0.5,1,2} x gamma in {1,5,10} = 12 tổ hợp)
- `walk_forward_backtest(returns, *, lookback_months=24, validation_months=6, param_grid=None, shrinkage="lw", fee=0.002, rf=0.0) -> BacktestResult`

Đã thêm vào `tests/test_backtest.py` (giữ nguyên 3 test `performance_metrics` cũ của Task 3):
- `test_build_windows_no_look_ahead`, `test_build_windows_count_matches_lookback`
- `test_simulate_period_first_period_full_turnover`, `test_simulate_period_turnover_against_prev_drifted`
- `test_walk_forward_backtest_runs_and_is_feasible`

TDD nghiêm ngặt theo brief: mỗi hàm viết test trước, chạy xác nhận FAIL (ImportError vì hàm
chưa tồn tại), rồi implement, chạy xác nhận PASS, trước khi sang hàm kế tiếp. Không nới lỏng
assertion nào so với brief, kể cả `test_build_windows_no_look_ahead` (test quan trọng nhất).

Sửa lỗi gõ đã biết trước trong code mẫu brief: `window_months = window_months = months[i - lookback_months : i]`
(gán trùng lặp) -> viết thành 1 lần gán duy nhất `window_months = months[i - lookback_months : i]`.

Một sai khác nhỏ so với code mẫu brief: bản mẫu gọi `.to_numpy()` trên `periods == target_month`
và `periods.isin(...)`. Trên phiên bản pandas của môi trường này, so sánh `PeriodIndex == PeriodIndex`
và `PeriodIndex.isin(...)` đã trả về `numpy.ndarray` trực tiếp (không phải pandas boolean array/Series),
nên gọi thêm `.to_numpy()` gây `AttributeError: 'numpy.ndarray' object has no attribute 'to_numpy'`.
Xác nhận bằng test trực tiếp (`type(periods == periods[0])` -> `numpy.ndarray`). Fix: bỏ `.to_numpy()`
thừa, dùng thẳng kết quả ndarray làm mask.

## Output pytest thật (đầy đủ suite, sau khi hoàn tất cả 3 hàm)

```
$ .venv/bin/python -m pytest tests/ -v
...
tests/test_backtest.py::test_metrics_constant_return PASSED              [  2%]
tests/test_backtest.py::test_metrics_known_drawdown_sequence PASSED      [  5%]
tests/test_backtest.py::test_metrics_sharpe_sign PASSED                  [  7%]
tests/test_backtest.py::test_build_windows_no_look_ahead PASSED          [ 10%]
tests/test_backtest.py::test_build_windows_count_matches_lookback PASSED [ 12%]
tests/test_backtest.py::test_simulate_period_first_period_full_turnover PASSED [ 15%]
tests/test_backtest.py::test_simulate_period_turnover_against_prev_drifted PASSED [ 17%]
tests/test_backtest.py::test_walk_forward_backtest_runs_and_is_feasible PASSED [ 20%]
tests/test_cvxpy_check.py::test_cvxpy_solve_feasible_and_matches_objective PASSED [ 23%]
tests/test_cvxpy_check.py::test_compare_columns_and_small_relgap PASSED  [ 25%]
tests/test_cvxpy_check.py::test_long_only_matches_cvxpy_on_real_data PASSED [ 28%]
tests/test_cvxpy_check.py::test_cvxpy_solve_long_only_feasible_and_matches_objective PASSED [ 30%]
tests/test_data_loader.py::test_weekend_rows_dropped_by_clean_prices PASSED [ 33%]
tests/test_data_loader.py::test_compute_returns_drops_suspected_holiday_row PASSED [ 35%]
tests/test_data_loader.py::test_compute_returns_keeps_genuine_flat_day_for_minority PASSED [ 38%]
tests/test_data_loader.py::test_compute_returns_no_nan_after_filtering PASSED [ 41%]
tests/test_estimators.py::test_mu_shape_and_value PASSED                 [ 43%]
tests/test_estimators.py::test_sigma_symmetric PASSED                    [ 46%]
tests/test_estimators.py::test_sigma_psd PASSED                          [ 48%]
tests/test_estimators.py::test_sqrt_reconstructs PASSED                  [ 51%]
tests/test_estimators.py::test_sqrt_symmetric PASSED                     [ 53%]
tests/test_estimators.py::test_shrinkage_between PASSED                  [ 56%]
tests/test_estimators.py::test_sqrt_on_psd_singular PASSED               [ 58%]
tests/test_estimators.py::test_estimate_all_shapes_and_order PASSED      [ 61%]
tests/test_prox_solver.py::test_closed_form_meanvar PASSED               [ 64%]
tests/test_prox_solver.py::test_sum_to_one PASSED                        [ 66%]
tests/test_prox_solver.py::test_sparsity_increases_with_lambda PASSED    [ 69%]
tests/test_prox_solver.py::test_prox_l1_simplex_eq_exact_zero_and_sum_one PASSED [ 71%]
tests/test_prox_solver.py::test_prox_l1_simplex_eq_reduces_to_projection_when_t_zero PASSED [ 74%]
tests/test_prox_solver.py::test_best_obj_monotone PASSED                 [ 76%]
tests/test_prox_solver.py::test_returns_best_not_last PASSED             [ 79%]
tests/test_prox_solver.py::test_robust_subgrad_zero_safe PASSED          [ 82%]
tests/test_prox_solver.py::test_simplex_projection_symmetric_case PASSED [ 84%]
tests/test_prox_solver.py::test_simplex_projection_dominant_component PASSED [ 87%]
tests/test_prox_solver.py::test_simplex_projection_already_feasible_is_fixed_point PASSED [ 89%]
tests/test_prox_solver.py::test_simplex_projection_random_always_feasible PASSED [ 92%]
tests/test_prox_solver.py::test_solve_long_only_uniform_when_isotropic PASSED [ 94%]
tests/test_prox_solver.py::test_solve_long_only_feasible_general_case PASSED [ 97%]
tests/test_prox_solver.py::test_solve_long_only_returns_best_not_last PASSED [100%]

======================= 39 passed, 27 warnings in 8.88s ========================
```

39/39 pass (34 cũ trước Task 4 + 5 test mới của Task 4). 27 warnings đều là RuntimeWarning
numpy benign (Apple Accelerate BLAS quirk trên `matmul` với ma trận gần suy biến/eigenvalue
cực nhỏ) đã ghi nhận từ các phase trước trong `.sdd/progress.md` -- không xuất hiện NaN nào
trong kết quả cuối, không phải lỗi mới.

Log FAIL xác nhận từng bước trước khi implement (TDD, đúng brief):
- `_build_rebalance_windows` chưa tồn tại -> `ImportError: cannot import name '_build_rebalance_windows'`.
- `_simulate_period` chưa tồn tại -> `ImportError: cannot import name '_simulate_period'`.
- `walk_forward_backtest` chưa tồn tại -> `ImportError: cannot import name 'walk_forward_backtest'`.
Sau mỗi lần implement, rerun xác nhận PASS trước khi sang bước kế tiếp.

## Smoke test DATA THẬT (`data/returns.parquet`, 965x98)

```
$ .venv/bin/python -c "
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

**Thời gian chạy thật: 28.81 giây** (< 5 phút, KHÔNG cần giảm `max_iter` mặc định).

**Số kỳ rebalance: 25** (965 phiên / 98 mã, lookback=24 tháng, validation=6 tháng, param_grid
mặc định 12 tổ hợp -> chọn tham số theo Sharpe validation cho mỗi kỳ).

Vài dòng đầu `rebalance_log` (đầy đủ 25 dòng, cột: kappa, gamma, turnover, cost, n_active,
val_sharpe):

```
            kappa  gamma  turnover      cost  n_active  val_sharpe
date
2024-07-01    0.0    5.0  1.000000  0.002000        12    0.559302
2024-08-15    2.0   10.0  1.727769  0.003456        15    1.471091
2024-09-04    2.0   10.0  0.057748  0.000115        13    0.519384
2024-10-01    0.0   10.0  1.373233  0.002746         9    1.130142
2024-11-01    0.0    5.0  0.749513  0.001499         7    2.851156
```

Kiểm tra bất biến trên data thật:
- `daily_returns.isna().any()` = False (không NaN nào trong toàn bộ chuỗi return net-of-fee).
- Kỳ đầu (2024-07-01): turnover = 1.000000 đúng như thiết kế (mua từ đầu, danh mục trước = 0).
- `weights` mỗi dòng: max|sum(w)-1| = 2.22e-16 (feasible sum=1 tới sai số máy), min(w) = 0.0
  (không có trọng số âm, đúng ràng buộc long-only).
- Turnover các kỳ sau dao động 0.058 - 1.79 tuỳ mức thay đổi tham số/nghiệm được chọn giữa các
  kỳ (turnover > 1 khả dĩ vì đổi từ danh mục cũ SANG danh mục hoàn toàn khác cấu trúc, ví dụ
  2024-08-15 danh mục trước đã trôi giá rồi rebalance sang tổ hợp kappa=2,gamma=10 khác biệt).

## Concerns

1. **RuntimeWarning benign lặp lại** (divide-by-zero/overflow/invalid trong `matmul` ở
   `matrix_sqrt_psd`, `portfolio_objective_long_only`, `_smooth_subgrad`, `_robust_subgrad`,
   và dòng mới `val_daily = Vwin.values @ w_val` trong `backtest.py`): xuất hiện trong CẢ smoke
   test thật lẫn suite pytest hiện có (`test_sparsity_increases_with_lambda`,
   `test_long_only_matches_cvxpy_on_real_data`) từ trước Task 4 -- đã ghi nhận trong
   `.sdd/progress.md` là "Apple Accelerate BLAS quirk", không sinh NaN thật trong kết quả cuối.
   Đã verify riêng: `daily_returns` và `weights` của smoke test data thật không có NaN nào. Không
   coi là bug mới do Task 4, nhưng ghi lại vì tần suất warning tăng lên (nhiều cửa sổ ước lượng
   hơn so với 1 lần gọi estimate_all trong test cũ).
2. **`weights.min() == 0.0` đúng như kỳ vọng long-only sparse** -- không phải lỗi, nhưng lưu ý
   n_active dao động mạnh (2 đến 17 trên 98 mã) tuỳ (kappa,gamma) được chọn mỗi kỳ qua Sharpe
   validation -- việc chọn tham số mỗi tháng có thể dao động khá nhiều (vd 2025-02-03 chỉ
   n_active=2), đây là hành vi nội tại của thiết kế (chọn theo Sharpe trên Vwin 6 tháng, mẫu nhỏ
   nên "best Sharpe" có thể nhạy với nhiễu) chứ không phải lỗi implement -- nêu ra để Task 6/7 (so
   sánh với equal-weight, phân tích ổn định tham số) lưu ý khi diễn giải kết quả.
3. Không có concern về look-ahead: `test_build_windows_no_look_ahead` PASS với assertion đầy đủ
   như brief (không nới lỏng), và code implementation dùng `periods.isin(e_months/v_months)` với
   `e_months`/`v_months` chỉ lấy từ `months[i - lookback_months : i]` (nghiêm ngặt trước
   `target_month`), nên bất biến chống look-ahead được đảm bảo cấu trúc (không phải chỉ đúng ngẫu
   nhiên trên test case cụ thể).
