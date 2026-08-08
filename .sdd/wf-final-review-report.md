# Walk-Forward Backtest — Final Review Report

**Ngày review**: 2026-07-26
**Phạm vi**: Chỉ phần MỚI (7 task) — long-only solver, walk-forward engine, equal-weight
benchmark, fig7/8/9, tích hợp notebook. Phần cũ (long-short) chỉ kiểm tra KHÔNG bị sửa.
**Trạng thái test**: `pytest tests/` → **40 passed** (11.97s). Khớp kỳ vọng.

## Kết luận tổng quát

**Phần MỚI về cơ bản SẠCH và ĐÚNG.** Không phát hiện lỗi correctness nào (Critical/
Important). Chống look-ahead đúng, toán long-only đúng, metrics đúng chuẩn tài chính,
số liệu notebook khớp output 100%. Chỉ có vài điểm Minor (dead code, edge-case degenerate,
ghi chú nhất quán). Chi tiết bên dưới.

---

## Critical

Không có.

## Important

Không có.

## Minor

1. **`src/backtest.py:17` — dead code**: `logger = logging.getLogger("backtest")` được
   khai báo nhưng KHÔNG dùng ở đâu (0 lời gọi `logger.*`). Cùng với `import logging`.
   Vô hại nhưng thừa.

2. **`src/backtest.py:283-294` — edge-case model selection degenerate**: nếu MỌI (kappa,
   gamma) trong 1 kỳ đều cho Sharpe = NaN (mọi std validation = 0), `best_sharpe` giữ
   `-np.inf` và `best_params` rơi về `param_grid[0]` → `val_sharpe` ghi vào rebalance_log
   là `-inf`. Fallback param_grid[0] hợp lý, nhưng `-inf` lọt vào DataFrame log là hơi
   xấu. Không xảy ra trên data thật (đã chạy: 25 kỳ đều có Sharpe hữu hạn). Tie-breaking
   dùng `>` (strict) → param đầu tiên thắng khi hoà, nhất quán và tất định. OK.

3. **`src/backtest.py:288` — bất nhất selection vs deployment (theo spec, không phải bug)**:
   Sharpe validation tính trên `Vwin.values @ w_val` với w GIỮ CỐ ĐỊNH (không drift, không
   phí) suốt 6 tháng; trong khi deployment (`_simulate_period`) lại drift trọng số từng
   ngày và trừ phí. Nghĩa là tiêu chí chọn tham số hơi lệch so với hành vi thực tế. Đây
   ĐÚNG theo spec §3 bước 3c (validation chỉ để xếp hạng param, cố ý đơn giản hoá) — ghi
   nhận là lựa chọn thiết kế, không phải lỗi.

4. **`src/viz.py:787 generate_all()` — chỉ tái sinh fig1-6**: fig7/8/9 KHÔNG nằm trong
   `generate_all()` (đúng, vì chúng cần `BacktestResult` chứ không chỉ estimate). Hệ quả:
   `figures/` chỉ có đủ 9 PNG nếu notebook/smoke-test đã chạy — README (dòng 42) nói "9 PNG
   đã sinh sẵn". Chỉ là ghi chú nhất quán vận hành, không phải lỗi code.

5. **`src/backtest.py:287` — hiệu năng (không phải correctness)**: vòng validation gọi
   `solve_long_only(...)` với `max_iter=5000` mặc định cho 12 param × 25 kỳ. Chạy được
   (test + notebook đã chạy thật) nhưng đây là điểm tốn thời gian nhất; không ảnh hưởng
   tính đúng.

---

## Xác minh chi tiết các điểm trọng tâm

### 1. Chống look-ahead (`_build_rebalance_windows`) — ĐÚNG
- `months = sorted(periods.unique())`; với target `months[i]`, cửa sổ = `months[i-24:i]`
  (24 tháng NGAY TRƯỚC, disjoint với target). E = 18 tháng đầu, Vwin = 6 tháng cuối của
  cửa sổ đó → cả E và Vwin nằm trong các tháng < target_month.
- `rebalance_date = index[target_mask][0]` (ngày giao dịch đầu tiên của target month) →
  mọi ngày của E/Vwin < rebalance_date. Test `test_build_windows_no_look_ahead` khẳng định
  `e_dates.max() < rebalance_date`, `v_dates.max() < rebalance_date`, `e < v`. PASS.
- `walk_forward_backtest` chỉ đọc `target` (dữ liệu tháng OOS) trong `_simulate_period`
  (bước tính return OOS), KHÔNG dùng vào estimate/validation/chọn param. `estimate_all`
  chạy trên E rồi trên `E∪Vwin`, không bao giờ chạm target. Sạch.
- Dữ liệu thật: 49 tháng → 25 kỳ rebalance (target = months[24..48]). Khớp.

### 2. `_simulate_period` — ĐÚNG
- Buy-and-hold đúng: return ngày = `w @ r_day` với w ĐẦU ngày, sau đó drift
  `w ← w(1+r)/Σ`. Kỳ sau tính turnover so với `w_end` (đã trôi) — đúng spec §4.
- Kỳ đầu `prev=None → zeros → turnover = Σ|w_start| = 1.0` (vì Σw_start=1). Test khẳng
  định turnover kỳ đầu = 1.0 (cả strategy lẫn benchmark). PASS.
- `cost = fee*turnover` trừ 1 lần vào `daily_net.iloc[0]`. Đúng.

### 3. Toán long-only — ĐÚNG
- `simplex_projection` (Duchi 2008): kiểm chứng tay ví dụ [0.5,0.5,0.5]→[1/3,1/3,1/3],
  [2,0.2,0.2]→[1,0,0], điểm feasible là fixed point. `rho`/`theta` index đúng
  (`theta=(cumsum[rho]-1)/(rho+1)`). 4 test PASS.
- `solve_long_only` TÁI DÙNG trực tiếp `_smooth_subgrad` (nhánh long-short, không copy lại
  công thức) → subgradient `-mu + 2γΣw + robust` khớp 100% với `solve()`. Điểm khác biệt
  duy nhất: thay `_prox_l1_simplex_eq` bằng `simplex_projection`. Cấu trúc best-iterate /
  patience / rel_change giống hệt `solve()`. Đúng thiết kế.
- CVXPY verify (`compare_long_only`, 4 bộ param, data thật): relgap < 0.5%, w≥0, Σw=1.
  `test_long_only_matches_cvxpy_on_real_data` PASS.

### 4. `portfolio_objective_long_only` — KHỚP với cái đang tối ưu
- Đúng 3 term `-μᵀw + κ‖Σ^½w‖₂ + γ wᵀΣw`, KHÔNG có λ. Formulation CVXPY
  (`cvxpy_solve_long_only`) khớp chính xác cùng 3 term + ràng buộc `w≥0, Σw=1`. Nhất quán.

### 5. Model selection — không có bug logic
- `np.isnan(sharpe) → -inf`; `>` strict tie-break; `best_params=param_grid[0]` fallback.
  param_grid rỗng: `param_grid[0]` sẽ IndexError, nhưng `None → DEFAULT_PARAM_GRID` (12 bộ)
  nên chỉ xảy ra nếu caller cố tình truyền `[]` — ngoài hợp đồng, chấp nhận được.

### 6. `performance_metrics` — ĐÚNG chuẩn
- Sharpe = (mean_daily − rf/252)/std(ddof=1) × √252; std≈0 (< `STD_ZERO_EPS=1e-12`) → NaN
  (tránh chia số cực nhỏ). annualized_return geometric `(1+cum)^(1/n_years)−1`.
  max_drawdown = min(equity/cummax − 1) ≤ 0. Edge chuỗi rỗng: n=0 → cum=0, sharpe/dd/
  ann_return = NaN, không crash. 3 test PASS.

### 7. Notebook — số liệu markdown KHỚP output đã lưu 100%
Đối chiếu cell 26 (markdown "Kết luận thực nghiệm") với output cell 24:
| | markdown | output cell 24 | khớp |
|---|---|---|---|
| Cum return | 23.9% / 14.7% | 0.238576 / 0.147189 | ✓ |
| Ann return | 11.5% / 7.3% | 0.115326 / 0.072559 | ✓ |
| Ann vol | 28.7% / 19.9% | 0.286765 / 0.198787 | ✓ |
| Sharpe | 0.524 / 0.452 | 0.524351 / 0.452280 | ✓ |
| Max DD | -37.9% / -20.9% | -0.378944 / -0.209146 | ✓ |
| n_days | 494 | 494 | ✓ |
Phí & số kỳ (không hiện trong output cell nhưng kiểm chứng lại bằng code):
- 25 kỳ rebalance ✓ (chạy `_build_rebalance_windows` trên data thật).
- Benchmark tổng phí = 0.4936% ✓ (≈ "0.49%").
- Strategy tổng phí 3.44% ⇒ turnover TB ≈ 68.8% ✓ (khớp "~69%/tháng").
Cell 25 có đủ 3 image/png (fig7/8/9). Cell 27 (Hạn chế & Kết luận) cập nhật đúng.

### 8. Dead code / docstring / path / cvxpy leak
- `src/prox_solver.py` & `src/backtest.py`: KHÔNG `import cvxpy` (chỉ có comment nhắc tên
  CVXPY trong docstring prox_solver — vô hại). cvxpy chỉ ở `src/cvxpy_check.py`. Sạch.
- Không có đường dẫn tuyệt đối hardcode trong src.
- Phần cũ (`solve`, `portfolio_objective`, `_prox_l1_simplex_eq`, `_robust_subgrad`,
  `_smooth_subgrad`) hiện diện đầy đủ, nhất quán docstring, và `_smooth_subgrad` được cả
  `solve` lẫn `solve_long_only` dùng chung — KHÔNG bị sửa lệch.
- Dead code duy nhất: `logger` trong backtest.py (Minor #1).

---

## Cảnh báo runtime (đã biết, benign)
`RuntimeWarning: invalid/divide/overflow in matmul` từ Apple Accelerate BLAS khi
`κ=0` (robust term tắt) — đã được tài liệu hoá từ Phase 2/3, không ảnh hưởng kết quả
(các test vẫn PASS, giá trị hữu hạn). Không phải lỗi mới.
