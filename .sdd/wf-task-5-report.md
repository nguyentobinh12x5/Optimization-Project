# Report — wf-Task 5: Equal-weight benchmark

**Status:** COMPLETE

**Files changed:**
- `src/backtest.py` — thêm `equal_weight_backtest()` (đúng chữ ký brief), tái dùng
  `_build_rebalance_windows`, `_simulate_period`, `BacktestResult` có sẵn, không sửa gì logic cũ.
- `tests/test_backtest.py` — thêm `test_equal_weight_backtest_matches_uniform_weights` (đúng nội dung brief).
- `.sdd/progress.md` — append 1 dòng ghi nhận hoàn tất.

**Tóm tắt:** TDD đúng quy trình (viết test → FAIL do `ImportError` → implement → PASS).
`.venv/bin/python -m pytest tests/ -v`: 40/40 pass (39 cũ + 1 mới).

**Concerns:** Không có. RuntimeWarning matmul (divide-by-zero/overflow) xuất hiện ở
`test_cvxpy_check.py`/`test_prox_solver.py` là benign, đã ghi nhận từ các task trước (Apple
Accelerate BLAS quirk), không liên quan tới thay đổi của task này.
