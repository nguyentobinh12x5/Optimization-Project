# wf-Task 3 Report — Performance metrics

**Status:** COMPLETE

**Files created:**
- `src/backtest.py` (module mới, docstring đầu file như brief; hàm `performance_metrics(daily_returns, rf=0.0) -> dict`)
- `tests/test_backtest.py` (3 test theo brief: constant return, known drawdown sequence, sharpe sign)

**Tóm tắt:** TDD đúng quy trình — viết test trước, chạy FAIL (ModuleNotFoundError vì `src/backtest.py` chưa tồn tại), implement, chạy PASS. pytest toàn repo: 34/34 pass (31 cũ + 3 mới), không regression.

**Concerns / deviation nhỏ so với code mẫu trong brief:**
- Code mẫu dùng `if daily_std > 0:` để quyết định Sharpe hay NaN. Trên thực tế `pandas.Series.std(ddof=1)` của một chuỗi return HẰNG SỐ (test `test_metrics_constant_return`, 252 giá trị `0.001`) không trả đúng `0.0` tuyệt đối mà ra một số cực nhỏ do sai số làm tròn floating-point (~2.17e-19 trong lần chạy thực tế) — khiến Sharpe bị chia cho số gần-0 và bùng nổ thành ~7.3e16 thay vì NaN, làm test đầu tiên FAIL.
- Đã fix bằng cách thêm hằng số `STD_ZERO_EPS = 1e-12` và so sánh `daily_std > STD_ZERO_EPS` thay vì `> 0`, đúng tinh thần thiết kế (vol≈0 → Sharpe không xác định → NaN). Đã cập nhật docstring giải thích rõ lý do. Không có thay đổi nào khác so với đặc tả brief.
- Không đụng tới solver (`solve()`, `portfolio_objective()`, `_prox_l1_simplex_eq`, `_robust_subgrad`) — không liên quan tới task này.
- Không chạy lệnh git nào (project không dùng git theo constraint).
