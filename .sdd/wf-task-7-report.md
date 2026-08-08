# Task 7 Report — Notebook integration + README

Status: **DONE** (hoàn tất bởi controller sau khi subagent bị rớt kết nối giữa chừng —
xem "Ghi chú sự cố" cuối file).

## Việc đã làm (bởi subagent trước khi rớt kết nối — đã verify lại toàn bộ)

- `notebook.ipynb`: thêm section "## 7. Walk-forward backtest ngoài mẫu" (markdown giải
  thích cơ chế + code cell chạy `walk_forward_backtest`/`equal_weight_backtest` + bảng
  metrics + 3 hình fig7-9 + markdown "Kết luận thực nghiệm" trung thực) và mở rộng "## 8.
  Hạn chế & Kết luận" với mục cập nhật + hạn chế riêng của backtest.
- `README.md`: mô tả module `src/backtest.py`, cấu trúc thư mục cập nhật (9 hình,
  `backtest.py`), phần "Cách chạy" thêm bước đăng ký kernel (đã có từ Phase 6).

## Việc controller hoàn tất thêm (phần bị đứt do subagent rớt kết nối)

1. Cập nhật section "Kết quả `pytest tests/ -v`" trong README — bản cũ dừng ở 18 test,
   đã thay bằng output pytest THẬT hiện tại (40 test).
2. Cập nhật đoạn mô tả "Kết quả lần chạy thật gần nhất" của notebook trong README — bản cũ
   ghi "22 cell, 6 hình, ~14-15s", đã thay bằng số thật hiện tại: 28 cell, 9 hình, ~50s.
3. Chạy lại `nbconvert --execute` một lần cuối để xác nhận sạch sau khi README thay đổi
   (README không ảnh hưởng notebook, nhưng chạy lại cho chắc).
4. Viết report này + append ledger.

## Verify thật (controller, độc lập)

```
pytest tests/ -v -> 40 passed, 27 warnings in 11.99s
nbconvert --to notebook --execute notebook.ipynb --output notebook.ipynb
  -> 0 lỗi, elapsed ~50s
  -> cells=28, imgs=9, errors=0
```

## Số liệu backtest thật lấy từ notebook (khớp với số controller đã verify độc lập ở
Task 6, sai lệch 0 vì cùng code/data xác định):

| Metric | Chiến lược | Equal-weight 1/N |
|---|---|---|
| Cumulative return | 23.9% | 14.7% |
| Annualized return | 11.5% | 7.3% |
| Annualized vol | 28.7% | 19.9% |
| Sharpe (rf=0) | 0.524 | 0.452 |
| Max drawdown | -37.9% | -20.9% |
| Tổng phí (25 kỳ) | 3.44% | 0.49% |

Kết luận trong notebook viết đúng tinh thần đã yêu cầu: chiến lược thắng cumulative return
VÀ Sharpe, nhưng đây là đánh đổi risk-adjusted thật (rủi ro/phí cao hơn đáng kể), không
phải "thắng" đơn giản — không có tuyên bố phóng đại.

## Ghi chú sự cố (minh bạch quy trình)

Subagent ban đầu (dispatch lúc đầu Task 7) bị lỗi hạ tầng "Connection closed mid-response"
giữa chừng — KHÔNG phải lỗi logic/task. Kiểm tra lại cho thấy nó đã hoàn thành phần việc
nặng nhất (chèn section notebook, viết kết luận trung thực, chạy nbconvert lần đầu thành
công 0 lỗi) trước khi rớt kết nối; phần còn dang dở chỉ là 2 đoạn text trong README (mục
pytest + mục mô tả notebook) chưa kịp đồng bộ số liệu mới, và chưa ghi report/ledger.
Controller đã hoàn tất trực tiếp phần còn lại (không dispatch subagent mới, vì phần việc
còn lại nhỏ và controller đã có sẵn mọi số liệu cần từ các bước verify độc lập trước đó).
