# Task 6 Report — Visualization (fig7, fig8, fig9)

## Status: COMPLETE

## Files thay đổi

- Modified: `src/viz.py` — thêm 3 hàm mới (`fig7_backtest_equity`, `fig8_drawdown`,
  `fig9_selected_params`), cập nhật `__all__`, thêm đoạn docstring module giải thích
  vì sao fig7-9 KHÔNG đưa vào `generate_all()`.
- New (generated, không phải code): `figures/fig7_backtest_equity.png`,
  `figures/fig8_drawdown.png`, `figures/fig9_selected_params.png`.
- Modified: `.sdd/progress.md` — append 1 dòng `wf-Task 6: complete (...)`.

## Tóm tắt mỗi hình (smoke test trên `data/returns.parquet`, 965×98)

- **fig7_backtest_equity**: đường cong tài sản tích luỹ (net phí), chiến lược
  long-only walk-forward (xanh) vs equal-weight 1/N (xanh lá, nét đứt). Cả 2 dao
  động quanh 1.0 trong ~1 năm đầu, chiến lược có drawdown sâu (~0.69) giữa
  2025-04 rồi phục hồi vượt benchmark cuối giai đoạn (~1.25-1.4 vs ~1.15-1.4).
- **fig8_drawdown**: drawdown (%) theo thời gian cho cả 2 đường — chiến lược có
  đáy sâu nhất ~-38% (quanh 2025-04), benchmark đáy ~-21% cùng giai đoạn;
  minh hoạ chiến lược biến động mạnh hơn benchmark dù Sharpe cuối kỳ có thể cao hơn.
- **fig9_selected_params**: panel trên — scatter κ*/γ* chọn mỗi kỳ (γ hầu hết
  ở 1, 5 hoặc 10; κ gần như luôn 0, chỉ 3/25 kỳ κ=0.5 hoặc 2 — khớp finding đã
  ghi ở wf-Task 4); panel dưới — turnover mỗi kỳ (%), dao động mạnh 6%-179%,
  đỉnh cao trùng các kỳ κ/γ đổi đột ngột.

## Kiểm chứng

- Smoke test thật (không phải pytest, đúng brief Step 5): chạy
  `walk_forward_backtest` + `equal_weight_backtest` trên data thật rồi gọi cả
  3 hàm vẽ — cả 3 file PNG được tạo (156K/147K/67K), mở lên xem hợp lý (không
  NaN/rỗng, drawdown âm, tham số + turnover hiển thị rõ như liệt kê ở trên).
  `strategy.daily_returns`/`benchmark.daily_returns`: 0 NaN trên 494 ngày OOS.
- `pytest tests/ -q`: 40/40 pass, không có test mới nào cần thêm (brief coi
  đây là hình vẽ, không yêu cầu pytest chính thức) — chạy full suite chỉ để
  xác nhận không có regression.
- Style: dùng lại đúng `_apply_style()`, `_save()` (thay vì gọi `fig.savefig`
  trực tiếp như code mẫu trong brief, để nhất quán với fig1-6 — `_save()` nội
  bộ vẫn gọi đúng `dpi=150, bbox_inches="tight"`), `CAT_COLORS[0]`/`CAT_COLORS[1]`
  (blue/green, đã validate all-pairs) cho 2 series, `BASELINE` cho đường y=0.

## Concerns

- Trong lúc chạy walk-forward thật, quan sát thấy `RuntimeWarning: divide by
  zero/overflow/invalid value encountered in matmul` xuất hiện trong
  `src/estimators.py:237` (sqrt_sigma) và `src/prox_solver.py` (robust_term,
  var_term, subgrad) ở một vài cửa sổ ước lượng — cùng warning cũng xuất hiện
  khi chạy `pytest tests/` sẵn có (không phải regression do Task 6 gây ra).
  Kết quả cuối (`daily_returns`) vẫn 0 NaN nên không chặn việc vẽ hình, nhưng
  đây là dấu hiệu numerical instability tiềm ẩn trong solver/estimators ở một
  số (E, κ, γ) cụ thể — nằm ngoài phạm vi Task 6 (solver bị đóng băng theo
  Global Constraints), nên chỉ ghi nhận để Task 7 (notebook) hoặc review sau
  cân nhắc, không tự ý sửa.
- fig9 vẽ κ và γ chung 1 trục y dù thang giá trị khác nhau (κ∈[0,2] vs
  γ∈[1,10]) — giữ đúng theo brief/thiết kế đã duyệt vì mục tiêu là thấy thời
  điểm đổi tham số qua các kỳ, không phải so sánh độ lớn tuyệt đối.
- `generate_all()` KHÔNG được mở rộng để gọi fig7-9 (cần backtest result làm
  tham số, và `walk_forward_backtest` tốn ~29s theo ghi nhận wf-Task 4) — nếu
  Task 7 (notebook) muốn tái sinh toàn bộ 9 hình bằng 1 lệnh, sẽ cần gọi
  `walk_forward_backtest`/`equal_weight_backtest` riêng rồi truyền kết quả vào
  fig7-9 thủ công (như trong smoke test), KHÔNG qua `generate_all()`.
