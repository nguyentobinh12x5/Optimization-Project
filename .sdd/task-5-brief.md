# Task Brief — Phase 5: Visualization

Đọc trước tiên. Sinh các hàm vẽ + tạo 6 hình từ dữ liệu/kết quả đã có. Sau phase này user sẽ chọn hình nào vào notebook (Gate 3).

## BẮT BUỘC trước khi vẽ
Gọi skill `dataviz` (tool Skill) và tuân theo hướng dẫn về palette/style/accessibility. Mục tiêu: 6 hình trông NHƯ MỘT hệ thống nhất quán (cùng palette, cùng font-size, trục có nhãn + đơn vị, colorblind-safe), phù hợp notebook học thuật.

## Môi trường
- `.venv/bin/python`. Có numpy/pandas/matplotlib/seaborn. Consume: `estimate_all`, `solve`, `cvxpy_solve`.
- Data: `data/returns.parquet`, `data/prices.parquet` (đã có, 97 mã). KHÔNG gọi mạng.
- Backend: dùng `matplotlib` với `Agg` (savefig ra file, không cần hiển thị).

## Output
- `src/viz.py`: mỗi hình = 1 hàm, nhận dữ liệu/kết quả làm tham số, trả `matplotlib.figure.Figure` VÀ lưu ra `figures/<tên>.png` (dpi 150, bbox_inches='tight'). Có hàm `generate_all()` chạy tất cả từ cache.
- Thư mục `figures/` chứa 6 PNG.
- Phase 6 (notebook) sẽ import các hàm này → thiết kế hàm tái sử dụng được, KHÔNG hardcode đường dẫn tuyệt đối (dùng path tương đối từ project root).

## 6 hình cần tạo
1. **fig1_data_overview**: tổng quan universe — heatmap ma trận tương quan của returns (97×97, có colorbar), HOẶC/VÀ giá chuẩn hóa (giá/giá đầu kỳ) một nhóm mã đại diện. Chọn cách rõ ràng nhất; nhãn trục, tiêu đề.
2. **fig2_convergence**: objective theo iteration (trục y log nếu hợp lý), vẽ 2-3 bộ tham số trên cùng axes, legend rõ. Dùng `obj_history` từ `solve`.
3. **fig3_sparsity_path**: số tài sản active (|wᵢ|>1e-4) theo λ, trục x log-scale (λ từ 1e-4 → 1e-1, ~15-20 điểm), giữ κ,γ cố định (vd κ=1,γ=5). Cho thấy λ tăng → thưa hơn.
4. **fig4_weights_bar**: bar chart trọng số w* tại 2-3 bộ tham số tiêu biểu (vd λ nhỏ vs λ lớn), trục x là mã (hoặc chỉ các mã active để đỡ rối), có đường y=0 (vì cho phép w âm).
5. **fig5_robust_effect**: so sánh danh mục κ=0 vs κ>0 (cùng γ,λ) — vd cạnh nhau so sánh độ tập trung (số mã active, hoặc ‖w‖₂ / Herfindahl), hoặc overlay trọng số. Nêu rõ robust term ảnh hưởng thế nào.
6. **fig6_prox_vs_cvxpy**: scatter w_hand (trục x) vs w_cvxpy (trục y) cho 1 bộ tham số, thêm đường y=x tham chiếu. Minh họa solver tay khớp CVXPY.

## Acceptance criteria
1. `.venv/bin/python -c "from src.viz import generate_all; generate_all()"` (hoặc `python -m src.viz`) chạy KHÔNG lỗi, tạo đủ 6 file PNG trong `figures/`.
2. Mỗi hình mở được, có tiêu đề + nhãn trục (+ đơn vị khi có ý nghĩa) + legend/colorbar khi cần. Style nhất quán giữa các hình (theo dataviz).
3. Các hàm import được và tái dùng trong notebook (Phase 6).
4. Dùng dữ liệu cache, chạy nhanh (< ~2 phút kể cả chạy solve cho sparsity path).

## Report vào `.sdd/task-5-report.md`
Đầy đủ: file tạo, danh sách 6 PNG + mô tả ngắn mỗi hình (1 dòng: hình cho thấy gì), xác nhận generate_all chạy sạch, palette/style đã theo dataviz thế nào. Trả về controller: status, danh sách file (gồm 6 PNG), 1 dòng tóm tắt, concerns.
