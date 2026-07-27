# Task 5 Report — Visualization

## File tạo

- `src/viz.py` — module chính: 6 hàm vẽ (`fig1_data_overview` .. `fig6_prox_vs_cvxpy`), mỗi hàm nhận dữ liệu/kết quả làm tham số, trả `matplotlib.figure.Figure` và lưu PNG (dpi 150, `bbox_inches='tight'`) ra `figures/<tên>.png` qua path TƯƠNG ĐỐI (`Path(__file__).resolve().parent.parent`, cùng quy ước với `PROJECT_ROOT` trong `src/data_loader.py` — không hardcode path tuyệt đối). Hàm `generate_all()` đọc `data/returns.parquet`, gọi `estimate_all` rồi chạy cả 6 hàm, trả `dict[str, Path]`. Entry point `python -m src.viz` chạy `generate_all()` + in tóm tắt.
- `figures/` — 6 PNG (đường dẫn tuyệt đối, tất cả trong `/Users/nguyentobinh12gmail.com/Documents/Optimization Project/figures/`):
  - `fig1_data_overview.png` (101 KB)
  - `fig2_convergence.png` (40 KB)
  - `fig3_sparsity_path.png` (48 KB)
  - `fig4_weights_bar.png` (68 KB)
  - `fig5_robust_effect.png` (71 KB)
  - `fig6_prox_vs_cvxpy.png` (58 KB)

## Xác nhận chạy sạch

Lệnh: `.venv/bin/python -m src.viz` (từ project root). Kết quả: **6/6 PNG sinh thành công trong 9.25s** (`generate_all()` báo `Đã sinh 6 hình trong 7.2s` + overhead import ~2s), tất cả `exists=True`. Có `RuntimeWarning: divide/overflow/invalid value encountered in matmul` — đây là noise BLAS (Apple Accelerate) đã biết TỪ TRƯỚC ở Phase 2/3/4 (xem docstring `src/prox_solver.py`, và cùng warning xuất hiện y hệt khi chạy `pytest` trên `tests/test_prox_solver.py` vốn không đụng tới `src/viz.py`) — KHÔNG phải lỗi do code vẽ gây ra, không ảnh hưởng kết quả (đã xác nhận bằng mắt tất cả 6 hình, không có NaN/giá trị vỡ). Toàn bộ test suite cũ (18 test) vẫn PASS sau khi thêm module mới.

## 6 hình — mô tả ngắn

1. **fig1_data_overview** — Heatmap tương quan Pearson của daily returns toàn bộ 97 mã VN100 (colormap diverging blue↔gray↔red, [-1,1], colorbar); cho thấy universe có tương quan dương khá đồng đều/mạnh giữa hầu hết các mã (đặc trưng thị trường large-cap VN), là input trực tiếp cho Sigma mà solver dùng.
2. **fig2_convergence** — Objective f(w_k) theo iteration (trục y log) cho 3 bộ tham số (κ=1,λ=0.01 / κ=0,λ=0.1 / κ=1,λ=0.001); cho thấy solver hội tụ nhanh (vài trăm–2 nghìn vòng) và ổn định (đường phẳng sau khi patience trigger), không đơn điệu tuyệt đối nhưng best-iterate hội tụ rõ.
3. **fig3_sparsity_path** — Số tài sản active theo λ (trục x log-scale, 18 điểm 1e-4→1e-1, κ=1,γ=5 cố định); cho thấy quan hệ đơn điệu giảm rõ ràng: λ tăng → danh mục thưa hơn, bão hoà ở λ≳4e-3 (giữ nguyên 16 mã active).
4. **fig4_weights_bar** — 2 panel small-multiples, trọng số w* tại λ nhỏ (0.001, dense hơn, 38 mã active) vs λ lớn (0.02, thưa hơn), cùng trục x = union mã active, đường y=0; cho thấy λ lớn vừa giảm số mã active vừa tập trung trọng số hơn vào vài mã đầu (KOS, KDC, VPI).
5. **fig5_robust_effect** — So sánh danh mục κ=0 vs κ=1 (γ=5, λ=0.02 cố định): panel trái = grouped bar trọng số theo mã, panel phải = số mã active (9 vs 16) và HHI×100 (20.6 vs 27.6); cho thấy robust term (κ>0) ở bộ tham số này vừa MỞ RỘNG active-set vừa dồn trọng số mạnh hơn vào 1 mã đơn lẻ (KOS: 0.04→0.49) — kết quả thật từ data, không phải giả định lý thuyết áp đặt.
6. **fig6_prox_vs_cvxpy** — Scatter w_hand vs w_cvxpy (κ=1,γ=5,λ=0.01, giống bộ đầu `DEFAULT_PARAM_GRID` Phase 4) + đường y=x tham chiếu; điểm nằm sát đường chéo (‖Δw‖∞=6.28e-4, khớp số liệu đã verify ở task-4-report.md), minh hoạ trực quan solver tay khớp CVXPY.

## Palette/style theo skill `dataviz`

- Gọi skill `dataviz` trước khi viết code (đọc `palette.md`, `color-formula.md`), dùng đúng palette đã VALIDATE bằng `scripts/validate_palette.js` của skill (không tự chọn màu eyeball):
  - Categorical 8-hue thứ tự cố định (`CAT_COLORS` trong `src/viz.py`), CHỈ dùng tối đa 4 slot đầu (blue/green/magenta/yellow) cho các hình có nhiều series cạnh nhau (bar/scatter/small-multiples) — đã chạy validator với `--pairs all --mode light`, kết quả: PASS mọi check cứng (lightness band, chroma floor, CVD ΔE all-pairs 13.0, normal-vision floor 19.6), chỉ WARN contrast của magenta/yellow trên nền sáng (2.1–2.6:1) — đã bù bằng "relief channel" bắt buộc: mọi hình có các slot đó đều có legend rõ (fig2 dùng magenta cho 1 trong 3 đường, luôn kèm legend).
  - Diverging blue↔gray↔red cho heatmap tương quan (fig1) — đúng job "diverging=polarity" (correlation có dấu +/-), KHÔNG dùng rainbow/jet.
  - Ink/surface nhất quán qua `_apply_style()` (rcParams áp 1 lần đầu mỗi hàm): nền `#fcfcfb`, chữ chính `#0b0b0b`, chữ phụ `#52514e`/`#898781`, gridline hairline `#e1e0d9`, baseline/axis `#c3c2b7` — cùng 1 bộ rcParams cho cả 6 hình nên trông như MỘT hệ thống (cùng font-size tiêu đề/trục, cùng độ dày line/spine, cùng style legend không khung).
  - Mỗi hình có tiêu đề, nhãn trục kèm đơn vị khi có ý nghĩa (vd "Hệ số tương quan (đơn vị: không thứ nguyên, [-1,1])", "λ (hệ số phạt L1, thang log)"), colorbar (fig1) hoặc legend (fig2, fig4, fig5, fig6) khi có ≥2 series; fig3 chỉ 1 series nên không cần legend (title đã nêu κ,γ cố định) — đúng quy tắc skill "1 series không cần legend box".
  - Đường y=0 (fig4, fig5, nét đứt màu baseline) và y=x (fig6, nét đứt màu baseline) dùng cùng 1 style tham chiếu nhất quán.
- Static PNG cho notebook học thuật (không phải trang HTML tương tác runtime) nên chỉ ship 1 theme (light) — không có dark-mode toggle vì đây là ảnh raster tĩnh, không phải trang web.

## Concerns

1. **Chỉ ship light theme** — do 6 hình là PNG tĩnh nhúng vào notebook học thuật (không phải artifact HTML có thể đổi theme runtime), không làm bản dark-mode riêng. Nếu Phase 6 cần hiển thị trong môi trường nền tối, cần regenerate với rcParams surface khác (dễ làm vì đã tách `_apply_style()` riêng).
2. **fig1 gần như toàn đỏ** — vì tương quan daily return giữa các mã large-cap VN100 hầu hết dương khá mạnh (đặc điểm thị trường thật, không phải lỗi màu); colorbar vẫn đúng thang [-1,1] đầy đủ, chỉ là dữ liệu thật không có nhiều giá trị âm để tô xanh.
3. **fig4/fig5 dùng `max_iter=3000`** (khác `max_iter=5000` mặc định của `solve()`/Phase 4) để giữ tổng runtime `generate_all()` nhanh (<10s) — mọi bộ tham số quan sát đều `converged=True` trong vài trăm–~2000 vòng (xem fig2), nên 3000 đủ dư; fig6 vẫn dùng `max_iter=5000` để khớp CHÍNH XÁC số liệu đã verify ở task-4-report.md.
4. **fig5 cho kết quả "phản trực giác nhẹ"**: robust term (κ=1) ở bộ tham số γ=5,λ=0.02 vừa tăng active-set (9→16) vừa tăng HHI (20.6→27.6, tức tập trung hơn) — đây là quan sát THẬT từ data (không chỉnh sửa để khớp kỳ vọng lý thuyết "robust luôn làm portfolio đa dạng hơn"); nên nêu rõ trong notebook rằng hiệu ứng robust term phụ thuộc bộ tham số cụ thể, không phải quy luật đơn điệu tổng quát.
5. Cảnh báo `RuntimeWarning` (BLAS matmul) xuất hiện khi chạy — đã xác nhận là artifact môi trường có từ trước (không phải do `src/viz.py`), không ảnh hưởng đến giá trị/hình ảnh sinh ra.

## Trạng thái

DONE. `generate_all()` chạy sạch (không lỗi, không exception), tạo đủ 6/6 PNG trong `figures/`, mỗi hình có tiêu đề + nhãn trục (kèm đơn vị khi có ý nghĩa) + legend/colorbar khi cần, style nhất quán theo skill `dataviz` (palette đã validate bằng script, không eyeball). Toàn bộ 6 hàm vẽ + `generate_all()` import được từ `src.viz` để Phase 6 (notebook) tái sử dụng trực tiếp.
