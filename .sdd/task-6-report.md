# Task 6 Report — Notebook deliverable + README

## File tạo/sửa

- `notebook.ipynb` (project root, mới) — 22 cell (11 markdown + 11 code), build bằng
  `nbformat` (script tạm ở scratchpad, không commit vào repo) rồi thực thi thật bằng
  `nbconvert --execute`.
- `README.md` (project root, mới).
- Đăng ký thêm 1 Jupyter kernel `optproj-venv` trỏ đúng `.venv/bin/python` (máy này trước
  đó chỉ có kernel `python3` trỏ tới Python 3.13 hệ thống khác, không phải venv của
  project) — cần thiết để `nbconvert --execute` chạy đúng interpreter có sẵn
  numpy/pandas/matplotlib/seaborn/cvxpy. Lệnh:
  `python -m ipykernel install --user --name optproj-venv --display-name "Python (Optimization Project venv)"`.
  README đã ghi lại bước này ở mục "Cách chạy" (bước 5) để tái lập trên máy khác.

## Cấu trúc notebook (đúng 8 phần theo brief)

1. Tiêu đề + phát biểu bài toán (công thức đầy đủ, bảng ý nghĩa từng số hạng, nêu rõ
   cho phép bán khống).
2. Dữ liệu: `load_from_cache()`, in shape/khoảng ngày, `fig1` (heatmap tương quan), nêu
   survivorship bias + giả định giá điều chỉnh.
3. Ước lượng: `estimate_all()`, verify `‖Σ^(1/2)²-Σ‖_F/‖Σ‖_F`.
4. Thuật toán: markdown đầy đủ công thức subgradient (kể cả chain-rule dẫn tới `Σw`
   thay vì `Σ^(1/2)w`), joint-prox qua Lagrangian + bisection, lý do `α0/√(k+1)`, lý do
   best-iterate; code chạy `solve()` 2 bộ tham số + `fig2` (convergence).
5. Kết quả sparsity & weights: `fig3` + `fig4`, bàn về λ điều khiển số mã nắm giữ.
6. Hiệu ứng robust: `fig5`, bàn phát hiện phản trực giác (đã verify số thật, xem dưới).
7. Verify chéo CVXPY: `compare()` với `DEFAULT_PARAM_GRID` (6 bộ, ≥5 theo yêu cầu), in
   DataFrame đầy đủ, `fig6`, kết luận relgap cực nhỏ + Jaccard.
8. Hạn chế & kết luận: nêu quá trình prox-then-project → joint-prox như bài học (đúng
   yêu cầu brief), survivorship bias, μ̂ nhiễu, robust finding, hướng mở rộng.

Cả 6 hình được sinh **bằng cách gọi lại hàm trong `src/viz.py`** (không phải đọc PNG tĩnh
qua `Image()`) — chọn cách này vì `generate_all()` full pipeline chỉ mất ~7-9s (đã đo ở
Phase 5), nằm rất xa dưới ngưỡng 3 phút, nên notebook tự tái sinh hình chứng minh khả năng
tái lập thay vì chỉ hiển thị ảnh có sẵn.

## Kết quả `nbconvert --execute` (chạy thật)

Lệnh: `.venv/bin/python -m nbconvert --to notebook --execute notebook.ipynb --output notebook.ipynb --ExecutePreprocessor.timeout=180`

```
[NbConvertApp] Converting notebook notebook.ipynb to notebook
[NbConvertApp] Writing 377881 bytes to notebook.ipynb
real: ~14.5s (12.71s user + 1.19s system, 96% cpu)
exit code: 0
```

Verify bằng script đọc lại `notebook.ipynb` (nbformat):
- Tổng số cell: **22** (11 code, 11 markdown).
- Số cell lỗi (`output_type == "error"`): **0**.
- Số output `image/png` (display_data/execute_result): **6/6** — đúng đủ fig1..fig6.
- Không cần mạng (chỉ đọc `data/*.parquet` qua `load_from_cache()`).

Đã chạy 2 lần (lần đầu phát hiện 1 câu markdown diễn giải sai — sửa ở dưới — rồi build +
execute lại lần 2, cả 2 lần đều sạch 0 lỗi, 6/6 ảnh, thời gian tương đương ~14-15s).

## Sửa 1 lỗi diễn giải trong quá trình build (không fabricate)

Bản build đầu tiên có câu kết luận verify chéo viết "Jaccard = 1.0 ở mọi bộ tham số" —
sau khi in bảng `compare()` thật, phát hiện bộ tham số `κ=1, γ=5, λ=0` có
`active_hand=97, active_cvx=96, jaccard≈0.9897` (không phải 1.0), vì `λ=0` tắt hoàn toàn
phạt L1 nên không có sparsity thật để so khớp (1 toạ độ biên quanh ngưỡng
`active_thresh=1e-4` khác nhau giữa 2 solver). Đã sửa markdown để phản ánh đúng số liệu
thật: relgap tối đa quan sát ~0.0007%, Jaccard=1.0 ở 5/6 bộ, ~0.99 ở bộ λ=0 (giải thích rõ
lý do, không phải solver sai). Số liệu đầy đủ (bảng 6 dòng) nằm trong notebook, cell in
`df_compare`.

Đã verify thêm bằng script độc lập (ngoài notebook) rằng finding phản trực giác ở fig5
(`κ=0,γ=5,λ=0.02` → active=9, HHI=0.2064; `κ=1` → active=16, HHI=0.2757) là số thật trên
`data/returns.parquet` hiện tại — κ tăng làm TĂNG cả active-set lẫn HHI, khớp đúng nội
dung markdown mục 5 của notebook.

## Kết quả `pytest tests/ -v` (chạy thật)

18/18 PASSED, 3 warnings (RuntimeWarning BLAS đã biết từ Phase 2/3, không phải lỗi),
5.25s. Output đầy đủ đã dán vào `README.md` mục "Kết quả pytest tests/ -v".

## Xác nhận docstring các module `src/`

`ast.get_docstring()` xác nhận cả 5 module (`data_loader.py`, `estimators.py`,
`prox_solver.py`, `cvxpy_check.py`, `viz.py`) đều có docstring module-level (2200-9600
ký tự mỗi file, kế thừa từ các phase trước) — không cần viết lại, chỉ xác nhận theo yêu
cầu brief.

## Trả về controller

- **Status**: DONE.
- **File tạo**: `notebook.ipynb`, `README.md` (cả 2 ở project root).
- **Tóm tắt 1 dòng**: nbconvert --execute chạy SẠCH (0 lỗi, 22 cell, 6/6 hình, ~14-15s,
  exit 0, không mạng); pytest 18/18 PASSED.
- **Concerns**:
  1. Máy này ban đầu không có Jupyter kernel trỏ đúng `.venv` — đã đăng ký kernel mới
     (`optproj-venv`) và ghi vào README; nếu chạy trên máy khác cần lặp lại bước đăng ký
     kernel đó (đã có sẵn trong README bước 5) trước khi `nbconvert --execute` hoặc mở
     Jupyter UI.
  2. Repo không có `requirements.txt` sẵn (đã kiểm tra, không tìm thấy) — README liệt kê
     trực tiếp package cần cài qua `pip install` thay vì trỏ tới file requirements không
     tồn tại; đã ghi chú cách tự tạo requirements.txt bằng `pip freeze` nếu cần.
  3. Bộ tham số `κ=1,γ=5,λ=0` trong bảng verify chéo có Jaccard≈0.99 (không phải 1.0
     tuyệt đối) — đã diễn giải rõ nguyên nhân trong notebook (λ=0 không tạo sparsity
     thật), không phải vấn đề của solver.
