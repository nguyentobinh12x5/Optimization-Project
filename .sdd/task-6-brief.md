# Task Brief — Phase 6: Notebook deliverable + README

Đọc trước tiên. Đây là deliverable CUỐI: gộp toàn bộ thành notebook chạy end-to-end + README. User đã chọn đưa CẢ 6 HÌNH vào notebook.

## Mục tiêu
`notebook.ipynb` chạy sạch từ đầu đến cuối, kể chuyện mạch lạc: bài toán → dữ liệu → thuật toán → kết quả → verify → kết luận. Import từ `src/`, KHÔNG copy-paste code dài vào cell.

## Môi trường
- `.venv/bin/python` (Python 3.14). Đã cài: numpy/pandas/matplotlib/seaborn/cvxpy/pytest + jupyter stack (nbconvert/nbformat/ipykernel).
- Modules có sẵn: `src/data_loader.py`, `src/estimators.py`, `src/prox_solver.py`, `src/cvxpy_check.py`, `src/viz.py`.
- Data cache: `data/returns.parquet`, `data/prices.parquet` (97 mã). Figures: `figures/fig1..6.png`.
- KHÔNG gọi mạng: notebook PHẢI dùng cache, không tải lại từ vnstock.

## Cấu trúc notebook (theo thứ tự cell, markdown + code xen kẽ)
1. **Tiêu đề + Phát biểu bài toán**: markdown giải thích `min -μ̂ᵀw + κ‖Σ^½w‖₂ + γwᵀΣw + λ‖w‖₁ s.t. 1ᵀw=1`, ý nghĩa TỪNG term (return, robust, risk, sparsity, ràng buộc ngân sách). Nêu cho phép bán khống.
2. **Dữ liệu**: markdown mô tả universe VN100 (97 mã, 4 năm daily, simple return, nguồn vnstock VCI). Code: `load_from_cache()` in shape + vài dòng đầu. Hiển thị **fig1** (heatmap tương quan). Nêu limitation: survivorship bias + giả định giá điều chỉnh.
3. **Ước lượng μ̂, Σ, Σ^½**: markdown ngắn (eigh + clip vì Σ chỉ PSD). Code: `estimate_all(returns)`, in shape + verify Σ^½²≈Σ.
4. **Thuật toán proximal-subgradient**: markdown giải thích công thức update (subgradient phần trơn+robust, JOINT PROX của L1+ràng buộc qua soft-threshold+bisection ν → số 0 chính xác + feasible), lý do step `α0/√(k+1)`, trả best-iterate. Code: chạy `solve()` 1-2 bộ tham số, in kết quả. Hiển thị **fig2** (convergence).
5. **Kết quả — tính thưa & trọng số**: Hiển thị **fig3** (sparsity path), **fig4** (weights bar). Bàn: λ điều khiển số mã nắm giữ.
6. **Hiệu ứng robust term**: Hiển thị **fig5**. Bàn phát hiện phản trực giác: κ>0 làm TĂNG active-set + HHI ở dữ liệu này (đưa ra giải thích hợp lý, không ép về lý thuyết).
7. **Verify chéo CVXPY**: markdown nêu vì sao verify. Code: `from src.cvxpy_check import compare` chạy bảng ≥5 bộ, in DataFrame. Hiển thị **fig6** (scatter). Kết luận: solver tay khớp nghiệm tối ưu (relgap ~0, active-set khớp).
8. **Hạn chế & Kết luận**: markdown — caveat prox-then-project ĐÃ được thay bằng joint-prox (nêu quá trình này như một điểm học được), survivorship bias, μ̂ nhiễu, robust finding, hướng mở rộng (long-only/simplex, Ledoit-Wolf, out-of-sample backtest).

Mỗi hình hiển thị bằng cách GỌI hàm viz tương ứng (tái tạo trong notebook) HOẶC `from IPython.display import Image; Image("figures/figN.png")`. Ưu tiên gọi hàm viz để notebook tự sinh hình (chứng minh tái lập được); nếu chậm thì dùng Image từ PNG có sẵn — chọn cách chạy nbconvert trong <~3 phút.

## README.md (mới, ở project root)
- Mô tả project 1 đoạn.
- Cấu trúc thư mục (src/, tests/, data/, figures/, notebook.ipynb).
- Cách chạy: tạo/kích hoạt venv, `pip install`, chạy `python -m src.data_loader` (nêu tải lâu ~ do vnstock, đã cache), chạy pytest, mở notebook.
- Dán output `pytest tests/ -v` (chạy thật, lấy số pass).
- Nêu yêu cầu môi trường (Python 3.10+, các package chính) + note về `.env` API key + note SSL proxy nếu máy sau corporate proxy (tham khảo .sdd/task-1b-report.md).

## Acceptance criteria (verify thật)
1. `.venv/bin/python -m nbconvert --to notebook --execute notebook.ipynb --output notebook.ipynb` (hoặc --stdout) chạy SẠCH, KHÔNG lỗi cell, KHÔNG cần mạng. Dán tóm tắt (số cell, thời gian, exit 0) vào report.
2. Notebook import từ `src/`, không nhúng lại toàn bộ logic solver/estimator.
3. Cả 6 hình hiển thị trong notebook.
4. `pytest tests/ -v` vẫn pass toàn bộ; output dán vào README + report.
5. README đầy đủ mục trên, hướng dẫn chạy chính xác.
6. Rà soát: mỗi module trong `src/` có docstring (đã có từ các phase trước — chỉ xác nhận, không cần viết lại).

## Report vào `.sdd/task-6-report.md`
Đầy đủ: file tạo (notebook.ipynb, README.md), kết quả nbconvert --execute thật, pytest thật, xác nhận 6 hình hiển thị. Trả về controller: status, danh sách file, 1 dòng tóm tắt (notebook chạy sạch chưa, pytest X/X), concerns.
