# Final Review Report — Sparse+Robust Portfolio Optimization (VN100)

Reviewer: final correctness + quality pass. Scope: `src/*.py`, `tests/*.py`,
`README.md`, `notebook.ipynb`. Focus: đúng-sai toán học của solver, khớp CVXPY,
chất lượng test, rò rỉ solver, dead code, doc drift.

**Kết luận tổng quát: KHÔNG có lỗi correctness nghiêm trọng.** Toán của solver
proximal-subgradient đúng, khớp CVXPY tới relgap ~1e-7 và Jaccard=1.0 trên 5/6
bộ tham số (bộ còn lại lam=0 lệch 1 toạ độ biên, đúng như notebook giải thích).
18/18 test pass. Các phát hiện bên dưới đều ở mức Minor (chủ yếu doc drift +
dead import). Xếp hạng: Critical / Important / Minor.

---

## Đã kiểm chứng ĐÚNG (không phải finding, ghi lại để rõ)

- **Subgradient robust** `κ·Σw/‖Σ^½w‖` (prox_solver.py:252): đúng. chain-rule
  qua u=Σ^½w cho tử số Σw (không phải Σ^½w) — docstring giải thích chuẩn xác.
- **Smooth subgrad** `-μ + 2γΣw` (prox_solver.py:347): đúng đạo hàm của γwᵀΣw.
- **Joint prox** `_prox_l1_simplex_eq` (prox_solver.py:255-333): đúng. KKT của
  min ½‖w-z‖²+t‖w‖₁ s.t. 1ᵀw=1 → wᵢ=soft(zᵢ-ν,t), tìm ν bằng bisection trên
  g(ν)=Σsoft(zᵢ+ν,t)=1 (g không giảm, bracket doubling đúng hướng). Edge case
  λ=0 → t=0 → chiếu Euclid; test xác nhận. ‖Σ^½w‖≈0 → subgrad 0, không NaN.
- **CVXPY formulation** (cvxpy_check.py:114-120): KHỚP CHÍNH XÁC bài toán tay.
  `-μ@w + κ·norm(Σ^½w,2) + γ·quad_form(w,psd_wrap(Σ)) + λ·norm1(w)`, ràng buộc
  `sum(w)==1`. obj so sánh tính lại bằng `portfolio_objective` (cùng hàm), không
  dùng `prob.value`. psd_wrap hợp lý cho Σ chỉ PSD tới sai số số học.
- **Không rò rỉ solver**: `prox_solver.py`, `estimators.py`, `data_loader.py`
  KHÔNG import cvxpy/scipy/sklearn (verified). cvxpy chỉ xuất hiện ở
  `cvxpy_check.py` và (gián tiếp) `viz.py`. Phần nộp chính sạch.
- **Test không rỗng**: closed-form N=2 KKT (test_prox_solver.py:41) dựng đúng hệ
  `[[2γΣ,1],[1ᵀ,0]][w;η]=[μ;1]` và so nghiệm < 1e-4 — kiểm tra thực chất, không
  tautology. Các test sparsity/best-iterate/sum-to-one đều assert đúng thứ cần.

---

## CRITICAL

Không có.

---

## IMPORTANT

**I1 — README mâu thuẫn với repo về `requirements.txt`** (README.md:46-48, 58-59)
README khẳng định *"Không có requirements.txt sẵn trong repo — cài trực tiếp qua
pip install"*, nhưng `requirements.txt` TỒN TẠI và đã pin đầy đủ (numpy, pandas,
cvxpy, vnstock, vnai, pyarrow, ...). Lệnh `pip install` liệt kê tay trong README
lại THIẾU `pyarrow` (bắt buộc để đọc parquet cache) và `vnai`. Người chấm làm
theo README sẽ bỏ qua file pin sẵn và có thể thiếu dependency → ảnh hưởng khả
năng tái lập. (Chỉ là doc, không phải lỗi code — nên vẫn dưới mức Critical.)

---

## MINOR

**M1 — Doc drift: fallback source "VCI → MSN → KBS"** (data_loader.py:41, 226)
Hai docstring nói thứ tự fallback là `VCI -> MSN -> KBS`, nhưng code thực tế
`QUOTE_SOURCE_FALLBACK = ("VCI", "KBS")` — MSN đã bị loại bỏ có chủ đích (giải
thích dài ở dòng 87-99 rằng MSN cắt cụt lịch sử). Docstring tự mâu thuẫn với
chính module. Vô hại về hành vi (chỉ ảnh hưởng path cần mạng), nhưng gây nhầm.

**M2 — Dead imports trong test** (test_prox_solver.py:19,17)
`SolveResult` và `pytest` được import nhưng không dùng ở đâu trong file. Xoá đi
cho sạch.

**M3 — Nhãn sai đơn vị relgap trong docstring fig6** (viz.py:570)
Ghi *"relgap=2.2e-7%"*. relgap là PHÂN SỐ (=2.2e-7), tức 0.000022% — dấu `%`
thừa làm sai đơn vị 100×. (Notebook/`_main` in đúng bằng format `:.6%` nên số
0.000692% ở nơi khác là chuẩn; chỉ docstring này nhầm.)

**M4 — `best_obj` docstring không khớp edge case khởi tạo** (prox_solver.py:361,
429-430) SolveResult.best_obj mô tả *"= min(obj_history)"*, nhưng best_obj khởi
tạo từ f(w0) (w0 uniform 1/N) — điểm này KHÔNG nằm trong obj_history. Nếu w0 tình
cờ tốt hơn mọi iterate (alpha0 quá lớn, iterate đầu overshoot), `result.w`=w0 và
`best_obj < min(obj_history)`, phá vỡ mô tả. Không xảy ra với data thật hiện tại
(iterate luôn cải thiện qua w0), nhưng là bất nhất tiềm ẩn giữa doc và code.

**M5 — Ledoit-Wolf dùng S ddof=1 trong công thức tiệm cận** (estimators.py:178,
149-150) `S = returns.cov()` (ddof=1, chia T-1) nhưng công thức LW pi_hat/delta
giả định S=(1/T)ΣxₜxₜT (ddof=0). Sai lệch nhỏ trong cường độ shrinkage. Không
dùng làm mặc định (shrinkage=None), nên chỉ ảnh hưởng nếu bật "lw". Đã ghi rõ
đơn giản hoá rho_hat=0 nhưng KHÔNG ghi điểm ddof này. Thuần lý thuyết, biên độ nhỏ.

**M6 — `seaborn` là dependency thừa** (requirements.txt, README.md:43-44)
seaborn được liệt kê là "package chính" nhưng KHÔNG được import ở bất kỳ file
`src/` hay `tests/` nào (viz.py chỉ dùng matplotlib thuần). Dependency dư.

**M7 — Ngữ nghĩa cờ `converged` có thể gây hiểu nhầm** (prox_solver.py:460-467)
`converged=True` chỉ nghĩa là best_obj không cải thiện quá `tol` trong `patience`
vòng liên tiếp — KHÔNG đảm bảo đã tới nghiệm tối ưu (subgradient có thể mắc kẹt
best sớm khi dao động). Docstring mô tả đúng cơ chế nhưng "converged" dễ bị đọc
là "đã hội tụ về optimum". Chỉ là vấn đề đặt tên/kỳ vọng, không phải lỗi.

---

## Ghi chú phụ (không tính finding)

- RuntimeWarning "divide by zero / overflow / invalid in matmul" khi chạy test:
  đúng là artefact Apple Accelerate BLAS trên ma trận gần suy biến (phát ra từ
  `eigvecs @ diag(sqrt) @ eigvecs.T`), vô hại, không sai kết quả — README giải
  thích hợp lý.
- Các claim số liệu trong notebook (max relgap 0.000692%, Jaccard 5/6 =1.0, bộ
  lam=0 active 97 vs 96, ~14-15s, 22 cell) KHỚP với output đã lưu trong notebook
  và với `DEFAULT_PARAM_GRID`. Không phát hiện claim bịa.
- N=2 closed-form, sparsity-tăng-theo-λ, best-not-last, prox-exact-zero: test
  thực chất, không rỗng.
