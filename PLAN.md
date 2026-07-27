# PLAN — Sparse + Robust Portfolio Optimization (VN100)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (khuyến nghị) hoặc superpowers:executing-plans để triển khai từng phase. Checklist dùng cú pháp `- [ ]` để theo dõi tiến độ.

**Goal:** Giải bài toán tối ưu danh mục sparse + robust trên rổ VN100 bằng proximal-subgradient method tự viết, verify chéo bằng CVXPY, kèm notebook trình bày dữ liệu và kết quả.

**Bài toán:**

```
min_w   -μ̂ᵀw + κ·‖Σ^(1/2) w‖₂ + γ·wᵀΣw + λ·‖w‖₁
s.t.    Σᵢ wᵢ = 1
```

- `-μ̂ᵀw`: kỳ vọng lợi nhuận (tuyến tính, trơn)
- `κ·‖Σ^(1/2)w‖₂`: robust term (không trơn tại w=0 → subgradient)
- `γ·wᵀΣw`: risk term (trơn, gradient = 2γΣw)
- `λ·‖w‖₁`: sparsity term (không trơn → proximal / soft-thresholding)
- Ràng buộc `1ᵀw = 1`: chiếu (projection) lên hyperplane sau mỗi bước

**Tech stack:** Python 3.14 (`.venv` hiện có), `vnstock>=4.0.5` (free tier, API namespaced v4: `Listing`, `Quote`), `numpy`, `pandas`, `cvxpy` (chỉ để verify), `matplotlib`, `pytest`.

## Global Constraints

- Phần nộp chính KHÔNG dùng solver có sẵn — proximal-subgradient viết tay bằng numpy thuần.
- CVXPY chỉ xuất hiện trong module verify, tách riêng khỏi solver chính.
- Nguồn dữ liệu: `vnstock` v4. `Quote` chỉ nhận source hợp lệ: `vci, msn, kbs, dnse, binance, fmp, fmarket` (TCBS/SSI KHÔNG hợp lệ). Ưu tiên `VCI`, fallback `MSN` rồi `KBS`. `Listing` chỉ nhận `KBS/VCI/MSN` → dùng VCI cho `symbols_by_group("VN100")`.
- Dữ liệu tải về phải cache ra file (parquet/csv) — notebook chạy lại không gọi API.
- API key đọc từ `.env` (`VNSTOCK_API_KEY`), không hardcode, không in ra log.
- Code có comment giải thích (người đọc là dân tối ưu, không chuyên lập trình tài chính).
- Short-selling: cho phép w < 0 (bài toán gốc không có ràng buộc w ≥ 0) — nếu muốn long-only phải confirm lại ở Gate 1 vì sẽ đổi phép chiếu.

## Cấu trúc file dự kiến (tổng ~10 file)

```
src/
  data_loader.py      # Phase 1: tải & cache dữ liệu VN100
  estimators.py       # Phase 2: μ̂, Σ, Σ^(1/2)
  prox_solver.py      # Phase 3: proximal-subgradient solver
  cvxpy_check.py      # Phase 4: verify bằng CVXPY
  viz.py              # Phase 5: các hàm vẽ
tests/
  test_estimators.py
  test_prox_solver.py
data/                 # cache parquet (git-ignored)
notebook.ipynb        # Phase 6: deliverable chính
PLAN.md
```

---

## Phase 0 — Chốt thông số bài toán 🔴 **[CẦN BẠN REVIEW/QUYẾT ĐỊNH TRƯỚC KHI ĐI TIẾP — Gate 1]**

**Mục tiêu:** Chốt các quyết định ảnh hưởng toàn bộ pipeline, tránh làm lại.

**Input:** Đề bài môn học, thảo luận với bạn.

**Output:** Mục "Quyết định đã chốt" điền vào cuối PLAN.md này.


**Cách verify:** Bạn xác nhận từng mục dưới đây.

- [x] **Khoảng thời gian dữ liệu**: ✅ ĐÃ CHỐT — 4 năm gần nhất (≈1000 phiên), daily. T/N ≈ 10 với N≈100 → Σ mẫu well-conditioned hơn đề xuất 2 năm ban đầu.
- [x] **Tần suất return**: ✅ ĐÃ CHỐT — simple daily return (bắt buộc về mặt toán vì μ̂ᵀw/wᵀΣw giả định return cộng được theo trọng số; log-return chỉ cộng được theo thời gian). Annualize (×252, ×√252) khi báo cáo.
- [x] **Universe**: ✅ ĐÃ CHỐT — snapshot VN100 hiện tại; mã niêm yết sau 2022 bị loại khi làm sạch (universe sạch dự kiến ~70-85 mã); ghi rõ survivorship bias là limitation trong notebook.
- [x] **Ước lượng Σ**: ✅ ĐÃ CHỐT — sample covariance mặc định + Ledoit-Wolf shrinkage là tham số tùy chọn (bật khi cần Σ well-conditioned hơn / hội tụ nhanh hơn).
- [x] **Cho phép bán khống?** ✅ ĐÃ CHỐT — cho phép w < 0 (đúng công thức đề bài, phép chiếu hyperplane); ghi chú trong notebook rằng thị trường VN thực tế chưa cho short.
- [x] **Dải tham số thí nghiệm**: ✅ ĐÃ CHỐT — κ ∈ {0, 0.5, 1, 2}, γ ∈ {1, 5, 10}, λ sweep log-scale 10⁻⁴ → 10⁻¹.

---

## Phase 1 — Data layer (vnstock)

**Mục tiêu:** Tải danh sách VN100 + lịch sử giá, làm sạch, cache local.

**Input:** Quyết định Phase 0; `.env` có `VNSTOCK_API_KEY`; `.venv` đã cài `vnstock==4.0.5`.

**Output:** `src/data_loader.py`; `data/vn100_symbols.csv`; `data/prices.parquet` (index=date, columns=symbol, giá đóng cửa điều chỉnh); `data/returns.parquet`.

**Cách verify:** Chạy `python -m src.data_loader` in ra: số mã tải được, khoảng ngày, % missing; chạy lại lần 2 phải đọc từ cache (không gọi API, nhanh < 1s).

**Số file:** 1 module + 2-3 file cache.

- [ ] Lấy danh sách VN100 qua `Listing` (API v4 namespaced); nếu vnstock không có group VN100 trực tiếp, fallback: lấy VNAllShare/HOSE rồi lọc top 100 vốn hóa — ghi chú rõ cách làm trong docstring.
- [ ] Tải lịch sử giá daily từng mã qua `Quote.history` với retry theo thứ tự nguồn VCI → MSN → KBS; rate-limit an toàn dưới 60 req/min (tier free).
- [ ] Làm sạch: loại mã có > 10% phiên thiếu hoặc niêm yết muộn hơn ngày bắt đầu; forward-fill lỗ hổng ≤ 3 phiên; log rõ mã nào bị loại và vì sao.
- [ ] Tính ma trận return theo quyết định Phase 0, lưu cache.
- [ ] Acceptance: ≥ 70 mã sạch còn lại (cửa sổ 4 năm loại mã niêm yết sau 2022); ma trận return không có NaN; shape (T, N) in ra khớp kỳ vọng.

---

## Phase 2 — Ước lượng μ̂, Σ, Σ^(1/2)

**Mục tiêu:** Sinh các đại lượng thống kê đầu vào cho solver, đảm bảo Σ đối xứng PSD.

**Input:** `data/returns.parquet`.

**Output:** `src/estimators.py` (hàm trả về μ̂ vector (N,), Σ matrix (N,N), Σ^(1/2) matrix (N,N)); `tests/test_estimators.py`.

**Cách verify:** `pytest tests/test_estimators.py -v` pass toàn bộ.

**Số file:** 1 module + 1 test file.

- [ ] Hàm ước lượng μ̂ (mean return) và Σ (sample covariance, tùy chọn shrinkage theo Phase 0).
- [ ] Tính Σ^(1/2) bằng eigendecomposition (clip eigenvalue âm do sai số số học về 0) — KHÔNG dùng Cholesky vì Σ có thể chỉ PSD, không PD.
- [ ] Test: Σ đối xứng (‖Σ−Σᵀ‖ < 1e-10); mọi eigenvalue ≥ −1e-8; ‖Σ^(1/2)·Σ^(1/2) − Σ‖_F / ‖Σ‖_F < 1e-6.
- [ ] Test trên dữ liệu giả lập (N=5, T=100, seed cố định) để không phụ thuộc dữ liệu thật.

---

## Phase 3 — Proximal-subgradient solver (viết tay) 🔴 **[CẦN BẠN REVIEW TRƯỚC KHI ĐI TIẾP — Gate 2]**

**Mục tiêu:** Solver chính của bài — đúng về mặt toán, hội tụ, có log lịch sử objective.

**Input:** μ̂, Σ, Σ^(1/2) từ Phase 2; tham số κ, γ, λ.

**Output:** `src/prox_solver.py` (hàm solve trả về: w*, lịch sử objective mỗi iteration, số iteration, cờ hội tụ); `tests/test_prox_solver.py`.

**Cách verify:** `pytest tests/test_prox_solver.py -v` pass; đường objective giảm (không tăng quá tolerance do subgradient).

**Số file:** 1 module + 1 test file.

**Thiết kế thuật toán (phần cần bạn review kỹ ở Gate 2 trước khi sang Phase 4):**

- [ ] Tách hàm mục tiêu: phần gradient/subgradient `g(w) = -μ̂ + 2γΣw + κ·∂‖Σ^(1/2)w‖₂` (subgradient của term robust: `Σw/‖Σ^(1/2)w‖₂` khi ≠ 0, chọn 0 khi = 0); phần prox: `λ‖w‖₁` → soft-thresholding.
- [ ] Xử lý ràng buộc: sau bước prox, chiếu lên hyperplane `{1ᵀw=1}`: `w ← w + (1 − 1ᵀw)/N · 1`. **Lưu ý cần review**: prox rồi chiếu ≠ prox của bài toán ràng buộc — đây là xấp xỉ kiểu projected proximal-subgradient; chấp nhận được cho môn học nhưng cần ghi rõ trong báo cáo, và CVXPY ở Phase 4 sẽ cho biết sai lệch thực tế bao nhiêu.
- [ ] Step size: diminishing `α_k = α₀/√(k+1)` (chuẩn cho subgradient method — hội tụ nhưng chậm); ghi lại lựa chọn α₀ sau khi thử.
- [ ] Điều kiện dừng: max_iter (mặc định 5000) HOẶC thay đổi tương đối của best objective < 1e-8 trong 100 iteration liên tiếp; luôn trả về w của best objective (không phải iterate cuối — subgradient không monotone).
- [ ] Test trên bài toán nhỏ nghiệm biết trước: N=2, λ=0, κ=0 → nghiệm mean-variance đóng có công thức; so sánh sai số < 1e-4.
- [ ] Test tính chất: `sum(w)=1` đúng đến 1e-12; λ lớn → số phần tử |wᵢ| > 1e-6 giảm dần (sparsity tăng); objective best-so-far giảm đơn điệu.

---

## Phase 4 — Verify chéo bằng CVXPY

**Mục tiêu:** Khẳng định solver tay cho nghiệm đúng (trong sai số chấp nhận được).

**Input:** Cùng μ̂, Σ, κ, γ, λ; `src/prox_solver.py`.

**Output:** `src/cvxpy_check.py` (giải cùng bài toán bằng CVXPY — bài toán convex, dùng `cp.norm(Σ^(1/2)@w, 2)`, `cp.quad_form`, `cp.norm1`); script so sánh in bảng chênh lệch.

**Cách verify:** Chạy so sánh trên ≥ 5 bộ (κ, γ, λ) khác nhau và dữ liệu thật: chênh lệch objective tương đối < 0.5%; ‖w_prox − w_cvxpy‖∞ nhỏ (báo cáo con số thực tế, kỳ vọng < 1e-2 — subgradient hội tụ chậm nên không đòi 1e-6).

**Số file:** 1 module.

- [ ] Cài `cvxpy` vào `.venv` (chỉ phục vụ verify).
- [ ] Viết formulation CVXPY khớp từng term với bài toán gốc.
- [ ] Bảng so sánh: objective, số tài sản active, ‖Δw‖∞ cho từng bộ tham số.
- [ ] Nếu lệch quá ngưỡng → quay lại Phase 3 debug (dùng superpowers:systematic-debugging), KHÔNG chỉnh ngưỡng cho qua.

---

## Phase 5 — Thí nghiệm & Visualization 🟡 **[REVIEW NHẸ — Gate 3: chọn biểu đồ nào vào notebook]**

**Mục tiêu:** Sinh kết quả và hình vẽ trả lời câu hỏi: λ ảnh hưởng sparsity thế nào, κ ảnh hưởng danh mục thế nào.

**Input:** Solver đã verify; dải tham số chốt ở Phase 0.

**Output:** `src/viz.py`; các figure PNG lưu `figures/`.

**Cách verify:** Mỗi hàm vẽ chạy không lỗi từ dữ liệu cache; hình mở lên đọc được (trục có nhãn, đơn vị, chú thích tiếng Việt hoặc Anh nhất quán).

**Số file:** 1 module + ~6 hình.

- [ ] Hình 1 — Dữ liệu: giá chuẩn hóa/heatmap tương quan của universe.
- [ ] Hình 2 — Convergence: objective theo iteration (log-scale), vài bộ tham số.
- [ ] Hình 3 — Sparsity path: số tài sản active theo λ (log-scale trục x).
- [ ] Hình 4 — Trọng số danh mục: bar chart w* tại 2-3 bộ tham số tiêu biểu.
- [ ] Hình 5 — Ảnh hưởng robust term: so sánh danh mục κ=0 vs κ>0 (độ tập trung, turnover).
- [ ] Hình 6 — So sánh prox vs CVXPY (scatter w_prox vs w_cvxpy).
- [ ] Trước khi vẽ: đọc skill `dataviz` để thống nhất palette/style.
- [ ] Gate 3: đưa bạn xem cả 6 hình, bạn chọn hình nào vào notebook chính.

---

## Phase 6 — Notebook deliverable & hoàn thiện

**Mục tiêu:** Gộp toàn bộ thành `notebook.ipynb` chạy end-to-end từ cache, kể chuyện mạch lạc: bài toán → dữ liệu → thuật toán → kết quả → kết luận.

**Input:** Toàn bộ module `src/`, figures, kết quả Gate 3.

**Output:** `notebook.ipynb` (import từ `src/`, không copy-paste code dài vào cell); README ngắn hướng dẫn chạy.

**Cách verify:** `jupyter nbconvert --execute notebook.ipynb` chạy sạch từ đầu đến cuối trên máy bạn, không cần mạng (dùng cache); mọi cell output hiển thị đúng.

**Số file:** 1 notebook + 1 README.

- [ ] Cấu trúc notebook: (1) phát biểu bài toán + ý nghĩa từng term, (2) mô tả dữ liệu + Hình 1, (3) trình bày thuật toán proximal-subgradient (công thức update, lý do chọn step size), (4) kết quả + các hình đã chọn, (5) verify CVXPY, (6) hạn chế & kết luận.
- [ ] Rà soát comment toàn bộ `src/` — mỗi hàm có docstring giải thích toán tương ứng.
- [ ] Chạy `pytest` toàn bộ lần cuối, đính kèm output vào README.
- [ ] (Tùy chọn) `git init` + commit để nộp bài có lịch sử — hỏi bạn trước khi làm.

---

## Quyết định đã chốt (điền sau Gate 1)

| # | Câu hỏi | Quyết định | Ngày |
|---|---------|-----------|------|
| 1 | Khoảng thời gian & tần suất | 4 năm gần nhất, daily (≈1000 phiên) | 2026-07-25 |
| 2 | Loại return | Simple daily return, annualize khi báo cáo | 2026-07-25 |
| 3 | Universe snapshot | Snapshot VN100 hiện tại, chấp nhận survivorship bias (ghi limitation) | 2026-07-25 |
| 4 | Shrinkage Σ | Sample covariance + tùy chọn Ledoit-Wolf shrinkage | 2026-07-25 |
| 5 | Bán khống | Cho phép w < 0 (chiếu hyperplane) | 2026-07-25 |
| 6 | Dải tham số | κ ∈ {0, 0.5, 1, 2}, γ ∈ {1, 5, 10}, λ log-scale 10⁻⁴ → 10⁻¹ | 2026-07-25 |
