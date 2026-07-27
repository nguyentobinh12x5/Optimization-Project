# Task Brief — Phase 7 (fix-round): Loại phiên cuối tuần giả + chạy lại toàn pipeline

## Bối cảnh lỗi (đã xác nhận, đã sửa code)
`data/prices.parquet` hiện tại chứa 71 "phiên Chủ nhật ma" — dữ liệu thô từ vnstock (nguồn VCI) lặp lại giá đóng cửa phiên liền trước (thường Thứ 6) và gắn nhãn ngày Chủ nhật. 68/1068 dòng trong `returns.parquet` (6.4%) là các phiên giả này, với return = 0.0 CHÍNH XÁC cho gần như mọi mã cùng lúc — làm méo mu_hat (kéo về 0) và Sigma (thổi phồng tương quan chéo giả).

**Code ĐÃ được sửa** ở `src/data_loader.py`, hàm `clean_prices()`: thêm dòng
```python
prices = prices.loc[prices.index.dayofweek < 5]
```
ngay sau khi cắt về cửa sổ [start,end], TRƯỚC khi tính `missing_frac`/`first_valid_index`. Đọc docstring hàm này (đã cập nhật) để hiểu đầy đủ lý do.

## Nhiệm vụ: tái tạo dữ liệu từ đầu + chạy lại toàn bộ pipeline phụ thuộc

### Bước 1 — Tái fetch dữ liệu THẬT (không tái dùng cache cũ, vì cache cũ nhiễm lỗi)
```bash
set -a && source .env && set +a
FORCE_REFRESH=1 .venv/bin/python -m src.data_loader
```
Lý do PHẢI re-fetch từ mạng (không chỉ lọc lại cache hiện có): việc loại phiên cuối tuần làm THAY ĐỔI mẫu số `n_sessions` dùng để tính `missing_frac`, có thể ảnh hưởng quyết định loại/giữ các mã biên (đặc biệt GEE, bị loại trước đây vì 10.1% thiếu — sát ngưỡng 10%). Chỉ re-fetch mới đảm bảo quyết định này được tính lại đúng trên toàn bộ ứng viên VN100 (không chỉ 97 mã đã giữ lại lần trước).

**Lưu ý môi trường (máy này, đã biết từ trước)**: mạng qua Cloudflare Zero Trust proxy (TLS-inspecting). Nếu gặp lỗi `SSL: self-signed certificate in certificate chain`, cần set `REQUESTS_CA_BUNDLE`/`SSL_CERT_FILE` trỏ tới certifi bundle đã nối thêm root CA từ macOS System Keychain (xem `.sdd/task-1b-report.md` để biết cách làm cụ thể đã dùng lần trước — KHÔNG dùng `verify=False`).

Việc tải có thể mất 10-50+ phút tùy độ ổn định VCI (rate-limit 1.2s/mã + retry backoff). Chạy NỀN, đợi xong, đừng bỏ dở giữa chừng.

### Bước 2 — Verify dữ liệu mới (bắt buộc, dán số thật vào report)
```python
import pandas as pd
prices = pd.read_parquet("data/prices.parquet")
returns = pd.read_parquet("data/returns.parquet")
weekend = prices.index[prices.index.dayofweek >= 5]
assert len(weekend) == 0, f"Vẫn còn {len(weekend)} phiên cuối tuần!"
```
Xác nhận: KHÔNG còn phiên cuối tuần nào. Ghi shape mới, universe mới (số mã, có đủ VIC/VHM/VNM/TCB/VRE/VIB không), so sánh với universe cũ (97 mã) — mã nào đổi trạng thái (nếu có, đặc biệt chú ý GEE).

### Bước 3 — So sánh mu_hat/Sigma TRƯỚC vs SAU fix (định lượng tác động thật)
Trước khi ghi đè, backup: `cp data/prices.parquet /tmp/prices_before_fix.parquet` (và returns tương tự) TRƯỚC khi chạy force_refresh — để có thể so sánh trước/sau. Sau khi có data mới:
```python
from src.estimators import estimate_all
mu_before, sigma_before, _ = estimate_all(pd.read_parquet("/tmp/returns_before_fix.parquet"))
mu_after, sigma_after, _ = estimate_all(pd.read_parquet("data/returns.parquet"))
# So trên các mã chung giữa 2 universe (dùng .reindex hoặc intersection columns)
# In: chênh lệch trung bình |mu| , chênh lệch trung bình phần tử ngoài-đường-chéo của Sigma (đo tương quan)
```
Ghi số cụ thể vào report: mu_hat thay đổi bao nhiêu %, tương quan trung bình (off-diagonal correlation) thay đổi bao nhiêu — đây là bằng chứng định lượng cho việc fix có ý nghĩa hay không.

### Bước 4 — Chạy lại toàn bộ downstream (KHÔNG cần sửa code các phase này, chỉ re-run)
```bash
.venv/bin/python -m pytest tests/ -v          # phải vẫn pass (test dùng data giả lập, không phụ thuộc file thật)
.venv/bin/python -m src.cvxpy_check           # bảng so sánh 6 bộ tham số trên data MỚI -- verify relgap vẫn <0.5%
.venv/bin/python -m src.viz                   # regenerate 6 hình trong figures/ trên data MỚI
.venv/bin/python -m nbconvert --to notebook --execute notebook.ipynb --output notebook.ipynb
```
Tất cả PHẢI chạy sạch. Nếu `cvxpy_check` cho relgap vượt 0.5% ở bộ tham số nào (không nên xảy ra vì solver/cvxpy code không đổi, chỉ data đổi) → điều tra, KHÔNG bỏ qua.

### Bước 5 — Cập nhật văn bản có số liệu cứng
- `notebook.ipynb`: các markdown cell có nhắc số liệu cũ (universe 97 mã, shape (1068,97), v.v.) cần khớp số liệu MỚI sau khi re-execute. Đặc biệt thêm 1 đoạn ngắn trong "Hạn chế & Kết luận" giải thích lỗi phiên cuối tuần đã phát hiện VÀ ĐÃ SỬA (khác với các hạn chế "chưa sửa" khác đã liệt kê) — nêu con số tác động thật từ Bước 3.
- `README.md`: nếu có số liệu (shape, số mã) bị hardcode, cập nhật khớp.
- KHÔNG cần sửa `.sdd/task-*-report.md` cũ (giữ nguyên làm lịch sử quyết định).

## Acceptance criteria
1. `prices.parquet` mới: 0 phiên cuối tuần (assert như Bước 2).
2. Universe mới vẫn chứa đủ blue-chip lớn (VIC/VHM/VNM/TCB/VRE/VIB), số mã ≥ 90 (kỳ vọng gần 97, có thể lệch 1-2 nếu GEE đổi trạng thái).
3. pytest 18/18 pass.
4. `src.cvxpy_check` max relgap < 0.5% trên data mới.
5. Notebook `nbconvert --execute` sạch, 0 lỗi.
6. Report có bảng so sánh ĐỊNH LƯỢNG trước/sau (mu_hat, correlation trung bình, số mã universe).

## Report vào `.sdd/task-7-report.md`
Đầy đủ: quá trình re-fetch (thời gian, số mã ok/fail), verify Bước 2, bảng so sánh Bước 3, kết quả Bước 4 (số thật), các file đã cập nhật ở Bước 5. Trả về controller: status, danh sách file thay đổi, 1 dòng tóm tắt (universe cũ→mới, tác động mu/Sigma đo được, mọi thứ downstream còn xanh không), concerns. KHÔNG fabricate — đây là việc sửa lỗi dữ liệu quan trọng, mọi số phải từ chạy thật.
