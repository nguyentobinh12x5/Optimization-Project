# Task 1b Report — Fix universe bias (MSN truncation)

**Status: DONE (not blocked).**

## Tóm tắt 1 dòng
Universe 75 → **97 mã**; đã có đủ VIC, VHM, VNM, TCB, VRE, VIB; cuối cùng chỉ loại 3 mã (DSE, VPL niêm yết muộn hợp lệ; GEE thiếu >10% phiên) trong tổng 100 mã VN100.

## Phần A — Patch `src/data_loader.py`

1. `QUOTE_SOURCE_FALLBACK` đổi từ `("VCI", "MSN", "KBS")` → `("VCI", "KBS")`. Comment giải thích chi tiết lý do (MSN cắt cụt ~365 phiên bất kể `start`) được ghi ngay tại chỗ khai báo hằng số.
2. Thêm `VCI_RETRY_BACKOFFS = (5.0, 15.0, 30.0)` và sửa `fetch_history_one_symbol()`: khi source đang thử là `"VCI"` và request lỗi/timeout, hàm retry chính VCI tối đa 3 lần (backoff 5s/15s/30s) TRƯỚC khi chuyển sang source kế tiếp trong `sources`. Docstring giải thích rõ lý do (timeout VCI là tạm thời; source khác có thể "thành công" nhưng cắt cụt âm thầm).
3. Không đổi tên file cache, không đổi chữ ký hàm public hiện có (chỉ thêm tham số optional `vci_retry_backoffs` với default).
4. Thêm hàm mới `repair_universe()` (không thay hàm nào đang có) — logic Phần B.

## Phần B — `repair_universe()` (chạy thật, không giả lập)

Danh sách 25 mã cần tải lại (đúng theo brief): `DSE, GEE, GVR, PNJ, POW, PVD, REE, SCS, SHB, SIP, SJS, SZC, TCB, VCG, VGC, VHC, VHM, VIB, VIC, VIX, VNM, VPL, VRE, VSC, VTP`.

Cấu hình: source=`("VCI",)` ONLY, retry 3 lần/backoff 5/15/30s, sleep 1.5s giữa các mã, cửa sổ `2022-07-25 → 2026-07-25` (giống 75 mã cũ).

### Sự cố môi trường gặp phải (không phải lỗi code)
Lần chạy đầu tiên, **toàn bộ** request HTTPS tới `trading.vietcap.com.vn` và `vnstocks.com` đều lỗi `SSL: CERTIFICATE_VERIFY_FAILED — self-signed certificate in certificate chain`. Chẩn đoán bằng `openssl s_client` cho thấy đây là chứng chỉ gốc **"Gateway CA - Cloudflare Managed G1"** — máy này đang chạy sau một Cloudflare Zero Trust Gateway (TLS-inspecting proxy) do chính sách quản lý thiết bị cài đặt, và root CA này ĐÃ được cài và tin cậy trong System Keychain của macOS, nhưng bundle CA riêng của `certifi` (dùng bởi `requests`/`vnstock`) không có root này nên từ chối handshake. Đây KHÔNG phải VCI timeout — retry logic mới thêm sẽ không tự sửa được lỗi này vì lỗi lặp lại y hệt ở mọi lần thử.

Đã dừng lần chạy đầu (không có mã nào merge vào cache — `repair_universe()` raise trước khi gọi `save_to_cache` nếu bị BLOCKED; đã verify `data/*.parquet` không đổi byte-for-byte trước khi retry). Xử lý: export root CA đó từ System Keychain (`security find-certificate`), nối vào bundle `certifi` thành 1 file PEM gộp, rồi set `REQUESTS_CA_BUNDLE`/`SSL_CERT_FILE` trỏ vào file gộp đó CHỈ cho tiến trình chạy `repair_universe()`. Đây KHÔNG phải tắt xác thực SSL (không dùng `verify=False`) — chỉ dạy Python tin thêm 1 root CA mà chính macOS đã tin cậy sẵn (chính sách MDM hợp pháp của tổ chức). Không sửa `src/data_loader.py` để làm việc này (không hardcode cert vào code) — đây là cấu hình môi trường máy, không phải logic nghiệp vụ. Ghi chú lại đây để lần chạy sau trên máy này (nếu vẫn gặp lỗi SSL này) biết cách xử lý tương tự.

Sau khi set 2 biến môi trường trên, chạy lại từ đầu: **VCI hoạt động bình thường**, kể cả 1 timeout thật (`VHC`, `Read timed out (30s)`) được retry logic mới xử lý thành công ngay ở lần thử tiếp theo — xác nhận patch Phần A hoạt động đúng.

### Kết quả tải 25 mã (log thật, không fabricate)
- **Fetch thất bại hoàn toàn (0 mã):** không có.
- **Coverage không đủ, bị loại trước khi merge (2 mã):**
  - `DSE`: `first_valid=2024-07-01 > cutoff=2022-08-24 (rows=516)` — niêm yết 2024-07, đúng như brief dự đoán (loại hợp lệ).
  - `VPL`: `first_valid=2025-05-13 > cutoff=2022-08-24 (rows=302)` — IPO 2025-05, đúng như brief dự đoán (loại hợp lệ).
- **Fetch OK, merge vào ma trận giá (23 mã):** `GEE, GVR, PNJ, POW, PVD, REE, SCS, SHB, SIP, SJS, SZC, TCB, VCG, VGC, VHC, VHM, VIB, VIC, VIX, VNM, VRE, VSC, VTP` (rows đa số 998, `GEE`=980, `SIP`=993, `VTP`=991).

### `clean_prices()` trên universe hợp nhất (75 cũ + 23 mã fetch OK)
- Chỉ **1 mã bị loại thêm** ở bước clean: `GEE: too_many_missing (10.1% > 10%)` — đúng như brief dự đoán ("GEE có thể vẫn bị loại").
- 22 mã còn lại được giữ trong universe cuối cùng.
- 75 mã cũ: **giữ nguyên 100%**, không mã nào bị mất (verify: `old_75 ⊆ set(prices.columns)` → True).

## Acceptance criteria — verify thật, số liệu thật

1. `{'VIC','VHM','VNM','TCB','VRE','VIB'} ⊆ set(returns.columns)` → **True** (verify bằng script Python đọc thẳng `data/returns.parquet`).
2. Số mã sạch cuối cùng: **97** (≥ 90, đạt; nằm trong khoảng kỳ vọng ~95 của brief, thậm chí tốt hơn vì chỉ mất đúng 3 mã: DSE/VPL/GEE).
3. `data/returns.parquet`: `returns.isna().any().any()` → **False** (không NaN). Kiểm tra nhất quán: recompute `prices.pct_change(fill_method=None).dropna(how="any")` từ `data/prices.parquet` và so với `data/returns.parquet` → **max abs diff = 0.0**, shape khớp tuyệt đối `(1068, 97)`. Index: `is_monotonic_increasing=True`, dtype `datetime64[ns]`.
4. Shape mới: `prices.parquet` = **(1090, 97)**, `returns.parquet` = **(1068, 97)** (trước đó: 75 mã, `prices` (1090,75)/`returns` (1069,75)). `data/vn100_symbols.csv` cập nhật đúng 97 dòng, khớp 100% với cột của `prices.parquet`.
5. `pytest tests/test_estimators.py -v` → **8 passed** (không đổi, vì test dùng dữ liệu giả lập độc lập với `data/`).
6. Log đầy đủ đã in ở trên: mã bị loại và lý do (`GEE`: too_many_missing 10.1%; `DSE`, `VPL`: coverage insufficient / niêm yết muộn thật).

## Universe cuối cùng (97 mã, sorted)
```
ACB, ANV, BAF, BCM, BID, BMP, BSI, BSR, BVH, BWE, CII, CMG, CTD, CTG, CTR, CTS,
DBC, DCM, DGW, DIG, DPM, DXG, DXS, EIB, EVF, FPT, FRT, FTS, GAS, GEX, GMD, GVR,
HAG, HCM, HDB, HDC, HDG, HHV, HPG, HSG, HT1, IMP, KBC, KDC, KDH, KOS, LPB, MBB,
MSB, MSN, MWG, NAB, NKG, NLG, NT2, NVL, OCB, PAN, PC1, PDR, PHR, PLX, PNJ, POW,
PVD, PVT, REE, SAB, SBT, SCS, SHB, SIP, SJS, SSB, SSI, STB, SZC, TCB, TCH, TPB,
VCB, VCG, VCI, VGC, VHC, VHM, VIB, VIC, VIX, VJC, VND, VNM, VPB, VPI, VRE, VSC, VTP
```

## Mã bị loại khỏi universe (3 mã, lý do hợp lệ)
- `DSE` — niêm yết muộn (first_valid 2024-07-01, ngoài dung sai cửa sổ 4 năm).
- `VPL` — niêm yết muộn / IPO 2025-05 (first_valid 2025-05-13).
- `GEE` — thiếu 10.1% phiên trong cửa sổ (> ngưỡng 10%).

## File đã thay đổi
- `src/data_loader.py`: patch `QUOTE_SOURCE_FALLBACK`, thêm `VCI_RETRY_BACKOFFS`, sửa `fetch_history_one_symbol()` (retry VCI), thêm hàm mới `repair_universe()`.
- `data/prices.parquet`, `data/returns.parquet`, `data/vn100_symbols.csv`: ghi đè với universe 97 mã mới (backup trước khi ghi đè: đã lưu ở scratchpad, không nằm trong repo).

## Concerns
1. **Môi trường máy này cần custom CA bundle cho HTTPS** (do Cloudflare Zero Trust Gateway ở tầng mạng công ty/tổ chức) — không phải vấn đề của code `data_loader.py`, nhưng nếu chạy lại `main()`/`repair_universe()` trên máy này trong tương lai mà gặp lại lỗi `self-signed certificate in certificate chain`, cần set `REQUESTS_CA_BUNDLE`/`SSL_CERT_FILE` trỏ tới bundle gộp (certifi + root CA của gateway, lấy từ System Keychain macOS) trước khi chạy. Không cần thay đổi gì trong `src/data_loader.py` vì đây thuần là cấu hình biến môi trường hệ thống, không phải logic ứng dụng.
2. `GEE` bị loại đúng theo dự đoán của brief (thiếu 10.1% phiên) — nếu muốn giữ `GEE` trong tương lai, cần nới `MAX_MISSING_FRAC` hoặc điều tra riêng vì sao thiếu nhiều phiên (không thuộc phạm vi task này).
3. `repair_universe()` hiện nhận `missing_symbols` optional — nếu gọi không truyền tham số, nó tự tính `set(load_vn100_symbols()) - set(prices.columns)` (cần mạng để gọi `Listing`); lần chạy thực tế ở đây đã truyền tường minh đúng 25 mã theo brief.
