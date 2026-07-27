# Task Brief — Phase 1b: FIX universe bias (MSN truncation)

## Bối cảnh lỗi (đã xác nhận)
Lần chạy Phase 1 trước: VCI timeout nhiều → 23 mã fallback sang MSN. MSN CHỈ trả ~365 phiên gần nhất bất kể `start`, nên `clean_prices` gán nhãn `listed_late` SAI cho các blue-chip lớn và loại bỏ chúng. Universe 75 mã hiện thiếu VIC, VHM, VNM, TCB, VRE, VIB, SHB, POW, PVD, PNJ, GVR, REE, SCS, SIP, SJS, SZC, VCG, VGC, VHC, VIX, VSC, VTP (và vài mã khác). Loại HỢP LỆ thật sự: DSE (niêm yết 2024-07), VPL (IPO 2025-05), GEE (thiếu >10% phiên).

## Môi trường
- `.venv/bin/python` (Python 3.14). Có numpy/pandas/pyarrow/matplotlib/seaborn/pytest/python-dotenv. KHÔNG scipy/sklearn.
- API key ở `.env` (`VNSTOCK_API_KEY`) — nạp bằng dotenv, KHÔNG in, KHÔNG hardcode.
- Project: `/Users/nguyentobinh12gmail.com/Documents/Optimization Project`. Module đã có: `src/data_loader.py`.
- Đọc `src/data_loader.py` hiện tại TRƯỚC để hiểu cấu trúc (`fetch_history_one_symbol`, `fetch_all_prices`, `clean_prices`, `compute_returns`, cache I/O). GIỮ nguyên các hàm/chữ ký public đang có (Phase 2 không phụ thuộc, nhưng giữ nhất quán).

## Phần A — Patch `src/data_loader.py` (để lần sau không tái phát)
1. **Bỏ MSN khỏi fallback cho lịch sử dài**: đổi `QUOTE_SOURCE_FALLBACK` từ `("VCI","MSN","KBS")` thành `("VCI","KBS")`. Lý do: MSN cắt cụt ~365 phiên nên KHÔNG dùng được cho cửa sổ 4 năm. Ghi comment giải thích rõ.
2. **Validate coverage + retry chính source**: trong `fetch_history_one_symbol` (hoặc tầng gọi), sau khi một source trả dữ liệu, kiểm tra độ phủ: nếu `series.index.min()` trễ hơn `start` một khoảng lớn (vd > 60 ngày) **và** đây KHÔNG phải mã niêm yết muộn thật (không thể biết chắc, nên dùng heuristic: coi là "nghi cắt cụt" khi số phiên trả về < 50% số phiên kỳ vọng ~ (số phiên mã tham chiếu). Đơn giản hơn: thêm tham số `min_expected_rows` và nếu `len(series) < min_expected_rows` thì coi source đó CHƯA đạt, thử source kế / retry.
   - Cụ thể, cách làm được chấp nhận (đơn giản, đủ tốt): thêm retry cho CHÍNH VCI khi VCI ném lỗi/timeout: thử VCI tối đa 3 lần, mỗi lần backoff tăng dần (vd 5s, 15s, 30s), TRƯỚC khi chuyển sang KBS. Vì nguyên nhân gốc là VCI timeout chứ VCI không thiếu dữ liệu.
   - Ghi docstring: MSN bị loại vì truncation; VCI được retry vì timeout là tạm thời.
3. Không phá vỡ cache logic, không đổi tên file output.

## Phần B — Targeted re-fetch (KHÔNG tải lại cả 100 mã)
Viết một hàm/script (có thể là `repair_universe()` trong `data_loader.py`, hoặc script riêng `src/repair_universe.py`) làm:
1. Đọc `data/prices.parquet` hiện có (75 mã full-history từ VCI — GIỮ NGUYÊN, không tải lại).
2. Danh sách mã cần tải lại = 25 mã đã bị loại lần trước:
   `DSE, GEE, GVR, PNJ, POW, PVD, REE, SCS, SHB, SIP, SJS, SZC, TCB, VCG, VGC, VHC, VHM, VIB, VIC, VIX, VNM, VPL, VRE, VSC, VTP`
   (Lấy động từ so sánh `set(load_vn100_symbols())` trừ `set(prices.columns)` cũng được — nhưng danh sách trên là chuẩn.)
3. Tải TỪ VCI ONLY (source="VCI") cho 25 mã này, với retry kiên nhẫn (3 lần, backoff 5/15/30s), sleep 1.5s giữa các mã. Cửa sổ giống cũ: start = today - 4 năm, end = today (dùng `_default_date_window()`).
4. Với mỗi mã tải được có ĐỦ lịch sử (kiểm tra `series.index.min() <= start + ~30 ngày`), ghép cột đó vào ma trận giá. Mã nào VCI vẫn không trả đủ history → bỏ qua, log rõ.
5. Ghép: `combined = pd.concat([prices_cũ_75mã, các_series_mới], axis=1)` theo outer join ngày, sort index.
6. Chạy lại `clean_prices(combined, start, end)` rồi `compute_returns(...)` trên TOÀN BỘ universe hợp nhất. Bây giờ DSE/VPL vẫn sẽ bị loại (niêm yết muộn thật), GEE có thể vẫn bị loại (thiếu >10%) — chấp nhận. Các blue-chip VIC/VHM/VNM/TCB... giờ PHẢI được giữ.
7. Ghi đè cache: `save_to_cache(...)` với universe mới.

## Acceptance criteria (verify thật, dán vào report)
1. Universe sạch cuối cùng chứa các mã lớn: kiểm tra `{'VIC','VHM','VNM','TCB','VRE','VIB'} ⊆ set(returns.columns)` → PHẢI True. In ra danh sách mã được thêm so với 75 cũ.
2. Số mã sạch mới ≥ 90 (kỳ vọng ~95: 100 trừ DSE/VPL/GEE và có thể 1-2 mã VCI lỗi).
3. `data/returns.parquet` mới: KHÔNG NaN, `returns == pct_change(ffill(prices))` nhất quán (diff ≈ 0), index datetime tăng dần.
4. `data/prices.parquet`, `returns.parquet`, `vn100_symbols.csv` được cập nhật; shape mới (T, N) in ra.
5. `pytest tests/test_estimators.py -v` VẪN pass (không đụng estimators, chỉ chắc chắn không vỡ gì).
6. In log rõ mã nào cuối cùng vẫn bị loại + lý do.

## Lưu ý thời gian
Chỉ ~25 mã × (tải + có thể retry) → dự kiến vài phút tới ~15 phút tuỳ VCI. Nếu VCI timeout dày, retry sẽ kéo dài — chấp nhận, miễn KHÔNG fabricate. Nếu > 10 mã trong số 25 vẫn không tải được từ VCI sau retry → dừng, báo BLOCKED kèm log, ĐỪNG nhận MSN cắt cụt.

## Report vào `.sdd/task-1b-report.md`
Đầy đủ: patch đã làm, danh sách mã re-fetch thành công/thất bại, universe cuối (số mã, có blue-chip chưa), shape mới, output verify thật. Trả về controller: status, 1 dòng tóm tắt (universe cũ 75 → mới N, đã có VIC/VHM/VNM/TCB chưa), concerns.
