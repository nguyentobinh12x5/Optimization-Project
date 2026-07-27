# Task Brief — Phase 1: Data Layer (vnstock)

Đây là YÊU CẦU của bạn. Dùng đúng các giá trị verbatim dưới đây. Đọc file này trước tiên.

## Mục tiêu
Viết `src/data_loader.py` để tải danh sách VN100 + lịch sử giá 4 năm, làm sạch, và cache ra file local. Notebook/lần chạy sau phải đọc từ cache, KHÔNG gọi lại API.

## Môi trường (đã verify sẵn — dùng đúng)
- Python: `.venv/bin/python` (Python 3.14.5). LUÔN dùng interpreter này, KHÔNG dùng python hệ thống.
- `vnstock` đã cài (v4). API key nằm ở `.env` biến `VNSTOCK_API_KEY` — nạp bằng `set -a && source .env && set +a` trước khi chạy, HOẶC dùng `python-dotenv` trong code. KHÔNG in key ra log, KHÔNG hardcode.
- Thư mục làm việc: `/Users/nguyentobinh12gmail.com/Documents/Optimization Project`.

## API vnstock THẬT (đã kiểm chứng — dùng chính xác, đừng đoán)
- Lấy VN100: `from vnstock import Listing; syms = Listing(source="VCI").symbols_by_group("VN100")` → trả về ~100 mã (list/Series các ticker như 'ACB','ANV','BAF',...). `Listing` CHỈ nhận source `KBS/VCI/MSN`.
- Lấy giá: `from vnstock import Quote; df = Quote(source="VCI", symbol="FPT").history(start="YYYY-MM-DD", end="YYYY-MM-DD", interval="1D")`.
  - Cột trả về: `['time','open','high','low','close','volume']`. `time` là ngày (dùng làm index), `close` là giá đóng cửa.
  - `Quote` CHỈ nhận source hợp lệ: `vci, msn, kbs, dnse, binance, fmp, fmarket`. **TCBS và SSI KHÔNG hợp lệ** — sẽ ném ValueError. Đừng dùng chúng.
- vnstock in banner quảng cáo ra stdout khi khởi tạo — bình thường, bỏ qua (có thể redirect nếu muốn log sạch).

## Quyết định thiết kế đã chốt (Gate 1)
- Khoảng thời gian: **4 năm gần nhất** tính từ hôm nay (2026-07-25) → start ≈ 2022-07-25, end = 2026-07-25. Daily, interval="1D".
- Return: **simple return** = `close.pct_change()` (KHÔNG dùng log-return). Tính theo giá `close`.
- Universe: snapshot VN100 hiện tại (chấp nhận survivorship bias).
- `close` của VCI được coi là giá đã điều chỉnh — ghi rõ giả định này trong docstring (nếu không chắc, note lại, đừng tự chế biến thêm).

## Rate limiting (QUAN TRỌNG — người dùng yêu cầu rõ)
Free tier vnstock giới hạn ~60 request/phút. BẮT BUỘC:
- `time.sleep(...)` giữa MỖI request tải giá từng mã — mặc định **1.2 giây** (tham số cấu hình được, ví dụ `sleep_sec=1.2`).
- Retry theo chuỗi fallback source **VCI → MSN → KBS** khi một mã lỗi (ConnectionError/RetryError/Exception mạng): thử source kế tiếp. Có backoff: nếu cả 3 source đều lỗi lần 1, sleep dài hơn (vd 5s) rồi thử lại tối đa 1 lần nữa.
- Log tiến độ dạng `[i/N] SYMBOL: ok (source=VCI, rows=...)` hoặc `... FAILED after all sources` để người dùng theo dõi khi chạy lâu.

## Làm sạch dữ liệu
- Ghép giá `close` mọi mã thành 1 DataFrame giá: index = ngày (giao của lịch giao dịch), columns = mã.
- Loại mã có > 10% phiên thiếu trong cửa sổ, HOẶC niêm yết muộn hơn ngày bắt đầu (không có dữ liệu ở đầu cửa sổ). Log rõ mã nào bị loại + lý do.
- Forward-fill lỗ hổng ≤ 3 phiên liên tiếp; lỗ hổng dài hơn → coi là thiếu (không fill bừa).
- Sau làm sạch, ma trận return = `prices.pct_change().dropna(how="any")` KHÔNG được còn NaN.

## Output (bắt buộc tạo)
- `src/data_loader.py` — module với các hàm rõ ràng và một hàm `main()`/`if __name__=="__main__"` để chạy `python -m src.data_loader`.
- `src/__init__.py` (rỗng, để `-m src.data_loader` chạy được).
- File cache:
  - `data/vn100_symbols.csv` — danh sách mã universe (trước và/hoặc sau làm sạch).
  - `data/prices.parquet` — index=date, columns=symbol, giá close đã làm sạch.
  - `data/returns.parquet` — simple daily returns, không NaN.
- `.gitignore` bổ sung `data/` nếu chưa có (data không commit).

## Cơ chế cache (bắt buộc)
- Nếu `data/prices.parquet` và `data/returns.parquet` đã tồn tại và không bị ép làm mới → đọc từ cache, KHÔNG gọi API. Cho một flag `force_refresh=False` (vd biến môi trường hoặc tham số hàm) để ép tải lại.
- Chạy lần 2 (có cache) phải xong < 1 giây và KHÔNG phát request mạng nào.

## Acceptance criteria (phải đạt hết)
1. `.venv/bin/python -m src.data_loader` lần đầu: tải xong, in ra số mã tải được, khoảng ngày (min→max), % missing tổng thể, số mã bị loại + lý do, và shape cuối cùng `(T, N)`.
2. Universe sạch còn **≥ 70 mã** (cửa sổ 4 năm sẽ loại mã niêm yết sau 2022).
3. `data/prices.parquet` và `data/returns.parquet` tạo ra được; `returns` không có NaN; `prices.index` là datetime tăng dần.
4. Chạy LẦN 2: đọc từ cache, < 1s, không gọi mạng (chứng minh bằng log rõ ràng "loaded from cache").
5. Code có docstring + comment tiếng Việt/Anh nhất quán, giải thích rate-limit và các bước làm sạch. Người đọc là dân tối ưu, không chuyên tài chính.

## Verify trước khi báo DONE
- Chạy thật `python -m src.data_loader` (chấp nhận mất vài phút vì rate-limit) và dán output thật (số mã, shape, thời gian) vào report. Nếu mạng chặn hoàn toàn không mã nào tải được → báo BLOCKED kèm log lỗi, ĐỪNG giả lập dữ liệu.
- Chạy lần 2 chứng minh cache hoạt động.

## Ghi report vào: `.sdd/task-1-report.md`
Ghi đầy đủ vào file đó: các file đã tạo, output thật của cả 2 lần chạy, số mã universe/sạch, quyết định/giả định đã đưa ra, và concerns nếu có. Trả về cho controller: status (DONE/DONE_WITH_CONCERNS/BLOCKED/NEEDS_CONTEXT), danh sách file, 1 dòng tóm tắt kết quả chạy, concerns.
