# Task Brief — Phase 2: Estimators (μ̂, Σ, Σ^(1/2))

Đây là YÊU CẦU của bạn. Dùng đúng các giá trị verbatim. Đọc trước tiên.

## Mục tiêu
Viết `src/estimators.py`: từ ma trận returns, sinh μ̂ (vector kỳ vọng), Σ (hiệp phương sai, có tùy chọn shrinkage), và Σ^(1/2) (căn bậc hai ma trận). Đây là input cho solver ở Phase 3. Kèm test `tests/test_estimators.py`.

## Môi trường
- Interpreter: `.venv/bin/python` (Python 3.14). KHÔNG dùng python hệ thống, KHÔNG tạo venv mới.
- Packages CÓ SẴN (đã verify): `numpy 2.2.6`, `pandas 2.3.3`, `pyarrow`, `matplotlib`, `seaborn`, `pytest 9.1.1`.
- Packages KHÔNG CÓ (đừng dùng, đừng pip install): **scipy KHÔNG có** → chỉ dùng `numpy.linalg.eigh` cho căn ma trận. **sklearn KHÔNG có** → Ledoit-Wolf phải TỰ CÀI ĐẶT bằng numpy (hoặc nhận δ float), KHÔNG import sklearn.
- Thư mục project: `/Users/nguyentobinh12gmail.com/Documents/Optimization Project`.
- KHÔNG cần mạng ở phase này.

## Input dữ liệu (đã tồn tại từ Phase 1)
- `data/returns.parquet`: simple daily returns, index=date (datetime), columns=symbol (75 mã), shape ~(1069, 75), KHÔNG có NaN. Đọc bằng `pd.read_parquet`.
- Đơn vị: return theo NGÀY (chưa annualize). Giữ nguyên đơn vị ngày trong estimators; việc annualize (nếu cần) để phần báo cáo lo.

## Hàm cần viết (interface — Phase 3/4 sẽ consume, giữ đúng tên/chữ ký)
```
estimate_mu(returns: pd.DataFrame) -> np.ndarray        # shape (N,), = mean theo cột (axis=0)
estimate_sigma(returns: pd.DataFrame, shrinkage: float | str | None = None) -> np.ndarray  # (N,N)
matrix_sqrt_psd(sigma: np.ndarray) -> np.ndarray        # (N,N), căn bậc hai PSD đối xứng
estimate_all(returns: pd.DataFrame, shrinkage=None) -> tuple[np.ndarray, np.ndarray, np.ndarray]
    # trả (mu, sigma, sigma_sqrt); thứ tự cột/asset khớp returns.columns
```
- Trả về numpy array (không phải DataFrame) để solver dùng đại số tuyến tính trực tiếp; NHƯNG cung cấp cách lấy thứ tự asset (ví dụ trả kèm hoặc tài liệu hóa rằng thứ tự = list(returns.columns)). Ghi rõ trong docstring rằng index/thứ tự asset = `returns.columns`.

## Yêu cầu chi tiết
1. **μ̂**: trung bình mẫu theo cột, `returns.mean(axis=0).values`, shape (N,).
2. **Σ**: sample covariance mặc định (`returns.cov()` — pandas dùng ddof=1, chấp nhận được; ghi rõ ddof trong docstring). Tùy chọn shrinkage:
   - `shrinkage=None` (mặc định): sample covariance thuần.
   - `shrinkage="lw"` hoặc `"ledoit-wolf"`: TỰ CÀI ĐẶT bằng numpy (KHÔNG có sklearn). Shrinkage tuyến tính về target F = trung bình phương sai × I: `Σ_shrunk = (1-δ)·S + δ·F`. Tính δ theo công thức Ledoit-Wolf (2004) chuẩn, HOẶC nếu thấy phức tạp thì cài đặt công thức LW rút gọn và ghi rõ; clamp δ về [0,1].
   - Nếu `shrinkage` là float trong [0,1]: dùng trực tiếp làm δ với cùng target F=mean_var·I.
   - Kết quả Σ phải đối xứng: ép `Σ = (Σ+Σ.T)/2` trước khi trả.
3. **Σ^(1/2)**: dùng eigendecomposition đối xứng (`np.linalg.eigh`), clip eigenvalue < 0 về 0 (do sai số số học Σ có thể có eigenvalue âm cực nhỏ), rồi `Σ^(1/2) = V · diag(sqrt(clip(λ,0,∞))) · V.T`. KHÔNG dùng Cholesky (Σ có thể chỉ PSD, không PD → Cholesky fail). Ép đối xứng kết quả.

## Test bắt buộc (`tests/test_estimators.py`, chạy được bằng pytest, KHÔNG cần mạng)
Dùng dữ liệu GIẢ LẬP có seed cố định (KHÔNG phụ thuộc data thật): `rng = np.random.default_rng(0)`, tạo returns giả N=5, T=200 (DataFrame với cột 'A'..'E').
1. `test_mu_shape_and_value`: μ̂ shape (5,); khớp `df.mean().values` (allclose).
2. `test_sigma_symmetric`: Σ đối xứng — `np.allclose(S, S.T, atol=1e-12)`.
3. `test_sigma_psd`: mọi eigenvalue của Σ ≥ -1e-8.
4. `test_sqrt_reconstructs`: `matrix_sqrt_psd(S) @ matrix_sqrt_psd(S)` xấp xỉ S — `‖A@A - S‖_F / ‖S‖_F < 1e-6`.
5. `test_sqrt_symmetric`: Σ^(1/2) đối xứng.
6. `test_shrinkage_between`: với shrinkage bật (lw hoặc δ=0.5), kết quả vẫn đối xứng, PSD, và (nếu δ float) nằm "giữa" S và target theo nghĩa đường chéo thay đổi đúng hướng. Tối thiểu: khẳng định Σ_shrunk đối xứng + PSD.
7. `test_sqrt_on_psd_singular`: tạo ma trận PSD suy biến (rank-deficient, ví dụ ngoài tích X.T@X với X có cột phụ thuộc tuyến tính) → `matrix_sqrt_psd` KHÔNG ném lỗi và trả ma trận thực, không NaN.

## Acceptance criteria
1. `.venv/bin/python -m pytest tests/test_estimators.py -v` PASS toàn bộ (dán output thật vào report).
2. Chạy thử trên data thật: `estimate_all(pd.read_parquet("data/returns.parquet"))` chạy không lỗi, in shape μ̂ (75,), Σ (75,75), Σ^(1/2) (75,75); verify `‖Σ^(1/2)² − Σ‖_F/‖Σ‖_F < 1e-6` và Σ^(1/2) đối xứng trên data thật. Dán số thật vào report.
3. Docstring/comment giải thích toán (vì sao eigh + clip thay vì Cholesky), người đọc dân tối ưu.

## Ghi report vào: `.sdd/task-2-report.md`
Đầy đủ: file tạo, output pytest thật, kết quả chạy trên data thật (các shape + số verify). Trả về controller: status, danh sách file, 1 dòng tóm tắt (pytest X/X pass, verify trên data thật), concerns. KHÔNG dán code dài vào phần trả về.
