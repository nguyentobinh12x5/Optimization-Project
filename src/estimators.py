"""
src/estimators.py
==================

Ước lượng thống kê cho bài toán sparse+robust portfolio optimization (VN100).

Module này biến ma trận simple daily returns (đầu ra của `src/data_loader.py`,
xem `data/returns.parquet`) thành 3 đối tượng đại số tuyến tính mà solver ở
Phase 3 (proximal-subgradient) và bước verify ở Phase 4 (CVXPY) sẽ dùng trực
tiếp:

- mu_hat     (N,)   : ước lượng vector kỳ vọng return NGÀY (chưa annualize).
- Sigma      (N,N)  : ước lượng ma trận hiệp phương sai return NGÀY, đối xứng,
                       PSD (tới sai số số học).
- Sigma_sqrt (N,N)  : căn bậc hai đối xứng PSD của Sigma, tức là ma trận A sao
                       cho A @ A ~= Sigma (A không nhất thiết là Cholesky
                       factor -- xem giải thích bên dưới).

Người đọc mục tiêu là dân TỐI ƯU HOÁ, không nhất thiết chuyên tài chính. Đơn
vị: mọi thứ ở đây giữ nguyên đơn vị NGÀY (daily), việc annualize (nếu cần,
thường là mu*252, Sigma*252) để cho các bước báo cáo/phân tích ở phase sau lo,
KHÔNG làm ở module này để tránh nhầm lẫn "đã annualize hay chưa" khi solver
consume trực tiếp các hàm dưới đây.

QUY ƯỚC THỨ TỰ ASSET: mọi vector/ma trận trả về có thứ tự hàng/cột KHỚP với
`list(returns.columns)` của DataFrame input, theo đúng thứ tự xuất hiện (không
sort lại). Caller (solver, verify) phải tự giữ mapping index <-> symbol nếu
cần tra ngược tên mã.

Vì sao dùng eigendecomposition (np.linalg.eigh) thay vì Cholesky cho căn bậc
hai ma trận?
- Cholesky (A = L L^T) đòi hỏi ma trận PHẢI positive DEFINITE (PD), tức mọi
  eigenvalue > 0 nghiêm ngặt. Sigma ước lượng từ dữ liệu thật (đặc biệt khi
  N gần bằng hoặc lớn hơn T, hoặc N lớn so với số quan sát độc lập) có thể
  chỉ positive SEMI-definite (PSD) -- có eigenvalue bằng 0 hoặc âm rất nhỏ do
  sai số số học dấu phẩy động. Cholesky sẽ NÉM LỖI (hoặc NaN) trong các
  trường hợp đó.
- eigh (dành riêng cho ma trận đối xứng/Hermitian) luôn trả về eigenvalue
  thực + eigenvector trực giao cho MỌI ma trận đối xứng, kể cả suy biến
  (rank-deficient). Ta chỉ cần CLIP các eigenvalue âm (do sai số số học) về 0
  trước khi lấy căn bậc hai, rồi dựng lại:
      Sigma = V diag(lambda) V^T
      Sigma^(1/2) := V diag(sqrt(clip(lambda, 0, inf))) V^T
  Đây chính là căn bậc hai đối xứng PSD (symmetric PSD square root) -- khác
  Cholesky factor L (tam giác dưới, không đối xứng) nhưng vẫn thỏa
  Sigma^(1/2) @ Sigma^(1/2) = Sigma, và có thêm tính chất đối xứng + PSD hữu
  ích cho solver ở Phase 3 (ví dụ khi cần chiếu/scale theo Sigma^(1/2) mà vẫn
  muốn giữ đối xứng).

Ledoit-Wolf shrinkage: sklearn không có sẵn trong môi trường này, nên công
thức được tự cài đặt bằng numpy (xem `_ledoit_wolf_delta` bên dưới), theo
Ledoit & Wolf (2004) "Honey, I Shrunk the Sample Covariance Matrix" (target
là F = mean(diag(S)) * I -- shrinkage về ma trận đường chéo hằng số, phiên
bản constant-variance-identity, KHÔNG phải bản constant-correlation phức tạp
hơn trong bài báo gốc; đây là lựa chọn đơn giản hóa hợp lý, ghi rõ ở đây thay
vì giấu trong code).

Chạy nhanh trên data thật: `.venv/bin/python -m src.estimators`
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "estimate_mu",
    "estimate_sigma",
    "matrix_sqrt_psd",
    "estimate_all",
]


def estimate_mu(returns: pd.DataFrame) -> np.ndarray:
    """Ước lượng vector kỳ vọng return (mu_hat) bằng trung bình mẫu theo cột.

    mu_hat[i] = mean_t(returns[t, i]), đơn vị NGÀY (không annualize).

    Parameters
    ----------
    returns : pd.DataFrame, shape (T, N)
        Simple daily returns, index = date, columns = symbol. Thứ tự asset
        của output = list(returns.columns) theo đúng thứ tự cột trong
        DataFrame (không sort lại).

    Returns
    -------
    np.ndarray, shape (N,), dtype float64.
    """
    return returns.mean(axis=0).to_numpy(dtype=np.float64)


def _mean_var_target(S: np.ndarray) -> np.ndarray:
    """Target F = mean(diag(S)) * I cho Ledoit-Wolf / shrinkage tuyến tính.

    Đây là target "constant variance, zero correlation": một ma trận chéo
    với phương sai bằng trung bình các phương sai mẫu trên đường chéo của S,
    và hiệp phương sai chéo (off-diagonal) bằng 0.
    """
    n = S.shape[0]
    mean_var = np.trace(S) / n
    return mean_var * np.eye(n)


def _ledoit_wolf_delta(X: np.ndarray, S: np.ndarray) -> float:
    """Tính hệ số shrinkage delta theo công thức Ledoit-Wolf (2004), target
    F = mean_var * I (constant-variance-identity target).

    X: ma trận demeaned observations, shape (T, N) (mỗi hàng là 1 quan sát,
       đã trừ mean theo cột).
    S: sample covariance ứng với X (ddof=1), shape (N, N).

    Công thức (rút gọn theo Ledoit & Wolf 2004, Sec 2, cho target hằng số *
    I):
        pi_hat   = sum_{i,j} mean_t[ (x_ti x_tj - S_ij)^2 ]   (ước lượng
                   "phi" -- phương sai của các phần tử S)
        rho_hat  = 0 (vì target off-diagonal = 0, on-diagonal các phần tử
                   không phụ thuộc lẫn nhau trong xấp xỉ này -- xem chú thích)
        gamma_hat = || S - F ||_F^2   (bình phương khoảng cách Frobenius
                    giữa sample covariance và target)
        kappa_hat = (pi_hat - rho_hat) / gamma_hat
        delta     = clip(kappa_hat / T, 0, 1)

    Đây là công thức shrinkage-intensity chuẩn của LW cho single-factor /
    identity target, được đơn giản hóa (rho_hat = 0) vì target không có
    thành phần chéo ngoài để "đồng shrink" -- một xấp xỉ hợp lý và phổ biến
    trong các cài đặt LW rút gọn (được ghi rõ ở đây theo yêu cầu brief thay
    vì cài đặt phiên bản đầy đủ constant-correlation phức tạp hơn nhiều).
    """
    T, n = X.shape
    F = _mean_var_target(S)

    # pi_hat: ước lượng sum of asymptotic variances của các phần tử S_ij,
    # dùng công thức mẫu: pi_hat = (1/T) * sum_t || x_t x_t^T - S ||_F^2
    # (tổng qua mọi i,j của phương sai từng phần tử).
    pi_mat = np.zeros((n, n))
    for t in range(T):
        xt = X[t, :]
        outer = np.outer(xt, xt)
        pi_mat += (outer - S) ** 2
    pi_mat /= T
    pi_hat = pi_mat.sum()

    gamma_hat = np.sum((S - F) ** 2)

    if gamma_hat <= 0:
        return 0.0

    kappa_hat = pi_hat / gamma_hat
    delta = kappa_hat / T
    return float(np.clip(delta, 0.0, 1.0))


def estimate_sigma(
    returns: pd.DataFrame, shrinkage: float | str | None = None
) -> np.ndarray:
    """Ước lượng ma trận hiệp phương sai (Sigma) của returns.

    Sample covariance mặc định dùng `returns.cov()` của pandas, tức
    ddof=1 (chia cho T-1, unbiased estimator) -- KHÔNG phải ddof=0.

    Parameters
    ----------
    returns : pd.DataFrame, shape (T, N)
        Simple daily returns. Thứ tự asset của output = list(returns.columns).
    shrinkage : float | str | None, default None
        - None: trả sample covariance thuần (không shrinkage).
        - "lw" hoặc "ledoit-wolf": tự tính hệ số shrinkage delta theo công
          thức Ledoit-Wolf (2004), target F = mean(diag(S)) * I (xem
          `_ledoit_wolf_delta`), rồi trả (1-delta)*S + delta*F.
        - float trong [0, 1]: dùng trực tiếp làm delta với cùng target F.

    Returns
    -------
    np.ndarray, shape (N, N), dtype float64, đối xứng (ép Sigma=(Sigma+Sigma.T)/2
    trước khi trả).
    """
    S = returns.cov().to_numpy(dtype=np.float64)  # pandas .cov(): ddof=1

    if shrinkage is None:
        Sigma = S
    elif isinstance(shrinkage, str):
        key = shrinkage.lower()
        if key in ("lw", "ledoit-wolf"):
            X = (returns - returns.mean(axis=0)).to_numpy(dtype=np.float64)
            F = _mean_var_target(S)
            delta = _ledoit_wolf_delta(X, S)
            Sigma = (1.0 - delta) * S + delta * F
        else:
            raise ValueError(
                f"shrinkage string không nhận diện được: {shrinkage!r}; "
                "dùng 'lw'/'ledoit-wolf', một float trong [0,1], hoặc None."
            )
    else:
        delta = float(shrinkage)
        if not (0.0 <= delta <= 1.0):
            raise ValueError(f"shrinkage (delta) phải trong [0,1], nhận {delta}")
        F = _mean_var_target(S)
        Sigma = (1.0 - delta) * S + delta * F

    # Ép đối xứng: loại sai số số học (dấu phẩy động) có thể làm Sigma lệch
    # đối xứng cực nhỏ sau các phép toán trên.
    Sigma = (Sigma + Sigma.T) / 2.0
    return Sigma


def matrix_sqrt_psd(sigma: np.ndarray) -> np.ndarray:
    """Căn bậc hai đối xứng PSD của một ma trận đối xứng PSD (hoặc gần PSD).

    Dùng eigendecomposition đối xứng (np.linalg.eigh) thay vì Cholesky vì
    Sigma có thể chỉ PSD (không PD nghiêm ngặt) -- xem giải thích chi tiết ở
    docstring module. Các eigenvalue âm (do sai số số học dấu phẩy động,
    thường cực nhỏ) được CLIP về 0 trước khi lấy căn bậc hai:

        sigma = V diag(lambda) V^T          (eigh, V trực giao, lambda thực)
        sqrt  = V diag(sqrt(clip(lambda,0))) V^T

    Kết quả thỏa: sqrt @ sqrt ~= sigma (tới sai số số học), sqrt đối xứng,
    PSD.

    Parameters
    ----------
    sigma : np.ndarray, shape (N, N)
        Ma trận đối xứng (hoặc gần đối xứng do sai số số học), PSD (hoặc gần
        PSD, có thể có vài eigenvalue âm cực nhỏ do sai số số học).

    Returns
    -------
    np.ndarray, shape (N, N), dtype float64, đối xứng, PSD.
    """
    # Ép đối xứng trước khi eigh để đảm bảo eigenvalue/eigenvector thực
    # (eigh giả định đối xứng và chỉ đọc nửa ma trận, nhưng ép ở đây để kết
    # quả ổn định ngay cả khi input lệch đối xứng nhẹ do sai số số học).
    sym = (sigma + sigma.T) / 2.0
    eigvals, eigvecs = np.linalg.eigh(sym)
    eigvals_clipped = np.clip(eigvals, 0.0, None)
    sqrt_sigma = eigvecs @ np.diag(np.sqrt(eigvals_clipped)) @ eigvecs.T
    # Ép đối xứng kết quả (cùng lý do sai số số học ở phép nhân ma trận).
    sqrt_sigma = (sqrt_sigma + sqrt_sigma.T) / 2.0
    return sqrt_sigma


def estimate_all(
    returns: pd.DataFrame, shrinkage: float | str | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Tiện ích gọi gộp: trả (mu_hat, Sigma, Sigma_sqrt) từ returns.

    Thứ tự asset của cả 3 output = list(returns.columns) (đúng thứ tự cột
    trong DataFrame input, không sort lại).

    Parameters
    ----------
    returns : pd.DataFrame, shape (T, N)
    shrinkage : float | str | None, default None
        Xem `estimate_sigma`.

    Returns
    -------
    (mu, sigma, sigma_sqrt) : tuple[np.ndarray, np.ndarray, np.ndarray]
        mu shape (N,), sigma shape (N,N), sigma_sqrt shape (N,N).
    """
    mu = estimate_mu(returns)
    sigma = estimate_sigma(returns, shrinkage=shrinkage)
    sigma_sqrt = matrix_sqrt_psd(sigma)
    return mu, sigma, sigma_sqrt


def _main() -> None:  # pragma: no cover - manual smoke test entry point
    from pathlib import Path

    path = Path(__file__).resolve().parent.parent / "data" / "returns.parquet"
    returns = pd.read_parquet(path)
    print(f"returns shape: {returns.shape}")

    mu, sigma, sigma_sqrt = estimate_all(returns)
    print(f"mu shape: {mu.shape}")
    print(f"sigma shape: {sigma.shape}")
    print(f"sigma_sqrt shape: {sigma_sqrt.shape}")

    recon = sigma_sqrt @ sigma_sqrt
    rel_err = np.linalg.norm(recon - sigma, "fro") / np.linalg.norm(sigma, "fro")
    print(f"||Sigma_sqrt^2 - Sigma||_F / ||Sigma||_F = {rel_err:.3e}")
    print(f"sigma_sqrt symmetric: {np.allclose(sigma_sqrt, sigma_sqrt.T, atol=1e-10)}")
    print(f"sigma symmetric: {np.allclose(sigma, sigma.T, atol=1e-12)}")


if __name__ == "__main__":  # pragma: no cover
    _main()
