"""
src/cvxpy_check.py
====================

Kiểm chứng chéo (cross-verification) cho solver proximal-subgradient tự viết
ở Phase 3 (`src/prox_solver.solve`), dùng CVXPY làm "ground truth" độc lập.

Đây là bước KIỂM CHỨNG, KHÔNG PHẢI thay thế solver tay:
    - `src/prox_solver.py` KHÔNG import cvxpy (và sẽ không bao giờ).
    - `cvxpy` CHỈ xuất hiện ở module NÀY, tách biệt hoàn toàn khỏi solver
      chính (đúng yêu cầu brief Phase 4).

Vì sao CVXPY làm ground truth?
-------------------------------
Bài toán gốc:

    min_w  -mu^T w + kappa*||Sigma^(1/2) w||_2 + gamma*w^T Sigma w + lam*||w||_1
    s.t.   1^T w = 1

là bài toán LỒI (tổng của: hàm affine + norm-2 SOC-representable + dạng toàn
phương PSD + norm-1, với một ràng buộc affine). CVXPY dịch trực tiếp công
thức này thành SOCP/QP chuẩn rồi gọi solver nội điểm (CLARABEL) -- một thuật
toán HOÀN TOÀN khác về bản chất so với proximal-subgradient tự viết ở Phase 3
(bậc hai/nội điểm vs bậc nhất/subgradient + prox chính xác qua bisection).
Nếu hai cách giải độc lập, khác thuật toán, khác cài đặt, cùng hội tụ về
cùng giá trị mục tiêu VÀ cùng active-set, đó là bằng chứng mạnh rằng solver
tay ĐÚNG -- không phải trùng hợp ngẫu nhiên hay "bug bù bug" (hai cài đặt độc
lập hiếm khi cho ra cùng kết quả sai theo cùng một cách).

Ý nghĩa `cp.psd_wrap(sigma)`
----------------------------
Sigma ước lượng từ dữ liệu thật (`src/estimators.estimate_sigma`) chỉ PSD tới
sai số số học dấu phẩy động (có thể có vài eigenvalue âm cực nhỏ, cỡ
1e-18..1e-19, do rounding tích luỹ qua `returns.cov()` + phép ép đối xứng).
`cp.quad_form(w, sigma)` mặc định của CVXPY kiểm tra PSD NGHIÊM NGẶT qua
eigendecomposition và sẽ NÉM LỖI (`DCPError`) nếu phát hiện bất kỳ eigenvalue
âm nào, kể cả cực nhỏ do sai số số học -- CLARABEL/CVXPY không tự biết đó là
nhiễu số học hay ma trận thực sự không PSD. `cp.psd_wrap(sigma)` báo cho
CVXPY "tin tưởng ma trận này PSD, bỏ qua kiểm tra DCP nghiêm ngặt cho biểu
thức này" -- đây KHÔNG phải "sửa" dữ liệu hay che giấu vấn đề, chỉ tắt một
kiểm tra quá thận trọng cho trường hợp sai số số học đã biết trước (cùng tinh
thần với việc `matrix_sqrt_psd` ở Phase 2 clip eigenvalue âm về 0 trước khi
lấy căn bậc hai -- xem `src/estimators.py`).

Vai trò trong pipeline
-----------------------
Module này KHÔNG phải solver production. Nó chạy `solve()` (Phase 3) và
`cvxpy_solve()` (module này) trên CÙNG bộ tham số, CÙNG dữ liệu, rồi so sánh
objective + active-set + nghiệm w -- kết quả dùng để XÁC NHẬN (hoặc bác bỏ)
tính đúng đắn của solver tay, không dùng để thay thế nó trong các phase sau.

Chạy trên data thật: `.venv/bin/python -m src.cvxpy_check`
"""

from __future__ import annotations

import warnings

import cvxpy as cp
import numpy as np
import pandas as pd

from src.estimators import estimate_all
from src.prox_solver import portfolio_objective, portfolio_objective_long_only, solve

__all__ = [
    "cvxpy_solve",
    "compare",
    "cvxpy_solve_long_only",
    "compare_long_only",
]


def cvxpy_solve(
    mu: np.ndarray,
    sigma: np.ndarray,
    sigma_sqrt: np.ndarray,
    kappa: float,
    gamma: float,
    lam: float,
) -> tuple[np.ndarray, float]:
    """Giải CÙNG bài toán sparse+robust portfolio bằng CVXPY (ground truth).

    Formulation KHỚP CHÍNH XÁC với `src/prox_solver.solve` (xem docstring
    module để biết lý do dùng CVXPY làm ground truth và ý nghĩa
    `cp.psd_wrap`):

        min_w  -mu^T w + kappa*||Sigma^(1/2) w||_2 + gamma*w^T Sigma w + lam*||w||_1
        s.t.   1^T w = 1

    Mặc định dùng solver CLARABEL (nội điểm, tốt cho SOCP+QP dạng này). Nếu
    CLARABEL lỗi (SolverError -- ví dụ không hội tụ số học trên một bộ tham
    số biên), tự động fallback sang SCS và in cảnh báo rõ ràng (KHÔNG âm
    thầm nuốt lỗi).

    Parameters
    ----------
    mu, sigma, sigma_sqrt : np.ndarray
        Xem `src.prox_solver.portfolio_objective`.
    kappa, gamma, lam : float
        Hệ số không âm của robust / variance / L1 term.

    Returns
    -------
    (w, obj) : tuple[np.ndarray, float]
        w: nghiệm tối ưu, shape (N,).
        obj: giá trị hàm mục tiêu tại w, tính lại bằng
             `portfolio_objective` (CÙNG công thức với solver tay) --
             KHÔNG dùng trực tiếp `prob.value` của CVXPY, để đảm bảo hai
             con số so sánh được tính từ CÙNG MỘT hàm, tránh sai lệch do
             khác biệt biểu diễn nội bộ (epigraph reformulation) của CVXPY.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    sigma_sqrt = np.asarray(sigma_sqrt, dtype=np.float64)
    n = mu.shape[0]

    w = cp.Variable(n)
    obj = (
        -mu @ w
        + kappa * cp.norm(sigma_sqrt @ w, 2)
        + gamma * cp.quad_form(w, cp.psd_wrap(sigma))
        + lam * cp.norm1(w)
    )
    prob = cp.Problem(cp.Minimize(obj), [cp.sum(w) == 1])

    solver_used = "CLARABEL"
    try:
        prob.solve(solver=cp.CLARABEL)
    except cp.error.SolverError as exc:
        warnings.warn(
            f"CLARABEL lỗi (kappa={kappa}, gamma={gamma}, lam={lam}): {exc!r} "
            "-- fallback sang SCS.",
            RuntimeWarning,
            stacklevel=2,
        )
        solver_used = "SCS (fallback)"
        prob.solve(solver=cp.SCS)

    if prob.status not in ("optimal", "optimal_inaccurate") or w.value is None:
        raise RuntimeError(
            f"CVXPY KHÔNG hội tụ optimal (status={prob.status}, solver={solver_used}) "
            f"cho kappa={kappa}, gamma={gamma}, lam={lam}"
        )
    if prob.status == "optimal_inaccurate":
        warnings.warn(
            f"CVXPY status=optimal_inaccurate (solver={solver_used}) cho "
            f"kappa={kappa}, gamma={gamma}, lam={lam} -- kết quả vẫn dùng "
            "nhưng nên xem xét thận trọng hơn.",
            RuntimeWarning,
            stacklevel=2,
        )

    w_opt = np.asarray(w.value, dtype=np.float64)
    obj_val = portfolio_objective(w_opt, mu, sigma, sigma_sqrt, kappa, gamma, lam)
    return w_opt, obj_val


def compare(
    mu: np.ndarray,
    sigma: np.ndarray,
    sigma_sqrt: np.ndarray,
    param_grid,
    *,
    active_thresh: float = 1e-4,
    max_iter: int = 5000,
) -> pd.DataFrame:
    """So sánh solver tay (`src.prox_solver.solve`) với CVXPY (`cvxpy_solve`)
    trên nhiều bộ tham số (kappa, gamma, lam).

    Parameters
    ----------
    mu, sigma, sigma_sqrt : np.ndarray
        Xem `src.prox_solver.portfolio_objective`.
    param_grid : sequence của dict {"kappa":.., "gamma":.., "lam":..} hoặc
        tuple (kappa, gamma, lam).
    active_thresh : float, default 1e-4
        Ngưỡng |w_i| để coi toạ độ i là "active" (không xem là 0) khi tính
        active-set và Jaccard.
    max_iter : int, default 5000
        Truyền thẳng cho `solve()` (solver tay). Tăng giá trị này nếu relgap
        quan sát được vượt ngưỡng chấp nhận (0.5%) -- xem docstring
        `src.prox_solver.solve` mục "Vì sao trả best-iterate": subgradient
        method cần đủ vòng lặp để hội tụ, KHÔNG nới ngưỡng relgap để "qua".

    Returns
    -------
    pd.DataFrame, 1 hàng / bộ tham số, cột:
        kappa, gamma, lam, f_hand, f_cvx, relgap, active_hand, active_cvx,
        jaccard, linf, n_iter, converged.

        relgap   = (f_hand - f_cvx) / |f_cvx|  (dương nghĩa là solver tay
                   tệ hơn CVXPY -- kỳ vọng ~0 vì CVXPY là nghiệm tối ưu toàn
                   cục của bài toán lồi, solver tay không thể tốt hơn về mặt
                   lý thuyết, chỉ có thể bằng hoặc kém hơn do hội tụ chưa
                   hoàn toàn).
        active_* = số toạ độ |w_i| > active_thresh.
        jaccard  = |active_hand ∩ active_cvx| / |active_hand ∪ active_cvx|
                   (quy ước = 1.0 nếu union rỗng, tức cả hai đều toàn 0 --
                   trường hợp suy biến không xảy ra trong thực tế vì
                   sum(w)=1).
        linf     = ||w_hand - w_cvx||_inf.
        n_iter, converged: lấy từ `SolveResult` của solver tay (Phase 3).
    """
    rows = []
    for params in param_grid:
        if isinstance(params, dict):
            kappa = float(params["kappa"])
            gamma = float(params["gamma"])
            lam = float(params["lam"])
        else:
            kappa, gamma, lam = (float(x) for x in params)

        result = solve(mu, sigma, sigma_sqrt, kappa, gamma, lam, max_iter=max_iter)
        w_hand = result.w
        f_hand = result.best_obj

        w_cvx, f_cvx = cvxpy_solve(mu, sigma, sigma_sqrt, kappa, gamma, lam)

        denom = abs(f_cvx) if abs(f_cvx) > 1e-12 else 1e-12
        relgap = (f_hand - f_cvx) / denom

        active_hand_mask = np.abs(w_hand) > active_thresh
        active_cvx_mask = np.abs(w_cvx) > active_thresh
        active_hand = int(active_hand_mask.sum())
        active_cvx = int(active_cvx_mask.sum())

        union = int(np.logical_or(active_hand_mask, active_cvx_mask).sum())
        inter = int(np.logical_and(active_hand_mask, active_cvx_mask).sum())
        jaccard = float(inter) / union if union > 0 else 1.0

        linf = float(np.max(np.abs(w_hand - w_cvx)))

        rows.append(
            {
                "kappa": kappa,
                "gamma": gamma,
                "lam": lam,
                "f_hand": f_hand,
                "f_cvx": f_cvx,
                "relgap": relgap,
                "active_hand": active_hand,
                "active_cvx": active_cvx,
                "jaccard": jaccard,
                "linf": linf,
                "n_iter": result.n_iter,
                "converged": result.converged,
            }
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Walk-forward Task 2: verify nhánh long-only (`prox_solver.solve_long_only`)
# ---------------------------------------------------------------------------
# Cùng lý do dùng CVXPY làm ground truth như phần long-short ở trên -- chỉ
# khác constraint set (w>=0, sum(w)=1 thay vì chỉ sum(w)=1) và KHÔNG có term
# lam*||w||_1 (dưới long-only, ||w||_1 = sum(w) = 1 là hằng số -- xem docstring
# `prox_solver.portfolio_objective_long_only`).
# ---------------------------------------------------------------------------


def cvxpy_solve_long_only(
    mu: np.ndarray,
    sigma: np.ndarray,
    sigma_sqrt: np.ndarray,
    kappa: float,
    gamma: float,
    max_weight: float | None = None,
) -> tuple[np.ndarray, float]:
    """Giải bài toán long-only bằng CVXPY (ground truth).

    Formulation KHỚP CHÍNH XÁC với `src.prox_solver.solve_long_only`:

        min_w  -mu^T w + kappa*||Sigma^(1/2) w||_2 + gamma*w^T Sigma w
        s.t.   w >= 0, 1^T w = 1   [, w <= max_weight nếu được truyền]

    KHÔNG có term lam*||w||_1 (xem docstring
    `src.prox_solver.portfolio_objective_long_only` -- dưới long-only,
    ||w||_1 = sum(w) = 1 là hằng số nên phạt L1 vô tác dụng). Dùng
    `cp.psd_wrap(sigma)` cùng lý do với `cvxpy_solve` (Sigma ước lượng từ
    dữ liệu thật chỉ PSD tới sai số số học, xem docstring module).

    RÀNG BUỘC `max_weight` (tuỳ chọn, VẪN LỒI -- không phải heuristic):
    thêm `w <= max_weight` (áp cho MỌI toạ độ) buộc toán học phải phân bổ
    vào TỐI THIỂU `ceil(1/max_weight)` mã (vì mỗi mã đóng góp tối đa
    `max_weight` vào tổng=1) -- vd max_weight=0.20 (20%) đảm bảo LUÔN CÓ ÍT
    NHẤT 5 mã active. Đây là cách kiểm soát tập trung danh mục ĐÚNG NGHĨA
    (nằm trong khung lồi, nghiệm vẫn là tối ưu toàn cục CHO bài toán đã
    ràng buộc), khác với cách "lấy top-K rồi chuẩn hoá lại" (heuristic hậu
    xử lý, không phải nghiệm tối ưu của bài toán đã ràng buộc). LƯU Ý:
    max_weight chỉ đảm bảo CẬN DƯỚI số mã (>= ceil(1/max_weight)), KHÔNG
    đảm bảo cận trên tuyệt đối -- số mã thực tế có thể nhiều hơn mức tối
    thiểu đó tuỳ dữ liệu.

    Parameters
    ----------
    mu, sigma, sigma_sqrt : np.ndarray
        Xem `src.prox_solver.portfolio_objective`.
    kappa, gamma : float, hệ số không âm.
    max_weight : float | None, default None
        Trần trọng số MỖI mã, trong (0, 1]; None = không ràng buộc (hành vi
        gốc). Truyền 0.20 để đảm bảo tối thiểu 5 mã, 1/6~=0.1667 cho tối
        thiểu 6 mã, v.v.

    Returns
    -------
    (w, obj) : tuple[np.ndarray, float]
        w: nghiệm tối ưu, shape (N,).
        obj: giá trị hàm mục tiêu tại w, tính lại bằng
             `portfolio_objective_long_only` (CÙNG công thức với solver tay)
             -- KHÔNG dùng trực tiếp `prob.value`, cùng lý do với
             `cvxpy_solve`. LƯU Ý: giá trị này KHÔNG gồm ảnh hưởng của ràng
             buộc max_weight (portfolio_objective_long_only tính trên công
             thức gốc) -- chỉ dùng để so sánh objective giữa các w cùng
             ràng buộc, không so sánh trực tiếp với nghiệm không có
             max_weight.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    sigma_sqrt = np.asarray(sigma_sqrt, dtype=np.float64)
    n = mu.shape[0]

    w = cp.Variable(n)
    obj = (
        -mu @ w
        + kappa * cp.norm(sigma_sqrt @ w, 2)
        + gamma * cp.quad_form(w, cp.psd_wrap(sigma))
    )
    constraints = [w >= 0, cp.sum(w) == 1]
    if max_weight is not None:
        constraints.append(w <= max_weight)
    prob = cp.Problem(cp.Minimize(obj), constraints)

    solver_used = "CLARABEL"
    try:
        prob.solve(solver=cp.CLARABEL)
    except cp.error.SolverError as exc:
        warnings.warn(
            f"CLARABEL lỗi (long-only, kappa={kappa}, gamma={gamma}): {exc!r} "
            "-- fallback sang SCS.",
            RuntimeWarning,
            stacklevel=2,
        )
        solver_used = "SCS (fallback)"
        prob.solve(solver=cp.SCS)

    if prob.status not in ("optimal", "optimal_inaccurate") or w.value is None:
        raise RuntimeError(
            f"CVXPY KHÔNG hội tụ optimal (status={prob.status}, solver={solver_used}) "
            f"cho long-only kappa={kappa}, gamma={gamma}"
        )
    if prob.status == "optimal_inaccurate":
        warnings.warn(
            f"CVXPY status=optimal_inaccurate (solver={solver_used}) cho "
            f"long-only kappa={kappa}, gamma={gamma} -- kết quả vẫn dùng "
            "nhưng nên xem xét thận trọng hơn.",
            RuntimeWarning,
            stacklevel=2,
        )

    w_opt = np.asarray(w.value, dtype=np.float64)
    obj_val = portfolio_objective_long_only(w_opt, mu, sigma, sigma_sqrt, kappa, gamma)
    return w_opt, obj_val


def compare_long_only(
    mu: np.ndarray,
    sigma: np.ndarray,
    sigma_sqrt: np.ndarray,
    param_grid,
    *,
    active_thresh: float = 1e-4,
    max_iter: int = 5000,
) -> pd.DataFrame:
    """So sánh `solve_long_only` (tay) với `cvxpy_solve_long_only` (ground
    truth) trên nhiều bộ tham số (kappa, gamma).

    Tương tự `compare()` (nhánh long-short) nhưng KHÔNG có cột `lam` (nhánh
    long-only không có tham số này -- xem `portfolio_objective_long_only`).

    Parameters
    ----------
    mu, sigma, sigma_sqrt : np.ndarray
        Xem `src.prox_solver.portfolio_objective`.
    param_grid : sequence của dict {"kappa":.., "gamma":..} hoặc
        tuple (kappa, gamma).
    active_thresh : float, default 1e-4
        Ngưỡng |w_i| để coi toạ độ i là "active" khi tính active-set/Jaccard.
    max_iter : int, default 5000
        Truyền thẳng cho `solve_long_only()`. Tăng nếu relgap vượt ngưỡng
        chấp nhận (0.5%).

    Returns
    -------
    pd.DataFrame, 1 hàng / bộ tham số, cột:
        kappa, gamma, f_hand, f_cvx, relgap, active_hand, active_cvx,
        jaccard, linf, n_iter, converged, w_hand, w_cvx.

        relgap  = (f_hand - f_cvx) / |f_cvx|  (cùng quy ước dấu với
                  `compare()`: dương nghĩa là solver tay tệ hơn CVXPY).
        w_hand, w_cvx: nghiệm đầy đủ (np.ndarray), tiện cho test kiểm tra
                  feasibility (w>=0, sum(w)=1) trực tiếp trên DataFrame.
    """
    from src.prox_solver import solve_long_only

    rows = []
    for params in param_grid:
        if isinstance(params, dict):
            kappa = float(params["kappa"])
            gamma = float(params["gamma"])
        else:
            kappa, gamma = (float(x) for x in params)

        result = solve_long_only(mu, sigma, sigma_sqrt, kappa, gamma, max_iter=max_iter)
        w_hand = result.w
        f_hand = result.best_obj

        w_cvx, f_cvx = cvxpy_solve_long_only(mu, sigma, sigma_sqrt, kappa, gamma)

        denom = abs(f_cvx) if abs(f_cvx) > 1e-12 else 1e-12
        relgap = (f_hand - f_cvx) / denom

        active_hand_mask = np.abs(w_hand) > active_thresh
        active_cvx_mask = np.abs(w_cvx) > active_thresh
        active_hand = int(active_hand_mask.sum())
        active_cvx = int(active_cvx_mask.sum())

        union = int(np.logical_or(active_hand_mask, active_cvx_mask).sum())
        inter = int(np.logical_and(active_hand_mask, active_cvx_mask).sum())
        jaccard = float(inter) / union if union > 0 else 1.0

        linf = float(np.max(np.abs(w_hand - w_cvx)))

        rows.append(
            {
                "kappa": kappa,
                "gamma": gamma,
                "f_hand": f_hand,
                "f_cvx": f_cvx,
                "relgap": relgap,
                "active_hand": active_hand,
                "active_cvx": active_cvx,
                "jaccard": jaccard,
                "linf": linf,
                "n_iter": result.n_iter,
                "converged": result.converged,
                "w_hand": w_hand,
                "w_cvx": w_cvx,
            }
        )

    return pd.DataFrame(rows)


# Bộ tham số bắt buộc theo brief Phase 4 (>=5 bộ, gồm các trường hợp biên:
# kappa=0 tắt robust term, lam nhỏ ít thưa, lam=0 không thưa).
DEFAULT_PARAM_GRID = [
    {"kappa": 1.0, "gamma": 5.0, "lam": 0.01},
    {"kappa": 0.0, "gamma": 5.0, "lam": 0.1},
    {"kappa": 1.0, "gamma": 5.0, "lam": 0.001},
    {"kappa": 2.0, "gamma": 10.0, "lam": 0.05},
    {"kappa": 0.5, "gamma": 1.0, "lam": 0.02},
    {"kappa": 1.0, "gamma": 5.0, "lam": 0.0},
]


def _main() -> None:  # pragma: no cover - manual verification entry point
    from pathlib import Path

    # RuntimeWarning benign từ Apple Accelerate BLAS (đã biết từ Phase 2/3,
    # xem docstring src/prox_solver.py) -- filter riêng warning NÀY, không
    # che các warning khác (vd cảnh báo fallback SCS / optimal_inaccurate
    # phát ra từ cvxpy_solve ở trên vẫn hiện đầy đủ).
    warnings.filterwarnings(
        "ignore", message=".*encountered in matmul.*", category=RuntimeWarning
    )

    path = Path(__file__).resolve().parent.parent / "data" / "returns.parquet"
    returns = pd.read_parquet(path)
    print(f"returns shape: {returns.shape}")

    mu, sigma, sigma_sqrt = estimate_all(returns)
    print(f"mu shape: {mu.shape}, sigma shape: {sigma.shape}\n")

    df = compare(mu, sigma, sigma_sqrt, DEFAULT_PARAM_GRID, active_thresh=1e-4)

    pd.set_option("display.float_format", lambda x: f"{x:.6e}")
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", None)
    print(df.to_string(index=False))

    max_relgap = df["relgap"].abs().max()
    print(f"\nmax |relgap| across param grid: {max_relgap:.6%}")
    print(f"all Jaccard == 1.0: {bool((df['jaccard'] == 1.0).all())}")
    print(f"all converged: {bool(df['converged'].all())}")


if __name__ == "__main__":  # pragma: no cover
    _main()
