"""
tests/test_prox_solver.py
==========================

Test cho src/prox_solver.py (solver proximal-subgradient tự viết, Phase 3).

Dữ liệu GIẢ LẬP với seed cố định (KHÔNG phụ thuộc data thật ở
data/returns.parquet), để test chạy độc lập, không cần mạng, ổn định qua
thời gian. Sigma luôn tạo qua `A @ A.T + eps*I` để đảm bảo PD.

Chạy: `.venv/bin/python -m pytest tests/test_prox_solver.py -v`
"""

from __future__ import annotations

import numpy as np

from src.estimators import matrix_sqrt_psd
from src.prox_solver import (
    _prox_l1_simplex_eq,
    _robust_subgrad,
    portfolio_objective,
    portfolio_objective_long_only,
    simplex_projection,
    solve,
    solve_long_only,
)


def _make_pd_sigma(rng: np.random.Generator, n: int, scale: float = 1.0) -> np.ndarray:
    """Sigma PD ngẫu nhiên: A @ A.T + eps*I, scale theo `scale`."""
    A = rng.normal(size=(n, n)) * np.sqrt(scale)
    sigma = A @ A.T + 1e-6 * scale * np.eye(n)
    return sigma


def _sqrt_psd(sigma: np.ndarray) -> np.ndarray:
    eigvals, eigvecs = np.linalg.eigh(sigma)
    eigvals = np.clip(eigvals, 0.0, None)
    return eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T


def test_closed_form_meanvar() -> None:
    """N=2, kappa=0, lam=0 -> QP với ràng buộc affine, nghiệm đóng qua KKT:
    [[2*gamma*Sigma, 1], [1^T, 0]] @ [w; eta] = [mu; 1].
    """
    rng = np.random.default_rng(0)
    n = 2
    # Scale O(1) (không phải scale returns thật ~1e-4) để hệ KKT tham chiếu
    # well-conditioned và test tập trung vào tính ĐÚNG của thuật toán, tách
    # biệt khỏi câu hỏi tinh chỉnh alpha0 cho scale returns thật (xem mục
    # "Chạy trên data thật" trong report cho việc đó).
    sigma = _make_pd_sigma(rng, n, scale=1.0)
    sigma_sqrt = _sqrt_psd(sigma)
    mu = rng.normal(loc=0.1, scale=0.05, size=n)
    gamma = 1.0

    # Nghiệm đóng qua hệ KKT.
    KKT = np.zeros((n + 1, n + 1))
    KKT[:n, :n] = 2.0 * gamma * sigma
    KKT[:n, n] = 1.0
    KKT[n, :n] = 1.0
    rhs = np.concatenate([mu, [1.0]])
    sol = np.linalg.solve(KKT, rhs)
    w_ref = sol[:n]

    # alpha0=1.0 tuned riêng cho scale O(1) của test này (KHÔNG phải
    # ALPHA0_DEFAULT của module, vốn tinh chỉnh cho scale returns thật
    # ~1e-3/1e-4 -- xem "Lựa chọn alpha0 mặc định" trong src/prox_solver.py
    # và mục tinh chỉnh alpha0 trong task-3-report.md). Test này chỉ kiểm
    # tra TÍNH ĐÚNG của thuật toán qua đối chiếu KKT, tách biệt khỏi câu
    # hỏi alpha0 nào phù hợp cho data thật.
    result = solve(
        mu, sigma, sigma_sqrt, kappa=0.0, gamma=gamma, lam=0.0,
        max_iter=20000, alpha0=1.0,
    )

    assert np.max(np.abs(result.w - w_ref)) < 1e-4


def test_sum_to_one() -> None:
    rng = np.random.default_rng(1)
    n = 6
    sigma = _make_pd_sigma(rng, n, scale=1e-4)
    sigma_sqrt = _sqrt_psd(sigma)
    mu = rng.normal(loc=1e-3, scale=5e-4, size=n)

    result = solve(mu, sigma, sigma_sqrt, kappa=1.0, gamma=5.0, lam=0.01, max_iter=2000)

    assert abs(np.sum(result.w) - 1.0) < 1e-9


def test_sparsity_increases_with_lambda() -> None:
    """Sau FIX joint prox (xem `_prox_l1_simplex_eq`), toạ độ bị
    soft-threshold về 0 KHÔNG còn bị "hồi sinh" bởi bước chiếu ràng buộc
    (khác bản heuristic prox-then-project cũ) -- nên sparsity phải tăng
    ĐƠN ĐIỆU và ỔN ĐỊNH theo lambda, với các toạ độ = 0 CHÍNH XÁC (không
    phải xấp xỉ nhỏ)."""
    rng = np.random.default_rng(14)
    n = 15
    sigma = _make_pd_sigma(rng, n, scale=1e-4)
    sigma_sqrt = _sqrt_psd(sigma)
    mu = rng.normal(loc=1e-3, scale=8e-4, size=n)

    # Dùng ALPHA0_DEFAULT (không truyền alpha0) -- đã dò thực nghiệm trên
    # data thật và ở đây: dãy active GIẢM NGHIÊM NGẶT 15 -> 11 -> 9, với số
    # toạ độ = 0 CHÍNH XÁC tăng dần 0 -> 4 -> 6.
    counts = []
    exact_zeros = []
    for lam in (1e-4, 3e-3, 1e-2):
        result = solve(mu, sigma, sigma_sqrt, kappa=0.5, gamma=5.0, lam=lam, max_iter=6000)
        active = int(np.sum(np.abs(result.w) > 1e-4))
        counts.append(active)
        exact_zeros.append(int(np.sum(result.w == 0.0)))

    # Không tăng khi lambda tăng.
    assert counts[1] <= counts[0]
    assert counts[2] <= counts[1]
    # Với cặp đủ tách biệt (nhỏ nhất vs lớn nhất), phải giảm thực sự.
    assert counts[2] < counts[0]

    # Sparsity THẬT: ở lambda đủ lớn, phải có ÍT NHẤT một toạ độ bằng 0
    # CHÍNH XÁC (== 0.0, không phải "nhỏ hơn eps"), và số toạ độ 0 chính
    # xác không giảm khi lambda tăng.
    assert exact_zeros[0] == 0
    assert exact_zeros[1] > 0
    assert exact_zeros[2] >= exact_zeros[1]


def test_prox_l1_simplex_eq_exact_zero_and_sum_one() -> None:
    """Test trực tiếp helper `_prox_l1_simplex_eq`: với z có sum(z) đã gần
    1 và vài toạ độ nhỏ nằm trong ngưỡng threshold, joint prox phải trả
    toạ độ = 0 CHÍNH XÁC cho những toạ độ đó (không bị offset "hồi sinh"),
    và sum(w) = 1 chính xác tới sai số bisection."""
    z = np.array([0.6, 0.6, 0.005, -0.005, -0.2])  # sum(z) = 1.0 đã gần 1
    t = 0.05  # ngưỡng đủ lớn để "nuốt" 2 toạ độ nhỏ (0.005, -0.005)

    w = _prox_l1_simplex_eq(z, t)

    assert abs(np.sum(w) - 1.0) < 1e-9
    # Hai toạ độ nhỏ (idx 2, 3) phải bị threshold về đúng 0 -- CHÍNH XÁC,
    # không phải giá trị nhỏ do offset chiếu hyperplane cộng vào (khác
    # hành vi của bản heuristic prox-then-project cũ).
    assert w[2] == 0.0
    assert w[3] == 0.0
    assert w[0] != 0.0 and w[1] != 0.0 and w[4] != 0.0


def test_prox_l1_simplex_eq_reduces_to_projection_when_t_zero() -> None:
    """t=0 -> soft_threshold(x,0)=x -> joint prox rút gọn về đúng phép chiếu
    Euclid lên hyperplane {1^T w=1} (không có sparsity)."""
    rng = np.random.default_rng(6)
    n = 8
    z = rng.normal(size=n)

    w = _prox_l1_simplex_eq(z, 0.0)
    w_expected = z + (1.0 - np.sum(z)) / n

    assert np.allclose(w, w_expected, atol=1e-9)
    assert abs(np.sum(w) - 1.0) < 1e-9


def test_best_obj_monotone() -> None:
    rng = np.random.default_rng(3)
    n = 5
    sigma = _make_pd_sigma(rng, n, scale=1e-4)
    sigma_sqrt = _sqrt_psd(sigma)
    mu = rng.normal(loc=1e-3, scale=5e-4, size=n)

    result = solve(mu, sigma, sigma_sqrt, kappa=1.0, gamma=5.0, lam=0.01, max_iter=1000)

    running_min = np.minimum.accumulate(result.obj_history)
    assert np.all(np.diff(running_min) <= 1e-12)
    assert np.isclose(result.best_obj, running_min[-1], rtol=1e-9, atol=1e-12)


def test_returns_best_not_last() -> None:
    """Dùng alpha0 lớn + max_iter nhỏ để tạo dao động (subgradient không đơn
    điệu) -> iterate cuối cùng tệ hơn best. Khẳng định result.w khớp
    best_obj, KHÔNG phải iterate cuối."""
    rng = np.random.default_rng(4)
    n = 5
    sigma = _make_pd_sigma(rng, n, scale=1e-4)
    sigma_sqrt = _sqrt_psd(sigma)
    mu = rng.normal(loc=1e-3, scale=5e-4, size=n)

    result = solve(
        mu, sigma, sigma_sqrt, kappa=1.0, gamma=5.0, lam=0.01,
        max_iter=20, alpha0=200.0, tol=0.0, patience=10**9,
    )

    last_obj = result.obj_history[-1]
    # Phải tồn tại dao động thật sự cho test này có ý nghĩa.
    assert last_obj > result.best_obj + 1e-12

    f_w = portfolio_objective(result.w, mu, sigma, sigma_sqrt, 1.0, 5.0, 0.01)
    assert np.isclose(f_w, result.best_obj, rtol=1e-9, atol=1e-12)
    assert f_w <= last_obj + 1e-12


def test_robust_subgrad_zero_safe() -> None:
    """Khi Sigma^1/2 w ~ 0 (vd w=0), subgradient robust phải trả 0 vector,
    KHÔNG chia cho 0 / KHÔNG NaN."""
    rng = np.random.default_rng(5)
    n = 4
    sigma = _make_pd_sigma(rng, n, scale=1e-4)
    sigma_sqrt = _sqrt_psd(sigma)

    w_zero = np.zeros(n)
    g = _robust_subgrad(w_zero, sigma, sigma_sqrt, kappa=2.0)
    assert not np.any(np.isnan(g))
    assert np.allclose(g, 0.0)

    # Đồng thời xác nhận solve() với kappa>0 không sinh NaN trong obj_history.
    mu = rng.normal(loc=1e-3, scale=5e-4, size=n)
    result = solve(mu, sigma, sigma_sqrt, kappa=2.0, gamma=5.0, lam=0.01, max_iter=500)
    assert not np.any(np.isnan(result.obj_history))


# ---------------------------------------------------------------------------
# Task 1 (walk-forward): simplex_projection + solve_long_only
# ---------------------------------------------------------------------------


def test_simplex_projection_symmetric_case():
    v = np.array([0.5, 0.5, 0.5])
    w = simplex_projection(v)
    assert np.allclose(w, np.array([1 / 3, 1 / 3, 1 / 3]), atol=1e-10)
    assert abs(w.sum() - 1.0) < 1e-10
    assert (w >= -1e-12).all()


def test_simplex_projection_dominant_component():
    v = np.array([2.0, 0.2, 0.2])
    w = simplex_projection(v)
    assert np.allclose(w, np.array([1.0, 0.0, 0.0]), atol=1e-10)


def test_simplex_projection_already_feasible_is_fixed_point():
    v = np.array([0.2, 0.3, 0.5])  # đã w>=0, sum=1
    w = simplex_projection(v)
    assert np.allclose(w, v, atol=1e-10)


def test_simplex_projection_random_always_feasible():
    rng = np.random.default_rng(1)
    for _ in range(20):
        v = rng.normal(0, 3, size=10)
        w = simplex_projection(v)
        assert (w >= -1e-9).all()
        assert abs(w.sum() - 1.0) < 1e-8


def test_solve_long_only_uniform_when_isotropic():
    """min w^T w s.t. w>=0, sum(w)=1 (mu=0, kappa=0, Sigma=I) có nghiệm đóng
    w = 1/N (do tính đối xứng -- đây là bài toán min-variance isotropic
    kinh điển, nghiệm là phân bổ đều)."""
    N = 6
    mu = np.zeros(N)
    sigma = np.eye(N)
    sigma_sqrt = np.eye(N)
    result = solve_long_only(mu, sigma, sigma_sqrt, kappa=0.0, gamma=1.0)
    assert np.allclose(result.w, np.full(N, 1.0 / N), atol=1e-3)
    assert abs(result.w.sum() - 1.0) < 1e-9
    assert (result.w >= -1e-9).all()


def test_solve_long_only_feasible_general_case():
    rng = np.random.default_rng(2)
    N = 8
    A = rng.normal(0, 1, size=(N, N))
    sigma = A @ A.T / N + 0.1 * np.eye(N)
    sigma_sqrt = matrix_sqrt_psd(sigma)
    mu = rng.normal(0, 0.001, size=N)
    result = solve_long_only(mu, sigma, sigma_sqrt, kappa=1.0, gamma=5.0)
    assert (result.w >= -1e-8).all()
    assert abs(result.w.sum() - 1.0) < 1e-6
    assert not np.isnan(result.obj_history).any()


def test_solve_long_only_returns_best_not_last():
    rng = np.random.default_rng(3)
    N = 5
    A = rng.normal(0, 1, size=(N, N))
    sigma = A @ A.T / N + 0.1 * np.eye(N)
    sigma_sqrt = matrix_sqrt_psd(sigma)
    mu = rng.normal(0, 0.001, size=N)
    result = solve_long_only(mu, sigma, sigma_sqrt, kappa=0.5, gamma=2.0)
    f_w = portfolio_objective_long_only(result.w, mu, sigma, sigma_sqrt, 0.5, 2.0)
    assert abs(f_w - result.best_obj) < 1e-6
