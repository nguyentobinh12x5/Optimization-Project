"""
tests/test_cvxpy_check.py
===========================

Test NHẸ (không bắt buộc theo brief Phase 4) cho src/cvxpy_check.py.
Dữ liệu GIẢ LẬP nhỏ, well-conditioned (N=6), seed cố định -- KHÔNG phụ
thuộc data thật (test riêng trên data thật + bảng số liệu đầy đủ nằm ở
`.venv/bin/python -m src.cvxpy_check` và `.sdd/task-4-report.md`).

Mục đích: verify nhanh interface/shape của `compare()` và `cvxpy_solve()`,
không thay thế cho verify trên data thật.

Chạy: `.venv/bin/python -m pytest tests/test_cvxpy_check.py -v`
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.cvxpy_check import compare, compare_long_only, cvxpy_solve, cvxpy_solve_long_only
from src.prox_solver import portfolio_objective


def _make_problem(seed: int = 0, n: int = 6):
    rng = np.random.default_rng(seed)
    mu = rng.normal(scale=1e-3, size=n)
    A = rng.normal(size=(n, n)) * 1e-2
    sigma = A @ A.T + 1e-6 * np.eye(n)  # PD đảm bảo, tránh nhiễu số học
    eigvals, eigvecs = np.linalg.eigh(sigma)
    sigma_sqrt = eigvecs @ np.diag(np.sqrt(np.clip(eigvals, 0.0, None))) @ eigvecs.T
    return mu, sigma, sigma_sqrt


def test_cvxpy_solve_feasible_and_matches_objective():
    """w trả về phải feasible (sum=1) và obj trả về khớp portfolio_objective."""
    mu, sigma, sigma_sqrt = _make_problem()
    w, obj = cvxpy_solve(mu, sigma, sigma_sqrt, kappa=1.0, gamma=5.0, lam=0.01)

    assert np.isclose(np.sum(w), 1.0, atol=1e-6)
    expected_obj = portfolio_objective(w, mu, sigma, sigma_sqrt, 1.0, 5.0, 0.01)
    assert np.isclose(obj, expected_obj, rtol=1e-8)


def test_compare_columns_and_small_relgap():
    """compare() trả đúng cột yêu cầu, relgap nhỏ trên bài toán nhỏ dễ hội tụ."""
    mu, sigma, sigma_sqrt = _make_problem()
    grid = [
        {"kappa": 1.0, "gamma": 5.0, "lam": 0.01},
        {"kappa": 0.0, "gamma": 5.0, "lam": 0.1},
    ]
    df = compare(mu, sigma, sigma_sqrt, grid, max_iter=20000)

    expected_cols = {
        "kappa", "gamma", "lam", "f_hand", "f_cvx", "relgap",
        "active_hand", "active_cvx", "jaccard", "linf", "n_iter", "converged",
    }
    assert expected_cols.issubset(set(df.columns))
    assert len(df) == 2
    assert isinstance(df, pd.DataFrame)
    # Solver tay không thể tốt hơn CVXPY (nghiệm tối ưu toàn cục) đáng kể --
    # cho phép sai số nhỏ do hội tụ chưa hoàn toàn (subgradient method).
    assert df["relgap"].abs().max() < 0.02  # 2%, dư so với ngưỡng production 0.5%
    assert (df["jaccard"] >= 0.0).all() and (df["jaccard"] <= 1.0).all()


def test_long_only_matches_cvxpy_on_real_data():
    """Verify solve_long_only (tay) khớp cvxpy_solve_long_only (ground-truth)
    trên data thật VN100 (965x98), shrinkage='lw' xuyên suốt (đúng convention
    backtest, xem wf-task-2-brief.md)."""
    from src.estimators import estimate_all

    returns = pd.read_parquet("data/returns.parquet")
    mu, sigma, sigma_sqrt = estimate_all(returns, shrinkage="lw")

    param_grid = [(0.0, 1.0), (0.5, 5.0), (1.0, 5.0), (2.0, 10.0)]
    df = compare_long_only(mu, sigma, sigma_sqrt, param_grid)

    assert (df["relgap"].abs() < 0.005).all(), df[["kappa", "gamma", "relgap"]]
    assert (df["w_hand"].apply(lambda w: (w >= -1e-6).all())).all()
    assert (df["w_hand"].apply(lambda w: abs(w.sum() - 1.0) < 1e-6)).all()


def test_cvxpy_solve_long_only_feasible_and_matches_objective():
    """w trả về phải feasible (w>=0, sum=1) và obj khớp portfolio_objective_long_only."""
    from src.prox_solver import portfolio_objective_long_only

    mu, sigma, sigma_sqrt = _make_problem()
    w, obj = cvxpy_solve_long_only(mu, sigma, sigma_sqrt, kappa=1.0, gamma=5.0)

    assert np.isclose(np.sum(w), 1.0, atol=1e-6)
    assert (w >= -1e-6).all()
    expected_obj = portfolio_objective_long_only(w, mu, sigma, sigma_sqrt, 1.0, 5.0)
    assert np.isclose(obj, expected_obj, rtol=1e-8)
