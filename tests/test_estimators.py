"""
tests/test_estimators.py
=========================

Test cho src/estimators.py, dùng dữ liệu GIẢ LẬP với seed cố định (KHÔNG phụ
thuộc data thật ở data/returns.parquet), để test chạy độc lập, không cần
mạng, ổn định qua thời gian.

Chạy: `.venv/bin/python -m pytest tests/test_estimators.py -v`
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.estimators import (
    estimate_all,
    estimate_mu,
    estimate_sigma,
    matrix_sqrt_psd,
)


@pytest.fixture
def fake_returns() -> pd.DataFrame:
    """DataFrame returns giả lập N=5 (cột 'A'..'E'), T=200, seed cố định."""
    rng = np.random.default_rng(0)
    N, T = 5, 200
    data = rng.normal(loc=0.0005, scale=0.02, size=(T, N))
    cols = list("ABCDE")
    return pd.DataFrame(data, columns=cols)


def test_mu_shape_and_value(fake_returns: pd.DataFrame) -> None:
    mu = estimate_mu(fake_returns)
    assert mu.shape == (5,)
    assert np.allclose(mu, fake_returns.mean().values)


def test_sigma_symmetric(fake_returns: pd.DataFrame) -> None:
    S = estimate_sigma(fake_returns)
    assert np.allclose(S, S.T, atol=1e-12)


def test_sigma_psd(fake_returns: pd.DataFrame) -> None:
    S = estimate_sigma(fake_returns)
    eigvals = np.linalg.eigvalsh(S)
    assert np.all(eigvals >= -1e-8)


def test_sqrt_reconstructs(fake_returns: pd.DataFrame) -> None:
    S = estimate_sigma(fake_returns)
    A = matrix_sqrt_psd(S)
    recon = A @ A
    rel_err = np.linalg.norm(recon - S, "fro") / np.linalg.norm(S, "fro")
    assert rel_err < 1e-6


def test_sqrt_symmetric(fake_returns: pd.DataFrame) -> None:
    S = estimate_sigma(fake_returns)
    A = matrix_sqrt_psd(S)
    assert np.allclose(A, A.T, atol=1e-10)


def test_shrinkage_between(fake_returns: pd.DataFrame) -> None:
    S = estimate_sigma(fake_returns, shrinkage=None)

    # shrinkage = "lw"
    S_lw = estimate_sigma(fake_returns, shrinkage="lw")
    assert np.allclose(S_lw, S_lw.T, atol=1e-12)
    eigvals_lw = np.linalg.eigvalsh(S_lw)
    assert np.all(eigvals_lw >= -1e-8)

    # shrinkage = 0.5 (float delta) -- kiểm tra nằm "giữa" S và target theo
    # đường chéo: target F = mean_var * I, nên diag(S_shrunk) phải nằm giữa
    # diag(S) và mean_var, theo đúng hướng nội suy tuyến tính.
    delta = 0.5
    S_half = estimate_sigma(fake_returns, shrinkage=delta)
    assert np.allclose(S_half, S_half.T, atol=1e-12)
    eigvals_half = np.linalg.eigvalsh(S_half)
    assert np.all(eigvals_half >= -1e-8)

    mean_var = np.trace(S) / S.shape[0]
    expected_diag = (1 - delta) * np.diag(S) + delta * mean_var
    assert np.allclose(np.diag(S_half), expected_diag)

    # delta=0 phải khớp sample covariance thuần; delta=1 phải khớp target.
    S_zero = estimate_sigma(fake_returns, shrinkage=0.0)
    assert np.allclose(S_zero, S)
    S_one = estimate_sigma(fake_returns, shrinkage=1.0)
    n = S.shape[0]
    target = mean_var * np.eye(n)
    assert np.allclose(S_one, target)


def test_sqrt_on_psd_singular() -> None:
    """Ma trận PSD suy biến (rank-deficient): X có cột phụ thuộc tuyến tính
    => X.T @ X có eigenvalue 0 (không PD). matrix_sqrt_psd không được ném lỗi
    và phải trả ma trận thực, không NaN."""
    rng = np.random.default_rng(1)
    T, N = 50, 6
    X = rng.normal(size=(T, N - 1))
    # Cột cuối = tổ hợp tuyến tính của 2 cột đầu -> rank-deficient.
    dependent_col = 2.0 * X[:, 0] - 0.5 * X[:, 1]
    X_full = np.column_stack([X, dependent_col])
    S_singular = X_full.T @ X_full  # (N, N), PSD, rank <= N-1 (suy biến)

    A = matrix_sqrt_psd(S_singular)

    assert A.shape == (N, N)
    assert not np.any(np.isnan(A))
    assert np.isrealobj(A)
    assert np.allclose(A, A.T, atol=1e-8)
    recon = A @ A
    rel_err = np.linalg.norm(recon - S_singular, "fro") / (
        np.linalg.norm(S_singular, "fro") + 1e-15
    )
    assert rel_err < 1e-6


def test_estimate_all_shapes_and_order(fake_returns: pd.DataFrame) -> None:
    """Bonus: kiểm tra estimate_all trả đúng shape và thứ tự asset khớp
    returns.columns."""
    mu, sigma, sigma_sqrt = estimate_all(fake_returns)
    n = fake_returns.shape[1]
    assert mu.shape == (n,)
    assert sigma.shape == (n, n)
    assert sigma_sqrt.shape == (n, n)
    assert np.allclose(mu, fake_returns.mean().values)
