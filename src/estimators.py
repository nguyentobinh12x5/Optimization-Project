"""
src/estimators.py
==================

Statistical estimators for the sparse+robust portfolio optimization
problem (VN100).

This module turns the simple daily returns matrix (output of
`src/data_loader.py`, see `data/returns.parquet`) into 3 linear-algebra
objects that the solver in Phase 3 (proximal-subgradient) and the verify
step in Phase 4 (CVXPY) consume directly:

- mu_hat     (N,)   : estimated vector of DAILY expected returns (not
                       annualized).
- Sigma      (N,N)  : estimated DAILY return covariance matrix, symmetric,
                       PSD (up to numerical error).
- Sigma_sqrt (N,N)  : symmetric PSD square root of Sigma, i.e. a matrix A
                       such that A @ A ~= Sigma (A is not necessarily the
                       Cholesky factor -- see the explanation below).

The intended reader is an OPTIMIZATION person, not necessarily a finance
specialist. Units: everything here stays in DAILY units, annualizing (if
needed, usually mu*252, Sigma*252) is left to the later reporting/analysis
phases, and is NOT done in this module, to avoid confusion about "already
annualized or not" when the solver consumes the functions below directly.

ASSET ORDERING CONVENTION: every returned vector/matrix has row/column
order MATCHING `list(returns.columns)` of the input DataFrame, in the
order it appears (not re-sorted). The caller (solver, verify) must keep
its own index <-> symbol mapping if it needs to look up a symbol by name.

Why use eigendecomposition (np.linalg.eigh) instead of Cholesky for the
matrix square root?
- Cholesky (A = L L^T) requires the matrix to be strictly positive
  DEFINITE (PD), i.e. every eigenvalue > 0 strictly. Sigma estimated from
  real data (especially when N is close to or larger than T, or N is
  large relative to the number of independent observations) can be only
  positive SEMI-definite (PSD) -- with an eigenvalue equal to zero or a
  very small negative value due to floating-point rounding error. Cholesky
  would RAISE AN ERROR (or return NaN) in those cases.
- eigh (specialized for symmetric/Hermitian matrices) always returns real
  eigenvalues + orthogonal eigenvectors for ANY symmetric matrix, even a
  degenerate (rank-deficient) one. We just need to CLIP the negative
  eigenvalues (due to numerical rounding error) to 0 before taking the
  square root, then reconstruct:
      Sigma = V diag(lambda) V^T
      Sigma^(1/2) := V diag(sqrt(clip(lambda, 0, inf))) V^T
  This is exactly the symmetric PSD square root -- different from the
  Cholesky factor L (lower triangular, not symmetric) but still satisfies
  Sigma^(1/2) @ Sigma^(1/2) = Sigma, and has the additional useful
  property of being symmetric + PSD for the solver in Phase 3 (e.g. when
  it needs to project/scale by Sigma^(1/2) while still preserving
  symmetry).

Ledoit-Wolf shrinkage: sklearn is not available in this environment, so
the formula was implemented by hand with numpy (see `_ledoit_wolf_delta`
below), following Ledoit & Wolf (2004) "Honey, I Shrunk the Sample
Covariance Matrix" (the target is F = mean(diag(S)) * I -- shrinkage
towards a constant-diagonal matrix, the constant-variance-identity
version, NOT the more complex constant-correlation version in the
original paper; this is a reasonable simplification, documented here
explicitly rather than hidden in the code).

Quick run on real data: `.venv/bin/python -m src.estimators`
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
    """Estimate the expected return vector (mu_hat) via the column-wise
    sample mean.

    mu_hat[i] = mean_t(returns[t, i]), DAILY units (not annualized).

    Parameters
    ----------
    returns : pd.DataFrame, shape (T, N)
        Simple daily returns, index = date, columns = symbol. The output's
        asset order = list(returns.columns), in the exact column order of
        the DataFrame (not re-sorted).

    Returns
    -------
    np.ndarray, shape (N,), dtype float64.
    """
    return returns.mean(axis=0).to_numpy(dtype=np.float64)


def _mean_var_target(S: np.ndarray) -> np.ndarray:
    """Target F = mean(diag(S)) * I for Ledoit-Wolf / linear shrinkage.

    This is the "constant variance, zero correlation" target: a diagonal
    matrix with variance equal to the average of the sample variances on
    the diagonal of S, and off-diagonal covariances equal to 0.
    """
    n = S.shape[0]
    mean_var = np.trace(S) / n
    return mean_var * np.eye(n)


def _ledoit_wolf_delta(X: np.ndarray, S: np.ndarray) -> float:
    """Compute the shrinkage coefficient delta using the Ledoit-Wolf (2004)
    formula, target F = mean_var * I (constant-variance-identity target).

    X: demeaned observation matrix, shape (T, N) (each row is 1
       observation, column means already subtracted).
    S: sample covariance corresponding to X (ddof=1), shape (N, N).

    Formula (condensed from Ledoit & Wolf 2004, Sec 2, for a constant * I
    target):
        pi_hat   = sum_{i,j} mean_t[ (x_ti x_tj - S_ij)^2 ]   (estimate of
                   "phi" -- the variance of the elements of S)
        rho_hat  = 0 (since the target's off-diagonal = 0, the on-diagonal
                   elements are not mutually dependent in this
                   approximation -- see note below)
        gamma_hat = || S - F ||_F^2   (squared Frobenius distance between
                    the sample covariance and the target)
        kappa_hat = (pi_hat - rho_hat) / gamma_hat
        delta     = clip(kappa_hat / T, 0, 1)

    This is the standard LW shrinkage-intensity formula for a
    single-factor / identity target, simplified (rho_hat = 0) since the
    target has no off-diagonal component to "co-shrink" against -- a
    reasonable and common approximation in condensed LW implementations
    (documented here explicitly per the brief's requirement, rather than
    implementing the much more complex full constant-correlation
    version).
    """
    T, n = X.shape
    F = _mean_var_target(S)

    # pi_hat: estimate of the sum of asymptotic variances of the elements
    # of S_ij, using the sample formula: pi_hat = (1/T) * sum_t
    # || x_t x_t^T - S ||_F^2 (summed over every i,j of each element's
    # variance).
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
    """Estimate the covariance matrix (Sigma) of returns.

    The default sample covariance uses pandas' `returns.cov()`, i.e.
    ddof=1 (divide by T-1, unbiased estimator) -- NOT ddof=0.

    Parameters
    ----------
    returns : pd.DataFrame, shape (T, N)
        Simple daily returns. The output's asset order =
        list(returns.columns).
    shrinkage : float | str | None, default None
        - None: return the plain sample covariance (no shrinkage).
        - "lw" or "ledoit-wolf": automatically compute the shrinkage
          coefficient delta using the Ledoit-Wolf (2004) formula, target
          F = mean(diag(S)) * I (see `_ledoit_wolf_delta`), then return
          (1-delta)*S + delta*F.
        - a float in [0, 1]: used directly as delta with the same target F.

    Returns
    -------
    np.ndarray, shape (N, N), dtype float64, symmetric (Sigma is forced
    to (Sigma+Sigma.T)/2 before being returned).
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
                f"unrecognized shrinkage string: {shrinkage!r}; "
                "use 'lw'/'ledoit-wolf', a float in [0,1], or None."
            )
    else:
        delta = float(shrinkage)
        if not (0.0 <= delta <= 1.0):
            raise ValueError(f"shrinkage (delta) must be in [0,1], got {delta}")
        F = _mean_var_target(S)
        Sigma = (1.0 - delta) * S + delta * F

    # Force symmetry: eliminates the tiny numerical (floating-point)
    # error that can make Sigma slightly asymmetric after the operations
    # above.
    Sigma = (Sigma + Sigma.T) / 2.0
    return Sigma


def matrix_sqrt_psd(sigma: np.ndarray) -> np.ndarray:
    """Symmetric PSD square root of a symmetric PSD (or near-PSD) matrix.

    Uses symmetric eigendecomposition (np.linalg.eigh) instead of Cholesky
    because Sigma may be only PSD (not strictly PD) -- see the detailed
    explanation in the module docstring. Negative eigenvalues (due to
    floating-point rounding error, usually tiny) are CLIPPED to 0 before
    taking the square root:

        sigma = V diag(lambda) V^T          (eigh, V orthogonal, lambda real)
        sqrt  = V diag(sqrt(clip(lambda,0))) V^T

    Result satisfies: sqrt @ sqrt ~= sigma (up to numerical error), sqrt
    is symmetric, PSD.

    Parameters
    ----------
    sigma : np.ndarray, shape (N, N)
        A symmetric (or near-symmetric due to numerical error) matrix,
        PSD (or near-PSD, possibly with a few tiny negative eigenvalues
        due to numerical error).

    Returns
    -------
    np.ndarray, shape (N, N), dtype float64, symmetric, PSD.
    """
    # Force symmetry before eigh to ensure real eigenvalues/eigenvectors
    # (eigh assumes symmetry and only reads half the matrix, but we force
    # it here so the result stays stable even when the input is slightly
    # asymmetric due to numerical error).
    sym = (sigma + sigma.T) / 2.0
    eigvals, eigvecs = np.linalg.eigh(sym)
    eigvals_clipped = np.clip(eigvals, 0.0, None)
    sqrt_sigma = eigvecs @ np.diag(np.sqrt(eigvals_clipped)) @ eigvecs.T
    # Force symmetry on the result (same numerical-error reason, from the
    # matrix multiplication).
    sqrt_sigma = (sqrt_sigma + sqrt_sigma.T) / 2.0
    return sqrt_sigma


def estimate_all(
    returns: pd.DataFrame, shrinkage: float | str | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convenience wrapper: returns (mu_hat, Sigma, Sigma_sqrt) from returns.

    All 3 outputs' asset order = list(returns.columns) (exact column order
    of the input DataFrame, not re-sorted).

    Parameters
    ----------
    returns : pd.DataFrame, shape (T, N)
    shrinkage : float | str | None, default None
        See `estimate_sigma`.

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
