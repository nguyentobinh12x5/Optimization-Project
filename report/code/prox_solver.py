"""
src/prox_solver.py
====================

HAND-WRITTEN (pure numpy) proximal-subgradient solver for the sparse+robust
portfolio optimization problem:

    min_w  f(w) = -mu^T w + kappa * ||Sigma^(1/2) w||_2 + gamma * w^T Sigma w
                  + lam * ||w||_1
    s.t.   1^T w = 1

w has NO non-negativity constraint (short-selling is allowed). Does NOT use
cvxpy, scipy.optimize, sklearn, or any off-the-shelf solver -- the entire
algorithm is implemented in pure numpy per the approved design (Gate 2).

The intended reader is an OPTIMIZATION person. Phase 4 cross-verifies the
result against CVXPY; the FIX version (see the "FIX-ROUND: joint prox"
section below) fixed a prox-then-project bug in the first version (caught
by the CVXPY cross-check, which broke sparsity) using an exact joint prox
via bisection -- the comparison numbers against CVXPY after the fix are in
task-3-report.md.

---------------------------------------------------------------------------
Subgradient formula (the key point, easy to get wrong)
---------------------------------------------------------------------------
f(w) has 3 components that are non-smooth / not directly differentiable:
the robust term `kappa*||Sigma^(1/2) w||_2` (not differentiable at w=0, but
usually w != 0 so it is differentiable almost everywhere) and the term
`lam*||w||_1` (not differentiable at w_i=0, handled separately via prox,
see below). The rest (`-mu^T w + gamma*w^T Sigma w`) is ordinary smooth
differentiable.

Derivative of `||Sigma^(1/2) w||_2` with respect to w: let u = Sigma^(1/2)
w. With g(w) = ||u||_2 = sqrt(u^T u) = sqrt(w^T Sigma^(1/2) Sigma^(1/2) w) =
sqrt(w^T Sigma w) (since Sigma^(1/2) is symmetric, so (Sigma^(1/2))^T
Sigma^(1/2) = Sigma^(1/2) Sigma^(1/2) = Sigma). Therefore:

    d/dw ||u||_2 = (Sigma w) / ||u||_2 = (Sigma w) / ||Sigma^(1/2) w||_2

IMPORTANT: the numerator is `Sigma @ w` (using the full Sigma), NOT
`Sigma^(1/2) @ w` -- confusing the two is a common mistake. Reason: the
chain rule through u = Sigma^(1/2) w gives Jacobian (Sigma^(1/2))^T =
Sigma^(1/2) (since it is symmetric), then (Sigma^(1/2))^T @ u =
Sigma^(1/2) @ (Sigma^(1/2) @ w) = Sigma @ w.

When ||Sigma^(1/2) w||_2 <= eps (w near 0 in the Sigma^(1/2) norm, which in
theory is only exactly true algebraically if w=0, since Sigma^(1/2) is
PSD), the function is not differentiable -- we choose subgradient = 0 (a
valid element of the subdifferential set at the singular point, since 0
always belongs to the subdifferential of a norm at the origin). This
avoids division by 0 / NaN. See `_robust_subgrad`.

---------------------------------------------------------------------------
Algorithm per iteration k (approved, FIX version after CVXPY cross-check --
see "FIX-ROUND: joint prox" below for the reason steps 3-4 changed)
---------------------------------------------------------------------------
1. Subgradient of the smooth + robust part (NOT including L1 -- L1 is
   handled via prox):
       v = -mu + 2*gamma*Sigma@w_k + robust_subgrad(w_k)
2. Descent step along the subgradient: z = w_k - alpha_k * v
3. JOINT PROX of (L1 + the indicator of the constraint {1^T w = 1}) at z,
   threshold t = alpha_k * lam -- A SINGLE step, see `_prox_l1_simplex_eq`:
       w_{k+1} = _prox_l1_simplex_eq(z, t)

Steps 2-3 together form a standard "proximal-subgradient step": a
subgradient step for the smooth+robust part, followed by the EXACT prox
(not decoupled) of the remaining non-smooth part (L1 + the affine
constraint, simultaneously). See the proof + derivation in "FIX-ROUND:
joint prox" below.

Decreasing step size: alpha_k = alpha0 / sqrt(k+1) -- the classic choice
for the subgradient method (ensures sum(alpha_k)=inf, sum(alpha_k^2)<inf,
a sufficient condition for convergence of the subgradient method on a
convex function).

---------------------------------------------------------------------------
FIX-ROUND: joint prox of lam*||w||_1 + I{1^T w=1} (replacing the decoupled
prox-then-project of the first version)
---------------------------------------------------------------------------
FIRST VERSION (bug caught by the controller/CVXPY cross-check):
soft-threshold L1, then PROJECT SEPARATELY onto the hyperplane by adding
an equal offset `(1-sum(w_half))/N` to EVERY coordinate. This offset gets
added to coordinates that were JUST soft-thresholded to exactly 0 as well
-- "reviving" every zero coordinate into a small identical value, breaking
true sparsity (e.g. on real data: active=48 while CVXPY active=9 for
kappa=0,gamma=5,lam=0.1; objective gap 2.7-6%). This is NOT the true prox
of (L1 + the affine constraint) -- it is just L1 prox followed by a
decoupled Euclidean projection, two operations that do NOT commute with
each other.

FIX VERSION: directly solve the EXACT prox of `lam*||w||_1 + I{1^T w=1}`
at point z (the result after the subgradient step), i.e.:

    w* = argmin_w  (1/2)||w - z||_2^2 + t*||w||_1   s.t.  1^T w = 1,
    with t = alpha_k * lam (the standard Moreau envelope scale: choosing
    t = alpha_k * lam makes this prox match exactly the role of the old
    step 3 in the proximal-subgradient scheme, i.e. it is still equivalent
    to "soft-threshold at threshold alpha_k*lam" when the constraint is
    dropped).

Introduce the constraint via a Lagrange multiplier nu (scalar):
    L(w, nu) = (1/2)||w-z||_2^2 + t*||w||_1 + nu*(1^T w - 1)
Separating by coordinate i (the problem fully decouples over i for a
fixed nu), minimize:
    (1/2)(w_i - z_i)^2 + nu*w_i + t*|w_i|
Complete the square: (1/2)(w_i-z_i)^2 + nu*w_i
    = (1/2)(w_i - (z_i - nu))^2 + a constant (not depending on w_i)
=> the per-coordinate problem becomes a standard L1 prox at the shifted
point (z_i - nu), threshold t:
    w_i(nu) = soft_threshold(z_i - nu, t)
            = sign(z_i - nu) * max(|z_i - nu| - t, 0)

(Sign convention: setting nu' = -nu, rewrite as w_i =
soft_threshold(z_i + nu', t) -- this is exactly the formula in
`_prox_l1_simplex_eq`, where the code variable `nu` corresponds to nu'
here; the sign of nu does not matter, it is just the variable to be found
via bisection.)

Choose nu (or nu') so that the constraint 1^T w = 1 is satisfied:
    g(nu) := sum_i soft_threshold(z_i + nu, t) = 1

g(nu) is a continuous, NON-DECREASING function of nu (each
soft_threshold(.,t) term is non-decreasing in its argument, and the
argument increases linearly in nu with coefficient 1) and unbounded in
both directions (nu -> -inf => g -> -inf, nu -> +inf => g -> +inf) => a nu*
exists (possibly not unique if g is exactly flat at 1, but any nu on that
flat segment gives a valid solution w) such that g(nu*) = 1, found via
BISECTION:
1. Bracket: start with [nu_lo, nu_hi] = [-1, 1], double nu_lo (if
   g(nu_lo) > 1) or nu_hi (if g(nu_hi) < 1) until the bracket points the
   right way.
2. Bisect for ~100 rounds or until |g(nu_mid) - 1| < 1e-12.
3. w = soft_threshold(z + nu*, t) -- the final vector, with coordinates
   EXACTLY = 0 for every i satisfying |z_i + nu*| <= t, and sum(w) = 1
   (up to bisection tolerance).

This is the EXACT prox (not a heuristic) of (L1 + the affine constraint)
at z -- the difference from the first version is that the offset is no
longer added equally to EVERY coordinate, but instead shifts the ARGUMENT
before soft-thresholding, so a coordinate already thresholded to 0 (where
|z_i+nu*| <= t) STAYS 0 after adding the offset -- no "revival". See
`_prox_l1_simplex_eq` for the implementation, and the "Cross-verify
against CVXPY after the fix" section in task-3-report.md for real numbers
confirming the significant improvement in active/gap compared to the
first version.

---------------------------------------------------------------------------
Why return the BEST-ITERATE, not the last iterate
---------------------------------------------------------------------------
The subgradient method (unlike the gradient method on a smooth function)
does NOT guarantee f(w_{k+1}) <= f(w_k) -- the sequence f(w_k) can
oscillate non-monotonically, especially in the early rounds when alpha_k
is still large. This is an inherent MATHEMATICAL property of the
subgradient method, not an implementation bug. We therefore track
best_obj = min_{j<=k} f(w_j) and return the corresponding best_w, instead
of the final w -- this is standard practice when using the subgradient
method (see Boyd, "Subgradient Methods" notes).

---------------------------------------------------------------------------
Choice of the default alpha0 (RE-TUNED after the joint prox FIX)
---------------------------------------------------------------------------
Daily returns are very small: mu ~ 1e-3, Sigma ~ 1e-4 -> the "smooth" part
of the subgradient (-mu + 2*gamma*Sigma@w) has magnitude ~ 1e-3..1e-2
depending on gamma, while the robust term kappa*Sigma@w/||.|| has
magnitude ~ kappa (order 1, since it is already divided by the norm).

With the OLD prox-then-project version, a large alpha0 caused sparsity to
be "revived" by the offset (see the history in the "FIX-ROUND" section
above) -- so previously alpha0 had to be chosen SMALL (0.1) to preserve
sparsity, trading off against slower convergence/worse best_obj. With the
joint prox (FIX version), that problem is GONE: a coordinate thresholded
to 0 stays 0 regardless of how large alpha0 is, so there is NO LONGER a
trade-off between sparsity and convergence speed -- a larger alpha0 both
converges faster (patience triggers sooner) and gives a lower (or equal)
best_obj, as well as equal or BETTER sparsity.

Real experiments on data/returns.parquet (97 assets, max_iter=20000, see
task-3-report.md section "Fix-round: re-tuning alpha0") with alpha0 in
{0.01, 0.05, 0.1, 0.3, 1, 3, 10, 30, 100, 300} for three parameter sets
(kappa=1,gamma=5,lam=0.01), (kappa=0,gamma=5,lam=0.1),
(kappa=1,gamma=5,lam=0.001): best_obj and active/sparsity both improved
MONOTONICALLY (or plateaued) as alpha0 increased from 0.01 -> 10, then
stabilized (alpha0=10, 30 give results nearly identical to alpha0=1..3 but
converge many times faster: a few hundred to a few thousand rounds instead
of running out max_iter). At alpha0=100 (the kappa=1,gamma=5,lam=0.001
set), the first step is too large, causing strong oscillation that traps
the best-iterate early at a POOR point (best_obj much worse, active=97) --
a false "converged" resembling the phenomenon already seen in the old
version. Hence alpha0 needs to be large enough for fast convergence but
not so large as to reach that unstable region.

CHOSEN: ALPHA0_DEFAULT = 10.0 -- sitting within the stable region (10, 30
give results nearly optimal as observed in the grid, while 100 already
became unstable for one parameter set) with a 3x safety margin before the
nearest observed instability threshold (30 -> 100). The caller can still
pass a different alpha0 if needed (e.g. a parameter set with a very
different scale from VN100 daily returns). Full numbers are in
task-3-report.md.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "SolveResult",
    "portfolio_objective",
    "solve",
    "simplex_projection",
    "portfolio_objective_long_only",
    "solve_long_only",
]

# Safe division-by-0 threshold when computing the subgradient of
# ||Sigma^(1/2) w||_2.
_EPS_NORM = 1e-12

# See "Choice of the default alpha0" in the module docstring.
ALPHA0_DEFAULT = 10.0


def portfolio_objective(
    w: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    sigma_sqrt: np.ndarray,
    kappa: float,
    gamma: float,
    lam: float,
) -> float:
    """f(w) = -mu^T w + kappa*||Sigma^(1/2) w||_2 + gamma*w^T Sigma w + lam*||w||_1.

    Returns the full value of the objective function (NOT including the
    constraint 1^T w = 1 -- the caller must ensure w is feasible on its
    own if it wants the value to be meaningful for the constrained
    problem; this function just computes f(w) at any w, including
    infeasible w, which is useful for Phase 4 when CVXPY needs to compare
    the objective value at its own solution).

    Parameters
    ----------
    w : np.ndarray, shape (N,)
    mu : np.ndarray, shape (N,)
    sigma : np.ndarray, shape (N,N)
    sigma_sqrt : np.ndarray, shape (N,N)  -- symmetric PSD square root of sigma
    kappa, gamma, lam : float, non-negative coefficients

    Returns
    -------
    float
    """
    w = np.asarray(w, dtype=np.float64)
    mean_term = -float(mu @ w)
    robust_term = kappa * float(np.linalg.norm(sigma_sqrt @ w))
    var_term = gamma * float(w @ sigma @ w)
    l1_term = lam * float(np.sum(np.abs(w)))
    return mean_term + robust_term + var_term + l1_term


def _robust_subgrad(
    w: np.ndarray,
    sigma: np.ndarray,
    sigma_sqrt: np.ndarray,
    kappa: float,
    eps: float = _EPS_NORM,
) -> np.ndarray:
    """Subgradient of kappa*||Sigma^(1/2) w||_2 with respect to w.

    = kappa * (Sigma @ w) / ||Sigma^(1/2) w||_2   if ||Sigma^(1/2) w||_2 > eps
    = 0 (vector)                                   if ||Sigma^(1/2) w||_2 <= eps

    NOTE: the numerator is Sigma @ w (the full Sigma matrix), NOT
    Sigma_sqrt @ w -- see the chain-rule explanation in the module
    docstring. The eps branch avoids division by 0 / NaN when w is near 0
    in the Sigma^(1/2) norm (choosing 0, a valid element of the
    subdifferential of the norm at the singular point).
    """
    u = sigma_sqrt @ w
    norm_u = float(np.linalg.norm(u))
    if norm_u <= eps:
        return np.zeros_like(w)
    return kappa * (sigma @ w) / norm_u


def _prox_l1_simplex_eq(
    z: np.ndarray,
    t: float,
    nu_init: float = 1.0,
    bisect_tol: float = 1e-12,
    max_bracket_expand: int = 200,
    max_bisect_iter: int = 200,
) -> np.ndarray:
    """EXACT prox of `t*||w||_1 + I{1^T w = 1}` at point z.

    Solves:
        w* = argmin_w  (1/2)||w-z||_2^2 + t*||w||_1   s.t.  1^T w = 1

    The full derivation (Lagrangian, separated by coordinate) is in the
    module docstring, section "FIX-ROUND: joint prox". Summary: with nu as
    the Lagrange multiplier (sign convention chosen so the formula is
    +nu, see the module docstring),

        w_i(nu) = soft_threshold(z_i + nu, t) = sign(z_i+nu)*max(|z_i+nu|-t, 0)

    and we need to find nu such that g(nu) := sum_i w_i(nu) = 1. g is
    continuous, NON-DECREASING in nu (each term is non-decreasing in its
    argument), unbounded in both directions -- a solution always exists,
    found via BISECTION (there is no closed form since soft_threshold is
    not globally linear due to the flat segment around 0).

    Result: a coordinate i with |z_i + nu*| <= t will be EXACTLY 0 (not
    "revived" by the constraint-projection part like the old heuristic
    prox-then-project version -- see the historical CAVEAT in the module
    docstring), and sum(w) = 1 up to `bisect_tol`.

    Parameters
    ----------
    z : np.ndarray, shape (N,)
        The point to take the prox of (the result after the subgradient
        step: z = w_k - alpha_k*v).
    t : float, >= 0
        The soft-threshold threshold (= alpha_k * lam in `solve`). t=0 ->
        the function reduces exactly to the Euclidean projection onto the
        hyperplane {1^T w=1} (no soft-thresholding), since
        soft_threshold(x, 0) = x.
    nu_init : float, default 1.0
        The initial bracket boundary [-nu_init, nu_init] before expansion.
    bisect_tol : float, default 1e-12
        The |g(nu)-1| threshold for stopping bisection.
    max_bracket_expand, max_bisect_iter : int
        Safety limits on the number of bracket-expansion / bisection
        rounds (avoids an infinite loop in numerically edge-case
        situations; theoretically it always converges).

    Returns
    -------
    np.ndarray, shape (N,), satisfying sum(w) ~= 1 (up to bisect_tol) and
    possibly having coordinates equal to exactly 0.
    """
    z = np.asarray(z, dtype=np.float64)

    def g(nu: float) -> float:
        x = z + nu
        return float(np.sum(np.sign(x) * np.maximum(np.abs(x) - t, 0.0)))

    nu_lo, nu_hi = -nu_init, nu_init
    expand = 0
    while g(nu_lo) > 1.0 and expand < max_bracket_expand:
        nu_lo *= 2.0
        expand += 1
    expand = 0
    while g(nu_hi) < 1.0 and expand < max_bracket_expand:
        nu_hi *= 2.0
        expand += 1

    nu_mid = 0.5 * (nu_lo + nu_hi)
    for _ in range(max_bisect_iter):
        nu_mid = 0.5 * (nu_lo + nu_hi)
        g_mid = g(nu_mid)
        if abs(g_mid - 1.0) < bisect_tol:
            break
        if g_mid < 1.0:
            nu_lo = nu_mid
        else:
            nu_hi = nu_mid

    x = z + nu_mid
    return np.sign(x) * np.maximum(np.abs(x) - t, 0.0)


def _smooth_subgrad(
    w: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    sigma_sqrt: np.ndarray,
    kappa: float,
    gamma: float,
    eps: float = _EPS_NORM,
) -> np.ndarray:
    """v = -mu + 2*gamma*Sigma@w + robust_subgrad(w) (NOT including the L1
    part, L1 is handled separately via the joint prox `_prox_l1_simplex_eq`
    in `solve`)."""
    return -mu + 2.0 * gamma * (sigma @ w) + _robust_subgrad(w, sigma, sigma_sqrt, kappa, eps)


@dataclass
class SolveResult:
    """Result returned by `solve`.

    w : np.ndarray, shape (N,)
        The BEST-ITERATE solution (NOT the final iterate -- see the
        module docstring "Why return the best-iterate").
    obj_history : np.ndarray, shape (n_iter,)
        f(w_k) at EVERY round actually run (w_k is already feasible since
        it has already been projected onto the hyperplane), in time
        order -- NOT a running-min.
    best_obj : float
        min(obj_history) = f(w) at the returned w.
    n_iter : int
        Number of rounds actually run (<= max_iter).
    converged : bool
        True if it stopped because the relative change of best_obj < tol
        for `patience` consecutive rounds; False if it stopped by hitting
        max_iter.
    """

    w: np.ndarray
    obj_history: np.ndarray
    best_obj: float
    n_iter: int
    converged: bool


def solve(
    mu: np.ndarray,
    sigma: np.ndarray,
    sigma_sqrt: np.ndarray,
    kappa: float,
    gamma: float,
    lam: float,
    *,
    max_iter: int = 5000,
    alpha0: float = ALPHA0_DEFAULT,
    tol: float = 1e-8,
    patience: int = 100,
    w0: np.ndarray | None = None,
) -> SolveResult:
    """Solve min_w f(w) s.t. 1^T w = 1 with the hand-written
    proximal-subgradient method.

    See the module docstring for the full algorithm formula, the
    derivation of the robust term, the reason for returning the
    best-iterate, and the "FIX-ROUND: joint prox" section (the L1 +
    affine-constraint step is solved EXACTLY via bisection, no longer the
    heuristic prox-then-project of the first version).

    Parameters
    ----------
    mu, sigma, sigma_sqrt : np.ndarray
        See `portfolio_objective`.
    kappa, gamma, lam : float
        Non-negative coefficients for the robust / variance / L1 terms.
    max_iter : int, default 5000
    alpha0 : float, default ALPHA0_DEFAULT (=10.0, see the module docstring)
        alpha_k = alpha0 / sqrt(k+1).
    tol : float, default 1e-8
        Relative-change threshold of best_obj to consider it "stabilized".
    patience : int, default 100
        Number of consecutive rounds with relative-change < tol before
        stopping early (converged=True).
    w0 : np.ndarray | None, default None
        Starting point; None -> the uniform vector 1/N (feasible: sum=1).

    Returns
    -------
    SolveResult
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    sigma_sqrt = np.asarray(sigma_sqrt, dtype=np.float64)
    n = mu.shape[0]

    if w0 is None:
        w = np.full(n, 1.0 / n, dtype=np.float64)
    else:
        w = np.asarray(w0, dtype=np.float64).copy()

    obj_hist = np.empty(max_iter, dtype=np.float64)

    best_obj = portfolio_objective(w, mu, sigma, sigma_sqrt, kappa, gamma, lam)
    best_w = w.copy()
    prev_best = best_obj
    stall_count = 0
    converged = False
    n_iter = 0

    for k in range(max_iter):
        alpha_k = alpha0 / np.sqrt(k + 1)

        v = _smooth_subgrad(w, mu, sigma, sigma_sqrt, kappa, gamma)
        z = w - alpha_k * v

        # EXACT joint prox of (lam*||.||_1 + I{1^T w=1}) at z, threshold
        # alpha_k*lam -- see the module docstring "FIX-ROUND: joint prox"
        # and `_prox_l1_simplex_eq`. Replaces the old version
        # (soft-threshold then decoupled projection) which broke sparsity
        # due to the offset being added equally to every coordinate.
        w = _prox_l1_simplex_eq(z, alpha_k * lam)

        f_w = portfolio_objective(w, mu, sigma, sigma_sqrt, kappa, gamma, lam)
        obj_hist[k] = f_w
        n_iter = k + 1

        if f_w < best_obj:
            best_obj = f_w
            best_w = w.copy()

        denom = max(abs(prev_best), 1e-300)
        rel_change = abs(prev_best - best_obj) / denom
        prev_best = best_obj

        if rel_change < tol:
            stall_count += 1
        else:
            stall_count = 0

        if stall_count >= patience:
            converged = True
            break

    return SolveResult(
        w=best_w,
        obj_history=obj_hist[:n_iter],
        best_obj=best_obj,
        n_iter=n_iter,
        converged=converged,
    )


# ---------------------------------------------------------------------------
# Walk-forward Task 1: long-only branch (w >= 0, sum(w) = 1)
# ---------------------------------------------------------------------------
# Under the long-only constraint, ||w||_1 = sum_i |w_i| = sum_i w_i = 1
# (since every w_i >= 0) IS A CONSTANT over the entire feasible domain --
# the penalty lam*||w||_1 no longer has any effect on controlling the
# solution (it just adds exactly "lam" to the objective, a constant not
# depending on w) so it is dropped entirely from this branch (there is no
# lam parameter in `portfolio_objective_long_only` / `solve_long_only`).
# See docs/superpowers/specs/2026-07-26-walk-forward-backtest-design.md
# section 2.
#
# The "smooth + robust" part of the subgradient (-mu + 2*gamma*Sigma@w +
# robust_subgrad(w)) KEEPS the SAME formula as the long-short branch above
# -- directly reuses `_smooth_subgrad` (not modified, not re-copied) to
# avoid drifting out of sync with `solve()` if that formula changes in the
# future. The ONLY difference from `solve()`: the "prox" step of (L1 +
# affine constraint) is replaced by a EUCLIDEAN PROJECTION onto the
# simplex `{w>=0, sum(w)=1}` (`simplex_projection`, the Duchi et al. 2008
# algorithm) -- since there is no more L1, there is no need for the
# bisection-based joint-prox like `_prox_l1_simplex_eq`.
# ---------------------------------------------------------------------------


def simplex_projection(v: np.ndarray) -> np.ndarray:
    """Euclidean projection of vector v onto the probability simplex
    {w : w>=0, sum(w)=1}.

    The Duchi, Shalev-Shwartz, Singer, Chandra (2008) algorithm "Efficient
    Projections onto the l1-Ball for Learning in High Dimensions",
    O(N log N): sort v in descending order into u; find rho = the largest
    index j such that u_j - (cumsum(u)_j - 1)/j > 0; theta =
    (cumsum(u)_rho - 1)/rho; w = max(v - theta, 0). The result ALWAYS
    satisfies w>=0 and sum(w)=1 (up to numerical error), regardless of
    what v is -- v does not need to already be "close" to the simplex.

    Parameters
    ----------
    v : np.ndarray, shape (N,)

    Returns
    -------
    np.ndarray, shape (N,), satisfying w>=0 (up to numerical error) and
    sum(w)=1.
    """
    v = np.asarray(v, dtype=np.float64)
    n = v.shape[0]
    u = np.sort(v)[::-1]
    cumsum_u = np.cumsum(u)
    j = np.arange(1, n + 1)
    cond = u - (cumsum_u - 1) / j > 0
    rho = np.nonzero(cond)[0][-1]  # 0-based index of the largest j satisfying cond
    theta = (cumsum_u[rho] - 1) / (rho + 1)
    return np.maximum(v - theta, 0.0)


def portfolio_objective_long_only(
    w: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    sigma_sqrt: np.ndarray,
    kappa: float,
    gamma: float,
) -> float:
    """f(w) = -mu^T w + kappa*||Sigma^(1/2) w||_2 + gamma*w^T Sigma w.

    Same as `portfolio_objective` but WITHOUT the `lam*||w||_1` term --
    under the long-only constraint (w>=0, sum(w)=1), ||w||_1 = sum(w) = 1
    is a constant, so the L1 penalty no longer has any effect on
    controlling the solution and is dropped (see the docstring in the
    "Walk-forward Task 1" block above).

    Parameters
    ----------
    w, mu, sigma, sigma_sqrt : np.ndarray -- see `portfolio_objective`.
    kappa, gamma : float, non-negative coefficients.

    Returns
    -------
    float
    """
    w = np.asarray(w, dtype=np.float64)
    mean_term = -float(mu @ w)
    robust_term = kappa * float(np.linalg.norm(sigma_sqrt @ w))
    var_term = gamma * float(w @ sigma @ w)
    return mean_term + robust_term + var_term


def solve_long_only(
    mu: np.ndarray,
    sigma: np.ndarray,
    sigma_sqrt: np.ndarray,
    kappa: float,
    gamma: float,
    *,
    max_iter: int = 5000,
    alpha0: float = ALPHA0_DEFAULT,
    tol: float = 1e-8,
    patience: int = 100,
    w0: np.ndarray | None = None,
) -> SolveResult:
    """Solve min_w f(w) s.t. w>=0, sum(w)=1 with the hand-written
    projected-subgradient method.

    Same iteration scheme as `solve()` (subgradient step on the
    smooth+robust part via `_smooth_subgrad`, NOT modified/re-copied, then
    the "prox" step), differing only in the prox step: here there is no
    more L1 term (see the "Walk-forward Task 1" docstring block above) so
    the prox step reduces to a Euclidean projection onto the simplex via
    `simplex_projection` (Duchi et al. 2008), replacing
    `_prox_l1_simplex_eq` of the long-short branch. Same reason for
    returning the BEST-ITERATE (the subgradient method does not guarantee
    monotonicity) as `solve()` -- see the module docstring section "Why
    return the BEST-ITERATE".

    Parameters
    ----------
    mu, sigma, sigma_sqrt : np.ndarray
        See `portfolio_objective`.
    kappa, gamma : float, non-negative coefficients.
    max_iter : int, default 5000
    alpha0 : float, default ALPHA0_DEFAULT
        alpha_k = alpha0 / sqrt(k+1).
    tol : float, default 1e-8
        Relative-change threshold of best_obj to consider it "stabilized".
    patience : int, default 100
        Number of consecutive rounds with relative-change < tol before
        stopping early (converged=True).
    w0 : np.ndarray | None, default None
        Starting point; None -> the uniform vector 1/N (feasible: w>=0,
        sum=1).

    Returns
    -------
    SolveResult
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    sigma_sqrt = np.asarray(sigma_sqrt, dtype=np.float64)
    n = mu.shape[0]

    if w0 is None:
        w = np.full(n, 1.0 / n, dtype=np.float64)
    else:
        w = np.asarray(w0, dtype=np.float64).copy()

    obj_hist = np.empty(max_iter, dtype=np.float64)

    best_obj = portfolio_objective_long_only(w, mu, sigma, sigma_sqrt, kappa, gamma)
    best_w = w.copy()
    prev_best = best_obj
    stall_count = 0
    converged = False
    n_iter = 0

    for k in range(max_iter):
        alpha_k = alpha0 / np.sqrt(k + 1)

        v = _smooth_subgrad(w, mu, sigma, sigma_sqrt, kappa, gamma)
        z = w - alpha_k * v

        # Euclidean projection onto the simplex {w>=0, sum(w)=1} --
        # replaces the joint prox (L1 + hyperplane) of solve(), since the
        # long-only branch has no L1.
        w = simplex_projection(z)

        f_w = portfolio_objective_long_only(w, mu, sigma, sigma_sqrt, kappa, gamma)
        obj_hist[k] = f_w
        n_iter = k + 1

        if f_w < best_obj:
            best_obj = f_w
            best_w = w.copy()

        denom = max(abs(prev_best), 1e-300)
        rel_change = abs(prev_best - best_obj) / denom
        prev_best = best_obj

        if rel_change < tol:
            stall_count += 1
        else:
            stall_count = 0

        if stall_count >= patience:
            converged = True
            break

    return SolveResult(
        w=best_w,
        obj_history=obj_hist[:n_iter],
        best_obj=best_obj,
        n_iter=n_iter,
        converged=converged,
    )
