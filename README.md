# Sparse + Robust Portfolio Optimization — VN100

A portfolio optimization project for the **VN100** basket (98 Vietnamese stocks after data
cleaning, 4 years of daily returns) using the **sparse + robust mean-variance** formulation:

```
min_w  -μ̂ᵀw + κ‖Σ^(1/2)w‖₂ + γ·wᵀΣw + λ‖w‖₁   s.t.  1ᵀw = 1
```

which combines 4 objectives in a single convex problem: maximizing expected return,
penalizing risk in both a robust sense (ℓ2-norm of `Σ^(1/2)w`) and a classic Markowitz
sense (quadratic), and encouraging a **sparse** portfolio (few names held) via an L1
penalty — short selling is allowed (no `w ≥ 0` constraint). The main solver is a
**proximal-subgradient method written from scratch in pure numpy** (no cvxpy/
scipy.optimize/sklearn), with an **exact** prox step for `L1 + budget constraint` solved
via bisection (soft-thresholding + finding the Lagrange multiplier `ν`). Results are
**cross-verified** against CVXPY (interior-point solver CLARABEL) as an independent
"ground truth".

Beyond the in-sample analysis above, the project also includes an **out-of-sample
walk-forward backtest** (module `src/backtest.py`): a 24-month rolling window (18 months
estimation + 6 months validation), monthly rebalancing, long-only (`w≥0, Σw=1`),
automatic reselection of `(κ,γ)` each period via Sharpe validation, transaction costs, and
comparison against an equal-weight 1/N benchmark — see the "Walk-forward backtest"
section below.

See `notebook.ipynb` for the full end-to-end story (data → estimation → algorithm →
results → verification → OOS backtest → conclusion), and the `.sdd/` folder for the
design log / report of each phase.

## Directory structure

```
.
├── src/                    # Core logic (all notebook imports come from here)
│   ├── data_loader.py      # Loads + cleans VN100 data via vnstock, caches to data/
│   ├── estimators.py       # Estimates μ̂, Σ, Σ^(1/2) (eigh + clip, custom Ledoit-Wolf)
│   ├── prox_solver.py      # Custom proximal-subgradient solver (pure numpy)
│   ├── cvxpy_check.py      # Cross-verification via CVXPY (the ONLY place that imports cvxpy)
│   ├── backtest.py         # Walk-forward OOS backtest + equal-weight 1/N benchmark
│   └── viz.py               # 9 plotting functions (fig1..fig9), saved to figures/
├── tests/                  # pytest for estimators / prox_solver / cvxpy_check / backtest
├── data/                   # Cached parquet/csv (returns.parquet, prices.parquet, symbols) — gitignored
├── figures/                # 9 pre-generated PNGs (fig1_data_overview.png .. fig9_selected_params.png)
├── notebook.ipynb          # End-to-end deliverable: imports from src/, runs cleanly start to finish
├── .sdd/                   # Task briefs + reports for each phase (design log)
└── .env                    # VNSTOCK_API_KEY (not committed — already in .gitignore)
```

## Environment requirements

- Python **3.10+** (tested on 3.14.5 in `.venv/`).
- Main packages: `numpy`, `pandas`, `pyarrow` (parquet read/write), `matplotlib`, `cvxpy`,
  `pytest`, `vnstock`, `python-dotenv`, `python-dateutil`, plus the jupyter stack
  (`ipykernel`, `nbconvert`, `nbformat`) to build/run the notebook.
- A `requirements.txt` (pinned to tested versions) is provided at the project root — the
  recommended install method is `pip install -r requirements.txt`.

## How to run

```bash
# 1. Create & activate a virtualenv
python3 -m venv .venv
source .venv/bin/activate   # or .venv/bin/python for individual commands without activating

# 2. Install dependencies (recommended)
pip install -r requirements.txt
# or install directly:
# pip install numpy pandas pyarrow matplotlib cvxpy pytest vnstock \
#             python-dotenv python-dateutil ipykernel nbconvert nbformat

# 3. (Only needed if data/*.parquet doesn't exist yet) Download + clean VN100 data
#    Requires VNSTOCK_API_KEY in a .env file at the project root (do not commit this file).
#    The FIRST download takes a while (tens of minutes) due to vnstock's free-tier rate
#    limit (~1.2s sleep between requests for each of the 97+ tickers) — only needs to run
#    ONCE; results are cached to data/prices.parquet + data/returns.parquet, and subsequent
#    runs (including the notebook) read straight from the cache, with NO further network calls.
python -m src.data_loader

# 4. Run the test suite
pytest tests/ -v

# 5. Open / run the notebook
#    Register a kernel pointing to this venv (only needs to be done once):
python -m ipykernel install --user --name optproj-venv --display-name "Python (Optimization Project venv)"
#    Re-run the entire notebook from scratch (no need to open the Jupyter UI):
python -m nbconvert --to notebook --execute notebook.ipynb --output notebook.ipynb
#    Or open interactively: jupyter notebook notebook.ipynb  (select the "optproj-venv" kernel)
```

### Environment notes

- **`.env` / API key**: `src/data_loader.py` reads `VNSTOCK_API_KEY` from a `.env` file at
  the project root via `python-dotenv` (never logs the key). If `data/*.parquet` already
  exists, `main()` reads straight from the cache and **does not need** `.env`/network
  access — this is only needed when reloading from scratch or calling
  `repair_universe()`.
- **SSL / corporate proxy**: if the machine runs behind a Cloudflare Zero Trust Gateway
  (or a similar TLS-inspection proxy enforced by an organization's device management
  policy), network calls to `vnstock` may fail with `SSL: CERTIFICATE_VERIFY_FAILED —
  self-signed certificate in certificate chain`. This is NOT a code bug — you need to
  export that gateway's root CA from the System Keychain (macOS: `security
  find-certificate`), merge it into `certifi`'s CA bundle, then set the
  `REQUESTS_CA_BUNDLE`/`SSL_CERT_FILE` environment variables to point at that merged
  bundle BEFORE running `python -m src.data_loader`. No changes to `src/data_loader.py`
  are needed (this is machine/network configuration, not application logic). See full
  details in `.sdd/task-1b-report.md`.

## `pytest tests/ -v` results (real run, after adding the walk-forward backtest)

```
============================= test session starts ==============================
platform darwin -- Python 3.14.5, pytest-9.1.1, pluggy-1.6.0 -- .venv/bin/python
cachedir: .pytest_cache
rootdir: /Users/nguyentobinh12gmail.com/Documents/Optimization Project
collecting ... collected 40 items

tests/test_backtest.py::test_metrics_constant_return PASSED              [  2%]
tests/test_backtest.py::test_metrics_known_drawdown_sequence PASSED      [  5%]
tests/test_backtest.py::test_metrics_sharpe_sign PASSED                  [  7%]
tests/test_backtest.py::test_build_windows_no_look_ahead PASSED          [ 10%]
tests/test_backtest.py::test_build_windows_count_matches_lookback PASSED [ 12%]
tests/test_backtest.py::test_simulate_period_first_period_full_turnover PASSED [ 15%]
tests/test_backtest.py::test_simulate_period_turnover_against_prev_drifted PASSED [ 17%]
tests/test_backtest.py::test_walk_forward_backtest_runs_and_is_feasible PASSED [ 20%]
tests/test_backtest.py::test_equal_weight_backtest_matches_uniform_weights PASSED [ 22%]
tests/test_cvxpy_check.py::test_cvxpy_solve_feasible_and_matches_objective PASSED [ 25%]
tests/test_cvxpy_check.py::test_compare_columns_and_small_relgap PASSED  [ 27%]
tests/test_cvxpy_check.py::test_long_only_matches_cvxpy_on_real_data PASSED [ 30%]
tests/test_cvxpy_check.py::test_cvxpy_solve_long_only_feasible_and_matches_objective PASSED [ 32%]
tests/test_data_loader.py::test_weekend_rows_dropped_by_clean_prices PASSED [ 35%]
tests/test_data_loader.py::test_compute_returns_drops_suspected_holiday_row PASSED [ 37%]
tests/test_data_loader.py::test_compute_returns_keeps_genuine_flat_day_for_minority PASSED [ 40%]
tests/test_data_loader.py::test_compute_returns_no_nan_after_filtering PASSED [ 42%]
tests/test_estimators.py::test_mu_shape_and_value PASSED                 [ 45%]
tests/test_estimators.py::test_sigma_symmetric PASSED                    [ 47%]
tests/test_estimators.py::test_sigma_psd PASSED                          [ 50%]
tests/test_estimators.py::test_sqrt_reconstructs PASSED                  [ 52%]
tests/test_estimators.py::test_sqrt_symmetric PASSED                     [ 55%]
tests/test_estimators.py::test_shrinkage_between PASSED                  [ 57%]
tests/test_estimators.py::test_sqrt_on_psd_singular PASSED               [ 60%]
tests/test_estimators.py::test_estimate_all_shapes_and_order PASSED      [ 62%]
tests/test_prox_solver.py::test_closed_form_meanvar PASSED               [ 65%]
tests/test_prox_solver.py::test_sum_to_one PASSED                        [ 67%]
tests/test_prox_solver.py::test_sparsity_increases_with_lambda PASSED    [ 70%]
tests/test_prox_solver.py::test_prox_l1_simplex_eq_exact_zero_and_sum_one PASSED [ 72%]
tests/test_prox_solver.py::test_prox_l1_simplex_eq_reduces_to_projection_when_t_zero PASSED [ 75%]
tests/test_prox_solver.py::test_best_obj_monotone PASSED                 [ 77%]
tests/test_prox_solver.py::test_returns_best_not_last PASSED             [ 80%]
tests/test_prox_solver.py::test_robust_subgrad_zero_safe PASSED          [ 82%]
tests/test_prox_solver.py::test_simplex_projection_symmetric_case PASSED [ 85%]
tests/test_prox_solver.py::test_simplex_projection_dominant_component PASSED [ 87%]
tests/test_prox_solver.py::test_simplex_projection_already_feasible_is_fixed_point PASSED [ 90%]
tests/test_prox_solver.py::test_simplex_projection_random_always_feasible PASSED [ 92%]
tests/test_prox_solver.py::test_solve_long_only_uniform_when_isotropic PASSED [ 95%]
tests/test_prox_solver.py::test_solve_long_only_feasible_general_case PASSED [ 97%]
tests/test_prox_solver.py::test_solve_long_only_returns_best_not_last PASSED [100%]

======================= 40 passed, 27 warnings in 11.99s =======================
```

The `RuntimeWarning`s (divide-by-zero/overflow/invalid in `matmul`) are known noise from
Apple Accelerate BLAS when multiplying near-singular matrices or in tests that
deliberately construct edge cases (`kappa=0`, `Sigma^(1/2)w≈0`) — not bugs, and they don't
affect the passing results (verified that no NaN/Inf leaks into the final output).

## Notebook

`notebook.ipynb` runs cleanly end-to-end via:

```bash
.venv/bin/python -m nbconvert --to notebook --execute notebook.ipynb --output notebook.ipynb
```

Result of the most recent real run: 28 cells (15 code + 13 markdown), **0 errors**, all
**9 figures** (`fig1`..`fig9`, including 3 new walk-forward backtest figures) regenerated
directly inside the notebook (calling functions in `src/viz.py`, not reading static PNGs)
and displayed inline, with a total execution time of ~50 seconds (no network calls — only
reads the `data/*.parquet` cache; most of this time is the walk-forward backtest re-running
its grid search of 12 parameter combinations × 25 rebalancing periods). The notebook does
not copy-paste logic — all computation is imported directly from `src/data_loader`,
`src/estimators`, `src/prox_solver`, `src/cvxpy_check`, `src/backtest`, `src/viz`.
