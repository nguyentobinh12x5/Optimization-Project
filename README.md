# Sparse + Robust Portfolio Optimization — VN100

**Nguyen To Binh** (1010051) & **Vu Van Giang** (1010036) — Optimization course project,
instructor Cai Yutong.

A portfolio optimization project for the **VN100** basket (98 Vietnamese stocks after data
cleaning, 4 years of daily returns). The goal is not to crown one "best" method: one convex
objective is stated and its terms are switched on/off/tuned to collapse into **four investment
strategies** — robust or classical, long-only or long-short — plus two passive benchmarks, then
the choice among them is settled empirically for VN100 by walk-forward backtest.

```
min_w  -μ̂ᵀw + κ‖Σ^(1/2)w‖₂ + γ·wᵀΣw + λ‖w‖₁   s.t.  1ᵀw = 1
```

Four terms, one convex problem: maximize expected return, penalize risk in both a robust sense
(exact worst-case return over an ellipsoidal uncertainty set on `μ̂`) and a classic Markowitz
sense (quadratic variance), and encourage a **sparse** portfolio (few names held) via an L1
penalty, which also caps gross leverage. The main solver is a **proximal-subgradient method
written from scratch in pure numpy** (no cvxpy/scipy.optimize/sklearn), with an **exact** prox
step for `L1 + budget constraint` solved via bisection (soft-thresholding + finding the Lagrange
multiplier `ν`). Results are **cross-verified** against CVXPY (interior-point solver CLARABEL) as
an independent "ground truth" — six significant figures, identical active sets.

Beyond the in-sample analysis, the project runs an **out-of-sample walk-forward backtest**
(`src/backtest.py`): a 24-month rolling window (18 months estimation + 6 months validation),
monthly rebalancing over 494 trading days / 25 rebalances, automatic reselection of
hyperparameters each period via Sharpe validation, transaction costs, and **six variants** —
A: Robust (long-only), B: Classical (long-only), C: Equal-weight 1/N, D: VN100 buy-&-hold,
E: Sparse-only (long-short), F: Full equation (long-short) — plus a per-ticker concentration cap
added after an early run lost 9.47% of capital in one ticker in one month.

See `notebook.ipynb` (Vietnamese) / `notebook_en.ipynb` (English) for the full end-to-end story
(data → estimation → algorithm → results → verification → OOS backtest → conclusion), and
**`report/main.pdf`** / **`slides/main.pdf`** for the write-up and presentation deck.

## Report & slides

- **Report**: [`report/main.pdf`](report/main.pdf), built from `report/main.tex` with
  [`tectonic`](https://tectonic-typesetting.github.io/) (`cd report && tectonic main.tex`).
  Structure: Problem Identification → Problem Modeling → Model Correctness → Algorithm → Data →
  In-Sample Results → Walk-Forward Backtest → Discussion/Conclusion, plus an appendix with
  well-commented code listings (`src/estimators.py`, `src/prox_solver.py`, `src/backtest.py`).
- **Slides**: [`slides/main.pdf`](slides/main.pdf), built the same way from `slides/main.tex`.
- **Repository**: <https://github.com/nguyentobinh12x5/Optimization-Project>

## Directory structure

```
.
├── src/                    # Core logic (all notebook imports come from here)
│   ├── data_loader.py      # Loads + cleans VN100 data via vnstock, caches to data/
│   ├── estimators.py       # Estimates μ̂, Σ, Σ^(1/2) (eigh + clip, custom Ledoit-Wolf)
│   ├── prox_solver.py      # Custom proximal-subgradient solver (pure numpy)
│   ├── cvxpy_check.py      # Cross-verification via CVXPY (the ONLY place that imports cvxpy)
│   ├── backtest.py         # Walk-forward OOS backtest: 6 variants + concentration cap
│   └── viz.py               # Plotting functions, saved to figures/
├── tests/                  # pytest for estimators / prox_solver / cvxpy_check / backtest
├── data/                   # Cached parquet/csv/xlsx (returns, prices, VN30 index, symbols) — gitignored
├── figures/                # Pre-generated PNGs (in-sample + 6-variant backtest figures)
├── notebook.ipynb          # End-to-end deliverable (Vietnamese), imports from src/
├── notebook_en.ipynb       # Same notebook, English
├── report/                 # LaTeX report (main.tex + sections/ + main.pdf)
├── slides/                 # LaTeX/Beamer presentation deck (main.tex + main.pdf)
├── .sdd/                   # Condensed project log (progress.md, final review reports)
└── .env                    # VNSTOCK_API_KEY (not committed — already in .gitignore)
```

## Environment requirements

- Python **3.10+** (tested on 3.14.5 in `.venv/`).
- Main packages: `numpy`, `pandas`, `pyarrow` (parquet read/write), `matplotlib`, `cvxpy`,
  `pytest`, `vnstock`, `python-dotenv`, `python-dateutil`, plus the jupyter stack
  (`ipykernel`, `nbconvert`, `nbformat`) to build/run the notebook.
- A `requirements.txt` (pinned to tested versions) is provided at the project root — the
  recommended install method is `pip install -r requirements.txt`.
- [`tectonic`](https://tectonic-typesetting.github.io/) (optional) to rebuild the report/slides
  PDFs from source.

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

# 6. (Optional) Rebuild the report / slides PDFs from source
cd report && tectonic main.tex && cd ..
cd slides && tectonic main.tex && cd ..
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
  are needed (this is machine/network configuration, not application logic).

## `pytest tests/ -v` results (real run)

```
============================= test session starts ==============================
platform darwin -- Python 3.14.5, pytest-9.1.1, pluggy-1.6.0 -- .venv/bin/python
cachedir: .pytest_cache
rootdir: /Users/nguyentobinh12gmail.com/Documents/Optimization Project
collecting ... collected 51 items

tests/test_backtest.py::test_metrics_constant_return PASSED              [  1%]
tests/test_backtest.py::test_metrics_known_drawdown_sequence PASSED      [  3%]
tests/test_backtest.py::test_metrics_sharpe_sign PASSED                  [  5%]
tests/test_backtest.py::test_metrics_wipeout_does_not_raise_and_floors_annualized_return PASSED [  7%]
tests/test_backtest.py::test_build_windows_no_look_ahead PASSED          [  9%]
tests/test_backtest.py::test_build_windows_count_matches_lookback PASSED [ 11%]
tests/test_backtest.py::test_simulate_period_first_period_full_turnover PASSED [ 13%]
tests/test_backtest.py::test_simulate_period_turnover_against_prev_drifted PASSED [ 15%]
tests/test_backtest.py::test_simulate_period_wipeout_caps_return_and_zeros_remaining_days PASSED [ 17%]
tests/test_backtest.py::test_walk_forward_backtest_runs_and_is_feasible PASSED [ 19%]
tests/test_backtest.py::test_walk_forward_backtest_max_weight_guarantees_minimum_active_count PASSED [ 21%]
tests/test_backtest.py::test_walk_forward_backtest_selects_max_return_when_requested PASSED [ 23%]
tests/test_backtest.py::test_walk_forward_backtest_selects_max_sharpe_by_default PASSED [ 25%]
tests/test_backtest.py::test_walk_forward_backtest_rejects_invalid_selection_metric PASSED [ 27%]
tests/test_backtest.py::test_walk_forward_backtest_long_short_runs_and_is_feasible PASSED [ 29%]
tests/test_backtest.py::test_walk_forward_backtest_long_short_deploys_with_correct_param_order PASSED [ 31%]
tests/test_backtest.py::test_walk_forward_backtest_long_short_full_runs_and_is_feasible PASSED [ 33%]
tests/test_backtest.py::test_equal_weight_backtest_matches_uniform_weights PASSED [ 35%]
tests/test_backtest.py::test_index_buy_and_hold_backtest_aligns_and_applies_fee_once PASSED [ 37%]
tests/test_backtest.py::test_index_buy_and_hold_backtest_warns_on_missing_dates PASSED [ 39%]
tests/test_cvxpy_check.py::test_cvxpy_solve_feasible_and_matches_objective PASSED [ 41%]
tests/test_cvxpy_check.py::test_compare_columns_and_small_relgap PASSED  [ 43%]
tests/test_cvxpy_check.py::test_long_only_matches_cvxpy_on_real_data PASSED [ 45%]
tests/test_cvxpy_check.py::test_cvxpy_solve_long_only_feasible_and_matches_objective PASSED [ 47%]
tests/test_data_loader.py::test_weekend_rows_dropped_by_clean_prices PASSED [ 49%]
tests/test_data_loader.py::test_compute_returns_drops_suspected_holiday_row PASSED [ 50%]
tests/test_data_loader.py::test_compute_returns_keeps_genuine_flat_day_for_minority PASSED [ 52%]
tests/test_data_loader.py::test_compute_returns_no_nan_after_filtering PASSED [ 54%]
tests/test_estimators.py::test_mu_shape_and_value PASSED                 [ 56%]
tests/test_estimators.py::test_sigma_symmetric PASSED                    [ 58%]
tests/test_estimators.py::test_sigma_psd PASSED                          [ 60%]
tests/test_estimators.py::test_sqrt_reconstructs PASSED                  [ 62%]
tests/test_estimators.py::test_sqrt_symmetric PASSED                     [ 64%]
tests/test_estimators.py::test_shrinkage_between PASSED                  [ 66%]
tests/test_estimators.py::test_sqrt_on_psd_singular PASSED               [ 68%]
tests/test_estimators.py::test_estimate_all_shapes_and_order PASSED      [ 70%]
tests/test_prox_solver.py::test_closed_form_meanvar PASSED               [ 72%]
tests/test_prox_solver.py::test_sum_to_one PASSED                        [ 74%]
tests/test_prox_solver.py::test_sparsity_increases_with_lambda PASSED    [ 76%]
tests/test_prox_solver.py::test_prox_l1_simplex_eq_exact_zero_and_sum_one PASSED [ 78%]
tests/test_prox_solver.py::test_prox_l1_simplex_eq_reduces_to_projection_when_t_zero PASSED [ 80%]
tests/test_prox_solver.py::test_best_obj_monotone PASSED                 [ 82%]
tests/test_prox_solver.py::test_returns_best_not_last PASSED             [ 84%]
tests/test_prox_solver.py::test_robust_subgrad_zero_safe PASSED          [ 86%]
tests/test_prox_solver.py::test_simplex_projection_symmetric_case PASSED [ 88%]
tests/test_prox_solver.py::test_simplex_projection_dominant_component PASSED [ 90%]
tests/test_prox_solver.py::test_simplex_projection_already_feasible_is_fixed_point PASSED [ 92%]
tests/test_prox_solver.py::test_simplex_projection_random_always_feasible PASSED [ 94%]
tests/test_prox_solver.py::test_solve_long_only_uniform_when_isotropic PASSED [ 96%]
tests/test_prox_solver.py::test_solve_long_only_feasible_general_case PASSED [ 98%]
tests/test_prox_solver.py::test_solve_long_only_returns_best_not_last PASSED [100%]

======================= 51 passed, 33 warnings in 8.93s ========================
```

The `RuntimeWarning`s (divide-by-zero/overflow/invalid in `matmul`) are known noise from
Apple Accelerate BLAS when multiplying near-singular matrices or in tests that
deliberately construct edge cases (`kappa=0`, `Sigma^(1/2)w≈0`) — not bugs, and they don't
affect the passing results (verified that no NaN/Inf leaks into the final output).

## Notebook

`notebook.ipynb` (and its English twin `notebook_en.ipynb`) runs cleanly end-to-end via:

```bash
.venv/bin/python -m nbconvert --to notebook --execute notebook.ipynb --output notebook.ipynb
```

The notebook does not copy-paste logic — all computation is imported directly from
`src/data_loader`, `src/estimators`, `src/prox_solver`, `src/cvxpy_check`, `src/backtest`,
`src/viz`, reading only the `data/*.parquet` cache (no network calls on a normal run).
