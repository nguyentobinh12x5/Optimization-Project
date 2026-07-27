"""
src/viz.py
===========

Visualization layer cho project sparse+robust portfolio optimization (VN100).

Phase 5: sinh 6 hình minh hoạ pipeline (data -> estimate -> solve -> verify)
từ dữ liệu cache + kết quả solver, KHÔNG gọi mạng. Mỗi hình = 1 hàm nhận dữ
liệu/kết quả làm tham số, trả về `matplotlib.figure.Figure` VÀ lưu PNG (dpi
150, bbox_inches='tight') ra thư mục `figures/` (path TƯƠNG ĐỐI từ project
root, xem `FIGURES_DIR` -- để Phase 6 (notebook) import và gọi lại được từ
bất kỳ working directory nào miễn cwd nằm trong/hoặc là project root, giống
quy ước `PROJECT_ROOT` của `src/data_loader.py`).

Palette & style (theo skill `dataviz`, xem `.sdd/task-5-report.md` mục
"Palette/style"):
- Categorical (nhận diện bộ tham số / series): 8 hue cố định thứ tự
  `CAT_COLORS` (blue, green, magenta, yellow, aqua, orange, violet, red) --
  đã validate bằng `scripts/validate_palette.js` (skill dataviz): 4 slot đầu
  (blue/green/magenta/yellow) pass ALL-PAIRS ở cả light/dark (dùng cho các
  hình có nhiều series cạnh nhau kiểu bar/scatter/small-multiples, đúng cảnh
  báo "series cap = 4 slot đầu" của skill). KHÔNG dùng quá 4 series categorical
  trong 1 axes; nhiều hơn -> facet (small multiples) như `fig4_weights_bar`.
- Diverging (tương quan, có dấu +/-): blue<->red quanh trung điểm gray, dùng
  cho heatmap tương quan (`fig1_data_overview`) vì correlation có "cực"
  (dương/âm) quanh 0 -- đúng job "diverging = polarity" của skill, KHÔNG
  dùng rainbow/jet.
- Ink/surface: nền `#fcfcfb` (light chart surface), chữ chính `#0b0b0b`,
  chữ phụ/trục `#52514e`/`#898781`, gridline hairline `#e1e0d9`, baseline/axis
  `#c3c2b7` -- áp qua `_apply_style()` (rcParams) để 6 hình nhất quán.
- Static PNG cho notebook học thuật (không phải trang HTML tương tác) nên
  chỉ ship 1 theme (light) -- không có toggle dark/light lúc runtime cho một
  file ảnh tĩnh; xem "Concerns" trong report.
- Contrast WARN của magenta/yellow trên nền sáng (skill dataviz mục
  categorical) được bù bằng "relief channel" bắt buộc: MỌI hình có các slot
  đó đều có legend + (khi hợp lý) direct value label, KHÔNG chỉ dựa vào màu.

Chạy nhanh trên data thật: `.venv/bin/python -m src.viz` (hoặc
`from src.viz import generate_all; generate_all()`).

Walk-forward backtest (fig7-9, xem
docs/superpowers/plans/2026-07-26-walk-forward-backtest.md Task 6): 3 hình
tiêu thụ trực tiếp `BacktestResult` (`src/backtest.py`, các trường
`daily_returns`/`rebalance_log`/`weights`) thay vì mu/Sigma như fig1-6 --
KHÔNG đưa vào `generate_all()` vì `walk_forward_backtest` tốn thời gian
(grid-search 12 tổ hợp tham số mỗi kỳ rebalance), không còn "chạy nhanh" như
mục tiêu của `generate_all`/`_main`. Gọi trực tiếp sau khi đã có kết quả
backtest (xem smoke test trong `.sdd/wf-task-6-report.md`).
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # không cần hiển thị, chỉ savefig -- đặt TRƯỚC pyplot

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from pathlib import Path

from src.cvxpy_check import cvxpy_solve
from src.estimators import estimate_all
from src.prox_solver import solve

__all__ = [
    "fig1_data_overview",
    "fig2_convergence",
    "fig3_sparsity_path",
    "fig4_weights_bar",
    "fig5_robust_effect",
    "fig6_prox_vs_cvxpy",
    "fig7_backtest_equity",
    "fig8_drawdown",
    "fig9_selected_params",
    "generate_all",
]

# ---------------------------------------------------------------------------
# Path tương đối từ project root (KHÔNG hardcode đường dẫn tuyệt đối cứng --
# `Path(__file__).resolve().parent.parent` luôn trỏ đúng project root bất kể
# cwd, giống quy ước `PROJECT_ROOT` trong src/data_loader.py).
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
FIGURES_DIR = PROJECT_ROOT / "figures"
RETURNS_PARQUET = DATA_DIR / "returns.parquet"

# ---------------------------------------------------------------------------
# Palette (skill dataviz, xem docstring module) -- categorical fixed order,
# 4 slot đầu đã validate ALL-PAIRS (scatter/bar/small-multiples).
# ---------------------------------------------------------------------------
CAT_COLORS = [
    "#2a78d6",  # 1 blue
    "#008300",  # 2 green
    "#e87ba4",  # 3 magenta
    "#eda100",  # 4 yellow
    "#1baf7a",  # 5 aqua
    "#eb6834",  # 6 orange
    "#4a3aa7",  # 7 violet
    "#e34948",  # 8 red
]
DIVERGING_BLUE = "#2a78d6"
DIVERGING_RED = "#e34948"
DIVERGING_MID = "#f0efec"

INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"
SURFACE = "#fcfcfb"

_DIVERGING_CMAP = LinearSegmentedColormap.from_list(
    "blue_gray_red", [DIVERGING_BLUE, DIVERGING_MID, DIVERGING_RED]
)


def _apply_style() -> None:
    """Áp rcParams chung cho cả 6 hình -- đảm bảo nhất quán font-size/màu/
    style (theo skill dataviz), gọi ở đầu mỗi hàm vẽ."""
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "font.family": "sans-serif",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.titlecolor": INK_PRIMARY,
            "axes.labelsize": 10,
            "axes.labelcolor": INK_SECONDARY,
            "axes.edgecolor": BASELINE,
            "xtick.color": INK_MUTED,
            "ytick.color": INK_MUTED,
            "text.color": INK_PRIMARY,
            "grid.color": GRIDLINE,
            "grid.linewidth": 0.7,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _save(fig: Figure, name: str) -> Path:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    return path


def _load_returns() -> pd.DataFrame:
    return pd.read_parquet(RETURNS_PARQUET)


# ---------------------------------------------------------------------------
# Fig 1 -- tổng quan universe: heatmap tương quan returns (97x97)
# ---------------------------------------------------------------------------

def fig1_data_overview(returns: pd.DataFrame | None = None, *, save: bool = True) -> Figure:
    """Heatmap ma trận tương quan (Pearson) của daily returns, 97x97.

    Chọn heatmap tương quan (thay vì giá chuẩn hoá) làm cách RÕ RÀNG NHẤT để
    thấy cấu trúc đồng biến động của toàn universe VN100 trong 1 hình duy
    nhất -- đây chính là input trực tiếp cho Sigma (`src.estimators`) mà
    solver dùng, nên minh hoạ đúng "tổng quan universe" theo góc nhìn tối ưu
    hoá. Dùng colormap DIVERGING (blue<->gray<->red) vì correlation có cực
    +/- quanh 0 (đúng job "diverging=polarity" của skill dataviz).

    Parameters
    ----------
    returns : pd.DataFrame | None
        Simple daily returns (T,N). None -> đọc từ `data/returns.parquet`.
    save : bool, default True
        Lưu PNG ra `figures/fig1_data_overview.png`.

    Returns
    -------
    matplotlib.figure.Figure
    """
    _apply_style()
    if returns is None:
        returns = _load_returns()

    corr = returns.corr()
    n = corr.shape[0]

    fig, ax = plt.subplots(figsize=(8.5, 7.5))
    im = ax.imshow(corr.to_numpy(), cmap=_DIVERGING_CMAP, vmin=-1.0, vmax=1.0, aspect="equal")

    # 97 nhãn không đọc được hết -> hiện mỗi ~10 mã để tránh chữ chồng lấp,
    # vẫn giữ đủ thông tin định hướng (đầu mỗi cụm) thay vì ẩn hết tick.
    step = max(1, n // 12)
    ticks = np.arange(0, n, step)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(corr.columns[ticks], rotation=90, fontsize=7)
    ax.set_yticklabels(corr.columns[ticks], fontsize=7)

    ax.set_xlabel(f"Mã cổ phiếu (N={n})")
    ax.set_ylabel(f"Mã cổ phiếu (N={n})")
    ax.set_title("VN100 Universe — Ma trận tương quan return ngày (Pearson)")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Hệ số tương quan (đơn vị: không thứ nguyên, [-1, 1])")

    fig.tight_layout()
    if save:
        _save(fig, "fig1_data_overview")
    return fig


# ---------------------------------------------------------------------------
# Fig 2 -- convergence: objective theo iteration, 2-3 bộ tham số
# ---------------------------------------------------------------------------

def fig2_convergence(
    mu: np.ndarray,
    sigma: np.ndarray,
    sigma_sqrt: np.ndarray,
    param_sets: list[dict] | None = None,
    *,
    max_iter: int = 3000,
    save: bool = True,
) -> Figure:
    """Objective f(w_k) theo iteration k cho 2-3 bộ tham số, cùng 1 axes.

    Minh hoạ hành vi hội tụ của proximal-subgradient solver (`src.prox_solver
    .solve`) -- KHÔNG đơn điệu giảm (đặc tính subgradient method, xem
    docstring `src/prox_solver.py`), nhưng best-iterate (đường trả về)
    hội tụ ổn định. Trục y log-scale CHỈ dùng khi mọi giá trị objective > 0
    trên toàn bộ 2-3 series (kiểm tra động); nếu có series âm (có thể xảy ra
    khi lam nhỏ, mean_term chiếm ưu thế), tự chuyển về trục y tuyến tính để
    tránh vẽ sai (log của số âm vô nghĩa).

    Parameters
    ----------
    mu, sigma, sigma_sqrt : np.ndarray
        Xem `src.prox_solver.portfolio_objective`.
    param_sets : list[dict] | None
        Mỗi dict {"kappa":, "gamma":, "lam":, "label": (tuỳ chọn)}. None ->
        3 bộ mặc định đại diện (kappa=0 vs kappa=1, lam nhỏ/lớn).
    max_iter : int, default 3000
        Truyền cho `solve()`.
    save : bool, default True

    Returns
    -------
    matplotlib.figure.Figure
    """
    _apply_style()
    if param_sets is None:
        param_sets = [
            {"kappa": 1.0, "gamma": 5.0, "lam": 0.01},
            {"kappa": 0.0, "gamma": 5.0, "lam": 0.1},
            {"kappa": 1.0, "gamma": 5.0, "lam": 0.001},
        ]

    fig, ax = plt.subplots(figsize=(7.5, 5))

    histories = []
    labels = []
    for i, p in enumerate(param_sets[:4]):  # cap 4 series (all-pairs slot cap)
        kappa, gamma, lam = float(p["kappa"]), float(p["gamma"]), float(p["lam"])
        label = p.get("label", f"κ={kappa:g}, γ={gamma:g}, λ={lam:g}")
        result = solve(mu, sigma, sigma_sqrt, kappa, gamma, lam, max_iter=max_iter)
        histories.append(result.obj_history)
        labels.append(label)

    all_positive = all(np.all(h > 0) for h in histories)

    for color, h, label in zip(CAT_COLORS, histories, labels):
        ax.plot(np.arange(1, len(h) + 1), h, color=color, linewidth=1.6, label=label)

    if all_positive:
        ax.set_yscale("log")
        ax.set_ylabel("Objective f(w_k)  (thang log)")
    else:
        ax.set_ylabel("Objective f(w_k)")

    ax.set_xlabel("Iteration k")
    ax.set_title("Hội tụ solver proximal-subgradient — f(w_k) theo iteration")
    ax.grid(True, alpha=0.6)
    ax.legend(title="Bộ tham số", loc="best")

    fig.tight_layout()
    if save:
        _save(fig, "fig2_convergence")
    return fig


# ---------------------------------------------------------------------------
# Fig 3 -- sparsity path: số mã active theo lambda (log-scale)
# ---------------------------------------------------------------------------

def fig3_sparsity_path(
    mu: np.ndarray,
    sigma: np.ndarray,
    sigma_sqrt: np.ndarray,
    *,
    kappa: float = 1.0,
    gamma: float = 5.0,
    lambdas: np.ndarray | None = None,
    active_thresh: float = 1e-4,
    max_iter: int = 3000,
    save: bool = True,
) -> Figure:
    """Số tài sản active (|w_i| > active_thresh) theo lambda, trục x log.

    kappa, gamma giữ CỐ ĐỊNH (mặc định kappa=1, gamma=5 theo brief) để cô lập
    đúng 1 hiệu ứng: lambda tăng -> L1 penalty mạnh hơn -> danh mục thưa hơn
    (số active giảm). Đây là 1 series duy nhất -> không cần legend (title đã
    nêu tên series, đúng quy tắc "1 series không cần legend box" của skill).

    Parameters
    ----------
    mu, sigma, sigma_sqrt : np.ndarray
    kappa, gamma : float, default 1.0, 5.0 -- giữ cố định theo brief.
    lambdas : np.ndarray | None
        None -> np.logspace(-4, -1, 18) (15-20 điểm theo brief).
    active_thresh : float, default 1e-4
    max_iter : int, default 3000
    save : bool, default True

    Returns
    -------
    matplotlib.figure.Figure
    """
    _apply_style()
    if lambdas is None:
        lambdas = np.logspace(-4, -1, 18)

    n_active = []
    for lam in lambdas:
        result = solve(mu, sigma, sigma_sqrt, kappa, gamma, float(lam), max_iter=max_iter)
        n_active.append(int(np.sum(np.abs(result.w) > active_thresh)))

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(lambdas, n_active, color=CAT_COLORS[0], marker="o", markersize=5, linewidth=1.8)
    ax.set_xscale("log")
    ax.set_xlabel("λ (hệ số phạt L1, thang log)")
    ax.set_ylabel(f"Số tài sản active (|w_i| > {active_thresh:g})")
    ax.set_title(f"Sparsity path — κ={kappa:g}, γ={gamma:g} cố định")
    ax.grid(True, alpha=0.6)

    fig.tight_layout()
    if save:
        _save(fig, "fig3_sparsity_path")
    return fig


# ---------------------------------------------------------------------------
# Fig 4 -- bar chart trọng số w* tại 2-3 bộ tham số tiêu biểu
# ---------------------------------------------------------------------------

def fig4_weights_bar(
    mu: np.ndarray,
    sigma: np.ndarray,
    sigma_sqrt: np.ndarray,
    symbols: list[str],
    param_sets: list[dict] | None = None,
    *,
    active_thresh: float = 1e-4,
    max_iter: int = 3000,
    save: bool = True,
) -> Figure:
    """Bar chart trọng số w* tại 2-3 bộ tham số (vd lambda nhỏ vs lambda lớn).

    Dùng SMALL MULTIPLES (1 subplot / bộ tham số, chia sẻ trục x = union các
    mã active) thay vì chồng nhiều nhóm bar cạnh nhau trên cùng 1 axes -- với
    có thể hàng chục mã active, grouped-bar chồng 3 bộ tham số cạnh nhau trên
    1 axes sẽ quá rối; small multiples giữ mỗi panel dễ đọc trong khi vẫn so
    sánh được qua trục x chung. Mỗi panel có đường y=0 (nét đứt) vì w có thể
    âm (short được phép).

    Parameters
    ----------
    mu, sigma, sigma_sqrt : np.ndarray
    symbols : list[str]
        Tên mã theo đúng thứ tự cột của mu/sigma (list(returns.columns)).
    param_sets : list[dict] | None
        None -> 2 bộ mặc định: lambda nhỏ (dense hơn) vs lambda lớn (thưa hơn),
        cùng kappa=1, gamma=5.
    active_thresh : float, default 1e-4
    max_iter : int, default 3000
    save : bool, default True

    Returns
    -------
    matplotlib.figure.Figure
    """
    _apply_style()
    if param_sets is None:
        param_sets = [
            {"kappa": 1.0, "gamma": 5.0, "lam": 0.001, "label": "λ nhỏ (0.001) — dense hơn"},
            {"kappa": 1.0, "gamma": 5.0, "lam": 0.02, "label": "λ lớn (0.02) — thưa hơn"},
        ]
    param_sets = param_sets[:3]  # cap 3 panel cho dễ đọc

    ws = []
    labels = []
    for p in param_sets:
        kappa, gamma, lam = float(p["kappa"]), float(p["gamma"]), float(p["lam"])
        label = p.get("label", f"κ={kappa:g}, γ={gamma:g}, λ={lam:g}")
        result = solve(mu, sigma, sigma_sqrt, kappa, gamma, lam, max_iter=max_iter)
        ws.append(result.w)
        labels.append(label)

    # Union các mã active qua mọi bộ tham số, sắp theo trọng số trung bình
    # giảm dần để dễ nhìn (mã "lớn" ở panel nào đó lên trước).
    active_mask = np.zeros(len(symbols), dtype=bool)
    for w in ws:
        active_mask |= np.abs(w) > active_thresh
    idx = np.where(active_mask)[0]
    order = idx[np.argsort(-np.mean([ws[i][idx] for i in range(len(ws))], axis=0))]
    active_symbols = [symbols[i] for i in order]

    n_panels = len(ws)
    fig, axes = plt.subplots(n_panels, 1, figsize=(max(8, len(order) * 0.25), 3.0 * n_panels), sharex=True)
    if n_panels == 1:
        axes = [axes]

    x = np.arange(len(order))
    for ax, w, label, color in zip(axes, ws, labels, CAT_COLORS):
        vals = w[order]
        ax.bar(x, vals, color=color, width=0.7)
        ax.axhline(0.0, color=BASELINE, linewidth=1.0, linestyle="--")
        ax.set_ylabel("Trọng số wᵢ")
        ax.set_title(label, fontsize=10, loc="left", fontweight="normal", color=INK_SECONDARY)
        ax.grid(True, axis="y", alpha=0.5)

    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(active_symbols, rotation=90, fontsize=7)
    axes[-1].set_xlabel(f"Mã cổ phiếu (active tại ít nhất 1 bộ tham số, N={len(order)})")

    fig.suptitle("Trọng số danh mục w* tại các bộ tham số tiêu biểu", fontsize=12, fontweight="bold", y=1.0)
    fig.tight_layout()
    if save:
        _save(fig, "fig4_weights_bar")
    return fig


# ---------------------------------------------------------------------------
# Fig 5 -- so sánh robust kappa=0 vs kappa>0
# ---------------------------------------------------------------------------

def _herfindahl(w: np.ndarray) -> float:
    """Herfindahl-Hirschman index của |w| chuẩn hoá tổng=1 (đo độ tập trung).

    HHI = sum((|w_i| / sum(|w_j|))^2) -- nằm trong (1/N, 1], 1/N = phân bổ
    đều tuyệt đối, càng gần 1 càng tập trung vào ít mã.
    """
    abs_w = np.abs(w)
    total = abs_w.sum()
    if total <= 0:
        return float("nan")
    shares = abs_w / total
    return float(np.sum(shares**2))


def fig5_robust_effect(
    mu: np.ndarray,
    sigma: np.ndarray,
    sigma_sqrt: np.ndarray,
    symbols: list[str],
    *,
    gamma: float = 5.0,
    lam: float = 0.02,
    kappa_values: tuple[float, float] = (0.0, 1.0),
    active_thresh: float = 1e-4,
    max_iter: int = 3000,
    save: bool = True,
) -> Figure:
    """So sánh danh mục κ=0 (không robust) vs κ>0 (có robust), cùng γ, λ.

    2 panel: (a) grouped bar trọng số trên union mã active của cả 2 kappa
    (minh hoạ robust term định hình lại phân bổ thế nào); (b) bar so sánh 2
    chỉ số tóm tắt độ tập trung -- số mã active và Herfindahl index (HHI) --
    để lượng hoá "robust làm danh mục tập trung hơn/kém hơn" thay vì chỉ nhìn
    định tính qua panel (a).

    Parameters
    ----------
    mu, sigma, sigma_sqrt : np.ndarray
    symbols : list[str]
    gamma, lam : float, default 5.0, 0.02 -- giữ cố định để cô lập hiệu ứng κ.
    kappa_values : tuple[float, float], default (0.0, 1.0)
    active_thresh : float, default 1e-4
    max_iter : int, default 3000
    save : bool, default True

    Returns
    -------
    matplotlib.figure.Figure
    """
    _apply_style()
    ws = []
    for kappa in kappa_values:
        result = solve(mu, sigma, sigma_sqrt, float(kappa), gamma, lam, max_iter=max_iter)
        ws.append(result.w)

    labels = [f"κ={k:g}" for k in kappa_values]

    active_mask = np.zeros(len(symbols), dtype=bool)
    for w in ws:
        active_mask |= np.abs(w) > active_thresh
    idx = np.where(active_mask)[0]
    order = idx[np.argsort(-np.mean([ws[i][idx] for i in range(len(ws))], axis=0))]
    active_symbols = [symbols[i] for i in order]

    n_active = [int(np.sum(np.abs(w) > active_thresh)) for w in ws]
    hhi = [_herfindahl(w) for w in ws]

    fig, (ax_bar, ax_metrics) = plt.subplots(1, 2, figsize=(13, 5.5), gridspec_kw={"width_ratios": [2.2, 1]})

    x = np.arange(len(order))
    width = 0.8 / len(ws)
    for i, (w, label, color) in enumerate(zip(ws, labels, CAT_COLORS)):
        offset = (i - (len(ws) - 1) / 2) * width
        ax_bar.bar(x + offset, w[order], width=width, color=color, label=label)
    ax_bar.axhline(0.0, color=BASELINE, linewidth=1.0, linestyle="--")
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(active_symbols, rotation=90, fontsize=7)
    ax_bar.set_xlabel(f"Mã cổ phiếu (active tại ít nhất 1 κ, N={len(order)})")
    ax_bar.set_ylabel("Trọng số wᵢ")
    ax_bar.set_title("Trọng số theo mã", fontsize=10, loc="left", color=INK_SECONDARY)
    ax_bar.legend(title=f"γ={gamma:g}, λ={lam:g}", loc="best")
    ax_bar.grid(True, axis="y", alpha=0.5)

    metric_x = np.arange(2)
    metric_width = 0.8 / len(ws)
    for i, (label, color) in enumerate(zip(labels, CAT_COLORS)):
        offset = (i - (len(ws) - 1) / 2) * metric_width
        vals = [n_active[i], hhi[i] * 100]  # HHI*100 để cùng thang hiển thị hợp lý với 2 trục phụ bên dưới
        bars = ax_metrics.bar(metric_x + offset, vals, width=metric_width, color=color, label=label)
        for b, v in zip(bars, vals):
            ax_metrics.annotate(f"{v:.1f}", (b.get_x() + b.get_width() / 2, b.get_height()),
                                 ha="center", va="bottom", fontsize=8, color=INK_PRIMARY)
    ax_metrics.set_xticks(metric_x)
    ax_metrics.set_xticklabels(["Số mã active", "HHI × 100"])
    ax_metrics.set_ylabel("Giá trị chỉ số")
    ax_metrics.set_title("Độ tập trung danh mục", fontsize=10, loc="left", color=INK_SECONDARY)
    ax_metrics.legend(loc="best")
    ax_metrics.grid(True, axis="y", alpha=0.5)

    fig.suptitle(
        f"Hiệu ứng robust term: κ=0 vs κ={kappa_values[1]:g} (γ={gamma:g}, λ={lam:g} cố định)",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout()
    if save:
        _save(fig, "fig5_robust_effect")
    return fig


# ---------------------------------------------------------------------------
# Fig 6 -- scatter w_hand vs w_cvxpy + đường y=x
# ---------------------------------------------------------------------------

def fig6_prox_vs_cvxpy(
    mu: np.ndarray,
    sigma: np.ndarray,
    sigma_sqrt: np.ndarray,
    *,
    kappa: float = 1.0,
    gamma: float = 5.0,
    lam: float = 0.01,
    max_iter: int = 5000,
    save: bool = True,
) -> Figure:
    """Scatter w_hand (solver tay) vs w_cvxpy (ground truth) + đường y=x.

    Minh hoạ trực quan kết luận Phase 4 (`src/cvxpy_check.py`): solver tay
    khớp CVXPY -- các điểm phải nằm rất sát đường y=x nếu 2 nghiệm gần trùng
    nhau. Bộ tham số mặc định (κ=1, γ=5, λ=0.01) khớp bộ đầu tiên trong
    `DEFAULT_PARAM_GRID` (đã verify relgap≈2.2e-7 tức 0.000022%, Jaccard=1.0 -- xem
    task-4-report.md).

    Parameters
    ----------
    mu, sigma, sigma_sqrt : np.ndarray
    kappa, gamma, lam : float, default 1.0, 5.0, 0.01
    max_iter : int, default 5000 -- truyền cho `solve()` (khớp mặc định
        dùng ở Phase 4 để tái lập đúng kết quả đã verify).
    save : bool, default True

    Returns
    -------
    matplotlib.figure.Figure
    """
    _apply_style()
    result = solve(mu, sigma, sigma_sqrt, kappa, gamma, lam, max_iter=max_iter)
    w_hand = result.w
    w_cvx, _ = cvxpy_solve(mu, sigma, sigma_sqrt, kappa, gamma, lam)

    lo = float(min(w_hand.min(), w_cvx.min()))
    hi = float(max(w_hand.max(), w_cvx.max()))
    pad = 0.05 * (hi - lo) if hi > lo else 0.01
    lo, hi = lo - pad, hi + pad

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([lo, hi], [lo, hi], color=BASELINE, linewidth=1.2, linestyle="--", label="y = x (khớp hoàn hảo)")
    ax.scatter(w_hand, w_cvx, color=CAT_COLORS[0], s=28, alpha=0.85, edgecolor="white", linewidth=0.4,
               label="Tài sản (97 mã)")

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("w_hand — solver tay (proximal-subgradient)")
    ax.set_ylabel("w_cvxpy — CVXPY (CLARABEL, ground truth)")
    ax.set_title(f"Solver tay vs CVXPY — κ={kappa:g}, γ={gamma:g}, λ={lam:g}")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.5)

    linf = float(np.max(np.abs(w_hand - w_cvx)))
    ax.annotate(
        f"‖w_hand − w_cvxpy‖∞ = {linf:.2e}",
        xy=(0.03, 0.97), xycoords="axes fraction", ha="left", va="top",
        fontsize=9, color=INK_SECONDARY,
    )

    fig.tight_layout()
    if save:
        _save(fig, "fig6_prox_vs_cvxpy")
    return fig


# ---------------------------------------------------------------------------
# Fig 7 -- backtest walk-forward: đường cong tài sản tích luỹ vs benchmark
# ---------------------------------------------------------------------------

def fig7_backtest_equity(strategy_result, benchmark_result, *, save: bool = True) -> Figure:
    """Đường cong tài sản tích luỹ (net phí) — chiến lược walk-forward vs
    benchmark equal-weight 1/N, cùng giai đoạn out-of-sample.

    2 series trên cùng 1 axes (dưới cap 4 series all-pairs của palette) --
    đủ để so sánh trực quan; đường nét đứt cho benchmark để phân biệt ngay
    cả khi in đen trắng (không chỉ dựa vào màu, đúng "relief channel" theo
    skill dataviz). Cả 2 `daily_returns` được chuẩn hoá về gốc 1.0 độc lập
    (không giả định cùng ngày bắt đầu tuyệt đối, dù trong pipeline này 2 kết
    quả dùng cùng danh sách rebalance windows nên trùng khớp).

    Parameters
    ----------
    strategy_result : BacktestResult
        Kết quả `walk_forward_backtest` (`src/backtest.py`).
    benchmark_result : BacktestResult
        Kết quả `equal_weight_backtest`.
    save : bool, default True
        Lưu PNG ra `figures/fig7_backtest_equity.png`.

    Returns
    -------
    matplotlib.figure.Figure
    """
    _apply_style()
    fig, ax = plt.subplots(figsize=(11, 6))

    strat_curve = (1 + strategy_result.daily_returns).cumprod()
    bench_curve = (1 + benchmark_result.daily_returns).cumprod()

    ax.plot(strat_curve.index, strat_curve.values, color=CAT_COLORS[0], linewidth=1.8,
             label="Chiến lược (long-only, walk-forward)")
    ax.plot(bench_curve.index, bench_curve.values, color=CAT_COLORS[1], linewidth=1.6,
             linestyle="--", label="Equal-weight 1/N")

    ax.set_title("Backtest walk-forward — Đường cong tài sản tích luỹ (net phí)")
    ax.set_xlabel("Ngày")
    ax.set_ylabel("Giá trị danh mục (chuẩn hoá, bắt đầu = 1.0)")
    ax.grid(True, alpha=0.5)
    ax.legend(loc="best")

    fig.tight_layout()
    if save:
        _save(fig, "fig7_backtest_equity")
    return fig


# ---------------------------------------------------------------------------
# Fig 8 -- drawdown theo thời gian: chiến lược vs benchmark
# ---------------------------------------------------------------------------

def fig8_drawdown(strategy_result, benchmark_result, *, save: bool = True) -> Figure:
    """Drawdown (%) theo thời gian — chiến lược walk-forward vs benchmark
    equal-weight 1/N, tính từ CHÍNH `daily_returns` của mỗi kết quả (đỉnh
    tích luỹ riêng của từng đường, không so đỉnh chung) -- đúng định nghĩa
    max_drawdown dùng trong `performance_metrics` (`src/backtest.py`).

    Parameters
    ----------
    strategy_result : BacktestResult
    benchmark_result : BacktestResult
    save : bool, default True
        Lưu PNG ra `figures/fig8_drawdown.png`.

    Returns
    -------
    matplotlib.figure.Figure
    """
    _apply_style()
    fig, ax = plt.subplots(figsize=(11, 5))

    for result, label, color, ls in [
        (strategy_result, "Chiến lược", CAT_COLORS[0], "-"),
        (benchmark_result, "Equal-weight 1/N", CAT_COLORS[1], "--"),
    ]:
        curve = (1 + result.daily_returns).cumprod()
        drawdown = curve / curve.cummax() - 1
        ax.plot(drawdown.index, drawdown.values * 100, color=color, linewidth=1.6,
                 linestyle=ls, label=label)

    ax.axhline(0.0, color=BASELINE, linewidth=1.0)
    ax.set_title("Drawdown theo thời gian")
    ax.set_xlabel("Ngày")
    ax.set_ylabel("Drawdown (%)")
    ax.grid(True, alpha=0.5)
    ax.legend(loc="best")

    fig.tight_layout()
    if save:
        _save(fig, "fig8_drawdown")
    return fig


# ---------------------------------------------------------------------------
# Fig 9 -- tham số (kappa, gamma) đã chọn + turnover mỗi kỳ rebalance
# ---------------------------------------------------------------------------

def fig9_selected_params(strategy_result, *, save: bool = True) -> Figure:
    """2 panel chia sẻ trục x (ngày rebalance): (a) scatter κ*/γ* được chọn
    qua Sharpe validation mỗi kỳ (`rebalance_log` cột `kappa`/`gamma`, xem
    `walk_forward_backtest`); (b) bar turnover mỗi kỳ (%).

    Mục đích: kiểm tra trực quan finding đã ghi ở `.sdd/progress.md` (wf-Task
    4) -- (kappa, gamma) có thể đổi mạnh giữa các kỳ do mẫu validation 6
    tháng nhỏ, kéo theo turnover cao bất thường ở những kỳ đổi tham số đột
    ngột. κ và γ vẽ chung 1 panel bằng marker khác nhau (chấp nhận khác thang
    giá trị κ∈[0,2] vs γ∈[1,10] vì đây là scatter rời rạc theo lưới tham số
    cố định, không phải đường liên tục -- mục tiêu là thấy THỜI ĐIỂM đổi tham
    số, không phải so sánh độ lớn tuyệt đối giữa κ và γ).

    Parameters
    ----------
    strategy_result : BacktestResult
        Phải có `rebalance_log` với cột kappa, gamma, turnover (kết quả
        `walk_forward_backtest`; KHÔNG dùng được với `equal_weight_backtest`
        vì benchmark không có cột kappa/gamma).
    save : bool, default True
        Lưu PNG ra `figures/fig9_selected_params.png`.

    Returns
    -------
    matplotlib.figure.Figure
    """
    _apply_style()
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    log = strategy_result.rebalance_log

    axes[0].scatter(log.index, log["kappa"], color=CAT_COLORS[0], s=40, label="κ đã chọn")
    axes[0].scatter(log.index, log["gamma"], color=CAT_COLORS[1], s=40, marker="x", label="γ đã chọn")
    axes[0].set_ylabel("Giá trị tham số")
    axes[0].set_title("Tham số (κ, γ) được chọn qua validation mỗi kỳ")
    axes[0].grid(True, alpha=0.5)
    axes[0].legend(loc="best")

    axes[1].bar(log.index, log["turnover"] * 100, width=15, color=CAT_COLORS[0])
    axes[1].set_ylabel("Turnover (%)")
    axes[1].set_xlabel("Ngày rebalance")
    axes[1].set_title("Turnover mỗi kỳ rebalance")
    axes[1].grid(True, axis="y", alpha=0.5)

    fig.tight_layout()
    if save:
        _save(fig, "fig9_selected_params")
    return fig


# ---------------------------------------------------------------------------
# generate_all -- chạy tất cả từ cache
# ---------------------------------------------------------------------------

def generate_all() -> dict[str, Path]:
    """Chạy cả 6 hàm vẽ từ dữ liệu cache (`data/returns.parquet`), KHÔNG gọi
    mạng. Dùng làm entry point cho `python -m src.viz` và cho Phase 6 notebook
    (gọi lại để tái sinh toàn bộ figures/ nếu cần).

    Returns
    -------
    dict[str, Path]
        {tên_hình: đường dẫn PNG đã lưu}.
    """
    returns = _load_returns()
    symbols = list(returns.columns)
    mu, sigma, sigma_sqrt = estimate_all(returns)

    paths: dict[str, Path] = {}

    fig1 = fig1_data_overview(returns)
    paths["fig1_data_overview"] = FIGURES_DIR / "fig1_data_overview.png"
    plt.close(fig1)

    fig2 = fig2_convergence(mu, sigma, sigma_sqrt)
    paths["fig2_convergence"] = FIGURES_DIR / "fig2_convergence.png"
    plt.close(fig2)

    fig3 = fig3_sparsity_path(mu, sigma, sigma_sqrt)
    paths["fig3_sparsity_path"] = FIGURES_DIR / "fig3_sparsity_path.png"
    plt.close(fig3)

    fig4 = fig4_weights_bar(mu, sigma, sigma_sqrt, symbols)
    paths["fig4_weights_bar"] = FIGURES_DIR / "fig4_weights_bar.png"
    plt.close(fig4)

    fig5 = fig5_robust_effect(mu, sigma, sigma_sqrt, symbols)
    paths["fig5_robust_effect"] = FIGURES_DIR / "fig5_robust_effect.png"
    plt.close(fig5)

    fig6 = fig6_prox_vs_cvxpy(mu, sigma, sigma_sqrt)
    paths["fig6_prox_vs_cvxpy"] = FIGURES_DIR / "fig6_prox_vs_cvxpy.png"
    plt.close(fig6)

    return paths


def _main() -> None:  # pragma: no cover - manual smoke test entry point
    import time

    t0 = time.monotonic()
    paths = generate_all()
    elapsed = time.monotonic() - t0
    print(f"Đã sinh {len(paths)} hình trong {elapsed:.1f}s:")
    for name, path in paths.items():
        exists = path.exists()
        size_kb = path.stat().st_size / 1024 if exists else 0
        print(f"  - {name}: {path} (exists={exists}, {size_kb:.0f} KB)")


if __name__ == "__main__":  # pragma: no cover
    _main()
