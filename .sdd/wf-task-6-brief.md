# Task Brief — Walk-Forward Backtest: Task 6

Đây là brief trích từ plan đầy đủ: `docs/superpowers/plans/2026-07-26-walk-forward-backtest.md`.

## Global Constraints (áp dụng cho toàn bộ plan, đọc kỹ)

- Spec đầy đủ: `docs/superpowers/specs/2026-07-26-walk-forward-backtest-design.md`.
- Solver long-short hiện tại (`solve()`, `portfolio_objective()`, `_prox_l1_simplex_eq`, `_robust_subgrad`) GIỮ NGUYÊN KHÔNG SỬA — đã CVXPY-verify, các phase trước phụ thuộc vào nó.
- Long-only: ràng buộc `w≥0, Σw=1`. Vì `‖w‖₁=Σw=1` là hằng số dưới ràng buộc này, **λ vô tác dụng** → KHÔNG đưa λ vào đường long-only.
- Rolling window: `lookback_months=24` (`E`=18 tháng đầu, `Vwin`=6 tháng cuối), rebalance **hàng tháng** (ngày giao dịch đầu tiên mỗi tháng).
- Param grid mặc định: κ ∈ {0, 0.5, 1, 2} × γ ∈ {1, 5, 10} = 12 tổ hợp. Chọn theo **Sharpe (rf=0)** trên `Vwin`.
- Shrinkage `"lw"` (Ledoit-Wolf) BẬT xuyên suốt backtest (cả lúc chọn tham số lẫn giải nghiệm cuối).
- Phí giao dịch mặc định `fee=0.002` (0.20%) × turnover mỗi kỳ; kỳ đầu turnover=1.
- KHÔNG look-ahead: mọi ước lượng/validation của kỳ `t` chỉ dùng dữ liệu có ngày < ngày rebalance `t`. Phải có test cấu trúc khẳng định điều này.
- Môi trường: `.venv/bin/python` (Python 3.14). KHÔNG scipy/sklearn — mọi thuật toán (kể cả simplex projection) viết bằng numpy thuần. Không pip install gì thêm (cvxpy đã có).
- Data thật: `data/returns.parquet` (965×98, đã sạch — xem `src/data_loader.py`).
- Project KHÔNG dùng git — KHÔNG chạy `git init`/`git add`/`git commit`. Ghi nhận hoàn tất từng task bằng cách append 1 dòng vào `.sdd/progress.md` (đúng quy ước ledger đã dùng xuyên suốt các phase trước của project).
- Style code/docstring: tiếng Việt/Anh nhất quán như các module hiện có, người đọc là dân tối ưu hoá.

---

### Task 6: Visualization (fig7, fig8, fig9)

**Files:**
- Modify: `src/viz.py`

**Interfaces:**
- Consumes: `BacktestResult` (Task 4/5).
- Produces: `fig7_backtest_equity(strategy_result, benchmark_result, *, save=True) -> Figure`, `fig8_drawdown(strategy_result, benchmark_result, *, save=True) -> Figure`, `fig9_selected_params(strategy_result, *, save=True) -> Figure`.

- [ ] **Step 1: Đọc `src/viz.py` hiện có** để tái dùng đúng `_apply_style()`/palette/pattern lưu file (`figures/<tên>.png`, dpi 150, tight) đã dùng ở fig1-6.

- [ ] **Step 2: Implement `fig7_backtest_equity`**

```python
def fig7_backtest_equity(strategy_result, benchmark_result, *, save: bool = True):
    _apply_style()
    fig, ax = plt.subplots(figsize=(11, 6))

    strat_curve = (1 + strategy_result.daily_returns).cumprod()
    bench_curve = (1 + benchmark_result.daily_returns).cumprod()

    ax.plot(strat_curve.index, strat_curve.values, label="Chiến lược (long-only, walk-forward)")
    ax.plot(bench_curve.index, bench_curve.values, label="Equal-weight 1/N", linestyle="--")

    ax.set_title("Backtest walk-forward — Đường cong tài sản tích luỹ (net phí)")
    ax.set_xlabel("Ngày")
    ax.set_ylabel("Giá trị danh mục (chuẩn hoá, bắt đầu = 1.0)")
    ax.legend()

    if save:
        fig.savefig("figures/fig7_backtest_equity.png", dpi=150, bbox_inches="tight")
    return fig
```

- [ ] **Step 3: Implement `fig8_drawdown`**

```python
def fig8_drawdown(strategy_result, benchmark_result, *, save: bool = True):
    _apply_style()
    fig, ax = plt.subplots(figsize=(11, 5))

    for result, label, ls in [
        (strategy_result, "Chiến lược", "-"),
        (benchmark_result, "Equal-weight 1/N", "--"),
    ]:
        curve = (1 + result.daily_returns).cumprod()
        drawdown = curve / curve.cummax() - 1
        ax.plot(drawdown.index, drawdown.values * 100, label=label, linestyle=ls)

    ax.set_title("Drawdown theo thời gian")
    ax.set_xlabel("Ngày")
    ax.set_ylabel("Drawdown (%)")
    ax.legend()

    if save:
        fig.savefig("figures/fig8_drawdown.png", dpi=150, bbox_inches="tight")
    return fig
```

- [ ] **Step 4: Implement `fig9_selected_params`**

```python
def fig9_selected_params(strategy_result, *, save: bool = True):
    _apply_style()
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    log = strategy_result.rebalance_log

    axes[0].scatter(log.index, log["kappa"], label="κ đã chọn")
    axes[0].scatter(log.index, log["gamma"], label="γ đã chọn", marker="x")
    axes[0].set_ylabel("Giá trị tham số")
    axes[0].set_title("Tham số (κ, γ) được chọn qua validation mỗi kỳ")
    axes[0].legend()

    axes[1].bar(log.index, log["turnover"] * 100, width=15)
    axes[1].set_ylabel("Turnover (%)")
    axes[1].set_xlabel("Ngày rebalance")
    axes[1].set_title("Turnover mỗi kỳ rebalance")

    if save:
        fig.savefig("figures/fig9_selected_params.png", dpi=150, bbox_inches="tight")
    return fig
```

- [ ] **Step 5: Smoke test thủ công (không cần pytest chính thức, đây là hình vẽ) trên data thật**

```bash
.venv/bin/python -c "
import pandas as pd
from src.backtest import walk_forward_backtest, equal_weight_backtest
from src.viz import fig7_backtest_equity, fig8_drawdown, fig9_selected_params

returns = pd.read_parquet('data/returns.parquet')
strat = walk_forward_backtest(returns)
bench = equal_weight_backtest(returns)
fig7_backtest_equity(strat, bench)
fig8_drawdown(strat, bench)
fig9_selected_params(strat)
print('done, check figures/fig7-9_*.png')
"
```

Xác nhận cả 3 file PNG được tạo, mở lên xem có hợp lý (đường cong không NaN/rỗng, drawdown âm, tham số hiển thị rõ).

- [ ] **Step 6: Ghi nhận hoàn tất task**

Project này KHÔNG dùng git (xem Global Constraints) — thay vì `git commit`, append 1 dòng vào `.sdd/progress.md` theo đúng quy ước ledger project đã dùng xuyên suốt các phase trước:
`Task <tên task>: complete (feat: add backtest visualization (equity curve, drawdown, selected params))`

---

