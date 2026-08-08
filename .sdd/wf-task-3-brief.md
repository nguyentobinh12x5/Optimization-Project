# Task Brief — Walk-Forward Backtest: Task 3

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

### Task 3: Performance metrics

**Files:**
- Create: `src/backtest.py` (chỉ phần `performance_metrics` ở task này; các hàm khác thêm ở Task 4/5)
- Create: `tests/test_backtest.py`

**Interfaces:**
- Produces: `performance_metrics(daily_returns: pd.Series, rf: float = 0.0) -> dict` với keys:
  `cumulative_return, annualized_return, annualized_vol, sharpe, max_drawdown, n_days` (dùng ở Task 4, 5, 6).

- [ ] **Step 1: Viết test cho `performance_metrics`**

```python
import numpy as np
import pandas as pd
from src.backtest import performance_metrics


def test_metrics_constant_return():
    r = 0.001
    n = 252
    daily = pd.Series([r] * n)
    m = performance_metrics(daily, rf=0.0)
    expected_cum = (1 + r) ** n - 1
    assert abs(m["cumulative_return"] - expected_cum) < 1e-9
    assert abs(m["annualized_vol"] - 0.0) < 1e-9
    assert np.isnan(m["sharpe"])  # vol=0 -> sharpe không xác định
    assert abs(m["max_drawdown"] - 0.0) < 1e-9
    assert m["n_days"] == n


def test_metrics_known_drawdown_sequence():
    # Giá trị danh mục: 1.0 -> 1.10 -> 1.045 -> 0.9928 (giảm) -> 1.20 (hồi phục)
    daily = pd.Series([0.10, -0.05, -0.05, 0.209])
    m = performance_metrics(daily, rf=0.0)
    curve = (1 + daily).cumprod()
    running_max = curve.cummax()
    expected_dd = float((curve / running_max - 1).min())
    assert abs(m["max_drawdown"] - expected_dd) < 1e-9
    assert m["max_drawdown"] < 0  # phải âm (giảm giá trị)


def test_metrics_sharpe_sign():
    rng = np.random.default_rng(0)
    daily_pos = pd.Series(rng.normal(0.001, 0.01, 500))
    daily_neg = pd.Series(rng.normal(-0.001, 0.01, 500))
    assert performance_metrics(daily_pos)["sharpe"] > 0
    assert performance_metrics(daily_neg)["sharpe"] < 0
```

- [ ] **Step 2: Chạy test, xác nhận FAIL**

Run: `.venv/bin/python -m pytest tests/test_backtest.py -k metrics -v`
Expected: FAIL (module `src/backtest.py` hoặc hàm chưa tồn tại).

- [ ] **Step 3: Implement `performance_metrics`**

```python
"""
src/backtest.py
=================
Walk-forward backtest out-of-sample cho project sparse+robust portfolio
optimization (VN100). Xem thiết kế đầy đủ ở
docs/superpowers/specs/2026-07-26-walk-forward-backtest-design.md.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger("backtest")

ANNUALIZATION_FACTOR = 252


def performance_metrics(daily_returns: pd.Series, rf: float = 0.0) -> dict:
    """Tính các chỉ số hiệu suất chuẩn từ chuỗi return NGÀY của 1 danh mục.

    rf: lãi suất phi rủi ro NĂM (annualized), mặc định 0 theo thiết kế đã
    chốt. Sharpe = (mean(excess_daily)) / std(daily) * sqrt(252); nếu
    std(daily) == 0 (return hằng số, vd chuỗi giả lập trong test), Sharpe
    KHÔNG xác định -> trả NaN thay vì chia cho 0.

    max_drawdown: mức sụt giảm lớn nhất từ đỉnh giá trị danh mục tích luỹ
    (giá trị ÂM hoặc 0, không phải giá trị dương).
    """
    daily_returns = daily_returns.dropna()
    n = len(daily_returns)
    cumulative_return = float((1 + daily_returns).prod() - 1)

    n_years = n / ANNUALIZATION_FACTOR
    if n_years > 0:
        annualized_return = float((1 + cumulative_return) ** (1 / n_years) - 1)
    else:
        annualized_return = float("nan")

    daily_std = float(daily_returns.std(ddof=1)) if n > 1 else 0.0
    annualized_vol = daily_std * np.sqrt(ANNUALIZATION_FACTOR)

    if daily_std > 0:
        rf_daily = rf / ANNUALIZATION_FACTOR
        sharpe = float(
            (daily_returns.mean() - rf_daily) / daily_std * np.sqrt(ANNUALIZATION_FACTOR)
        )
    else:
        sharpe = float("nan")

    equity_curve = (1 + daily_returns).cumprod()
    running_max = equity_curve.cummax()
    drawdown = equity_curve / running_max - 1
    max_drawdown = float(drawdown.min()) if n > 0 else float("nan")

    return {
        "cumulative_return": cumulative_return,
        "annualized_return": annualized_return,
        "annualized_vol": annualized_vol,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "n_days": n,
    }
```

- [ ] **Step 4: Chạy test, xác nhận PASS**

Run: `.venv/bin/python -m pytest tests/test_backtest.py -v`
Expected: PASS toàn bộ 3 test.

- [ ] **Step 5: Ghi nhận hoàn tất task**

Project này KHÔNG dùng git (xem Global Constraints) — thay vì `git commit`, append 1 dòng vào `.sdd/progress.md` theo đúng quy ước ledger project đã dùng xuyên suốt các phase trước:
`Task <tên task>: complete (feat: add performance_metrics for backtest evaluation)`

---

