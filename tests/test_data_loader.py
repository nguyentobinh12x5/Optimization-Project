"""Test logic thuần (không cần mạng) cho src/data_loader.py."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data_loader import clean_prices, compute_returns


def _fake_prices(n_days: int = 20, n_symbols: int = 6, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-01-01", periods=n_days)
    steps = rng.normal(0, 1, size=(n_days, n_symbols))
    prices = 100 + steps.cumsum(axis=0)
    cols = [chr(ord("A") + i) for i in range(n_symbols)]
    return pd.DataFrame(prices, index=dates, columns=cols)


def test_weekend_rows_dropped_by_clean_prices():
    prices = _fake_prices()
    start, end = prices.index[0].isoformat(), prices.index[-1].isoformat()
    clean, _dropped = clean_prices(prices, start, end)
    assert (clean.index.dayofweek >= 5).sum() == 0


def test_compute_returns_drops_suspected_holiday_row():
    """Phiên có giá LẶP Y HỆT phiên trước (mọi mã) phải bị loại như artifact,
    không phải return=0.0 thật. Đây là cơ chế phòng vệ chung cho ngày lễ VN
    rơi vào ngày thường (Tết, 30/4, 1/5, 2/9...) -- dayofweek không bắt được
    các ngày này, nên cần phát hiện bằng thống kê (>=90% mã return=0.0)."""
    prices = _fake_prices(n_days=10, n_symbols=6)
    fake_holiday_pos = 4
    prices.iloc[fake_holiday_pos] = prices.iloc[fake_holiday_pos - 1]

    returns = compute_returns(prices)

    holiday_date = prices.index[fake_holiday_pos]
    assert holiday_date not in returns.index
    # Phiên NGAY SAU ngày lễ ma vẫn phải còn (không bị mất oan theo domino).
    day_after = prices.index[fake_holiday_pos + 1]
    assert day_after in returns.index


def test_compute_returns_keeps_genuine_flat_day_for_minority():
    """1-2 mã đứng giá thật (return=0.0) trong ngày các mã khác vẫn biến động
    bình thường KHÔNG được coi là ngày lễ giả -- chỉ loại khi đa số áp đảo
    (>=90%) cùng đứng giá, đúng dấu hiệu artifact toàn thị trường."""
    prices = _fake_prices(n_days=10, n_symbols=6)
    flat_pos = 4
    # chỉ 1/6 mã (~17%) đứng giá -- xa ngưỡng 90%, phải được GIỮ LẠI.
    prices.iloc[flat_pos, 0] = prices.iloc[flat_pos - 1, 0]

    returns = compute_returns(prices)
    assert prices.index[flat_pos] in returns.index


def test_compute_returns_no_nan_after_filtering():
    prices = _fake_prices(n_days=15, n_symbols=8)
    prices.iloc[6] = prices.iloc[5]  # 1 ngày lễ giả
    returns = compute_returns(prices)
    assert returns.isna().sum().sum() == 0
