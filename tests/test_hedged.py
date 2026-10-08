#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""β对冲测试：合成组合=1.3×基准+正漂移噪声 → 对冲后市场相关≈0、夏普为正"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import pytest

from portfolio.hedged import beta_hedge


def test_hedge_removes_market_exposure():
    rng = np.random.default_rng(3)
    n = 500
    dates = pd.bdate_range('2024-01-01', periods=n)
    r_b = pd.Series(rng.normal(0.0002, 0.012, n), index=dates)
    bench = (1 + r_b).cumprod()
    # 多头 = 1.3β + 日均+0.1% alpha
    r_p = 1.3 * r_b + 0.001 + rng.normal(0, 0.003, n)
    equity = (1 + r_p).cumprod()

    res = beta_hedge(equity, bench, beta_window=60, cost_annual=0.0)
    assert abs(res.market_correlation) < 0.25, f"市场相关 {res.market_correlation:.2f} 应接近0"
    assert res.sharpe > 1.0, "真alpha对冲后夏普应为正且高"
    # beta 序列应收敛到 ~1.3
    assert 1.0 < res.beta_series.iloc[-1] < 1.7


def test_cost_reduces_return():
    rng = np.random.default_rng(5)
    n = 300
    dates = pd.bdate_range('2024-01-01', periods=n)
    r_b = pd.Series(rng.normal(0, 0.01, n), index=dates)
    bench = (1 + r_b).cumprod()
    r_p = r_b + 0.002  # β=1 纯alpha
    equity = (1 + r_p).cumprod()

    free = beta_hedge(equity, bench, cost_annual=0.0)
    paid = beta_hedge(equity, bench, cost_annual=0.08)
    assert paid.total_return < free.total_return
    assert free.total_return > 0.30  # 0.002×300日
