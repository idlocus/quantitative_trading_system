#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
选股扫描器测试：连续强度评分
覆盖：评分区间、强度区分度、AND/OR 逻辑门、操作符强度
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import pytest
from datetime import date, timedelta

from selection.scanner import StockScanner
from selection.config import SelectionConfig, IndicatorRule


def make_ohlcv(n=120, start_price=10.0, drift=0.0, vol=0.02, seed=42):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(end=date.today(), periods=n)
    close = start_price * np.cumprod(1 + drift + rng.normal(0, vol, n))
    return pd.DataFrame({
        'date': dates,
        'open': close * 0.998,
        'high': close * 1.015,
        'low': close * 0.985,
        'close': close,
        'volume': rng.integers(1e6, 5e6, n),
    })


class StaticSource:
    """返回预置K线的数据源"""

    def __init__(self, mapping):
        self.mapping = mapping

    def get_kline(self, symbol, start, end):
        df = self.mapping[symbol]
        return df[(df['date'] >= pd.Timestamp(start)) & (df['date'] <= pd.Timestamp(end))]


class TestContinuousScoring:
    """连续评分：score = 60 + 40 × 加权平均强度"""

    def test_score_range_and_differentiation(self):
        """同一条件（RSI<40）下，RSI 更低的股票应得更高分"""
        # 构造两只 RSI 差异明显的股票
        weak = make_ohlcv(120, drift=-0.002, vol=0.01, seed=1)   # 温和下跌 RSI≈35
        strong = make_ohlcv(120, drift=-0.010, vol=0.02, seed=2) # 急跌 RSI≈20
        src = StaticSource({'WEAK': weak, 'STRONG': strong})

        cfg = SelectionConfig(
            name='test',
            indicators=[IndicatorRule(name='rsi', operator='<', threshold=40, weight=1.0)],
            logic='AND', min_score=0,
        )
        results = StockScanner(src).scan(['WEAK', 'STRONG'], cfg)
        assert len(results) == 2
        scores = {r.symbol: r.score for r in results}
        assert all(60 <= s <= 100 for s in scores.values()), f"分数应在60-100区间: {scores}"
        assert scores['STRONG'] >= scores['WEAK'], \
            f"更强的超卖信号应得更高分: {scores}"

    def test_and_gate_rejects_partial(self):
        """AND 逻辑：任一条件不满足 → score=0 被 min_score 过滤"""
        up = make_ohlcv(120, drift=+0.01, seed=5)  # 上涨，RSI 高
        src = StaticSource({'UP': up})
        cfg = SelectionConfig(
            name='test',
            indicators=[
                IndicatorRule(name='rsi', operator='<', threshold=10, weight=1.0),  # 几乎不可能满足
                IndicatorRule(name='rsi', operator='>', threshold=5, weight=1.0),
            ],
            logic='AND', min_score=60,
        )
        results = StockScanner(src).scan(['UP'], cfg)
        assert results == []

    def test_or_gate_requires_one(self):
        """OR 逻辑：至少一个条件满足"""
        up = make_ohlcv(120, drift=+0.01, seed=5)
        src = StaticSource({'UP': up})
        cfg = SelectionConfig(
            name='test',
            indicators=[
                IndicatorRule(name='rsi', operator='<', threshold=10, weight=1.0),  # 不满足
                IndicatorRule(name='rsi', operator='>', threshold=5, weight=1.0),   # 满足
            ],
            logic='OR', min_score=0,
        )
        results = StockScanner(src).scan(['UP'], cfg)
        assert len(results) == 1
        assert results[0].score >= 60

    def test_conditions_carry_strength(self):
        """满足的条件附带 strength 字段（0-1）"""
        down = make_ohlcv(120, drift=-0.008, seed=9)
        src = StaticSource({'DOWN': down})
        cfg = SelectionConfig(
            name='test',
            indicators=[IndicatorRule(name='rsi', operator='<', threshold=50, weight=1.0)],
            logic='AND', min_score=0,
        )
        results = StockScanner(src).scan(['DOWN'], cfg)
        assert len(results) == 1
        assert len(results[0].conditions) == 1
        s = results[0].conditions[0]['strength']
        assert 0.0 <= s <= 1.0

    def test_distance_strength_scale_free(self):
        """相对距离强度与指标量纲无关"""
        assert StockScanner._distance_strength(50, 40) == pytest.approx(10 / 90, abs=1e-6)
        assert StockScanner._distance_strength(5000, 4000) == pytest.approx(1000 / 9000, abs=1e-6)
        assert StockScanner._distance_strength(40, 40) == pytest.approx(0.0)
        assert StockScanner._distance_strength(1e9, 0) == pytest.approx(1.0)
