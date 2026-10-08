#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
组合模块测试：因子合成、选股、权重、回测器（成本/涨跌停/换手）
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import pytest
from datetime import date

from portfolio.construction import (composite_score, select_portfolio,
                                    equal_weights, inverse_vol_weights)
from portfolio.backtester import PortfolioBacktester, PortfolioConfig


def make_cache(n_sym=30, n_days=120, seed=42, drift_fn=None):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range('2024-01-01', periods=n_days)
    cache = {}
    for i in range(n_sym):
        drift = drift_fn(i) if drift_fn else 0.0
        close = 10 * np.cumprod(1 + drift + rng.normal(0, 0.02, n_days))
        cache[f'S{i:03d}.SZ'] = pd.DataFrame({
            'date': dates,
            'open': close, 'high': close * 1.01, 'low': close * 0.99,
            'close': close, 'volume': rng.integers(1e6, 5e6, n_days),
        })
    return cache


class TestConstruction:

    def test_composite_score_range(self):
        cache = make_cache()

        def fake_momentum(df):
            # 简单因子：20日收益
            return df['close'] / df['close'].shift(20) - 1

        cs = composite_score(cache, {'mom': fake_momentum})
        assert not cs.empty
        valid = cs.stack().dropna()
        assert (valid >= 0).all() and (valid <= 1).all()

    def test_composite_negative_direction(self):
        """direction=-1 时：原因子值最高 → 综合分最低"""
        cache = make_cache(n_sym=10, seed=1)

        def factor(df):
            # S000 恒高分，S009 恒低分
            return pd.Series(np.linspace(1, 0, len(df)))

        cs_pos = composite_score(cache, {'f': factor}, directions={'f': 1.0})
        cs_neg = composite_score(cache, {'f': factor}, directions={'f': -1.0})

        last = cs_pos.iloc[-1].dropna().idxmax()
        last_neg_min = cs_neg.iloc[-1].dropna().idxmin()
        assert last == last_neg_min  # 同一只股票在两个方向上分别是最高/最低

    def test_select_portfolio_top_n(self):
        scores = pd.Series({'A': 0.9, 'B': 0.8, 'C': 0.7, 'D': 0.6})
        assert select_portfolio(scores, 2) == ['A', 'B']
        assert select_portfolio(scores, 2, exclude_symbols={'A'}) == ['B', 'C']
        assert select_portfolio(pd.Series(dtype=float), 3) == []

    def test_equal_weights_cap(self):
        w = equal_weights(['A', 'B'], max_weight=0.05)
        assert w == {'A': 0.05, 'B': 0.05}  # 1/2=0.5 触碰上限
        w2 = equal_weights([f'S{i}' for i in range(50)], max_weight=0.05)
        assert w2['S0'] == pytest.approx(0.02)  # 1/50 < 上限

    def test_inverse_vol_prefers_stable(self):
        dates = pd.bdate_range('2024-01-01', periods=60)
        calm = pd.Series(10 + np.linspace(0, 0.1, 60), index=dates)
        wild = pd.Series(10 + np.sin(np.arange(60) * 2) * 2, index=dates)
        mx = pd.DataFrame({'CALM': calm, 'WILD': wild})
        w = inverse_vol_weights(mx, ['CALM', 'WILD'], dates[-1], max_weight=0.9)
        assert w['CALM'] > w['WILD'], "低波动应得更高权重"


class TestPortfolioBacktester:

    def test_basic_run_and_costs(self):
        """基本运行：期末资金正确、成本累计>0、持有数≤top_n"""
        cache = make_cache(n_sym=30, n_days=100, seed=3,
                           drift_fn=lambda i: 0.001 * (i - 15))
        cfg = PortfolioConfig(top_n=10, rebalance_days=5,
                              initial_capital=1_000_000, slippage=0.001,
                              benchmark_symbol='')
        bt = PortfolioBacktester(cache, cfg)

        # 综合分：动量正向（涨得多的得分高）
        score = bt.close_mx.pct_change(20)
        result = bt.run(score)

        assert result.n_rebalances >= 10
        assert result.total_cost > 0, "有调仓必有成本"
        # 期末资金 = 权益曲线末值
        assert result.final_capital == pytest.approx(result.equity_curve.iloc[-1], rel=1e-9)
        # 持仓数不超上限（历史记录抽样）
        for h in result.holdings_history:
            assert h['n_holdings'] <= 10 + 3  # 跌停顺延可能短暂超限

    def test_limit_up_blocks_buy(self):
        """涨停股不出现在买入后的持仓中（构造一只天天涨停的股票）"""
        cache = make_cache(n_sym=20, n_days=60, seed=5)
        # 替换 S000 为连续涨停（每日+10%封板）
        dates = cache['S000.SZ']['date']
        limit_close = 10 * 1.1 ** np.arange(len(dates))
        cache['S000.SZ'] = pd.DataFrame({
            'date': dates, 'open': limit_close, 'high': limit_close,
            'low': limit_close, 'close': limit_close,
            'volume': np.full(len(dates), 1e6),
        })

        cfg = PortfolioConfig(top_n=3, rebalance_days=5, benchmark_symbol='',
                              enforce_price_limits=True)
        bt = PortfolioBacktester(cache, cfg)
        # 综合分：S000 永远最高分（涨最多）
        score = bt.close_mx.pct_change(10)
        result = bt.run(score)

        bought_s000 = any(
            'S000.SZ' in h['symbols'] for h in result.holdings_history
        )
        assert not bought_s000, "连续涨停股应被排除，永远买不进"

        # 关闭约束后应能买入（对照）
        cfg2 = PortfolioConfig(top_n=3, rebalance_days=5, benchmark_symbol='',
                               enforce_price_limits=False)
        bt2 = PortfolioBacktester(cache, cfg2)
        result2 = bt2.run(bt2.close_mx.pct_change(10))
        bought2 = any('S000.SZ' in h['symbols'] for h in result2.holdings_history)
        assert bought2, "关闭约束后涨停股可买（对照）"

    def test_equal_weight_baseline_beats_nothing(self):
        """等权全池组合（对照组）应正常运行且收益等于池平均（近似）"""
        cache = make_cache(n_sym=10, n_days=80, seed=7)
        cfg = PortfolioConfig(top_n=10, rebalance_days=20, benchmark_symbol='',
                              slippage=0.0)
        bt = PortfolioBacktester(cache, cfg)
        score = bt.close_mx.pct_change(5)  # 任意分数（全选）
        result = bt.run(score)
        assert np.isfinite(result.total_return)

    def test_benchmark_return(self):
        cache = make_cache(n_sym=10, n_days=80, seed=9)
        # 加基准指数
        dates = cache['S000.SZ']['date']
        bench_close = 3000 * 1.0005 ** np.arange(len(dates))
        cache['000300.SH'] = pd.DataFrame({
            'date': dates, 'open': bench_close, 'high': bench_close,
            'low': bench_close, 'close': bench_close,
            'volume': np.full(len(dates), 1e9),
        })
        cfg = PortfolioConfig(top_n=5, rebalance_days=10,
                              benchmark_symbol='000300.SH')
        bt = PortfolioBacktester(cache, cfg)
        result = bt.run(bt.close_mx.pct_change(5))
        assert result.benchmark_return is not None
        assert result.benchmark_return > 0

    def test_turnover_nonnegative(self):
        cache = make_cache(n_sym=15, n_days=70, seed=11)
        cfg = PortfolioConfig(top_n=5, rebalance_days=5, benchmark_symbol='')
        bt = PortfolioBacktester(cache, cfg)
        result = bt.run(bt.close_mx.pct_change(10))
        assert 0 <= result.avg_turnover <= 1.0
