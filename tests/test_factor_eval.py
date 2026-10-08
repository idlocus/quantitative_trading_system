#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
因子评价模块测试
核心验证：已知预测力的合成因子 → IC 符号与量级正确；无信息因子 → IC≈0
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import pytest

from research.factor_eval import FactorEvaluator, build_factor_matrix


def make_universe(n_sym=60, n_days=150, seed=42):
    """合成股票池：带截面差异的随机游走"""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range('2024-01-01', periods=n_days)
    cache = {}
    for i in range(n_sym):
        drift = rng.normal(0, 0.002)          # 截面漂移差异
        close = 10 * np.cumprod(1 + drift + rng.normal(0, 0.02, n_days))
        cache[f'S{i:03d}.TEST'] = pd.DataFrame({
            'date': dates,
            'open': close, 'high': close * 1.01, 'low': close * 0.99,
            'close': close, 'volume': rng.integers(1e6, 5e6, n_days),
        })
    return cache


def noisy_lookahead_factor(cache, noise=0.05, seed=7):
    """构造因子 = 未来5日收益 + 噪声（peeking factor，仅测试用）
    合成宇宙5日收益std≈4.5%，噪声std=5% → 理论IC≈0.67"""
    rng = np.random.default_rng(seed)
    factor_data = {}
    for sym, df in cache.items():
        close = df['close'].values
        fwd = np.full(len(close), np.nan)
        fwd[:-5] = close[5:] / close[:-5] - 1
        factor_data[sym] = fwd + rng.normal(0, noise, len(close))
    return factor_data


class TestFactorEvaluator:

    def test_informative_factor_positive_ic(self):
        """含未来信息的因子：IC 显著为正"""
        cache = make_universe()
        ev = FactorEvaluator(cache)

        peek = noisy_lookahead_factor(cache, noise=0.05)
        # 注册为可调用：build_factor_matrix 要求返回 Series 并对齐
        def factor_func(df):
            sym = None
            # 通过 df 内容反查 symbol 不可靠 → 用闭包表替代：
            # 直接测试 evaluate 太绕，改为构造矩阵后再测内部逻辑
            raise NotImplementedError

        # 更直接：手工构造因子矩阵路径 —— 用公开 API 的替代方案：
        # 把 peek 因子伪装成指标：按 close 序列匹配
        keys = list(peek.keys())
        close_to_sym = {tuple(np.round(df['close'].values, 6)): sym for sym, df in cache.items()}

        def factor_func(df):
            sym = close_to_sym.get(tuple(np.round(df['close'].values, 6)))
            if sym is None:
                return pd.Series(dtype=float)
            return pd.Series(peek[sym])

        result = ev.evaluate(factor_func, name='peek', period=5, quantiles=5)
        assert result is not None
        assert result.ic_mean > 0.3, f"含信息因子 IC 应显著为正, got {result.ic_mean}"
        assert result.rankic_mean > 0.3
        assert result.long_short_annual > 0

    def test_useless_factor_zero_ic(self):
        """纯噪声因子：IC 接近 0"""
        cache = make_universe(seed=11)
        ev = FactorEvaluator(cache)
        rng = np.random.default_rng(5)

        def noise_factor(df):
            n = len(df)
            return pd.Series(rng.normal(0, 1, n))

        result = ev.evaluate(noise_factor, name='noise', period=5, quantiles=5)
        assert result is not None
        assert abs(result.ic_mean) < 0.15, f"噪声因子 IC 应接近 0, got {result.ic_mean}"

    def test_momentum_factor_on_trending_universe(self):
        """动量因子在构造的动量宇宙中应有正 IC（强者恒强）"""
        rng = np.random.default_rng(3)
        dates = pd.bdate_range('2024-01-01', periods=200)
        cache = {}
        n = 50
        # 每只股票固定漂移从 -0.003 到 +0.003 线性排列 → 截面动量结构
        for i in range(n):
            drift = -0.003 + 0.006 * i / (n - 1)
            close = 10 * np.cumprod(1 + drift + rng.normal(0, 0.01, len(dates)))
            cache[f'M{i:03d}.TEST'] = pd.DataFrame({
                'date': dates, 'open': close, 'high': close * 1.01,
                'low': close * 0.99, 'close': close,
                'volume': np.full(len(dates), 1e6),
            })

        from indicators.registry import IndicatorRegistry
        roc = IndicatorRegistry.get('roc')

        ev = FactorEvaluator(cache)
        result = ev.evaluate(lambda df: roc(df, period=20), name='roc20', period=5, quantiles=5)
        assert result is not None
        assert result.rankic_mean > 0.2, f"动量宇宙中 ROC 应有正预测力, got {result.rankic_mean}"
        # 分层应单调（Q5 > Q1）
        q_rets = result.quantile_mean_ret
        assert q_rets[max(q_rets)] > q_rets[min(q_rets)]

    def test_build_factor_matrix_alignment(self):
        """因子矩阵日期对齐：值为指标在当日的值"""
        cache = make_universe(n_sym=5, n_days=60)
        from indicators.registry import IndicatorRegistry
        rsi = IndicatorRegistry.get('rsi')

        mx = build_factor_matrix(cache, rsi)
        assert not mx.empty
        assert mx.shape[1] == 5
        # RSI 值域检查（逐有效单元格；pandas 3 的 stack 保留 NaN 需 dropna）
        valid = mx.stack().dropna()
        assert len(valid) > 100
        assert (valid >= 0).all() and (valid <= 100).all()

    def test_summary_markdown_renders(self):
        cache = make_universe(n_sym=40, n_days=100)
        ev = FactorEvaluator(cache)
        rng = np.random.default_rng(9)

        def noise_factor(df):
            return pd.Series(rng.normal(0, 1, len(df)))

        result = ev.evaluate(noise_factor, name='noise', period=5)
        assert result is not None
        md = result.summary_markdown()
        assert '因子评价' in md and 'RankIC' in md
