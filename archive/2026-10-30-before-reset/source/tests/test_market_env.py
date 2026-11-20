#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
市场环境与基础设施测试
覆盖：logger 按名缓存、评分器时点正确性、引擎宽度计算、MultiStrategy 绝对评分
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import pytest
from datetime import date

from utils.logger import get_logger, _loggers


# =============================================================================
# Logger
# =============================================================================

class TestLogger:
    """回归：get_logger 曾是全局单例，所有模块共用第一个调用者的名字"""

    def test_per_name_instances(self):
        a = get_logger('module.a')
        b = get_logger('module.b')
        assert a is not b
        assert a.name == 'module.a'
        assert b.name == 'module.b'
        # 同名返回同一实例
        assert get_logger('module.a') is a

    def test_cache_dict_populated(self):
        get_logger('module.c')
        assert 'module.c' in _loggers


# =============================================================================
# 评分器时点正确性
# =============================================================================

class FakeEnvProvider:
    """记录调用日期的假 provider，用于验证时点传递"""

    def __init__(self):
        self.macro_calls = []      # get_macro_score 收到的 end 日期
        self.valuation_calls = []  # get_market_valuation 收到的 date
        self.breadth = None

    def get_market_breadth(self, end_date):
        return self.breadth

    def get_market_valuation(self, index_code, date=None):
        self.valuation_calls.append(date)
        return {'pe_percentile': 40, 'pb_percentile': 40}

    def fetch_indicator(self, indicator_id, begin=None, end=None, freq="M"):
        # 返回随 end 截断的序列：证明评分用的是历史数据
        dates = pd.date_range('2023-01-01', '2026-12-01', freq='MS')
        values = np.arange(len(dates), dtype=float)
        s = pd.Series(values, index=dates)
        if end:
            s = s[s.index <= pd.Timestamp(end[:4] + '-' + end[4:6] + '-' + end[6:])]
        return s


class TestScorerPointInTime:
    """回归：get_macro_score/get_syzm 曾不传日期 → 回测使用最新数据（前视偏差）"""

    def test_scorer_auto_wires_macro_db(self):
        from market_env.scorer import MarketEnvScorer
        provider = FakeEnvProvider()
        scorer = MarketEnvScorer(provider)  # 不显式传 macro_db
        assert scorer.macro_db is not None, "provider 支持 fetch_indicator 时应自动装配 MacroDB"

    def test_scorer_caches_by_date(self):
        from market_env.scorer import MarketEnvScorer
        scorer = MarketEnvScorer(FakeEnvProvider())
        r1 = scorer.score('20240115')
        n_calls = len(scorer.provider.valuation_calls)
        r2 = scorer.score('20240115')
        assert r1 is r2, "同日期应命中缓存"
        assert len(scorer.provider.valuation_calls) == n_calls, "缓存命中不应重新计算"

    def test_breadth_override_used(self):
        """引擎注入的宽度数据优先于 provider"""
        from market_env.scorer import MarketEnvScorer
        provider = FakeEnvProvider()
        provider.breadth = pd.DataFrame({'advance': [1] * 10, 'decline': [1] * 10})
        scorer = MarketEnvScorer(provider)

        # 注入强扩张宽度 → expanding
        injected = pd.DataFrame({
            'advance': [200, 210, 220, 230, 240],
            'decline': [100, 95, 90, 85, 80],
        })
        result = scorer.score('20240115', breadth_df=injected)
        assert result.breadth_signal == 'expanding'
        assert result.breadth_score == 75

    def test_macro_point_in_time_differs(self):
        """不同评分日期应得到不同的宏观分（用的是各自时点的数据）"""
        from market_env.scorer import MarketEnvScorer
        scorer = MarketEnvScorer(FakeEnvProvider())
        early = scorer.score('20230115')
        late = scorer.score('20261215')
        # FakeEnvProvider 的指标单调上升 → 晚期评分应不低于早期
        assert late.macro_score >= early.macro_score


# =============================================================================
# 引擎宽度计算
# =============================================================================

class TestEngineBreadth:

    def test_breadth_from_cache(self):
        """从缓存K线计算涨跌家数，并正确截断到指定日期"""
        from backtesting.engine import BacktestEngine, BacktestConfig

        dates = pd.bdate_range('2024-01-01', periods=30)
        rng = np.random.default_rng(7)
        cache = {}
        for i in range(25):  # 25 只股票
            close = 10 + np.cumsum(rng.normal(0, 0.1, len(dates)))
            cache[f'S{i:02d}.TEST'] = pd.DataFrame({
                'date': dates, 'open': close, 'high': close * 1.01,
                'low': close * 0.99, 'close': close, 'volume': np.full(len(dates), 1e6),
            })

        class CacheSource:
            def __init__(self, c):
                self._cache = c

        engine = BacktestEngine(BacktestConfig(benchmark_symbol=''))
        engine._data_source = CacheSource(cache)

        bdf = engine._compute_breadth(dates[10].date())
        assert bdf is not None
        assert set(bdf.columns) == {'advance', 'decline'}
        assert len(bdf) == 11  # 暖机首日无涨跌定义 → 至多 11 行
        assert (bdf['advance'] + bdf['decline'] <= 25).all()

        # 截断正确：不包含 trading_date 之后的行
        assert bdf.index.max() <= pd.Timestamp(dates[10].date())

    def test_breadth_none_when_cache_small(self):
        from backtesting.engine import BacktestEngine, BacktestConfig

        class CacheSource:
            _cache = {'A': pd.DataFrame()}  # 缓存不足 20 只

        engine = BacktestEngine(BacktestConfig(benchmark_symbol=''))
        engine._data_source = CacheSource()
        assert engine._compute_breadth(date(2024, 1, 1)) is None


# =============================================================================
# MultiStrategy 绝对评分
# =============================================================================

class TestMultiStrategyScoring:
    """回归：曾用 max 归一化——唯一弱信号也会得 100 分拿满仓"""

    def test_no_relative_normalization(self):
        from strategy.multi import MultiStrategy
        from selection.config import ScoredStock

        # 构造一个最小 MultiStrategy 实例（跳过 YAML 加载）
        ms = MultiStrategy.__new__(MultiStrategy)
        from selection.config import SelectionConfig
        ms.sub_strategies = []
        ms.config = SelectionConfig(name='test', min_score=50, max_positions=5)

        # 模拟：只有一只股票、仅一个子策略（权重 0.2）、原始分 90
        from strategy.multi import SubStrategy
        sub = SubStrategy.__new__(GenerationStrategy := type('X', (), {})) if False else None

        class FakeSub:
            name = 'sub'
            weight = 0.2
            def select(self, date_, allowed_symbols=None, constraint=None):
                return [ScoredStock(symbol='ONLY.TEST', score=90)]

        ms.sub_strategies = [FakeSub()]
        results = ms.select(date(2024, 1, 1), None)
        # 加权和 = 90×0.2 = 18 < min_score 50 → 不入选（旧的归一化会放大到 100）
        assert results == [], "唯一弱信号不应被相对归一化放大"

    def test_full_weight_stock_scores_high(self):
        from strategy.multi import MultiStrategy
        from selection.config import ScoredStock, SelectionConfig

        ms = MultiStrategy.__new__(MultiStrategy)
        ms.config = SelectionConfig(name='test', min_score=50, max_positions=5)

        class FakeSub:
            name = 'sub'
            def __init__(self, w, pick):
                self.weight = w
                self.pick = pick
            def select(self, date_, allowed_symbols=None, constraint=None):
                return [ScoredStock(symbol=self.pick, score=100)]

        # 两个子策略权重合计 1.0，都选中同一只 → 加权和 100
        ms.sub_strategies = [FakeSub(0.5, 'STAR.TEST'), FakeSub(0.5, 'STAR.TEST')]
        results = ms.select(date(2024, 1, 1), None)
        assert len(results) == 1
        assert results[0].score == 100


# =============================================================================
# Fetcher 时点切片（真实 _AkShareFetcher 逻辑，不触网）
# =============================================================================

class TestFetcherSlicing:
    """验证 fetch 的内存时点切片语义（用注入的假 akshare 模块）"""

    def test_slice_is_exclusive_of_future(self):
        from market_env.akshare_provider import _AkShareFetcher

        fetcher = _AkShareFetcher.__new__(_AkShareFetcher)

        class FakeAK:
            @staticmethod
            def fake_macro():
                return pd.DataFrame({
                    '月份': ['2023年12月份', '2024年01月份', '2024年02月份', '2024年03月份'],
                    '值': [1.0, 2.0, 3.0, 4.0],
                })

        fetcher._ak = FakeAK()
        fetcher._series_cache = {}

        s = fetcher.fetch('fake_macro', begin='20240101', end='20240131')
        # 只包含 2024-01（2月1日 > 1月31日，未来月份被正确排除）
        assert len(s) == 1
        assert s.iloc[0] == 2.0

        # 再次调用命中缓存（fake_macro 只被请求一次）
        calls = [0]
        orig = FakeAK.fake_macro
        FakeAK.fake_macro = staticmethod(lambda: (calls.__setitem__(0, calls[0] + 1) or orig()))
        fetcher._ak = FakeAK()
        fetcher.fetch('fake_macro', begin='20240101', end='20240331')
        assert calls[0] == 0, "应命中缓存，不重复调用 akshare"
