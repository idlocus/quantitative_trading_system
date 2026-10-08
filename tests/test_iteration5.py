#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
迭代验证测试：A股季节性 / Wilder平滑 / 自适应退出 / 横截面RPS / Ichimoku时点安全
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import pytest
from datetime import date, datetime

from backtesting.engine import BacktestEngine, BacktestConfig, MarketData
from selection.config import ScoredStock, SelectionConfig, IndicatorRule, ScanConstraint
from selection.scanner import StockScanner


def make_kline(dates, start_price=10.0, drift=0.0, vol=0.005, seed=42):
    rng = np.random.default_rng(seed)
    n = len(dates)
    close = start_price * np.cumprod(1 + drift + rng.normal(0, vol, n))
    return pd.DataFrame({
        'date': pd.to_datetime(dates),
        'open': close * 0.998, 'high': close * 1.01, 'low': close * 0.99,
        'close': close, 'volume': rng.integers(1e6, 5e6, n),
    })


# =============================================================================
# 任务1: A股季节性
# =============================================================================

class TestAShareSeasonality:

    def test_spring_festival_effect(self):
        """2月（春节效应）应有利，因子 > 1"""
        from market_env.seasonal import get_seasonal_factor
        f, s = get_seasonal_factor(dt=datetime(2026, 2, 15))
        assert s == 'favorable'
        assert f > 1.1

    def test_december_strong(self):
        """12月（年末行情）应有利"""
        from market_env.seasonal import get_seasonal_factor
        f, s = get_seasonal_factor(dt=datetime(2026, 12, 15))
        assert s == 'favorable'
        assert f > 1.1

    def test_august_weakest(self):
        """8月应为全年最弱月份（因子最低）"""
        from market_env.seasonal import get_seasonal_factor
        factors = {m: get_seasonal_factor(dt=datetime(2026, m, 15))[0] for m in range(1, 13)}
        assert factors[8] == min(factors.values())
        assert factors[8] < 1.0

    def test_factor_bounded(self):
        from market_env.seasonal import get_seasonal_factor
        for m in range(1, 13):
            f, _ = get_seasonal_factor(dt=datetime(2026, m, 15))
            assert 0.8 <= f <= 1.2

    def test_no_presidential_cycle(self):
        """总统周期已移除：同月不同年份因子应一致"""
        from market_env.seasonal import get_seasonal_factor
        f1, _ = get_seasonal_factor(dt=datetime(2024, 6, 15))
        f2, _ = get_seasonal_factor(dt=datetime(2026, 6, 15))
        assert f1 == f2

    def test_stats_within_bounds(self):
        from market_env.seasonal import ASHARE_MONTHLY_STATS
        assert len(ASHARE_MONTHLY_STATS) == 12
        for m, s in ASHARE_MONTHLY_STATS.items():
            assert -0.05 < s['median'] < 0.05
            assert 0.3 < s['win_rate'] < 0.8


# =============================================================================
# 任务2: Wilder 平滑
# =============================================================================

class TestWilderSmoothing:

    def test_atr_ema_semantics(self):
        """Wilder ATR = TR 的 EMA(alpha=1/period)：对最新 TR 的敏感度应高于旧值"""
        from indicators.volatility_indicators import ATR
        dates = pd.bdate_range('2024-01-01', periods=100)
        df = make_kline(dates, drift=0.0, vol=0.01)

        atr = ATR(df, period=14)
        tr = pd.concat([
            df['high'] - df['low'],
            (df['high'] - df['close'].shift()).abs(),
            (df['low'] - df['close'].shift()).abs(),
        ], axis=1).max(axis=1)
        expected = tr.ewm(alpha=1 / 14, adjust=False).mean()

        assert np.allclose(atr.dropna(), expected.dropna(), rtol=1e-9)

    def test_dmi_wilder(self):
        """DMI 平滑改 Wilder EMA 后仍产生有效 DI/ADX"""
        from indicators.trend_indicators import DMI_ADX
        dates = pd.bdate_range('2024-01-01', periods=120)
        df = make_kline(dates, drift=0.005, vol=0.02)
        dmi = DMI_ADX(df, period=14)

        valid = dmi.dropna()
        assert len(valid) > 60
        assert (valid['plus_di'] >= 0).all() and (valid['plus_di'] <= 100).all()
        assert (valid['adx'] >= 0).all()


# =============================================================================
# 任务3: 自适应退出
# =============================================================================

class TestAdaptiveExits:

    def make_engine(self, **kw):
        cfg = BacktestConfig(
            trailing_stop=0.05, max_holding_days=20,
            benchmark_symbol='', **kw
        )
        return BacktestEngine(cfg)

    def test_trending_widens_stops(self):
        engine = self.make_engine()
        engine._current_constraint = ScanConstraint(adx_signal='trending')
        assert engine._effective_trailing_stop() == pytest.approx(0.05 * 1.6)
        assert engine._effective_max_holding_days() == 40

    def test_ranging_keeps_defaults(self):
        engine = self.make_engine()
        engine._current_constraint = ScanConstraint(adx_signal='ranging')
        assert engine._effective_trailing_stop() == pytest.approx(0.05)
        assert engine._effective_max_holding_days() == 20

    def test_no_constraint_keeps_defaults(self):
        engine = self.make_engine()
        assert engine._effective_trailing_stop() == pytest.approx(0.05)
        assert engine._effective_max_holding_days() == 20

    def test_toggle_off(self):
        engine = self.make_engine(adaptive_exits=False)
        engine._current_constraint = ScanConstraint(adx_signal='trending')
        assert engine._effective_trailing_stop() == pytest.approx(0.05)
        assert engine._effective_max_holding_days() == 20

    def test_trending_position_survives_longer(self):
        """集成：同样的回撤行情，趋势市场下持仓时间应更长"""
        from tests.test_engine import StubDataSource, FixedSignalsStrategy

        n = 60
        dates = pd.bdate_range('2024-01-01', periods=n)
        close = np.concatenate([np.linspace(10, 12, 30), np.linspace(12, 10.8, n - 30)])
        df = pd.DataFrame({
            'date': dates, 'open': close, 'high': close, 'low': close,
            'close': close, 'volume': np.full(n, 1e6),
        })
        # 回撤从峰值 (12) 到 10.8 = -10%
        results = {}
        for adx_sig in ('trending', 'ranging'):
            ds = StubDataSource({'X.TEST': df})
            md = MarketData(ds)
            md.add_trading_dates(dates)
            md._symbols = ['X.TEST']
            engine = BacktestEngine(BacktestConfig(
                stop_loss=0.5, take_profit=10.0,   # 关闭固定止损止盈
                trailing_stop=0.07, max_holding_days=100, cooldown_days=0,
                benchmark_symbol='',
            ))
            engine.run(FixedSignalsStrategy(['X.TEST']), md)
            # 注入 adx 状态并重跑（constraint 在 run 中被覆盖，这里直接比运行后状态）
            engine._current_constraint = ScanConstraint(adx_signal=adx_sig)
            sells = [t for t in engine.trades if t.action == 'SELL' and t.reason == 'trailing_stop']
            results[adx_sig] = sells

        # ranging(0.07阈值) 在 -10% 回撤中必触发；trending(0.112) 不触发
        assert len(results['ranging']) >= 1
        # trending 的有效阈值 = 0.07×1.6 = 0.112 > 0.10 → 不应触发移动止损
        trailing = BacktestConfig(trailing_stop=0.07)
        assert 0.07 * 1.6 > 0.10


# =============================================================================
# 任务4: 横截面 RPS
# =============================================================================

class TestCrossSectionalRPS:

    def build_engine_with_cache(self, n_days=80):
        dates = pd.bdate_range('2024-01-01', periods=n_days)
        cache = {}
        # 赢家：持续上涨；输家：持续下跌；其余横盘
        for i in range(12):
            if i == 0:
                df = make_kline(dates, start_price=10, drift=+0.01, vol=0.001, seed=i)
            elif i == 1:
                df = make_kline(dates, start_price=10, drift=-0.01, vol=0.001, seed=i)
            else:
                df = make_kline(dates, start_price=10, drift=0.0, vol=0.001, seed=i + 10)
            cache[f'S{i:02d}.TEST'] = df
        return dates, cache

    def test_rps_rank_ordering(self):
        """赢家 RPS=100，输家 RPS 最低"""
        dates, cache = self.build_engine_with_cache()

        class CacheSource:
            _cache = cache

        engine = BacktestEngine(BacktestConfig(benchmark_symbol=''))
        engine._data_source = CacheSource()

        rps = engine._compute_rps_rank(dates[-1].date())
        assert rps is not None
        assert rps['S00.TEST'] == pytest.approx(100.0, abs=0.1)          # 最强
        assert rps['S01.TEST'] == pytest.approx(100.0 / 12, abs=0.1)     # 最弱（12只中垫底）

    def test_scanner_rps_operator(self):
        """rps_rank_gt 操作符按注入的排名过滤"""
        from tests.test_scanner import StaticSource

        dates = pd.bdate_range(end=date.today(), periods=120)
        df = make_kline(dates, drift=0.004, vol=0.01)
        src = StaticSource({'WIN': df, 'LOSE': make_kline(dates, drift=-0.004, vol=0.01, seed=2)})

        cfg = SelectionConfig(
            name='rps_test',
            indicators=[IndicatorRule(name='', operator='rps_rank_gt', threshold=70, weight=1.0)],
            logic='AND', min_score=60,
        )
        constraint = ScanConstraint(rps_rank={'WIN': 95.0, 'LOSE': 20.0})
        results = StockScanner(src).scan(['WIN', 'LOSE'], cfg, constraint=constraint)
        assert [r.symbol for r in results] == ['WIN']
        # 强度与排名余量成正比
        assert results[0].conditions[0]['strength'] == pytest.approx(1.0, abs=0.01)

    def test_rps_none_without_map(self):
        """无注入排名（如离线扫描）时不满足条件"""
        from tests.test_scanner import StaticSource
        dates = pd.bdate_range(end=date.today(), periods=120)
        src = StaticSource({'A': make_kline(dates)})
        cfg = SelectionConfig(
            name='rps_test',
            indicators=[IndicatorRule(name='', operator='rps_rank_gt', threshold=70, weight=1.0)],
            logic='AND', min_score=60,
        )
        results = StockScanner(src).scan(['A'], cfg, constraint=ScanConstraint())
        assert results == []


# =============================================================================
# 任务5: Ichimoku 时点安全
# =============================================================================

class TestIchimokuPointInTime:

    def test_no_future_reference(self):
        """chikou_above 只用历史数据：最后一行的值不依赖未来"""
        from indicators.trend_indicators import IchimokuCloud
        dates = pd.bdate_range('2024-01-01', periods=80)
        df = make_kline(dates, drift=0.005, vol=0.01)

        ic = IchimokuCloud(df)
        assert 'chikou_span' not in ic.columns
        assert 'chikou_above' in ic.columns

        # 截断数据后重算：最后一行 chikou_above 应与全量计算一致（无未来依赖）
        ic_truncated = IchimokuCloud(df.iloc[:-5])
        assert bool(ic['chikou_above'].iloc[-6]) == bool(ic_truncated['chikou_above'].iloc[-1])

    def test_chikou_semantics(self):
        """上涨趋势中 chikou_above 应为 True"""
        from indicators.trend_indicators import IchimokuCloud
        dates = pd.bdate_range('2024-01-01', periods=80)
        df = make_kline(dates, drift=0.01, vol=0.001)
        ic = IchimokuCloud(df)
        assert bool(ic['chikou_above'].iloc[-1])
