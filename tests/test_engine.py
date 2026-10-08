#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
回测引擎测试：合成数据验证核心行为
覆盖：构造回归、期末估值、T+1、移动止损、时间止损、冷却期、滑点、基准、日历推导
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import pytest
from datetime import date, timedelta

from backtesting.engine import (
    BacktestEngine, BacktestConfig, MarketData, BacktestResult
)
from selection.config import ScoredStock


# =============================================================================
# 合成数据工具
# =============================================================================

def make_kline(dates, start_price=10.0, drift=0.0, vol=0.005, seed=42):
    """按给定日期序列生成 OHLCV DataFrame（每日漂移 + 噪声）"""
    rng = np.random.default_rng(seed)
    n = len(dates)
    close = start_price * np.cumprod(1 + drift + rng.normal(0, vol, n))
    return pd.DataFrame({
        'date': pd.to_datetime(dates),
        'open': close * 0.998,
        'high': close * 1.01,
        'low': close * 0.99,
        'close': close,
        'volume': rng.integers(1e6, 5e6, n),
    })


class StubDataSource:
    """最小数据源：直接持有每只股票的 K线 DataFrame"""

    def __init__(self, cache):
        self.cache = cache  # symbol -> DataFrame

    def get_kline(self, symbol, start, end):
        if symbol not in self.cache:
            return pd.DataFrame()
        df = self.cache[symbol]
        return df[(df['date'] >= pd.Timestamp(start)) & (df['date'] <= pd.Timestamp(end))]

    def get_close(self, symbol, trading_date):
        if symbol not in self.cache:
            return 0.0
        df = self.cache[symbol]
        mask = df['date'] <= pd.Timestamp(trading_date)
        return float(df.loc[mask, 'close'].iloc[-1]) if mask.any() else 0.0


class FixedSignalsStrategy:
    """每日返回固定信号（score 可配置），用于隔离选股逻辑只测引擎"""

    def __init__(self, symbols, score=80):
        self.symbols = symbols
        self.score = score

    def select(self, date, data, allowed_symbols=None, constraint=None):
        return [ScoredStock(symbol=s, score=self.score) for s in self.symbols]


@pytest.fixture
def bdays():
    return pd.bdate_range('2024-01-01', periods=60)


def build_engine(ds, symbols, dates, **cfg_overrides):
    """构造引擎 + 市场数据的公共辅助（overrides 覆盖默认值）"""
    md = MarketData(ds)
    md.add_trading_dates(dates)
    md._symbols = list(symbols)
    defaults = dict(
        initial_capital=1_000_000,
        commission=0.0003,
        max_positions=5,
        trailing_stop=0.0,
        max_holding_days=0,
        cooldown_days=0,
        benchmark_symbol='',
    )
    defaults.update(cfg_overrides)
    return BacktestEngine(BacktestConfig(**defaults)), md


# =============================================================================
# 构造与估值
# =============================================================================

class TestEngineBasics:
    """引擎构造（回归：RiskManager 签名不匹配曾导致构造崩溃）与期末估值"""

    def test_engine_constructs(self):
        engine, _ = build_engine(StubDataSource({}), [], pd.bdate_range('2024-01-01', periods=5))
        assert engine.risk_manager is not None

    def test_final_capital_marked_at_last_date(self, bdays):
        """期末持仓按最后交易日收盘价估值（而非入场价）"""
        up = make_kline(bdays, start_price=10.0, drift=+0.01, seed=1)
        ds = StubDataSource({'UP.TEST': up})
        engine, md = build_engine(ds, ['UP.TEST'], bdays, max_positions=1)
        result = engine.run(FixedSignalsStrategy(['UP.TEST']), md)

        last_close = float(up['close'].iloc[-1])
        expected = engine.cash + last_close * engine.positions['UP.TEST'].quantity
        assert result.final_capital == pytest.approx(expected, abs=0.01)

    def test_equity_curve_indexed_by_dates(self, bdays):
        up = make_kline(bdays, drift=+0.005)
        ds = StubDataSource({'UP.TEST': up})
        engine, md = build_engine(ds, ['UP.TEST'], bdays)
        result = engine.run(FixedSignalsStrategy(['UP.TEST']), md)
        assert list(result.equity_curve.index) == [d.date() for d in bdays]


# =============================================================================
# 退出机制
# =============================================================================

class TestExitMechanisms:
    """移动止损 / 时间止损 / T+1 / 冷却期"""

    def test_t_plus_1_no_same_day_sell(self, bdays):
        """当天买入不能当天卖出（即使触发止损）"""
        # 首日暴跌数据：第0天 100，第1天 50（-50%，必触发止损）
        close = np.where(np.arange(len(bdays)) == 0, 100.0, 50.0)
        df = pd.DataFrame({
            'date': pd.to_datetime(bdays), 'open': close, 'high': close,
            'low': close, 'close': close, 'volume': np.full(len(bdays), 1e6),
        })
        ds = StubDataSource({'CRASH.TEST': df})
        engine, md = build_engine(ds, ['CRASH.TEST'], bdays, stop_loss=0.03)
        engine.run(FixedSignalsStrategy(['CRASH.TEST']), md)

        buys = [t for t in engine.trades if t.action == 'BUY']
        sells = [t for t in engine.trades if t.action == 'SELL']
        assert len(buys) >= 1
        # 所有卖出都在买入之后至少1个交易日
        for s in sells:
            assert s.date > buys[0].date

    def test_trailing_stop_triggers(self, bdays):
        """从最高收盘回撤超过 trailing_stop 触发移动止损"""
        n = len(bdays)
        # 前30天从10涨到13，之后跌回11（从峰值回撤 >15%）
        close = np.concatenate([
            np.linspace(10, 13, 30),
            np.linspace(13, 11, n - 30),
        ])
        df = pd.DataFrame({
            'date': pd.to_datetime(bdays), 'open': close, 'high': close,
            'low': close, 'close': close, 'volume': np.full(n, 1e6),
        })
        ds = StubDataSource({'HNS.TEST': df})
        engine, md = build_engine(
            ds, ['HNS.TEST'], bdays,
            stop_loss=0.5, take_profit=10.0,  # 关闭固定止损止盈，隔离移动止损
            trailing_stop=0.10,
        )
        engine.run(FixedSignalsStrategy(['HNS.TEST']), md)

        trailing_exits = [t for t in engine.trades if t.reason == 'trailing_stop']
        assert len(trailing_exits) >= 1, "从峰值回撤>10%应触发移动止损"

    def test_time_stop_triggers(self, bdays):
        """持仓超过 max_holding_days 触发时间止损"""
        up = make_kline(bdays, drift=+0.001, vol=0.001)  # 温和无波动
        ds = StubDataSource({'FLAT.TEST': up})
        engine, md = build_engine(
            ds, ['FLAT.TEST'], bdays,
            stop_loss=0.5, take_profit=10.0, trailing_stop=0.0,
            max_holding_days=10,
        )
        engine.run(FixedSignalsStrategy(['FLAT.TEST']), md)

        time_exits = [t for t in engine.trades if t.reason == 'time_stop']
        assert len(time_exits) >= 1, "持仓10个交易日后应触发时间止损"

    def test_cooldown_blocks_rebuy_after_stop(self, bdays):
        """止损卖出后，冷却期内禁止回买同一标的"""
        n = len(bdays)
        # 阶梯下跌：每天 -5%，持续触发固定止损
        close = 100 * 0.95 ** np.arange(n)
        df = pd.DataFrame({
            'date': pd.to_datetime(bdays), 'open': close, 'high': close,
            'low': close, 'close': close, 'volume': np.full(n, 1e6),
        })
        ds = StubDataSource({'DOWN.TEST': df})
        engine, md = build_engine(
            ds, ['DOWN.TEST'], bdays,
            stop_loss=0.03, take_profit=10.0, trailing_stop=0.0,
            cooldown_days=5,
        )
        engine.run(FixedSignalsStrategy(['DOWN.TEST']), md)

        # 验证：任何两次同标的的买/卖切换满足冷却间隔
        events = sorted(
            [(t.date, t.action) for t in engine.trades if t.symbol == 'DOWN.TEST'],
            key=lambda x: bdays[list(pd.bdate_range('2024-01-01', periods=1)[0:1].append(pd.DatetimeIndex([pd.Timestamp(x[0])])))[0]] if False else x[0]
        )
        dates_list = [d.date() for d in bdays]
        idx_of = {d: i for i, d in enumerate(dates_list)}
        for i, (d, action) in enumerate(events):
            if action == 'SELL':
                # 找下一次买入
                for d2, a2 in events[i + 1:]:
                    if a2 == 'BUY' and d2 > d:
                        gap = idx_of[d2] - idx_of[d]
                        assert gap > 5, f"冷却期内回买：间隔仅 {gap} 个交易日"
                        break

    def test_exit_on_ma_break(self, bdays):
        """exit_on_ma_break：跌破MA20触发信号退出"""
        n = len(bdays)
        close = np.concatenate([
            np.linspace(10, 12, 40),   # 上涨建立趋势
            np.linspace(12, 9, n - 40),  # 深度回调跌破MA20
        ])
        df = pd.DataFrame({
            'date': pd.to_datetime(bdays), 'open': close, 'high': close,
            'low': close, 'close': close, 'volume': np.full(n, 1e6),
        })
        ds = StubDataSource({'BRK.TEST': df})
        engine, md = build_engine(
            ds, ['BRK.TEST'], bdays,
            stop_loss=0.5, take_profit=10.0, trailing_stop=0.0,
            exit_on_ma_break=True, exit_ma_period=20,
        )
        engine.run(FixedSignalsStrategy(['BRK.TEST']), md)

        ma_exits = [t for t in engine.trades if t.reason == 'signal_exit_ma']
        assert len(ma_exits) >= 1, "跌破MA20应触发信号退出"


# =============================================================================
# 执行真实性
# =============================================================================

class TestExecutionRealism:

    def test_slippage_applied(self, bdays):
        """买入价 = close × (1+slippage)，卖出价 = close × (1-slippage)"""
        flat = make_kline(bdays, drift=0.0, vol=0.0)
        ds = StubDataSource({'F.TEST': flat})
        engine, md = build_engine(
            ds, ['F.TEST'], bdays,
            stop_loss=0.5, take_profit=10.0, trailing_stop=0.0,
            max_holding_days=3, slippage=0.001,
        )
        engine.run(FixedSignalsStrategy(['F.TEST']), md)

        buys = [t for t in engine.trades if t.action == 'BUY']
        sells = [t for t in engine.trades if t.action == 'SELL']
        assert buys and sells
        for t in buys:
            raw = ds.get_close(t.symbol, t.date)
            assert t.price == pytest.approx(raw * 1.001, rel=1e-6)
        for t in sells:
            raw = ds.get_close(t.symbol, t.date)
            assert t.price == pytest.approx(raw * 0.999, rel=1e-6)


# =============================================================================
# 基准对比
# =============================================================================

class TestBenchmark:

    def test_benchmark_return_computed(self, bdays):
        """基准收益 = 基准指数末值/首值 - 1"""
        stock = make_kline(bdays, drift=+0.001)
        bench = make_kline(bdays, start_price=4000.0, drift=+0.002, seed=7)
        ds = StubDataSource({'S.TEST': stock, '000300.SH': bench})
        engine, md = build_engine(
            ds, ['S.TEST'], bdays,
            stop_loss=0.5, take_profit=10.0, trailing_stop=0.0, max_holding_days=3,
            benchmark_symbol='000300.SH',
        )
        result = engine.run(FixedSignalsStrategy(['S.TEST']), md)

        expected = float(bench['close'].iloc[-1] / bench['close'].iloc[0] - 1)
        assert result.benchmark_return == pytest.approx(expected, abs=1e-9)
        assert result.benchmark_symbol == '000300.SH'

    def test_no_benchmark_graceful(self, bdays):
        """基准代码无数据时 benchmark_return 为 None，不崩溃"""
        stock = make_kline(bdays)
        ds = StubDataSource({'S.TEST': stock})
        engine, md = build_engine(ds, ['S.TEST'], bdays, benchmark_symbol='XXX.SH')
        result = engine.run(FixedSignalsStrategy(['S.TEST']), md)
        assert result.benchmark_return is None
        assert result.benchmark_symbol == ''


# =============================================================================
# 交易日历
# =============================================================================

class TestTradingCalendar:

    def test_market_data_normalizes_timestamps(self, bdays):
        """add_trading_date 接受 Timestamp 并归一为 date，去重排序"""
        md = MarketData(None)
        md.add_trading_dates(list(bdays) + list(bdays))  # 重复添加
        assert all(isinstance(d, date) for d in md.trading_dates)
        assert len(md.trading_dates) == len(bdays)
        assert md.date_index(bdays[10].date()) == 10
        assert md.date_index(date(2020, 1, 1)) is None

    def test_cache_derives_trading_dates(self, bdays):
        """CachedDataSource.get_trading_dates 返回缓存日期并集（排除区间外）"""
        from data.base.data_loader import CachedDataSource

        df = make_kline(bdays[:40])
        df2 = make_kline(bdays[20:], start_price=5.0, seed=3)
        cache = CachedDataSource(kline_source=None, cache={'A': df, 'B': df2})

        dates = cache.get_trading_dates(bdays[0].date(), bdays[-1].date())
        assert len(dates) == len(bdays)  # 并集 = 完整区间
        assert all(isinstance(d, pd.Timestamp) for d in dates)

        # 区间过滤
        partial = cache.get_trading_dates(bdays[5].date(), bdays[15].date())
        assert len(partial) == 11
