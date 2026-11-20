#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
回测诚实性测试：涨跌停不可成交 + 真实成本模型（印花税/过户费）
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import pytest
from datetime import date

from backtesting.engine import BacktestEngine, BacktestConfig, MarketData
from risk.constants import BacktestParams
from selection.config import ScoredStock


def limit_kline(dates, prices):
    """按给定收盘价序列构造K线"""
    n = len(prices)
    arr = np.asarray(prices, dtype=float)
    return pd.DataFrame({
        'date': pd.to_datetime(dates), 'open': arr, 'high': arr,
        'low': arr, 'close': arr, 'volume': np.full(n, 1e6),
    })


class StubDataSource:
    def __init__(self, cache):
        self.cache = cache

    def get_kline(self, symbol, start, end):
        df = self.cache[symbol]
        return df[(df['date'] >= pd.Timestamp(start)) & (df['date'] <= pd.Timestamp(end))]

    def get_close(self, symbol, trading_date):
        df = self.cache[symbol]
        mask = df['date'] <= pd.Timestamp(trading_date)
        return float(df.loc[mask, 'close'].iloc[-1]) if mask.any() else 0.0


class AlwaysBuy:
    def __init__(self, symbols):
        self.symbols = symbols

    def select(self, date, data, allowed_symbols=None, constraint=None):
        return [ScoredStock(symbol=s, score=80) for s in self.symbols]


class BuyOn:
    """只在指定日期发出信号（用于隔离单日行为）"""

    def __init__(self, symbols, day):
        self.symbols = symbols
        self.day = day

    def select(self, date, data, allowed_symbols=None, constraint=None):
        if date == self.day:
            return [ScoredStock(symbol=s, score=80) for s in self.symbols]
        return []


def build(dates, cache, **cfg):
    ds = StubDataSource(cache)
    md = MarketData(ds)
    md.add_trading_dates(dates)
    md._symbols = list(cache.keys())
    defaults = dict(
        initial_capital=1_000_000, commission=0.0003,
        stop_loss=0.5, take_profit=10.0, trailing_stop=0.0,
        max_holding_days=0, cooldown_days=0, benchmark_symbol='',
    )
    defaults.update(cfg)
    return BacktestEngine(BacktestConfig(**defaults)), md


BDAYS = list(pd.bdate_range('2024-01-01', periods=8))


class TestPriceLimitBoards:
    """板块涨跌停幅度规则"""

    def test_board_ratios(self):
        assert BacktestEngine._price_limit_ratio('600519.SH') == 0.10
        assert BacktestEngine._price_limit_ratio('000001.SZ') == 0.10
        assert BacktestEngine._price_limit_ratio('688126.SH') == 0.20
        assert BacktestEngine._price_limit_ratio('300408.SZ') == 0.20
        assert BacktestEngine._price_limit_ratio('830799.BJ') == 0.30
        assert BacktestEngine._price_limit_ratio('9868.HK') is None


class TestLimitUpBlocksBuy:

    def test_no_buy_at_limit_up(self):
        """信号日收盘封涨停（主板+10%）→ 不可买入"""
        dates = BDAYS[:3]
        limit_day = dates[2].date()

        # 正常日 +4.76% 信号 → 可买
        prices = [10.0, 10.5, 11.0]
        normal, md1 = build(dates, {'OK.TEST': limit_kline(dates, prices)})
        normal.run(BuyOn(['OK.TEST'], limit_day), md1)
        assert any(t.action == 'BUY' for t in normal.trades), "未涨停应正常买入"

        # 涨停日 +10% 信号 → 不可买
        prices_limit = [10.0, 10.5, 11.55]
        limited, md2 = build(dates, {'LU.TEST': limit_kline(dates, prices_limit)})
        limited.run(BuyOn(['LU.TEST'], limit_day), md2)
        assert not any(t.action == 'BUY' for t in limited.trades), "封涨停日不应成交"

    def test_toggle_allows_buy(self):
        """关闭约束后涨停可买（用于对照/研究）"""
        dates = BDAYS[:3]
        limit_day = dates[2].date()
        prices_limit = [10.0, 10.5, 11.55]
        engine, md = build(dates, {'LU.TEST': limit_kline(dates, prices_limit)},
                           enforce_price_limits=False)
        engine.run(BuyOn(['LU.TEST'], limit_day), md)
        assert any(t.action == 'BUY' for t in engine.trades)

    def test_gem_20_percent(self):
        """创业板 20% 涨停幅度：+10% 不封板可买"""
        dates = BDAYS[:3]
        limit_day = dates[2].date()
        prices = [10.0, 10.5, 11.55]  # +10% 对创业板未封板
        engine, md = build(dates, {'300001.SZ': limit_kline(dates, prices)})
        engine.run(BuyOn(['300001.SZ'], limit_day), md)
        assert any(t.action == 'BUY' for t in engine.trades), "创业板+10%不封板，应可买"


class TestLimitDownBlocksSell:

    def test_stop_loss_deferred_at_limit_down(self):
        """触发止损但当日跌停 → 延后卖出"""
        # 横盘3日后连续两天 -10% 跌停（第4、5日）
        # 买在第2日（无涨跌），第4日 -20% 触发止损（stop_loss=0.05）但跌停无法卖
        # 第6日反弹到 8.5（未跌停）→ 当日可卖
        prices = [10.0, 10.0, 10.0, 9.0, 8.1, 8.5]
        dates = BDAYS[:6]

        engine, md = build(dates, {'LD.TEST': limit_kline(dates, prices)},
                           stop_loss=0.05, cooldown_days=0)
        engine.run(AlwaysBuy(['LD.TEST']), md)

        sells = [t for t in engine.trades if t.action == 'SELL']
        assert len(sells) >= 1, "跌停解除后应完成退出"
        # 第4日（index 3）跌停 -10%：止损信号存在但当日不可成交
        # 卖出日期不应是跌停日（第4/5日）
        d4, d5 = dates[3].date(), dates[4].date()
        for s in sells:
            assert s.date not in (d4, d5), f"跌停日 {s.date} 不应成交卖出"


class TestRealCostModel:

    def test_stamp_tax_on_sell(self):
        """卖出费用 = 佣金 + 过户费 + 印花税；买入 = 佣金 + 过户费"""
        dates = BDAYS[:5]
        prices = [10.0, 10.0, 10.0, 10.0, 10.0]  # 价格不变避免数量漂移
        engine, md = build(dates, {'F.TEST': limit_kline(dates, prices)},
                           max_holding_days=2, slippage=0.0)
        engine.run(AlwaysBuy(['F.TEST']), md)

        buys = [t for t in engine.trades if t.action == 'BUY']
        sells = [t for t in engine.trades if t.action == 'SELL']
        assert buys and sells

        buy = buys[0]
        buy_value = buy.price * buy.quantity
        assert buy.taxes == pytest.approx(buy_value * BacktestParams.TRANSFER_FEE)
        assert buy.commission == pytest.approx(buy_value * 0.0003)

        sell = sells[0]
        sell_value = sell.price * sell.quantity
        expected_taxes = sell_value * (BacktestParams.TRANSFER_FEE + BacktestParams.STAMP_TAX)
        assert sell.taxes == pytest.approx(expected_taxes)

    def test_cost_reduces_final_capital(self):
        """相同行情下，含税费的资金 < 免税费的资金"""
        dates = BDAYS[:5]
        prices = [10.0, 10.0, 10.0, 10.0, 10.0]

        results = {}
        for comm in (0.0, 0.0003):
            engine, md = build(dates, {'C.TEST': limit_kline(dates, prices)},
                               commission=comm, max_holding_days=2, slippage=0.0)
            results[comm] = engine.run(AlwaysBuy(['C.TEST']), md)

        # 有交易的情况下，佣金+印花税使期末资金更低
        assert results[0.0003].final_capital < results[0.0].final_capital
