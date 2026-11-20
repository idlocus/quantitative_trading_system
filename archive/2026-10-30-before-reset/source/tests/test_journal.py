#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""每日交易日志测试：结构完整性、追加语义、归因计算"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import date

import numpy as np
import pytest

from live import Ledger
from live.broker import Order, Fill
from live.journal import JournalWriter


class FakeDS:
    """最小数据源桩：40个交易日基准 + 个股昨/今两档价"""
    def __init__(self, stock_px=10.0, stock_prev=None):
        pd = __import__('pandas')
        dates = pd.bdate_range(end=date.today(), periods=40)
        bench_close = list(4000 + 10 * np.arange(40) if False else np.linspace(3900, 4050, 40))
        self.bench = pd.DataFrame({
            'date': dates, 'open': bench_close, 'high': bench_close,
            'low': bench_close, 'close': bench_close, 'volume': [1e9] * 40})
        self.stock_prev = stock_prev if stock_prev is not None else stock_px
        self.stock_px = stock_px

    def preload_batch(self, symbols, start, end):
        pd = __import__('pandas')
        out = {'000300.SH': self.bench}
        for s in symbols:
            if s != '000300.SH':
                n = 40
                closes = [self.stock_prev] * (n - 1) + [self.stock_px]
                out[s] = self.bench.assign(close=closes)
        return out


class TestJournal:

    def test_am_writes_structure(self, tmp_path, monkeypatch):
        import live.journal as J
        monkeypatch.setattr(J, 'JOURNAL_DIR', str(tmp_path))
        led = Ledger(str(tmp_path / 'l.json'))
        led.initialize(1_000_000)
        w = JournalWriter(led, top_n=10)

        # 桩掉重计算（信号生成依赖全市场，单测只验证结构）
        monkeypatch.setattr('live.signals.generate_target',
                            lambda ds, as_of, top_n=10: ([f'S{i}' for i in range(10)],
                                                         __import__('pandas').Series(range(10), index=[f'S{i}' for i in range(10)])))
        ctx = w.write_am(FakeDS())
        text = open(w.path, encoding='utf-8').read()
        for section in ['盘前市场展望', '盘前决策依据', '盘前风控检查']:
            assert section in text, f"缺章节: {section}"
        assert '沪深300' in text
        assert ctx['targets']

    def test_pm_appends_and_attributes(self, tmp_path, monkeypatch):
        import live.journal as J
        monkeypatch.setattr(J, 'JOURNAL_DIR', str(tmp_path))
        led = Ledger(str(tmp_path / 'l.json'))
        led.initialize(1_000_000)
        led.apply_fills([Fill(Order('A', 'BUY', 1.0, quantity=10000), price=10.0,
                              quantity=10000, commission=3, taxes=0.1,
                              trade_date=date.today())])
        w = JournalWriter(led, top_n=10)

        fill = Fill(Order('B', 'BUY', 1.0, quantity=1000), price=5.0, quantity=1000,
                    commission=1.5, taxes=0.05, trade_date=date.today())
        path = w.write_pm(FakeDS(stock_px=10.5, stock_prev=10.0), [fill],
                          {'targets': ['A', 'B'], 'prev_trading': date.today() - __import__('datetime').timedelta(days=1)})
        text = open(path, encoding='utf-8').read()
        for section in ['成交记录', '持仓与归因', '收盘点评', '当日贡献']:
            assert section in text, f"缺章节: {section}"
        assert '| A |' in text          # 持仓表
        assert '+5.0%' in text          # 当日涨跌（10.5/10.0）

    def test_no_trade_day(self, tmp_path, monkeypatch):
        import live.journal as J
        monkeypatch.setattr(J, 'JOURNAL_DIR', str(tmp_path))
        led = Ledger(str(tmp_path / 'l.json'))
        led.initialize(100000)
        w = JournalWriter(led)
        path = w.write_pm(FakeDS(), [], {'targets': [], 'prev_trading': date.today()})
        assert '今日无交易' in open(path, encoding='utf-8').read()
