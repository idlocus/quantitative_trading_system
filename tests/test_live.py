#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""实盘模块测试：Paper 撮合、账本入账/净值、订单缓冲带"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import date

import pytest

from live.broker import Order, PaperBroker, Fill
from live.ledger import Ledger
from live.signals import build_orders


class TestPaperBroker:

    def test_buy_sell_with_costs(self):
        b = PaperBroker(slippage=0.001, commission=0.0003, stamp_tax=0.0005,
                        transfer_fee=0.00001)
        orders = [Order('A', 'BUY', 0.1, amount=100000)]
        fills = b.execute(orders, {'A': 10.0}, date(2026, 10, 8))
        f = fills[0]
        assert f.status == 'filled'
        assert f.price == pytest.approx(10.01)          # 买入滑点
        # 整手规则：数量必须是100的整数倍
        assert f.quantity % 100 == 0
        assert f.quantity > 0
        # 卖出含印花税
        fills2 = b.execute([Order('A', 'SELL', 0, quantity=f.quantity)],
                           {'A': 10.0}, date(2026, 10, 8))
        f2 = fills2[0]
        assert f2.price == pytest.approx(9.99)
        assert f2.taxes > f2.commission  # 印花税使税费大于佣金

    def test_limit_up_rejects_buy(self):
        b = PaperBroker()
        fills = b.execute([Order('A', 'BUY', 0.1, amount=10000)],
                          {'A': 11.0}, date(2026, 10, 8), prev_prices={'A': 10.0})
        assert '涨停' in fills[0].status

    def test_zero_qty_rejected(self):
        b = PaperBroker()
        fills = b.execute([Order('A', 'BUY', 0.1, amount=10)], {'A': 100.0}, date(2026, 10, 8))
        assert '不足一手' in fills[0].status


class TestLedger:

    def test_round_trip(self, tmp_path):
        led = Ledger(str(tmp_path / 'l.json'))
        led.initialize(1_000_000)
        buy = Fill(Order('A', 'BUY', 0.5, quantity=10000), price=10.0, quantity=10000,
                   commission=30.0, taxes=1.0, trade_date=date(2026, 10, 8))
        led.apply_fills([buy])
        assert led.cash == pytest.approx(1_000_000 - 10000 * 10.0 - 31.0)
        assert led.positions['A'] == 10000

        sell = Fill(Order('A', 'SELL', 0, quantity=10000), price=11.0, quantity=10000,
                    commission=33.0, taxes=55.0 + 1.0, trade_date=date(2026, 10, 9))
        led.apply_fills([sell])
        assert 'A' not in led.positions
        # 盈亏 = (11-10)×10000 - 全部费用
        expected = 1_000_000 + 10000 - 31.0 - 33.0 - 56.0
        assert led.cash == pytest.approx(expected)

    def test_nav_mark(self, tmp_path):
        led = Ledger(str(tmp_path / 'l.json'))
        led.initialize(100000)
        led.apply_fills([Fill(Order('A', 'BUY', 1.0, quantity=5000), price=10.0,
                              quantity=5000, commission=15, taxes=0.5,
                              trade_date=date(2026, 10, 8))])
        nav = led.mark_to_market({'A': 12.0}, date(2026, 10, 8))
        assert nav == pytest.approx(led.cash + 5000 * 12.0)
        assert len(led.nav_series()) == 1

    def test_no_double_init(self, tmp_path):
        led = Ledger(str(tmp_path / 'l.json'))
        led.initialize(100)
        with pytest.raises(RuntimeError):
            led.initialize(100)


class TestOrders:

    def test_band_reduces_turnover(self):
        targets = [f'S{i}' for i in range(10)]
        # 当前持有前10只且权重正好 → 无订单
        cur = {s: 0.1 for s in targets}
        assert build_orders(targets, cur, 1_000_000, top_n=10) == []
        # 权重漂移10%以内 → 缓冲带吸收
        cur2 = {s: 0.11 for s in targets}
        assert build_orders(targets, cur2, 1_000_000, top_n=10) == []
        # 换掉3只 → 3卖3买
        cur3 = {f'S{i}': 0.1 for i in range(10)}
        new_targets = targets[:7] + ['X1', 'X2', 'X3']
        orders = build_orders(new_targets, cur3, 1_000_000, top_n=10)
        sells = [o for o in orders if o.side == 'SELL']
        buys = [o for o in orders if o.side == 'BUY']
        assert len(sells) == 3 and len(buys) == 3
