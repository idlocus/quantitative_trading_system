#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""实盘模块测试：Paper 撮合、账本入账/净值、订单缓冲带"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import date

import pytest

from live.broker import Order, PaperBroker, Fill
from live.ledger import Ledger
from portfolio.orders import build_orders


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
        fills = b.execute([Order('600000.SH', 'BUY', 0.1, amount=10000)],
                          {'600000.SH': 11.0}, date(2026, 10, 8),
                          prev_prices={'600000.SH': 10.0})
        assert '涨停' in fills[0].status

    def test_zero_qty_rejected(self):
        b = PaperBroker()
        fills = b.execute([Order('A', 'BUY', 0.1, amount=10)], {'A': 100.0}, date(2026, 10, 8))
        assert '不足最低申报' in fills[0].status

    def test_full_liquidation_amount_sell_fills(self):
        """审计 2026-11-19 三-6：清仓卖出（金额口径）不得被费用除价缩量——
        100股@10、目标归零必须全部卖出，费用从成交所得扣除。"""
        b = PaperBroker(slippage=0, commission=0.0003, stamp_tax=0.0005,
                        transfer_fee=0.00001)
        fills = b.execute([Order('A', 'SELL', 0, amount=1000)],
                          {'A': 10.0}, date(2026, 10, 8), positions={'A': 100})
        f = fills[0]
        assert f.quantity == 100 and f.status == 'filled'
        assert f.price == 10.0
        assert f.commission + f.taxes > 0    # 费用照收，从所得中扣

    def test_known_empty_positions_reject_sell(self):
        """已知空仓（positions={}）≠ 未提供持仓（None）：空仓卖单必须被拒，
        不得凭空卖出释放现金。"""
        b = PaperBroker()
        fills = b.execute([Order('A', 'SELL', 0, quantity=100)],
                          {'A': 10.0}, date(2026, 10, 8), positions={})
        assert fills[0].quantity == 0
        assert fills[0].status.startswith('rejected')

    def test_batch_decrements_sellable_per_fill(self):
        """同批次同品种两笔卖出：第二笔只能卖余量（零股一次性），不得
        复用第一笔已卖股数。"""
        b = PaperBroker()
        fills = b.execute([Order('A', 'SELL', 0, quantity=100),
                           Order('A', 'SELL', 0, quantity=100)],
                          {'A': 10.0}, date(2026, 10, 8), positions={'A': 150})
        assert fills[0].quantity == 100
        assert fills[1].quantity == 50

    def test_unknown_positions_keep_legacy_sell_semantics(self):
        """positions=None（持仓未知，回测场景）：卖出数量按订单原样放行。"""
        b = PaperBroker()
        fills = b.execute([Order('A', 'SELL', 0, quantity=700)],
                          {'A': 10.0}, date(2026, 10, 8))
        assert fills[0].quantity == 700 and fills[0].status == 'filled'


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
        led.initialize(100000, as_of=date(2026, 10, 8))
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
        assert build_orders({s: 0.1 for s in targets}, cur, 1_000_000, band=0.2) == []
        # 权重漂移10%以内 → 缓冲带吸收
        cur2 = {s: 0.11 for s in targets}
        assert build_orders({s: 0.1 for s in targets}, cur2, 1_000_000, band=0.2) == []
        # 换掉3只 → 3卖3买
        cur3 = {f'S{i}': 0.1 for i in range(10)}
        new_targets = targets[:7] + ['X1', 'X2', 'X3']
        orders = build_orders({s: 0.1 for s in new_targets}, cur3, 1_000_000, band=0.2)
        sells = [o for o in orders if o.side == 'SELL']
        buys = [o for o in orders if o.side == 'BUY']
        assert len(sells) == 3 and len(buys) == 3


class TestLedgerOversellGuard:
    """超额卖出防线：卖量>持仓 → 阻止入账（现金不凭空增加）+ 记异常队列待人工核对"""

    def _ledger_with_position(self, tmp_path, qty=1000):
        led = Ledger(str(tmp_path / 'l.json'))
        led.initialize(1_000_000)
        led.apply_fills([Fill(Order('A', 'BUY', 1.0, quantity=qty), price=10.0,
                              quantity=qty, commission=0, taxes=0,
                              trade_date=date(2026, 10, 8))])
        return led

    def test_oversell_blocked_and_recorded(self, tmp_path):
        led = self._ledger_with_position(tmp_path, qty=1000)
        cash_before = led.cash
        oversell = Fill(Order('A', 'SELL', 0, quantity=1500), price=10.0,
                        quantity=1500, commission=0, taxes=0,
                        trade_date=date(2026, 10, 12))
        booked, dupes, blocked = led.apply_fills([oversell])
        assert blocked == [oversell] and booked == [] and dupes == []
        # 现金不变、持仓不变（不再出现负持仓被清除+现金虚增）
        assert led.cash == cash_before
        assert led.positions['A'] == 1000
        # 异常队列记录
        exs = led.open_exceptions()
        assert len(exs) == 1
        assert exs[0]['type'] == 'oversell' and exs[0]['quantity'] == 1500
        assert exs[0]['holding'] == 1000 and exs[0]['status'] == 'open'

    def test_exact_position_sell_still_works(self, tmp_path):
        led = self._ledger_with_position(tmp_path, qty=1000)
        booked, _, blocked = led.apply_fills([
            Fill(Order('A', 'SELL', 0, quantity=1000), price=10.0, quantity=1000,
                 commission=0, taxes=0, trade_date=date(2026, 10, 12))])
        assert len(booked) == 1 and blocked == []
        assert 'A' not in led.positions
        assert led.open_exceptions() == []

    def test_sequential_sell_then_oversell(self, tmp_path):
        """合法卖300后再卖800（余700）→ 第二笔被阻止"""
        led = self._ledger_with_position(tmp_path, qty=1000)
        f1 = Fill(Order('A', 'SELL', 0, quantity=300), price=10.0, quantity=300,
                  commission=0, taxes=0, trade_date=date(2026, 10, 12))
        f2 = Fill(Order('A', 'SELL', 0, quantity=800), price=10.0, quantity=800,
                  commission=0, taxes=0, trade_date=date(2026, 10, 12))
        booked, _, blocked = led.apply_fills([f1, f2])
        assert len(booked) == 1 and len(blocked) == 1
        assert led.positions['A'] == 700
        assert len(led.open_exceptions()) == 1

    def test_exceptions_persist_and_resolvable(self, tmp_path):
        led = self._ledger_with_position(tmp_path, qty=100)
        led.apply_fills([Fill(Order('A', 'SELL', 0, quantity=200), price=10.0,
                              quantity=200, commission=0, taxes=0,
                              trade_date=date(2026, 10, 12))])
        # 重载（持久化）
        led2 = Ledger(led.path)
        assert len(led2.open_exceptions()) == 1
        led2.resolve_exception(0, note='重复回报，已核实')
        assert led2.open_exceptions() == []
        assert led2.exceptions()[0]['status'] == 'resolved'


class TestLedgerFillValidation:
    """无效成交防线（审计 2026-11-19 三-2）：方向/价格/数量/费用非法的成交
    在写账前拒绝——未知方向曾落入卖出分支凭空加现金（1万→1.1万）、NaN 价格
    会把现金永久污染成 NaN。拒绝=记异常队列，不抛异常崩溃。"""

    def _fill(self, side='BUY', price=10.0, quantity=100, commission=0.0,
              taxes=0.0):
        return Fill(Order('A', side, 0, quantity=quantity), price=price,
                    quantity=quantity, commission=commission, taxes=taxes,
                    trade_date=date(2026, 10, 12))

    def test_unknown_side_does_not_credit_cash(self, tmp_path):
        led = Ledger(str(tmp_path / 'l.json'))
        led.initialize(10_000)
        bogus = self._fill(side='MYSTERY')
        booked, dupes, blocked = led.apply_fills([bogus])
        assert blocked == [bogus] and booked == [] and dupes == []
        assert led.cash == 10_000            # 不得按卖出分支加现金
        assert led.positions == {}
        exs = led.open_exceptions()
        assert len(exs) == 1 and exs[0]['type'] == 'invalid_fill'
        assert exs[0]['status'] == 'open'

    def test_nan_price_rejected_without_poisoning_cash(self, tmp_path):
        led = Ledger(str(tmp_path / 'l.json'))
        led.initialize(10_000)
        nan_fill = self._fill(price=float('nan'))
        booked, _, blocked = led.apply_fills([nan_fill])
        assert blocked == [nan_fill] and booked == []
        assert led.cash == 10_000            # 现金不得变成 NaN
        assert len(led.open_exceptions()) == 1

    @pytest.mark.parametrize('kwargs', [
        {'price': -5.0}, {'price': 0.0}, {'price': float('inf')},
        {'commission': float('nan')}, {'commission': -1.0},
        {'taxes': float('inf')},
        {'quantity': float('nan')}, {'quantity': 100.5},
    ])
    def test_invalid_numbers_rejected(self, tmp_path, kwargs):
        led = Ledger(str(tmp_path / 'l.json'))
        led.initialize(10_000)
        bad = self._fill(**kwargs)
        booked, _, blocked = led.apply_fills([bad])
        assert blocked == [bad] and booked == []
        assert led.cash == 10_000 and led.positions == {}
        assert len(led.open_exceptions()) == 1

    def test_valid_fill_still_books(self, tmp_path):
        led = Ledger(str(tmp_path / 'l.json'))
        led.initialize(10_000)
        good = self._fill(side='SELL', commission=5.0, taxes=3.0)
        # 先建仓再卖出，验证合法路径不受防线影响
        led.apply_fills([self._fill(side='BUY', quantity=100)])
        booked, _, blocked = led.apply_fills([good])
        assert len(booked) == 1 and blocked == []
        assert led.open_exceptions() == []


class TestLedgerNavBackfill:
    """审计 2026-11-19 三-7：补记旧日期估值曾直接覆盖最新净值
    （记18、19再补18 → 历史变两条18日、19日消失）。"""

    def _ledger_with_position(self, tmp_path):
        led = Ledger(str(tmp_path / 'l.json'))
        led.initialize(10_000, as_of=date(2026, 10, 10))
        led.apply_fills([Fill(Order('A', 'BUY', 1.0, quantity=100), price=10.0,
                              quantity=100, commission=0, taxes=0,
                              trade_date=date(2026, 10, 10))])
        return led

    def test_remark_same_old_date_updates_in_place(self, tmp_path):
        """二轮复审#15 语义升级：历史日期（即使已记录）一律拒绝重标——
        当前持仓改写历史净值等于事后改史；只允许重标最新一日。"""
        led = self._ledger_with_position(tmp_path)
        led.mark_to_market({'A': 10.0}, date(2026, 10, 18))
        led.mark_to_market({'A': 11.0}, date(2026, 10, 19))
        with pytest.raises(ValueError, match='拒绝重标历史日期'):
            led.mark_to_market({'A': 12.0}, date(2026, 10, 18))
        # 拒绝的请求不得污染 last_prices（下一日缺价估值不受影响）
        assert led._d['meta']['last_prices']['A'] == 11.0
        dates = [h['date'] for h in led.nav_series()]
        assert dates == ['2026-10-10', '2026-10-18', '2026-10-19']
        # 最新一日重标仍允许（当日估值修正）
        led.mark_to_market({'A': 11.5}, date(2026, 10, 19))
        assert led.nav_series()[-1]['nav'] == pytest.approx(10_000 - 1000 + 100 * 11.5)

    def test_backfill_unknown_old_date_rejected(self, tmp_path):
        led = self._ledger_with_position(tmp_path)
        led.mark_to_market({'A': 10.0}, date(2026, 10, 18))
        led.mark_to_market({'A': 11.0}, date(2026, 10, 19))
        with pytest.raises(ValueError, match='拒绝重标历史日期'):
            led.mark_to_market({'A': 9.0}, date(2026, 10, 17))
        assert [h['date'] for h in led.nav_series()] == \
            ['2026-10-10', '2026-10-18', '2026-10-19']
        assert led._d['meta']['last_prices']['A'] == 11.0, \
            '被拒绝的补记不得改变估值锚'

    def test_interleaved_instances_conflict_detected(self, tmp_path):
        """两实例交错保存：后写的旧状态实例必须被拒（丢成交防护）"""
        a = Ledger(str(tmp_path / 'l.json'))
        a.initialize(10_000)
        b = Ledger(str(tmp_path / 'l.json'))     # 同一文件的另一实例
        a.deposit(1)                             # a 推进磁盘版本
        with pytest.raises(RuntimeError, match='重新加载'):
            b.deposit(2)                         # b 基于旧状态 → 拒绝覆盖
        # 重载后可继续
        c = Ledger(str(tmp_path / 'l.json'))
        c.deposit(3)
        assert c.cash == pytest.approx(10_004)


class TestRoundTwoLedgerGuards:
    """二轮复审 #9/#10/#11：历史账单拒绝回写、数值有限性、并发写不丢。"""

    def test_nan_deposit_rejected(self, tmp_path):
        led = Ledger(str(tmp_path / 'l.json'))
        led.initialize(10_000)
        with pytest.raises(ValueError, match='有限数'):
            led.deposit(float('nan'))
        assert led.cash == 10_000

    def test_historical_statement_fix_refused_after_later_trades(self, tmp_path):
        led = Ledger(str(tmp_path / 'l.json'))
        led.initialize(10_000)
        led.apply_fills([Fill(Order('A', 'BUY', 0, quantity=100), price=10.0,
                              quantity=100, commission=0, taxes=0,
                              trade_date=date(2026, 10, 12))])
        from live.reconcile import BrokerStatement, Reconciler
        stmt = BrokerStatement(as_of='2026-10-10', cash=10_000, positions={})
        rec = Reconciler()
        rep = rec.reconcile(led, stmt)
        with pytest.raises(ValueError, match='拒绝修正历史账单'):
            rec.apply_adjustments(led, stmt, rep)
        assert led.positions.get('A') == 100, '历史账单不得清掉后续买入'

    def test_concurrent_saves_serialized_by_lock(self, tmp_path):
        """锁内重读版本：两实例先后写，后写者必须冲突报错（不丢写入）。"""
        a = Ledger(str(tmp_path / 'l.json'))
        a.initialize(10_000)
        b = Ledger(str(tmp_path / 'l.json'))
        a.deposit(100)
        with pytest.raises(RuntimeError, match='重新加载'):
            b.deposit(200)
        c = Ledger(str(tmp_path / 'l.json'))
        assert c.cash == pytest.approx(10_100)
