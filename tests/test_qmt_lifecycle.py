#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""QMT 订单生命周期测试：部分成交不丢失/超时撤单/增量补记/幂等/崩溃恢复/台账清退"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import date, datetime, timedelta

import pytest

from live.broker import Order, Fill
from live.ledger import Ledger
from live.qmt_broker import QMTBroker, QMTOrderState

AS_OF = date(2026, 10, 12)


class FakeQMT(QMTBroker):
    """注入假原语：_orders = {oid: (status, traded, px)}，可随测试推进改动"""

    def __init__(self, state_path, outcomes=None, order_timeout=0.2):
        """outcomes: 每次提交弹出一个 (status, traded, px) 作为该订单的查询结果"""
        super().__init__(state_path=state_path, order_timeout=order_timeout,
                         enforce_window=False)   # 测试桩：绕过收盘窗口判定
        self._orders = {}
        self._outcomes = list(outcomes or [])
        self._cancel_calls = []

    def _ensure_connected(self):
        self._connected = True          # 测试桩：跳过 xtquant 连接

    def _submit_order(self, order, qty, price):
        oid = f'ORD{len(self._orders) + 1:04d}'
        outcome = self._outcomes.pop(0) if self._outcomes \
            else (QMTOrderState.SUBMITTED, 0, 0.0)
        self.last_submitted_qty = int(qty)   # 供测试断言提交侧申报合法化
        self._orders[oid] = outcome
        # 复用真实登记逻辑
        self._xt_trader = None
        self._pending[oid] = {'symbol': order.symbol, 'side': order.side,
                              'qty': int(qty), 'booked': 0, 'status': 49,
                              'submitted_at': str(datetime.now())}
        self._save_state()
        return oid

    def _query_order(self, order_id):
        # 与真实语义一致：查无此单 = 不可确认（-1），而非默认在途
        return self._orders.get(order_id, (-1, -1, 0.0))

    def _cancel_order(self, order_id):
        self._cancel_calls.append(order_id)
        return True


@pytest.fixture
def ledger(tmp_path):
    led = Ledger(str(tmp_path / 'l.json'))
    led.initialize(1_000_000)
    return led


class TestPartialFillLifecycle:

    def test_partial_fill_at_timeout_is_booked(self, tmp_path, ledger):
        """核心场景：超时前已部分成交 → 撤单 → 真实成交量立即入账（旧代码返回0丢失）"""
        broker = FakeQMT(str(tmp_path / 'orders.json'),
                         outcomes=[(QMTOrderState.PARTIAL, 600, 10.0)])  # 始终600/1000
        fills = broker.execute([Order('600000.SH', 'BUY', 0.1, quantity=1000)],
                               {'600000.SH': 10.0}, AS_OF, available_cash=100_000)
        f = fills[0]
        assert f.quantity == 600, "部分成交600股必须入账（旧代码丢为0）"
        assert 'partial' in f.status
        assert broker._cancel_calls, "超时后必须撤单（不留隔夜在途单）"
        booked, _, _blk = ledger.apply_fills(fills)
        assert len(booked) == 1 and ledger.positions['600000.SH'] == 600

    def test_delta_booking_across_syncs(self, tmp_path, ledger):
        """增量补记：已入账600 → 涨到1000 → 补400；重复同步不重复入账（幂等）"""
        state = str(tmp_path / 'orders.json')
        broker = FakeQMT(state)
        # 模拟此前已入账600（ledger 已有持仓），台账 booked=600
        ledger._d['positions']['600000.SH'] = 600
        ledger._d['cost']['600000.SH'] = 10.0
        broker._pending['ORD0001'] = {'symbol': '600000.SH', 'side': 'BUY',
                                      'qty': 1000, 'booked': 600, 'booked_cost': 6000,
                                      'status': 50,
                                      'submitted_at': str(datetime.now())}
        broker._orders['ORD0001'] = (QMTOrderState.PARTIAL, 1000, 10.0)
        f1 = broker.sync_pending(ledger)
        assert len(f1) == 1 and f1[0].quantity == 400          # 增量400
        assert f1[0].trade_id == 'ORD0001#1000'
        assert ledger.positions['600000.SH'] == 1000
        # 再同步（成交量未变）→ 幂等无新增
        assert broker.sync_pending(ledger) == []
        assert ledger.positions['600000.SH'] == 1000

    def test_fill_dedup_by_cumulative_trade_id(self, tmp_path, ledger):
        """同一订单的重复回报：trade_id=oid#累计量 相同 → 账本去重"""
        state = str(tmp_path / 'orders.json')
        broker = FakeQMT(state)
        ledger._d['positions']['600000.SH'] = 600
        ledger._d['cost']['600000.SH'] = 10.0
        broker._pending['ORD0001'] = {'symbol': '600000.SH', 'side': 'BUY',
                                      'qty': 1000, 'booked': 600, 'booked_cost': 6000, 'status': 50,
                                      'submitted_at': str(datetime.now())}
        broker._orders['ORD0001'] = (QMTOrderState.FILLED, 1000, 10.0)
        broker.sync_pending(ledger)                             # 补400 → 1000
        # 模拟回报重放：手工把 booked 回退再同步（同一累计量）
        broker._pending['ORD0001'] = dict(broker._pending.get('ORD0001') or
                                          {'symbol': '600000.SH', 'side': 'BUY',
                                           'qty': 1000, 'booked': 600, 'booked_cost': 6000, 'status': 50,
                                           'submitted_at': str(datetime.now())})
        broker._pending['ORD0001']['booked'] = 600
        broker._pending['ORD0001']['status'] = 50
        f2 = broker.sync_pending(ledger)
        # trade_id 相同 ORD0001#1000 → 账本去重，持仓不变
        assert ledger.positions['600000.SH'] == 1000

    def test_crash_recovery_from_state_file(self, tmp_path, ledger):
        """崩溃/重启恢复：台账持久化，新实例续接补记"""
        state = str(tmp_path / 'orders.json')
        b1 = FakeQMT(state)
        b1._pending['ORD0007'] = {'symbol': '600000.SH', 'side': 'SELL',
                                  'qty': 500, 'booked': 300, 'booked_cost': 3000, 'status': 50,
                                  'submitted_at': str(datetime.now())}
        b1._save_state()
        del b1

        b2 = FakeQMT(state)                                    # 新实例加载台账
        assert 'ORD0007' in b2._pending
        b2._orders['ORD0007'] = (QMTOrderState.FILLED, 500, 10.0)
        # 先造持仓300股可卖
        ledger._d['positions']['600000.SH'] = 300
        ledger._d['cost']['600000.SH'] = 9.0
        f = b2.sync_pending(ledger)
        assert len(f) == 1 and f[0].quantity == 200            # 补剩余200
        assert ledger.positions['600000.SH'] == 100            # 300-200... 卖出500-300=200
        assert 'ORD0007' not in b2._pending                    # 终态清退

    def test_cancel_residual_caught_by_sync(self, tmp_path, ledger):
        """撤单瞬间的成交残量：CANCELLED 不立即清退，sync_pending 复核后清"""
        state = str(tmp_path / 'orders.json')
        broker = FakeQMT(state)
        ledger.apply_fills([Fill(Order('600000.SH', 'BUY', 0.0),
                                  10.0, 600, 0.0, 0.0, date.today(),
                                  trade_id='ORD0001#600')])
        # 执行时撤单后查询显示 CANCELLED@600 → booked 600，台账保留（复核机制）
        broker._pending['ORD0001'] = {'symbol': '600000.SH', 'side': 'BUY',
                                      'qty': 1000, 'booked': 600,
                                      'status': 52,
                                      'submitted_at': str(datetime.now())}
        # 次日复核发现实际成交800（撤单瞬间残量）
        broker._orders['ORD0001'] = (QMTOrderState.CANCELLED, 800, 10.0)
        f = broker.sync_pending(ledger)
        assert len(f) == 1 and f[0].quantity == 200
        assert 'ORD0001' not in broker._pending                # 复核后清退

    def test_ttl_prunes_zombie_records(self, tmp_path, ledger):
        """二轮复审#6：超期记录先查询再处置——券商已查不到（-1）时转入
        待对账（保留+阻断），不得静默删除；明确终态且无残量才清退。"""
        state = str(tmp_path / 'orders.json')
        broker = FakeQMT(state)
        broker._pending['OLD0001'] = {'symbol': '600000.SH', 'side': 'BUY',
                                      'qty': 100, 'booked': 0, 'status': 49,
                                      'submitted_at': str(date.today() - timedelta(days=5))}
        # 券商查无此单（真实语义 -1）→ 保留待对账 + 阻断
        broker.sync_pending(ledger)
        assert 'OLD0001' in broker._pending, '查询失败的超期单不得静默删除'
        assert broker._pending['OLD0001'].get('ttl_expired_unresolved') is True
        assert broker.query_failed
        # 明确终态（撤单、零成交）→ 正常清退
        broker2 = FakeQMT(state + '2')
        broker2._pending['OLD0002'] = {'symbol': '600000.SH', 'side': 'BUY',
                                       'qty': 100, 'booked': 0, 'status': 49,
                                       'submitted_at': str(date.today() - timedelta(days=5))}
        broker2._orders['OLD0002'] = (QMTOrderState.CANCELLED, 0, 0.0)
        broker2.sync_pending(ledger)
        assert 'OLD0002' not in broker2._pending

    def test_full_fill_booked_immediately(self, tmp_path, ledger):
        broker = FakeQMT(str(tmp_path / 'orders.json'),
                         outcomes=[(QMTOrderState.FILLED, 1000, 10.0)])
        fills = broker.execute([Order('600000.SH', 'BUY', 0.1, quantity=1000)],
                               {'600000.SH': 10.0}, AS_OF, available_cash=100_000)
        assert fills[0].quantity == 1000 and fills[0].status == 'filled'
        assert fills[0].trade_id == 'ORD0001#1000'
        ledger.apply_fills(fills)
        assert ledger.positions['600000.SH'] == 1000


class TestSubmitOrderingAndBatchStop:
    """审计 2026-11-19 三-4：①下单意图先持久化再提交（消除 ack 与落盘
    之间退出的丢单窗口）；②超时撤单后仍在途/非终态部分成交 → 停止本批
    后续委托，不得复用被前单占用的预算。"""

    def test_intent_persisted_before_broker_submit(self, tmp_path, monkeypatch):
        broker = QMTBroker(state_path=str(tmp_path / 'q.json'),
                           enforce_window=False)
        events = []

        class FakeXt:
            def order_stock_async(self, account, symbol, direction, qty,
                                  price_type, price_val, strategy, remark):
                events.append(('submit', remark))
                return 7

        broker._xt_trader = FakeXt()
        real_save = broker._save_state

        def save_spy():
            events.append(('save', sorted(broker._pending.keys())))
            real_save()

        monkeypatch.setattr(broker, '_save_state', save_spy)
        ref = broker._submit_order(Order('600000.SH', 'BUY', 0), 100, 10.0)
        assert ref is not None
        # 第一次 save 必须先于 submit，且下单意图已在台账中
        assert events[0][0] == 'save' and ref in events[0][1]
        assert events[1] == ('submit', ref)
        assert broker._pending[ref]['seq'] == 7

    def test_crash_after_submit_leaves_recoverable_intent(self, tmp_path, monkeypatch):
        """提交成功后、最终落盘前退出：意图记录仍在（磁盘首存为 submitting）"""
        broker = QMTBroker(state_path=str(tmp_path / 'q.json'),
                           enforce_window=False)
        calls = {'n': 0}

        class FakeXt:
            def order_stock_async(self, *a):
                return 7

        broker._xt_trader = FakeXt()

        def save_crash():
            calls['n'] += 1
            if calls['n'] >= 2:
                raise OSError('crash between submit and final save')

        monkeypatch.setattr(broker, '_save_state', save_crash)
        with pytest.raises(OSError):
            broker._submit_order(Order('600000.SH', 'BUY', 0), 100, 10.0)
        assert len(broker._pending) == 1
        rec = next(iter(broker._pending.values()))
        assert rec['symbol'] == '600000.SH' and rec['booked'] == 0

    def test_nonterminal_after_timeout_stops_batch(self, tmp_path, monkeypatch):
        broker = QMTBroker(state_path=str(tmp_path / 'q.json'), enforce_window=False)
        submitted = []
        monkeypatch.setattr(broker, '_ensure_connected', lambda: None)

        def execute_single(order, prices, trade_date, cash, **kw):
            submitted.append(order.symbol)
            return Fill(order, 10, 0, 0, 0, trade_date,
                        status=f'open({QMTOrderState.SUBMITTED})')

        monkeypatch.setattr(broker, '_execute_single', execute_single)
        fills = broker.execute(
            [Order('A', 'BUY', .01, amount=100), Order('B', 'BUY', .01, amount=100)],
            {'A': 10, 'B': 10}, date.today(), available_cash=1000)
        assert submitted == ['A'], "前单未确认终态，后续订单必须停止"
        assert broker.query_failed
        assert fills[-1].status.startswith('rejected')

    def test_nonterminal_partial_marked_open(self, tmp_path):
        """非终态部分成交必须标 open(...)，不得伪装成终态 partial(n/m)"""
        broker = QMTBroker(state_path=str(tmp_path / 'q.json'), enforce_window=False)
        broker._pending['ref'] = {'symbol': 'A', 'side': 'BUY', 'qty': 100,
                                  'booked': 0, 'submitted_at': str(datetime.now())}
        fill = broker._book_order_fill(
            Order('A', 'BUY', 0), 'ref', QMTOrderState.SUBMITTED, 50, 10.0,
            10.0, 100, date.today(), ledger=None)
        assert fill.status.startswith('open')
        assert '50/100' in fill.status
        assert fill.quantity == 50

    def test_terminal_partial_keeps_partial_status(self, tmp_path):
        """撤单后的终态部分成交仍是 partial(n/m)（历史口径不变）"""
        broker = QMTBroker(state_path=str(tmp_path / 'q.json'), enforce_window=False)
        broker._pending['ref'] = {'symbol': 'A', 'side': 'BUY', 'qty': 100,
                                  'booked': 0, 'submitted_at': str(datetime.now())}
        fill = broker._book_order_fill(
            Order('A', 'BUY', 0), 'ref', QMTOrderState.CANCELLED, 50, 10.0,
            10.0, 100, date.today(), ledger=None)
        assert fill.status == 'partial(50/100)'


class TestRoundTwoQmtGuards:
    """二轮复审 #4/#5/#7：非法方向不进券商、回执丢失保留意图、矛盾回报
    不得当终态。"""

    def test_invalid_side_never_reaches_broker(self, tmp_path):
        broker = QMTBroker(state_path=str(tmp_path / 'q.json'), enforce_window=False)
        submitted = []
        monkey = type('M', (), {})()
        broker._ensure_connected = lambda: None
        broker._submit_order = lambda order, qty, price: submitted.append(order.side) or 'REF'
        broker._await_order = lambda oid, timeout=None, cancel_on_timeout=True: (
            QMTOrderState.FILLED, 0, 0.0)
        fills = broker.execute([Order('A', 'TYPO', .01, quantity=100)],
                               {'A': 10.0}, AS_OF, available_cash=10_000)
        assert submitted == [], '非法方向必须在券商调用前被拒'
        assert fills[0].status.startswith('rejected(invalid side)')

    def test_submit_timeout_keeps_intent(self, tmp_path):
        broker = QMTBroker(state_path=str(tmp_path / 'q.json'), enforce_window=False)

        class TimeoutXt:
            def order_stock_async(self, *a):
                raise TimeoutError('broker ack lost')

        broker._xt_trader = TimeoutXt()
        ref = broker._submit_order(Order('A', 'BUY', 0), 100, 10.0)
        assert ref is not None, '回执丢失≠拒绝：意图必须保留'
        assert broker._pending[ref]['status'] == 'unknown'

    def test_contradictory_filled_zero_traded_blocked(self, tmp_path):
        broker = QMTBroker(state_path=str(tmp_path / 'q.json'), enforce_window=False)
        broker._pending['ref'] = {'symbol': 'A', 'side': 'BUY', 'qty': 100,
                                  'booked': 0, 'submitted_at': str(datetime.now())}
        fill = broker._book_order_fill(
            Order('A', 'BUY', 0), 'ref', QMTOrderState.FILLED, 0, 0.0,
            10.0, 100, date.today(), ledger=None)
        assert fill.status.startswith('query_failed(contradictory')
        assert broker.query_failed and 'ref' in broker._pending
