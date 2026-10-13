#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""QMT 订单生命周期测试：部分成交不丢失/超时撤单/增量补记/幂等/崩溃恢复/台账清退"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import date, datetime, timedelta

import pytest

from live.broker import Order
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
        return self._orders.get(order_id, (49, 0, 0.0))

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
                                      'qty': 1000, 'booked': 600,
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
                                      'qty': 1000, 'booked': 600, 'status': 50,
                                      'submitted_at': str(datetime.now())}
        broker._orders['ORD0001'] = (QMTOrderState.FILLED, 1000, 10.0)
        broker.sync_pending(ledger)                             # 补400 → 1000
        # 模拟回报重放：手工把 booked 回退再同步（同一累计量）
        broker._pending['ORD0001'] = dict(broker._pending.get('ORD0001') or
                                          {'symbol': '600000.SH', 'side': 'BUY',
                                           'qty': 1000, 'booked': 600, 'status': 50,
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
                                  'qty': 500, 'booked': 300, 'status': 50,
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
        state = str(tmp_path / 'orders.json')
        broker = FakeQMT(state)
        broker._pending['OLD0001'] = {'symbol': '600000.SH', 'side': 'BUY',
                                      'qty': 100, 'booked': 0, 'status': 49,
                                      'submitted_at': str(date.today() - timedelta(days=5))}
        broker._orders['OLD0001'] = (49, 0, 0.0)               # 券商已查不到
        broker.sync_pending(ledger)
        assert 'OLD0001' not in broker._pending

    def test_full_fill_booked_immediately(self, tmp_path, ledger):
        broker = FakeQMT(str(tmp_path / 'orders.json'),
                         outcomes=[(QMTOrderState.FILLED, 1000, 10.0)])
        fills = broker.execute([Order('600000.SH', 'BUY', 0.1, quantity=1000)],
                               {'600000.SH': 10.0}, AS_OF, available_cash=100_000)
        assert fills[0].quantity == 1000 and fills[0].status == 'filled'
        assert fills[0].trade_id == 'ORD0001#1000'
        ledger.apply_fills(fills)
        assert ledger.positions['600000.SH'] == 1000


class TestWiring:

    def test_sync_pending_called_before_signal(self):
        """源码守护：qmt 执行流必须先补账本再生成信号（风控/订单基于最新持仓）"""
        import inspect
        import run_daily
        src = inspect.getsource(run_daily.run_pm)
        assert 'sync_pending(ledger)' in src
        assert src.index('sync_pending(ledger)') < src.index('generate_production_signal(ds')
        import run_live
        src2 = inspect.getsource(run_live.main)
        assert 'sync_pending(ledger)' in src2
        assert src2.index('sync_pending(ledger)') < src2.index('generate_production_signal(ds')
