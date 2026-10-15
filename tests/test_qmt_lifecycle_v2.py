#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""QMT 订单生命周期重写验证（2026-10-15 三阻断项修复）

A. 客户端引用(order_remark) ↔ 真实订单号映射：回调 + 兜底扫描
B. 成交→入账原子性：先入账成功、后推进/清退台账（R4）
C. 台账持久化原子写：临时文件 + os.replace
"""
import json
import os
import sys
import types
from datetime import date

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from live import qmt_broker as qb
from live.broker import Fill, Order
from live.ledger import Ledger


class ScriptedXt:
    """模拟 xtquant：订单表可控可见（模拟 OES 异步报送时序）"""

    def __init__(self):
        self.orders = {}
        self.visible = False
        self.cancelled = []
        self._next_oid = 1000

    def order_stock_async(self, account, code, direction, qty,
                          ptype, price, strat, remark):
        oid = str(self._next_oid)
        self._next_oid += 1
        self.orders[oid] = {'status': 49, 'traded': 0, 'px': 0.0,
                            'remark': remark, 'code': code}
        self.last = {'seq': 777, 'oid': oid, 'remark': remark}
        return self.last['seq']

    def query_stock_orders(self, account):
        if not self.visible:
            return []
        return [types.SimpleNamespace(
            order_id=oid, order_status=v['status'],
            traded_volume=v['traded'], traded_price=v['px'],
            order_remark=v['remark']) for oid, v in self.orders.items()]

    def cancel_order_stock_async(self, account, oid):
        self.cancelled.append(oid)
        self.orders[oid]['status'] = 54
        return 0


def make_broker(tmp_path, xt):
    b = qb.QMTBroker(state_path=str(tmp_path / 'q.json'),
                     enforce_window=False, order_timeout=0.2)
    b._connected = True
    b._xt_trader = xt
    return b


class TestRefMapping:
    """A：引用 ↔ 真实订单号映射"""

    def test_unresolved_ref_is_unknown_not_pending(self, tmp_path):
        """回调未达且券商表不可见 → (-1,-1,0) 不可确认，绝不当'已报未成交'"""
        xt = ScriptedXt()
        b = make_broker(tmp_path, xt)
        ref = b._submit_order(Order('600000.SH', 'BUY', 0.01), 100, 10.0)
        assert ref and ref.startswith('cr')
        assert b._query_order(ref) == (-1, -1, 0.0)

    def test_callback_maps_ref_to_real_oid(self, tmp_path):
        """on_stock_order(order_remark=引用) → rec['oid'] 落账，此后可查询"""
        xt = ScriptedXt()
        b = make_broker(tmp_path, xt)
        ref = b._submit_order(Order('600000.SH', 'BUY', 0.01), 100, 10.0)
        real_oid = xt.last['oid']
        b.on_stock_order(types.SimpleNamespace(
            order_id=real_oid, order_remark=ref, stock_code='600000.SH'))
        xt.visible = True
        assert b._pending[ref]['oid'] == real_oid
        st, traded, px = b._query_order(ref)
        assert st == 49 and traded == 0

    def test_remark_scan_fallback_when_callback_missed(self, tmp_path):
        """回调丢失（断线）→ 兜底按 order_remark 扫券商订单表完成关联"""
        xt = ScriptedXt()
        b = make_broker(tmp_path, xt)
        ref = b._submit_order(Order('600000.SH', 'BUY', 0.01), 100, 10.0)
        # 不触发回调，直接让订单表可见
        xt.visible = True
        st, traded, px = b._query_order(ref)
        assert st == 49, "兜底扫描应完成 ref→oid 关联并查询成功"
        assert b._pending[ref]['oid'] == xt.last['oid']

    def test_cancel_resolves_real_oid(self, tmp_path):
        xt = ScriptedXt()
        b = make_broker(tmp_path, xt)
        ref = b._submit_order(Order('600000.SH', 'BUY', 0.01), 100, 10.0)
        b.on_stock_order(types.SimpleNamespace(
            order_id=xt.last['oid'], order_remark=ref, stock_code='600000.SH'))
        assert b._cancel_order(ref) is True
        assert xt.cancelled == [xt.last['oid']], "撤单必须用真实订单号"


class TestAtomicBooking:
    """B：成交→入账原子性（R4：入账成功后才推进/清退台账）"""

    def _filled_ctx(self, tmp_path, ledger_blocked=False):
        xt = ScriptedXt()
        b = make_broker(tmp_path, xt)
        led = Ledger(str(tmp_path / 'l.json'))
        led.initialize(1_000_000)
        ref = b._submit_order(Order('600000.SH', 'BUY', 0.01), 100, 10.0)
        real_oid = xt.last['oid']
        xt.orders[real_oid].update(status=56, traded=100, px=10.0)
        b._pending[ref]['oid'] = real_oid
        if ledger_blocked:
            # 让入账被阻：账本持仓远小于卖出量（超额卖出防线）
            b._pending[ref]['side'] = 'SELL'
            order = Order('600000.SH', 'SELL', 0.0)
        else:
            order = Order('600000.SH', 'BUY', 0.01)
        return b, led, ref, order

    def test_booking_success_advances_and_pops(self, tmp_path):
        b, led, ref, order = self._filled_ctx(tmp_path)
        fill = b._book_order_fill(order, ref, 56, 100, 10.0, 10.0, 100,
                                  date(2026, 10, 16), ledger=led)
        assert fill.quantity == 100
        assert ref not in b._pending, "入账成功后清退终态记录"
        assert led.positions.get('600000.SH') == 100

    def test_booking_blocked_retains_record(self, tmp_path):
        """入账被阻（超额卖出防线）→ 台账不推进、记录保留待复核"""
        b, led, ref, order = self._filled_ctx(tmp_path, ledger_blocked=True)
        fill = b._book_order_fill(order, ref, 56, 100, 10.0, 10.0, 100,
                                  date(2026, 10, 16), ledger=led)
        assert fill.quantity == 0 and '入账被阻' in fill.status
        assert ref in b._pending, "记录保留（否则成交丢失）"
        assert b._pending[ref]['booked'] == 0, "booked 不推进"
        assert led.positions.get('600000.SH') is None

    def test_crash_before_booking_recoverable(self, tmp_path):
        """模拟入账前崩溃：新进程从台账恢复，未入账差额可补记"""
        b, led, ref, order = self._filled_ctx(tmp_path)
        # 不调用 _book_order_fill（=崩溃在查询后入账前）
        state = json.load(open(b.state_path, encoding='utf-8'))
        assert state[ref]['booked'] == 0, "台账仍为旧值——可恢复"
        # 新实例 + sync_pending 完成补记（券商订单表可见）
        b._xt_trader.visible = True
        b2 = qb.QMTBroker(state_path=b.state_path, enforce_window=False)
        b2._connected = True
        b2._xt_trader = b._xt_trader
        b2._pending[ref]['oid'] = b._pending[ref]['oid']
        booked = b2.sync_pending(led)
        assert len(booked) == 1 and led.positions.get('600000.SH') == 100


class TestAtomicStateWrite:
    """C：台账持久化原子写"""

    def test_save_leaves_no_tmp_and_valid_json(self, tmp_path):
        xt = ScriptedXt()
        b = make_broker(tmp_path, xt)
        b._submit_order(Order('600000.SH', 'BUY', 0.01), 100, 10.0)
        assert not os.path.exists(b.state_path + '.tmp')
        json.load(open(b.state_path, encoding='utf-8'))

    def test_failed_write_does_not_corrupt_old_state(self, tmp_path):
        """写入中途失败（模拟崩溃截断）→ 旧状态文件必须完好"""
        xt = ScriptedXt()
        b = make_broker(tmp_path, xt)
        b._submit_order(Order('600000.SH', 'BUY', 0.01), 100, 10.0)
        good = open(b.state_path, encoding='utf-8').read()

        import live.qmt_broker as mod
        orig_json = mod.json          # 模块引用（恢复必须还原模块，而非函数）

        def boom(*a, **kw):
            a[0].write('{"trunc')     # 写一半
            raise IOError('模拟崩溃')
        mod.json = types.SimpleNamespace(dump=boom, loads=orig_json.loads)
        try:
            with pytest.raises(RuntimeError):
                b._save_state()
        finally:
            mod.json = orig_json
        assert open(b.state_path, encoding='utf-8').read() == good, \
            "旧台账被截断=在途订单全部遗忘（原子的意义所在）"
