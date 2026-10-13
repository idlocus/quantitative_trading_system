#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""账实核对测试：去重/部分成交/公司行动/资金流水/对账差异/修正/幂等"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import date

import pytest

from live.broker import Order, Fill
from live.ledger import Ledger
from live.reconcile import (BrokerStatement, Reconciler, ReconReport,
                            load_statement, write_statement_template,
                            DIFF_CASH, DIFF_POSITION, DIFF_TRADE_MISSING_LEDGER,
                            DIFF_TRADE_MISSING_BROKER, DIFF_TRADE_DUP, DIFF_FEE)

AS_OF = '2026-10-12'


def fill(sym, side, qty, price, commission=30.0, taxes=5.0, status='filled',
         trade_id='', d=AS_OF):
    return Fill(Order(sym, side, 0.0), price=price, quantity=qty,
                commission=commission, taxes=taxes, trade_date=d,
                status=status, trade_id=trade_id)


def stmt(cash=100000.0, positions=None, trades=None):
    return BrokerStatement(as_of=AS_OF, cash=cash,
                           positions=positions or {}, trades=trades or [])


class TestLedgerDedup:

    def test_duplicate_fill_booked_once(self, tmp_path):
        led = Ledger(str(tmp_path / 'l.json'))
        led.initialize(1_000_000)
        f = fill('A', 'BUY', 10000, 10.0)
        booked, dupes, _blocked = led.apply_fills([f])
        assert len(booked) == 1 and not dupes
        # 同一成交流入两次（重复回报/重复执行）
        booked2, dupes2, _b2 = led.apply_fills([fill('A', 'BUY', 10000, 10.0)])
        assert booked2 == [] and len(dupes2) == 1
        assert led.positions['A'] == 10000          # 只入账一次
        assert led.cash == pytest.approx(1_000_000 - 10000 * 10.0 - 35.0)

    def test_trade_id_overrides_element_fingerprint(self, tmp_path):
        """同一要素但券商回报ID不同 → 视为两笔独立成交（如分笔部分成交）"""
        led = Ledger(str(tmp_path / 'l.json'))
        led.initialize(1_000_000)
        led.apply_fills([fill('A', 'BUY', 5000, 10.0, trade_id='T1001')])
        booked, _, _b = led.apply_fills([fill('A', 'BUY', 5000, 10.0, trade_id='T1002')])
        assert len(booked) == 1
        assert led.positions['A'] == 10000

    def test_partial_fill_booked_by_actual_quantity(self, tmp_path):
        """部分成交（600/1000）按实际600股入账——旧版此处会静默丢账"""
        led = Ledger(str(tmp_path / 'l.json'))
        led.initialize(1_000_000)
        booked, _, _b = led.apply_fills([fill('A', 'BUY', 600, 10.0, status='partial(600/1000)')])
        assert len(booked) == 1
        assert led.positions['A'] == 600
        assert led._d['trades'][0]['status'].startswith('partial')

    def test_cancelled_zero_qty_not_booked(self, tmp_path):
        led = Ledger(str(tmp_path / 'l.json'))
        led.initialize(1_000_000)
        booked, _, _b = led.apply_fills([fill('A', 'BUY', 0, 10.0, status='cancelled')])
        assert booked == [] and not led.positions


class TestCorporateActions:

    def test_dividend_idempotent(self, tmp_path):
        led = Ledger(str(tmp_path / 'l.json'))
        led.initialize(100000)
        assert led.apply_dividend('A', 1200.50, ex_date='2026-10-10') is True
        assert led.cash == pytest.approx(100000 + 1200.50)
        assert led.apply_dividend('A', 1200.50, ex_date='2026-10-10') is False  # 重复
        assert led.cash == pytest.approx(100000 + 1200.50)
        assert led.cash_log()[-1]['type'] == 'dividend'

    def test_split_adjusts_qty_and_cost(self, tmp_path):
        led = Ledger(str(tmp_path / 'l.json'))
        led.initialize(1_000_000)
        led.apply_fills([fill('A', 'BUY', 1000, 20.0, commission=0, taxes=0)])
        assert led.apply_split('A', 1.3, ex_date='2026-10-10') is True
        assert led.positions['A'] == 1300
        assert led._d['cost']['A'] == pytest.approx(20.0 / 1.3)
        assert led.apply_split('A', 1.3, ex_date='2026-10-10') is False  # 幂等
        assert led.positions['A'] == 1300

    def test_deposit_withdraw_logged(self, tmp_path):
        led = Ledger(str(tmp_path / 'l.json'))
        led.initialize(100000)
        led.deposit(50000)
        led.withdraw(30000)
        assert led.cash == pytest.approx(120000)
        types = [e['type'] for e in led.cash_log()]
        assert types == ['deposit', 'deposit', 'withdraw']
        with pytest.raises(ValueError):
            led.withdraw(10**9)   # 超可用现金

    def test_old_format_ledger_compat(self, tmp_path):
        """旧版本账本文件（无cash_log/events键）可正常加载并补齐"""
        import json
        p = str(tmp_path / 'old.json')
        with open(p, 'w', encoding='utf-8') as f:
            json.dump({'cash': 100.0, 'positions': {'A': 10}, 'cost': {'A': 5.0},
                       'trades': [], 'nav_history': [], 'meta': {}}, f)
        led = Ledger(p)
        assert led.cash_log() == [] and led._d['events'] == []
        led.deposit(1)
        assert len(led.cash_log()) == 1


class TestReconcile:

    def test_all_match_ok(self, tmp_path):
        led = Ledger(str(tmp_path / 'l.json'))
        led.initialize(1_000_000)
        f = fill('A', 'BUY', 1000, 10.0, commission=3.0, taxes=0.1)
        ledger_cash_after = 1_000_000 - 1000 * 10.0 - 3.1
        led.apply_fills([f])
        s = stmt(cash=ledger_cash_after, positions={'A': 1000},
                 trades=[{'date': AS_OF, 'symbol': 'A', 'side': 'BUY',
                          'price': 10.0, 'quantity': 1000,
                          'commission': 3.0, 'taxes': 0.1}])
        rep = Reconciler().reconcile(led, s)
        assert rep.is_ok and rep.matched_trades == 1

    def test_cash_and_position_diff_detected(self, tmp_path):
        led = Ledger(str(tmp_path / 'l.json'))
        led.initialize(1_000_000)
        led.apply_fills([fill('A', 'BUY', 1000, 10.0)])
        s = stmt(cash=500000.0, positions={'A': 800, 'B': 200})
        rep = Reconciler().reconcile(led, s)
        kinds = {d.kind for d in rep.diffs}
        assert kinds == {DIFF_CASH, DIFF_POSITION}
        pos_syms = {d.symbol for d in rep.diffs if d.kind == DIFF_POSITION}
        assert pos_syms == {'A', 'B'}

    def test_trade_missing_in_ledger_detected(self, tmp_path):
        """券商有成交、账本漏记（如部分成交被丢弃）"""
        led = Ledger(str(tmp_path / 'l.json'))
        led.initialize(1_000_000)
        s = stmt(cash=990000.0, positions={'A': 1000},
                 trades=[{'date': AS_OF, 'symbol': 'A', 'side': 'BUY',
                          'price': 10.0, 'quantity': 1000}])
        rep = Reconciler().reconcile(led, s)
        assert any(d.kind == DIFF_TRADE_MISSING_LEDGER for d in rep.diffs)

    def test_trade_missing_in_broker_detected(self, tmp_path):
        """账本有、券商无 → 撤单后误记/重复入账嫌疑"""
        led = Ledger(str(tmp_path / 'l.json'))
        led.initialize(1_000_000)
        led.apply_fills([fill('A', 'BUY', 1000, 10.0, trade_id='X1')])
        # 券商流水里只有另一笔B的成交，没有账本里的A
        s = stmt(cash=990000.0, positions={'A': 1000},
                 trades=[{'date': AS_OF, 'symbol': 'B', 'side': 'BUY',
                          'price': 5.0, 'quantity': 2000}])
        rep = Reconciler().reconcile(led, s)
        assert any(d.kind == DIFF_TRADE_MISSING_BROKER for d in rep.diffs)

    def test_fee_diff_detected(self, tmp_path):
        led = Ledger(str(tmp_path / 'l.json'))
        led.initialize(1_000_000)
        led.apply_fills([fill('A', 'BUY', 1000, 10.0, commission=3.0, taxes=0.1)])
        s = stmt(cash=led.cash, positions={'A': 1000},
                 trades=[{'date': AS_OF, 'symbol': 'A', 'side': 'BUY',
                          'price': 10.0, 'quantity': 1000,
                          'commission': 5.0, 'taxes': 0.1}])
        rep = Reconciler(fee_tol=0.5).reconcile(led, s)
        assert any(d.kind == DIFF_FEE for d in rep.diffs)

    def test_broker_duplicate_trade_detected(self, tmp_path):
        led = Ledger(str(tmp_path / 'l.json'))
        led.initialize(1_000_000)
        t = {'date': AS_OF, 'symbol': 'A', 'side': 'BUY',
             'price': 10.0, 'quantity': 1000}
        s = stmt(cash=led.cash, positions={}, trades=[t, dict(t)])
        rep = Reconciler().reconcile(led, s)
        assert any(d.kind == DIFF_TRADE_DUP for d in rep.diffs)

    def test_apply_adjustments_syncs_to_broker(self, tmp_path):
        """--fix：补记漏单 + 持仓/现金对齐券商，且修正入流水留痕"""
        led = Ledger(str(tmp_path / 'l.json'))
        led.initialize(1_000_000)
        s = stmt(cash=990000.0, positions={'A': 1000},
                 trades=[{'date': AS_OF, 'symbol': 'A', 'side': 'BUY',
                          'price': 10.0, 'quantity': 1000,
                          'commission': 3.0, 'taxes': 0.1, 'trade_id': 'Q1'}])
        rec = Reconciler()
        rep = rec.reconcile(led, s)
        assert not rep.is_ok
        actions = rec.apply_adjustments(led, s, rep)
        assert any('补记' in a for a in actions)
        assert led.positions['A'] == 1000
        assert led.cash == pytest.approx(990000.0)
        # 修正后再对账 → 一致
        rep2 = Reconciler().reconcile(led, s)
        assert rep2.is_ok
        assert led._d['meta']['last_recon']['ok'] is True
        types = {e['type'] for e in led.cash_log()}
        assert 'trade' in types and 'cash_adj' in types

    def test_missing_in_broker_not_auto_fixed(self, tmp_path):
        """账本有券商无 → 不自动处理（人工核查），仅报告"""
        led = Ledger(str(tmp_path / 'l.json'))
        led.initialize(1_000_000)
        led.apply_fills([fill('A', 'BUY', 1000, 10.0, trade_id='X1')])
        s = stmt(cash=led.cash, positions={'A': 1000},
                 trades=[{'date': AS_OF, 'symbol': 'A', 'side': 'BUY',
                          'price': 10.0, 'quantity': 1000, 'trade_id': 'OTHER'}])
        rec = Reconciler()
        rep = rec.reconcile(led, s)
        rec.apply_adjustments(led, s, rep)
        assert led.positions['A'] == 1000       # 未被清掉
        assert led._d['meta']['last_recon']['ok'] is False

    def test_statement_roundtrip(self, tmp_path):
        p = write_statement_template(str(tmp_path / 's.json'))
        s = load_statement(p)
        assert s.source == 'manual' and s.cash == 0.0

    def test_render_markdown(self, tmp_path):
        led = Ledger(str(tmp_path / 'l.json'))
        led.initialize(1_000_000)
        s = stmt(cash=123.0)
        rep = Reconciler().reconcile(led, s)
        md = "\n".join(rep.render_markdown())
        assert '账实核对' in md and '现金不一致' in md
