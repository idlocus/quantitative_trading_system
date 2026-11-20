#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Kill Switch 测试：一键停止/恢复/闸门集成"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tempfile
from datetime import date

import pandas as pd
import pytest

from live.risk_gate import RiskGate
from live.ledger import Ledger
from live.broker import Order


@pytest.fixture(autouse=True)
def _restore_switch_globals():
    """恢复 kill_switch/trading_switch 的模块级路径全局。

    测试内重定向 KILL_FILE/STATE_FILE 到 tmp_path 后若不恢复，会泄漏到
    同会话后续测试文件（qmt_broker.execute 经 effective_state 读到残留
    CLOSE_ONLY → 买入被滤 → IndexError）。2026-10-15 全量套件
    test_qmt_lifecycle 顺序失败的根因即此。"""
    import scripts.kill_switch as ks
    import scripts.trading_switch as ts
    orig = (ks.KILL_FILE, ts.STATE_FILE)
    yield
    ks.KILL_FILE, ts.STATE_FILE = orig


class TestKillSwitch:

    def test_kill_and_unkill(self, tmp_path):
        from scripts.kill_switch import kill, unkill, is_killed, KILL_FILE
        # 用临时路径
        import scripts.kill_switch as ks
        orig = ks.KILL_FILE
        ks.KILL_FILE = str(tmp_path / 'KILL')

        ks.kill('test')
        assert ks.is_killed()
        assert 'reason: test' in open(ks.KILL_FILE).read()

        ks.unkill()
        assert not ks.is_killed()

        ks.KILL_FILE = orig

    def test_gate_halts_when_killed(self, tmp_path):
        """交易控制=HALT_ALL时，闸门直接halt（2026-10-15 起显式注入，
        不再运行时读全局状态文件——重放/回测不受 runtime 开关影响）"""
        led = Ledger(str(tmp_path / 'l.json'))
        led.initialize(1_000_000)

        dates = pd.bdate_range('2026-09-28', periods=3)
        bench = pd.DataFrame({'date': dates, 'close': [4000.0, 3990.0, 4000.0]})

        from live.risk_gate import GateConfig
        d = RiskGate(GateConfig(trading_switch='HALT_ALL'),
                     state_path=str(tmp_path / 'rg.json')).review(
            [], led, {}, {}, [],
            bench_df=bench, as_of=date(2026, 10, 1))

        assert d.action == 'halt'
        assert any('HALT_ALL' in i or '停止' in i for i in d.data_issues)

    def test_gate_close_only_blocks_buys(self, tmp_path):
        """CLOSE_ONLY时买入被拦截但卖出允许（显式注入）"""
        led = Ledger(str(tmp_path / 'l.json'))
        led.initialize(1_000_000)
        led._d['nav_history'] = [{'date': f'2026-09-{i:02d}', 'nav': 1_000_000} for i in range(1, 15)]
        led.save()

        dates = pd.bdate_range('2026-09-28', periods=3)
        bench = pd.DataFrame({'date': dates, 'close': [4000.0, 3990.0, 4000.0]})

        from live.risk_gate import GateConfig
        orders = [Order('S001.SH', 'BUY', 0.01, amount=5000),
                  Order('S002.SH', 'SELL', 0, quantity=100)]
        d = RiskGate(GateConfig(trading_switch='CLOSE_ONLY'),
                     state_path=str(tmp_path / 'rg.json')).review(
                orders, led, {'S001.SH': 10.0, 'S002.SH': 10.0},
                {'S001.SH': 10.0, 'S002.SH': 10.0}, ['S001.SH'],
                bench_df=bench, as_of=date(2026, 10, 1))

        # 买入应被拦截，卖出保留
        assert all(o.side != 'BUY' for o in d.orders), \
            "CLOSE_ONLY模式不应有买入订单"
        assert any(o.side == 'SELL' for o in d.orders), \
            "CLOSE_ONLY模式卖出应保留"
        assert any('CLOSE_ONLY' in r for r in d.reasons)


class TestSwitchFailSafe:
    """总闸 fail-safe（评审报告 §2.1，2026-10-15 定稿：
    状态文件损坏 → HALT_ALL，读不出状态=停）"""

    def test_corrupt_state_file_halts(self, tmp_path):
        import scripts.trading_switch as ts
        orig = ts.STATE_FILE
        ts.STATE_FILE = str(tmp_path / 'switch.json')
        try:
            (tmp_path / 'switch.json').write_text('{corrupt', encoding='utf-8')
            assert ts.get_state() == 'HALT_ALL', "损坏文件必须按停机处理"
        finally:
            ts.STATE_FILE = orig

    def test_missing_state_file_normal(self, tmp_path):
        import scripts.trading_switch as ts
        orig = ts.STATE_FILE
        ts.STATE_FILE = str(tmp_path / 'nonexistent.json')
        try:
            assert ts.get_state() == 'NORMAL', "无开关文件=正常运行"
        finally:
            ts.STATE_FILE = orig

    def test_replay_gate_ignores_ambient_switch(self, tmp_path):
        """默认 GateConfig 不受 runtime 实时开关影响（回测隔离）"""
        import scripts.trading_switch as ts
        orig = ts.STATE_FILE
        ts.STATE_FILE = str(tmp_path / 'switch.json')
        try:
            (tmp_path / 'switch.json').write_text(
                '{"state": "HALT_ALL"}', encoding='utf-8')
            led = Ledger(str(tmp_path / 'l.json'))
            led.initialize(1_000_000)
            dates = pd.bdate_range('2026-09-28', periods=3)
            bench = pd.DataFrame({'date': dates, 'close': [4000.0, 3990.0, 4000.0]})
            d = RiskGate(state_path=str(tmp_path / 'rg.json')).review(
                [], led, {}, {}, [], bench_df=bench, as_of=date(2026, 10, 1))
            assert d.action != 'halt', "回测/研究默认 NORMAL，不受实时开关污染"
        finally:
            ts.STATE_FILE = orig
