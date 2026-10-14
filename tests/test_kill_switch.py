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
        """Kill Switch 激活时，闸门直接halt"""
        import scripts.kill_switch as ks
        kill_file = str(tmp_path / 'KILL')
        ks.KILL_FILE = kill_file
        ks.kill('test_halt')

        # 创建简单账本
        led = Ledger(str(tmp_path / 'l.json'))
        led.initialize(1_000_000)

        dates = pd.bdate_range('2026-09-28', periods=3)
        bench = pd.DataFrame({'date': dates, 'close': [4000.0, 3990.0, 4000.0]})

        # Mock is_killed 使用我们的临时文件
        from unittest.mock import patch
        with patch('scripts.kill_switch.KILL_FILE', kill_file):
            with patch('scripts.kill_switch.is_killed', return_value=True):
                d = RiskGate(state_path=str(tmp_path / 'rg.json')).review(
                    [], led, {}, {}, [],
                    bench_df=bench, as_of=date(2026, 10, 1))

        assert d.action == 'halt'
        assert any('人工停止' in i or 'KILL' in i for i in d.data_issues)

        # 清理
        ks.unkill()
