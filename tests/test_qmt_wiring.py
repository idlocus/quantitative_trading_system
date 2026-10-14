#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""QMT 接通测试：单一 QMTBroker、执行器工厂、CLI 分支、环境不就绪不静默"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import inspect

import pytest

import live
import live.broker as broker_mod
import live.qmt_broker as qmt_mod
from live import make_broker, PaperBroker


class TestSingleQMTBroker:

    def test_live_exports_the_real_qmt_broker(self):
        """live.QMTBroker = qmt_broker.QMTBroker（完整实现，含 execute/connect/fetch_statement）"""
        assert live.QMTBroker is qmt_mod.QMTBroker
        for m in ('execute', 'connect', 'fetch_statement', 'check_env', 'cancel_all_pending'):
            assert hasattr(live.QMTBroker, m)

    def test_placeholder_removed_from_broker_module(self):
        """live/broker.py 的旧占位 QMTBroker 已删除（此前 from live.broker import QMTBroker 拿到坏类）"""
        assert not hasattr(broker_mod, 'QMTBroker')

    def test_all_list_resolves(self):
        """__all__ 每个名字都能从 live 包解析（此前 'QMTBroker' 声明但不存在 → ImportError）"""
        for name in live.__all__:
            assert hasattr(live, name), name


class TestMakeBroker:

    def test_paper_mode(self):
        b = make_broker('paper')
        assert isinstance(b, PaperBroker)

    def test_qmt_env_not_ready_raises_with_guidance(self, monkeypatch):
        """环境不就绪 → RuntimeError 带逐项修复指引（不静默跳过）"""
        # 绕过部署状态检查（单测不要求CANARY）
        monkeypatch.setattr('scripts.deployment.get_deployment_state', lambda: 'CANARY')
        monkeypatch.setattr(qmt_mod.QMTBroker, 'check_env',
                            lambda: {'ready': False,
                                     'issues': ['QMT_PATH 未设置', 'xtquant 未安装']})
        with pytest.raises(RuntimeError) as ei:
            make_broker('qmt')
        assert 'QMT_PATH 未设置' in str(ei.value)
        assert '不执行任何指令' in str(ei.value)

    def test_qmt_env_ready_returns_broker(self, monkeypatch):
        monkeypatch.setattr('scripts.deployment.get_deployment_state', lambda: 'CANARY')
        monkeypatch.setattr(qmt_mod.QMTBroker, 'check_env',
                            lambda: {'ready': True, 'issues': []})
        b = make_broker('qmt')
        assert isinstance(b, qmt_mod.QMTBroker)

    def test_unknown_mode_raises(self):
        with pytest.raises(ValueError):
            make_broker('ib')


class TestCliWiring:
    """源码守护：CLI 必须真正走 make_broker，不允许静默跳过或'待就位'占位文案"""

    def test_run_daily_pm_executes_both_modes(self):
        import run_daily
        src = inspect.getsource(run_daily.run_pm)
        assert 'make_broker(execute' in src                    # 工厂构造（含lot_map参数）
        assert "execute in ('paper', 'qmt')" in src
        assert "execute == 'paper'" not in src                 # 不再有只处理paper的分支
        assert '执行器不可用' in src                             # qmt失败必须显式报错

    def test_run_daily_has_yes_flag(self):
        import run_daily
        src = inspect.getsource(run_daily.main)
        assert '--yes' in src

    def test_run_live_uses_factory_and_confirms_qmt(self):
        import run_live
        src = inspect.getsource(run_live.main)
        assert 'make_broker(args.execute' in src                # 工厂构造（含lot_map参数）
        assert '待券商环境就位' not in src                       # 占位文案已删除
        assert '确认执行' in src                                 # 实盘二次确认存在

    def test_run_live_has_check_qmt(self):
        import run_live
        assert '--check-qmt' in inspect.getsource(run_live.main)


class TestCashConstraintWiring:
    """现金约束接线：所有执行点必须传 available_cash（run_live 曾遗漏，
    独立跑 --execute paper 时 PaperBroker 不检查现金、与主流程行为不一致）"""

    def test_all_execute_sites_pass_available_cash(self):
        import inspect
        import run_daily
        import run_live
        from portfolio import daily_replay
        for src, name in [(inspect.getsource(run_daily.run_pm), 'run_daily.run_pm'),
                          (inspect.getsource(run_live.main), 'run_live.main'),
                          (inspect.getsource(daily_replay.DailyReplayBacktester.run),
                           'DailyReplayBacktester.run')]:
            assert 'broker.execute(' in src, f"{name}: 应经 broker.execute 执行"
            assert 'available_cash=ledger.cash' in src, \
                f"{name}: broker.execute 必须传 available_cash=ledger.cash"

    def test_paper_broker_enforces_cash(self):
        """现金不足时买入被缩减或拒单，可用现金不透支"""
        from live.broker import Order, PaperBroker
        from datetime import date
        b = PaperBroker()
        fills = b.execute([Order('600000.SH', 'BUY', 0.5, amount=1_000_000)],
                          {'600000.SH': 100.0}, date(2026, 10, 12),
                          available_cash=5_000)
        f = fills[0]
        if f.quantity > 0:
            assert f.quantity * f.price <= 5_000 * 1.01   # 成交不超可用现金
        else:
            assert '资金不足' in f.status
        assert b is not None

    def test_qmt_path_shares_same_call_site(self):
        """paper 与 qmt 共用同一执行块（同一 available_cash 语义，不允许分叉）"""
        import inspect
        import run_live
        src = inspect.getsource(run_live.main)
        assert "args.execute in ('paper', 'qmt')" in src
        assert src.count('available_cash=ledger.cash') >= 1


class TestQMTFillCarriesTradeId:

    def test_fill_dataclass_has_trade_id(self):
        """QMT 成交回报带订单号 → 账本去重优先键（成交回报重放不重复入账）"""
        from live.broker import Fill, Order
        f = Fill(Order('A', 'BUY', 0), 10.0, 100, 3, 0, trade_date=None,
                 status='filled', trade_id='Q123')
        assert f.trade_id == 'Q123'
