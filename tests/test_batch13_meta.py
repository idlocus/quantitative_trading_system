#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""第十三批测试：工件元数据dirty/diff_hash/依赖、QMT注释一致性、收益分相字段、runtime目录"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json

import pytest

from research import artifacts


class TestArtifactMeta:

    def test_meta_has_dirty_and_deps(self, tmp_path):
        """修复#1：_meta 必须含 git_dirty/diff_hash/deps（同commit不同工作区可区分）"""
        meta = artifacts.build_meta(formal=False)
        assert 'git_dirty' in meta
        assert 'deps' in meta and 'pandas' in meta['deps'] and 'python' in meta['deps']
        # 当前工作区有未提交修改 → dirty=True 且有 diff_hash
        assert meta['git_dirty'] is True       # 本仓库此刻必然 dirty（审计轮次）
        assert meta['diff_hash'] is not None

    def test_formal_blocks_on_dirty(self, tmp_path):
        """正式工件在 dirty 工作区必须拒绝生成"""
        with pytest.raises(RuntimeError, match='干净工作区'):
            artifacts.build_meta(formal=True)

    def test_save_artifact_injects_full_meta(self, tmp_path, monkeypatch):
        monkeypatch.setattr(artifacts, 'DATA_DIR', str(tmp_path))
        path = artifacts.save_artifact('portfolio', 'test_meta', {'x': 1})
        d = json.load(open(path, encoding='utf-8'))
        meta = d['_meta']
        for k in ('code_version', 'generated_at', 'git_dirty', 'diff_hash',
                  'deps', 'payload_hash'):
            assert k in meta, f"_meta 缺 {k}"

    def test_payload_hash_differs_by_content(self):
        h1 = artifacts._payload_hash({'a': 1})
        h2 = artifacts._payload_hash({'a': 2})
        assert h1 != h2


class TestQmtDocConsistency:

    def test_window_doc_matches_behavior(self):
        """修复#2：check_execution_window 文档必须说'默认拒单'（与 enforce_window 行为一致）"""
        from live import qmt_broker as qb
        doc = qb.check_execution_window.__doc__ or ''
        assert '拒单' in doc, "文档应说明默认拒单而非只警告"
        assert '只警告不阻断' not in doc
        # 构造器文档同步
        init_doc = qb.QMTBroker.__init__.__doc__ or ''
        assert 'enforce_window' in init_doc


class TestPhaseFields:

    def test_liquidation_event_fields_in_payload(self):
        from portfolio.daily_replay import ReplayResult
        from research.artifacts import portfolio_payload
        from live.risk_gate import GateConfig
        cfg = __import__('portfolio.daily_replay', fromlist=['ReplayConfig']).ReplayConfig(
            gate_config=GateConfig(restart_after_liquidation=True))
        import pandas as pd
        r = ReplayResult(config=cfg, n_dates=10, initial_capital=1e6,
                         final_capital=9e5, total_return=-0.1, annual_return=-0.1,
                         sharpe=-1.0, max_drawdown=-0.15, n_rebalances=5,
                         n_trade_days=5, total_cost=100.0, avg_turnover=0.1,
                         n_stop_sells=10, n_halt_days=0, n_no_buy_days=0,
                         first_liquidation_date='2024-02-02',
                         n_liquidation_events=1, restart_mode=True,
                         equity_curve=pd.Series({pd.Timestamp('2026-01-0' + str(i)): 1e6 - i * 1e4 for i in range(1, 6)}))
        p = portfolio_payload(r)
        assert p['first_liquidation_date'] == '2024-02-02'
        assert p['n_liquidation_events'] == 1
        assert p['restart_mode'] is True


class TestRuntimeStateDir:

    def test_defaults_point_to_runtime(self):
        from live import risk_gate, qmt_broker
        assert 'runtime' in risk_gate.DEFAULT_STATE_PATH
        assert 'runtime' in qmt_broker.DEFAULT_ORDER_STATE
        gi = open('.gitignore', encoding='utf-8').read()
        assert 'runtime/' in gi
