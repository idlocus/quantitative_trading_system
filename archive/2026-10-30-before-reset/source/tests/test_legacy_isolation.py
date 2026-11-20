#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""遗留代码隔离测试：主链模块不得 import backtesting/strategy/selection（弃用包）"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ast


# 主链模块清单（生产/回测/数据层——这些不允许引用弃用包）
MAIN_CHAIN = [
    'live/signals.py', 'live/broker.py', 'live/ledger.py', 'live/journal.py',
    'live/risk_gate.py', 'live/reconcile.py', 'live/qmt_broker.py',
    'portfolio/backtester.py', 'portfolio/daily_replay.py',
    'portfolio/risk_manager.py', 'portfolio/construction.py',
    'portfolio/tradable_benchmark.py',
    'trading/config.py', 'trading/rules.py',
    'data/layered.py', 'data/wind.py', 'data/st_provider.py',
    'data/turnover.py', 'data/board_params.py',
    'run_daily.py', 'run_live.py', 'run_daily_replay.py',
]

# 弃用包名（risk 包=旧引擎残留，SLIPPAGE=0.0 与主链不可比且零测试——
# 评审报告 §2.7；主链风险定义在 trading/config + live/risk_gate）
DEPRECATED = ('backtesting', 'strategy', 'selection', 'risk')


class TestLegacyIsolation:

    def test_main_chain_does_not_import_deprecated(self):
        """主链模块不得 import 弃用包（backtesting/strategy/selection）"""
        violations = []
        for path in MAIN_CHAIN:
            full = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), path)
            if not os.path.exists(full):
                continue
            src = open(full, encoding='utf-8').read()
            tree = ast.parse(src)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        root = alias.name.split('.')[0]
                        if root in DEPRECATED:
                            violations.append(f"{path}: import {alias.name}")
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        root = node.module.split('.')[0]
                        if root in DEPRECATED:
                            violations.append(f"{path}: from {node.module} import ...")
        assert not violations, \
            f"主链模块引用了弃用包：\n" + "\n".join(violations) + \
            f"\n弃用包({DEPRECATED})是研究遗留，禁止进入主链。见 docs/唯一主链架构.md"

    def test_deprecated_headers_present(self):
        """弃用模块必须有 DEPRECATED 标记"""
        for path in ['backtesting/engine.py', 'strategy/__init__.py',
                     'selection/__init__.py']:
            full = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), path)
            if os.path.exists(full):
                src = open(full, encoding='utf-8').read()
                assert 'DEPRECATED' in src, f"{path} 缺少弃用标记"
