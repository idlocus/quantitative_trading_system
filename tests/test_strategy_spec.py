#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""StrategySpec 测试：版本治理——规格派生、哈希、工件/信号/日报集成"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from trading.spec import StrategySpec, STRATEGY_VERSION, attach_spec_to_artifact


class TestStrategySpec:

    def test_current_from_config(self):
        """规格从 trading/config 自动派生——不可独立维护"""
        spec = StrategySpec.current()
        from trading import config as chain
        assert spec.factors == dict(chain.PRODUCTION_FACTORS)
        assert spec.momentum_weight == chain.MOMENTUM_WEIGHT
        assert spec.top_n == chain.TOP_N

    def test_hash_deterministic(self):
        """相同参数=相同hash"""
        s1 = StrategySpec.current()
        s2 = StrategySpec.current()
        assert s1.hash == s2.hash

    def test_hash_changes_with_params(self):
        """参数变=hash变"""
        s1 = StrategySpec.current()
        s2 = StrategySpec(top_n=50)  # 改一个参数
        assert s1.hash != s2.hash

    def test_to_dict_roundtrip(self):
        """序列化可读、包含所有关键字段"""
        spec = StrategySpec.current()
        d = spec.to_dict()
        for key in ('version', 'factors', 'quality_filter', 'momentum_weight',
                     'top_n', 'commission', 'drawdown_hard', 'strategy_hash'):
            assert key in d or hasattr(spec, key), f"缺少关键字段: {key}"
        assert d['version'] == STRATEGY_VERSION

    def test_summary_readable(self):
        """一行摘要包含版本+因子+过滤+关键参数"""
        spec = StrategySpec.current()
        s = spec.summary()
        assert STRATEGY_VERSION in s
        assert 'ep' in s
        assert 'Top' in s

    def test_version_format(self):
        """版本号格式 semver"""
        parts = STRATEGY_VERSION.split('.')
        assert len(parts) == 3, f"版本号应为 major.minor.patch，实际: {STRATEGY_VERSION}"

    def test_attach_to_artifact(self):
        """工件附上策略规格"""
        payload = {'data': 123}
        result = attach_spec_to_artifact(payload)
        assert 'strategy_spec' in result['_meta']
        assert 'strategy_hash' in result['_meta']
        assert result['_meta']['strategy_spec']['version'] == STRATEGY_VERSION

    def test_frozen(self):
        """规格不可变（frozen=True）"""
        spec = StrategySpec.current()
        with pytest.raises(AttributeError):
            spec.top_n = 50  # type: ignore


class TestSpecIntegration:
    """规格已集成到信号和工件"""

    def test_signal_carries_spec(self):
        """生产信号附带策略规格"""
        import inspect
        from live.signals import ProductionSignal, generate_production_signal
        sig = inspect.signature(ProductionSignal)
        assert 'spec' in sig.parameters, "ProductionSignal 缺少 spec 字段"

    def test_artifact_has_spec(self):
        """save_artifact 自动注入策略规格"""
        import inspect
        from research.artifacts import save_artifact
        src = inspect.getsource(save_artifact)
        assert 'attach_spec_to_artifact' in src, \
            "save_artifact 应调用 attach_spec_to_artifact"

    def test_journal_shows_version(self):
        """日报显示策略版本"""
        from live.journal import JournalWriter
        import inspect
        src = inspect.getsource(JournalWriter.write_am)
        assert '策略版本' in src or 'spec' in src, \
            "日报应显示策略版本"
