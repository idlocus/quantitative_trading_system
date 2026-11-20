#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""单因子覆盖率闸门测试：任一必需因子崩塌→halt（防止单因子缺失被重新加权掩盖）

2026-10-30 重写：原版 patch 不完整（缺 PIT/动量适配，70天数据动量全NaN），
首个用例长期 xfail、边界用例断言恒真（len>0 or len==0）——无守护力。
现按 build_strategy_score 架构补全 patch（因子矩阵/Provider/足够长数据），
断言明确方向；并入 release_gate 门槛集（CLAUDE.md R3 条款）。
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from live.signals import generate_target, MIN_FACTOR_COVERAGE
from trading import config as chain

pytestmark = pytest.mark.release_gate

DAYS = 200   # ≥126+21 交易日，动量矩阵可用（否则综合分全NaN，测不到闸门）


def make_ds_with_factors(ep_cov=1.0, bp_cov=1.0, sp_cov=1.0,
                         base_price=10.0, n_syms=50):
    """
    构造假数据源：
    - K线：n_syms只股票，价格恒定（动量=0，不干扰价值截面）
    - EP/BP/SP因子矩阵：按覆盖率参数控制非NaN比例
    """
    dates = pd.bdate_range('2026-01-05', periods=DAYS)
    syms = [f'S{i:04d}.SH' for i in range(n_syms)]

    def factor_matrix(cov, seed):
        """按覆盖率生成因子矩阵（前cov比例非NaN，其余NaN）"""
        n_valid = int(n_syms * cov)
        vals = np.full(n_syms, np.nan)
        vals[:n_valid] = np.random.RandomState(seed).randn(n_valid)
        return pd.DataFrame([vals] * DAYS, index=dates, columns=syms)

    kline = pd.DataFrame({
        'date': dates,
        'open': [base_price] * DAYS,
        'high': [base_price] * DAYS,
        'low': [base_price] * DAYS,
        'close': [base_price] * DAYS,
        'volume': [500e4] * DAYS,
        'amount': [base_price * 500e4] * DAYS,
        'adjclose': [base_price] * DAYS,
    })

    class FakeDS:
        def list_symbols(self, as_of=None):
            return syms
        def preload_batch(self, syms_list, start, end):
            out = {s: kline.copy() for s in syms_list if s != '000300.SH'}
            out['000300.SH'] = kline.copy()
            return out
        def _db_src(self):
            return None

    ds = FakeDS()
    mats = {'ep': factor_matrix(ep_cov, 1), 'bp': factor_matrix(bp_cov, 2),
            'sp': factor_matrix(sp_cov, 3)}
    as_of = dates[-1].date()
    return ds, mats, syms, as_of


class FakePIT:
    """质量过滤全通过（ROE/杠杆不干扰覆盖率闸门测试）"""
    def __init__(self, ds):
        pass
    def get_factor_matrix(self, name, pool, s, e):
        value = 0.20 if name == 'roe' else 30.0
        return pd.DataFrame(value, index=pd.bdate_range(s, e), columns=pool)


@pytest.fixture
def patched_strategy(monkeypatch):
    """补全 patch：因子矩阵注入 + Provider 替换（build_strategy_score 架构）"""
    def _patch(mats):
        import trading.strategy as strat
        monkeypatch.setattr(
            strat, 'build_fundamental_matrix',
            lambda prov, f, pool, s, e: mats.get(f, pd.DataFrame()))
        monkeypatch.setattr(strat, 'WindFundamentalProvider', lambda ds: None)
        monkeypatch.setattr(strat, 'PITFinancialProvider', FakePIT)
        monkeypatch.setattr('data.fundamental_pit.PITFinancialProvider', FakePIT)
        return strat
    return _patch


class TestSingleFactorCoverage:
    """单因子覆盖率闸门：综合分覆盖正常但单因子崩塌→halt"""

    def test_all_factors_normal_passes(self, patched_strategy):
        """全部因子覆盖率100%→正常生成目标"""
        ds, mats, syms, as_of = make_ds_with_factors(
            ep_cov=1.0, bp_cov=1.0, sp_cov=1.0)
        patched_strategy(mats)
        targets, scores = generate_target(ds, as_of, top_n=30, st_syms=set())
        assert len(targets) == 30, \
            f"全覆盖时应生成30个目标（实际{len(targets)}）"

    def test_boundary_coverage_passes(self, patched_strategy):
        """恰好30%覆盖率应放行（>= 而非 >）——边界锁定"""
        ds, mats, syms, as_of = make_ds_with_factors(
            ep_cov=0.30, bp_cov=1.0, sp_cov=1.0)
        patched_strategy(mats)
        targets, scores = generate_target(ds, as_of, top_n=30, st_syms=set())
        assert len(targets) == 30, \
            f"ep恰{MIN_FACTOR_COVERAGE:.0%}覆盖应达标放行（实际{len(targets)}）"

    def test_single_factor_collapse_halts(self, patched_strategy):
        """ep覆盖率29%<30%→单因子闸门触发→空目标（数据停机，非降级继续）"""
        ds, mats, syms, as_of = make_ds_with_factors(
            ep_cov=0.29, bp_cov=1.0, sp_cov=1.0)
        patched_strategy(mats)
        targets, scores = generate_target(ds, as_of, top_n=30, st_syms=set())
        assert len(targets) == 0, \
            "单因子覆盖率29%<30%必须空目标halt（防崩塌被重新加权掩盖）"

    def test_min_target_count_halts(self, patched_strategy):
        """目标数<min(20, top_n)→空目标（R3：不能以残缺组合继续交易）"""
        ds, mats, syms, as_of = make_ds_with_factors(
            ep_cov=1.0, bp_cov=1.0, sp_cov=1.0, n_syms=15)
        patched_strategy(mats)
        targets, scores = generate_target(ds, as_of, top_n=30, st_syms=set())
        assert len(targets) == 0, \
            f"仅15只候选（<下限20）应空目标halt（实际{len(targets)}）"
