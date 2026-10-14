#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
唯一策略构建函数 —— 生产、重放、实验、日报全部经由这里

禁止任何路径自行维护因子列表、过滤阈值或动量权重。
所有参数来自 trading/config（单一来源）。

使用方式：
    from trading.strategy import build_strategy_score
    score, cache, mats = build_strategy_score(ds, pool, start, end)
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
from datetime import date

import pandas as pd

from data.fundamentals import WindFundamentalProvider, build_fundamental_matrix
from data.fundamental_pit import PITFinancialProvider
from portfolio.construction import composite_from_matrices
from trading import config as chain

logger = logging.getLogger(__name__)


def _make_loader(ds):
    """K线加载器（复制自 live/signals.py，避免循环导入）"""
    from data.base.data_loader import DataLoader
    loader = DataLoader.__new__(DataLoader)
    loader.data_source_type = 'layered'
    loader.data_source = ds
    loader._cache = None
    return loader


def build_strategy_score(ds, pool, start, end):
    """
    构建v2策略分数（质量过滤 + 价值合成 + 趋势叠加）。

    这是唯一的策略实现——生产、重放、实验调用同一函数，
    改策略只改 trading/config.py + 本函数，所有路径自动生效。

    Returns:
        (score_mx: DataFrame, cache: dict, factor_mats: dict)
        factor_mats 含原始因子矩阵（供单因子覆盖率检查）
    """
    cache = _make_loader(ds).preload_klines(pool + ['000300.SH'], start, end)._cache

    # ---- 价值合成（仅 PRODUCTION_FACTORS 声明的因子） ----
    prov = WindFundamentalProvider(ds)
    mats = {}
    for f in chain.PRODUCTION_FACTORS:
        fm = build_fundamental_matrix(prov, f, pool, start, end)
        if not fm.empty:
            # 契约校验（单位/值域/覆盖率）
            try:
                from data.contracts import validate_factor_matrix, get_factor_contract
                meta = validate_factor_matrix(fm, get_factor_contract(f))
                if not meta['passed']:
                    import logging
                    logging.getLogger(__name__).warning(
                        f"[strategy] 因子'{f}'契约校验未通过: {meta['issues']}")
            except Exception:
                pass
            mats[f] = fm
    score = composite_from_matrices(mats, directions=chain.PRODUCTION_FACTORS)
    score = score[score.columns.intersection(cache.keys())]

    # ---- 质量过滤（ROE + 杠杆，阈值来自 chain.QUALITY_FILTER） ----
    pit = PITFinancialProvider(ds)
    qf = chain.QUALITY_FILTER
    if qf.get('roe_min') is not None:
        roe = pit.get_factor_matrix('roe', pool, start, end)
        if not roe.empty:
            score = _filter_by_factor(score, roe, 'min', qf['roe_min'], 'ROE')
    if qf.get('leverage_max') is not None:
        lev = pit.get_factor_matrix('leverage', pool, start, end)
        if not lev.empty:
            score = _filter_by_factor(score, lev, 'max', qf['leverage_max'], '杠杆')

    # ---- 趋势叠加（6个月剔1个月 × MOMENTUM_WEIGHT） ----
    if chain.MOMENTUM_WEIGHT > 0:
        momentum = _build_momentum_matrix(cache, start, end)
        if not momentum.empty:
            score = _combine_value_momentum(score, momentum, chain.MOMENTUM_WEIGHT)

    return score, cache, mats


def _filter_by_factor(score, factor_mx, mode, threshold, label):
    """按因子值过滤（mode='min'排除低于阈值，'max'排除高于阈值）"""
    common_dates = score.index.intersection(factor_mx.index)
    common_cols = score.columns.intersection(factor_mx.columns)
    if len(common_dates) == 0 or len(common_cols) == 0:
        return score
    result = score.copy()
    f_aligned = factor_mx.reindex(index=common_dates, columns=common_cols)
    for dt in common_dates:
        row = f_aligned.loc[dt]
        if mode == 'min':
            fail = row[row < threshold].index
        else:
            fail = row[row > threshold].index
        if len(fail) > 0:
            result.loc[dt, fail] = pd.NA
    return result


def _build_momentum_matrix(cache, d0, d1):
    """6个月剔1个月复权收益矩阵"""
    from experiments.momentum_overlay import build_momentum_matrix
    return build_momentum_matrix(cache, d0, d1)


def _combine_value_momentum(score, momentum, weight):
    """价值+动量截面z-score加权合成"""
    common_dates = score.index.intersection(momentum.index)
    common_cols = score.columns.intersection(momentum.columns)
    if len(common_dates) == 0:
        return score
    result = score.copy()
    for dt in common_dates:
        v = score.loc[dt].reindex(common_cols)
        m = momentum.loc[dt].reindex(common_cols)
        v_z = (v - v.mean()) / (v.std() + 1e-10)
        m_z = (m - m.mean()) / (m.std() + 1e-10)
        result.loc[dt, common_cols] = (1 - weight) * v_z + weight * m_z
    return result
