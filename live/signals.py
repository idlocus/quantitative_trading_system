#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生产信号生成 —— 5因子合成（ep+bp+sp+size+leverage），PIT 到最新交易日

输出目标持仓与调仓订单（与账本持仓的差额），涨跌停不可买自动顺延。
"""

from datetime import date, timedelta
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from data.layered import LayeredDataSource
from data.fundamentals import WindFundamentalProvider, build_fundamental_matrix
from data.fundamental_pit import PITFinancialProvider
from portfolio.construction import composite_from_matrices

# 生产配置（与回测验收版本一致）
PRODUCTION_FACTORS = {'ep': 1.0, 'bp': 1.0, 'sp': 1.0, 'size': -1.0, 'leverage': -1.0}
DEFAULT_TOP_N = 100         # 与回测验证一致：Top100（Top30经回测验证跑输）
WARMUP_DAYS = 400


def generate_target(ds: LayeredDataSource, as_of: date,
                    top_n: int = DEFAULT_TOP_N) -> Tuple[List[str], pd.Series]:
    """
    生成 as_of 收盘后的目标持仓（信号用 ≤as_of 数据，PIT）

    Returns:
        (目标列表, 综合分Series)
    """
    start = as_of - timedelta(days=WARMUP_DAYS)
    pool = sorted(ds.list_symbols(as_of=as_of))

    loader_mod = _make_loader(ds)
    cache = loader_mod.preload_klines(pool, start, as_of)._cache
    prov = WindFundamentalProvider(ds)
    pit = PITFinancialProvider(ds)

    mats = {}
    for f in ['ep', 'bp', 'sp', 'size']:
        fm = build_fundamental_matrix(prov, f, pool, start, as_of)
        if not fm.empty:
            mats[f] = fm
    lev = pit.get_factor_matrix('leverage', pool, start, as_of)
    if not lev.empty:
        mats['leverage'] = lev

    score_mx = composite_from_matrices(mats, directions=PRODUCTION_FACTORS)
    if score_mx.empty:
        return [], pd.Series(dtype=float)
    ts = pd.Timestamp(as_of)
    valid_dates = score_mx.index[score_mx.index <= ts]
    if len(valid_dates) == 0:
        return [], pd.Series(dtype=float)
    scores = score_mx.loc[valid_dates[-1]].dropna()

    # 排除当日无价格（停牌）——用缓存尾部判断
    from research.factor_eval import FactorEvaluator
    ev = FactorEvaluator(cache)
    last_prices = ev.close_mx.loc[valid_dates[-1]] if len(valid_dates) else pd.Series()
    tradable = last_prices.dropna()
    scores = scores[scores.index.intersection(tradable.index)]

    targets = scores.sort_values(ascending=False).head(top_n).index.tolist()
    return targets, scores


def build_orders(targets: List[str], current_weights: Dict[str, float],
                 total_nav: float, top_n: int = DEFAULT_TOP_N,
                 band: float = 0.20) -> List[dict]:
    """
    目标持仓 vs 当前持仓 → 调仓指令（带缓冲带，降低换手）

    规则：目标权重=1/top_n；权重差超过 band×目标权重 才调；
    移出目标的持仓全部卖出。
    """
    from live.broker import Order
    orders = []
    target_w = 1.0 / max(top_n, 1)
    target_set = set(targets)

    # 卖出：不在目标中 或 超配
    for s, w in current_weights.items():
        if s not in target_set:
            orders.append(Order(s, 'SELL', 0.0, amount=w * total_nav,
                                note='移出组合'))
        elif w > target_w * (1 + band):
            excess = (w - target_w) * total_nav
            orders.append(Order(s, 'SELL', target_w, amount=excess, note='超配回收'))

    # 买入：新目标 或 低配
    cash_weight = 1.0 - sum(current_weights.values())
    for s in targets:
        w = current_weights.get(s, 0.0)
        if w < target_w * (1 - band):
            alloc = min(target_w - w, cash_weight + 0.05)
            if alloc > 0:
                orders.append(Order(s, 'BUY', target_w, amount=alloc * total_nav,
                                    note='新入/低配'))
    return orders


def _make_loader(ds):
    from data.base.data_loader import DataLoader
    loader = DataLoader.__new__(DataLoader)
    loader.data_source_type = 'layered'
    loader.data_source = ds
    loader._cache = None
    return loader
