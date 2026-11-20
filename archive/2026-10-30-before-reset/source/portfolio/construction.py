#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
组合构造 —— 因子合成、选股与权重（向量化，全市场规模可用）

与 selection/scanner 的逐股规则筛选不同，这里走因子矩阵路线：
每个因子在全历史上一次计算（data.factor_matrix.build_factor_matrix），
按日横截面 rank 合成综合分，TopN 构造组合。
"""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from data.factor_matrix import build_factor_matrix  # 数据层（2026-10-15 下沉，原 research.factor_eval）
from utils.logger import get_logger

logger = get_logger(__name__)


def composite_score(cache: Dict[str, pd.DataFrame],
                    factor_funcs: Dict[str, object],
                    weights: Optional[Dict[str, float]] = None,
                    directions: Optional[Dict[str, float]] = None) -> pd.DataFrame:
    """
    构建日频横截面综合因子分（dates × symbols，越高越值得买入）

    因子合成方式（业界标准做法）：各因子每日横截面百分位排名 rank(pct=True)，
    按 direction（+1 因子值越高越好 / -1 反向使用）与 weight 加权求均值。

    Args:
        cache: symbol -> K线 DataFrame
        factor_funcs: 因子名 -> 指标函数
        weights: 因子名 -> 权重（默认等权）
        directions: 因子名 -> +1/-1（默认 +1）

    Returns:
        DataFrame（dates × symbols），值域 [0,1]
    """
    weights = weights or {k: 1.0 for k in factor_funcs}
    directions = directions or {k: 1.0 for k in factor_funcs}

    composite = None
    total_w = 0.0
    for name, func in factor_funcs.items():
        fm = build_factor_matrix(cache, func)
        if fm.empty:
            logger.warning(f"[composite_score] factor '{name}' matrix empty, skipped")
            continue
        # 日频横截面百分位（向量化；rank pct 对 NaN 自动排除）
        ranked = fm.rank(axis=1, pct=True)
        weighted = ranked * directions.get(name, 1.0) * weights[name]
        # 方向为负时值域变为 [-1,0]，统一搬回 [0,1]
        if directions.get(name, 1.0) < 0:
            weighted = weighted + 1.0
        composite = weighted if composite is None else composite.add(weighted, fill_value=np.nan)
        total_w += abs(weights[name])

    if composite is None:
        return pd.DataFrame()
    return composite / total_w


def composite_from_matrices(matrices: Dict[str, pd.DataFrame],
                            weights: Optional[Dict[str, float]] = None,
                            directions: Optional[Dict[str, float]] = None,
                            experimental: bool = False) -> pd.DataFrame:
    """
    从因子矩阵直接合成综合分（基本面因子等外部矩阵入口）

    各矩阵日频横截面 rank(pct) 后按 direction/weight 加权（与 composite_score 同规则）。

    策略入口收口（评审 §2.4，2026-10-15）：directions 必须等于
    chain.PRODUCTION_FACTORS（生产组合），否则必须显式传 experimental=True
    ——无标记的实验合成是 10-15 十年回测口径事故的入口。
    """
    weights = weights or {k: 1.0 for k in matrices}
    directions = directions or {k: 1.0 for k in matrices}

    if not experimental:
        from trading import config as _chain
        if directions != _chain.PRODUCTION_FACTORS:
            raise ValueError(
                f"composite_from_matrices: directions={dict(directions)} ≠ "
                f"生产因子组合 {dict(_chain.PRODUCTION_FACTORS)}。"
                f"实验性合成必须显式传 experimental=True 并登记实验"
                f"（research/run_experiment.py）。")

    # NaN 语义（稀疏因子如一致预期覆盖不全）：按"可用因子"加权均值——
    # 未覆盖的因子不参与该股票的合成，而非使其出局
    weighted_sum = None
    weight_present = None
    for name, fm in matrices.items():
        if fm is None or fm.empty:
            continue
        ranked = fm.rank(axis=1, pct=True)
        weighted = ranked * directions.get(name, 1.0) * abs(weights[name])
        if directions.get(name, 1.0) < 0:
            weighted = weighted + 1.0
        w = abs(weights[name])
        val = weighted.fillna(0.0) * w
        mask = fm.notna().astype(float) * w
        weighted_sum = val if weighted_sum is None else weighted_sum.add(val, fill_value=0.0)
        weight_present = mask if weight_present is None else weight_present.add(mask, fill_value=0.0)

    if weighted_sum is None:
        return pd.DataFrame()
    return weighted_sum / weight_present.replace(0, np.nan)


def select_portfolio(scores: pd.Series, top_n: int,
                     exclude_symbols: Optional[set] = None,
                     band: Optional[tuple] = None) -> List[str]:
    """
    从某日综合分序列选出目标持仓（按分数降序取前N）

    Args:
        scores: 当日综合分（symbol -> score，含 NaN；值域约 [0,1]）
        top_n: 目标持仓数
        exclude_symbols: 排除的标的（如当日涨停不可买）
        band: 可选 (lo, hi) 分数带过滤——只从该区间选股。
              反转类因子在 A 股的经验：极端超卖（高分端）是"接飞刀"尾部，
              中度超卖区最好（因子评价分层数据支持），如 (0.2, 0.8)。
    """
    s = scores.dropna()
    if exclude_symbols:
        s = s[~s.index.isin(exclude_symbols)]
    if band is not None:
        lo, hi = band
        s = s[(s >= lo) & (s <= hi)]
    if s.empty:
        return []
    return s.sort_values(ascending=False).head(top_n).index.tolist()


def equal_weights(symbols: List[str], max_weight: float = 0.05) -> Dict[str, float]:
    """等权构造（带单票上限，超出部分留现金）"""
    if not symbols:
        return {}
    w = min(1.0 / len(symbols), max_weight)
    return {s: w for s in symbols}


def inverse_vol_weights(close_mx: pd.DataFrame, symbols: List[str],
                        rebalance_date, window: int = 20,
                        max_weight: float = 0.05) -> Dict[str, float]:
    """波动率倒数加权（低波动股票权重更高），截尾后归一到 ≤1"""
    if not symbols:
        return {}
    ts = pd.Timestamp(rebalance_date)
    idx = close_mx.index.searchsorted(ts, side='right')
    if idx < window + 1:
        return equal_weights(symbols, max_weight)
    window_close = close_mx[symbols].iloc[idx - window - 1: idx]
    vol = window_close.pct_change().dropna().std()
    inv = 1.0 / vol.replace(0, np.nan)
    inv = inv.dropna()
    if inv.empty:
        return equal_weights(symbols, max_weight)
    raw = inv / inv.sum()
    capped = raw.clip(upper=max_weight)
    return capped.to_dict()
