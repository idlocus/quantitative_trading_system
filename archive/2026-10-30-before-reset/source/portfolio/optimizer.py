#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
凸优化组合构造器（指增式约束）

目标: max α'w - λ·w'Σw - τ·||w - w_prev||₁
约束: Σw = 1；0 ≤ w ≤ 单票上限；|风格暴露·w - 基准暴露| ≤ 风格预算

协方差: 样本协方差 + 收缩到对角（稳健化）。
V1 无行业约束（行业分类非PIT，引入时标注）；TE 约束留接口。
"""

from typing import Dict, Optional

import numpy as np
import pandas as pd
import cvxpy as cp

from utils.logger import get_logger

logger = get_logger(__name__)


class OptimizerConfig:
    max_weight: float = 0.03          # 单票上限（Top100组合默认3%）
    risk_aversion: float = 8.0        # λ：风险厌恶
    turnover_penalty: float = 0.005   # τ：换手惩罚
    shrinkage: float = 0.10           # 协方差收缩系数（→对角）
    style_budget: float = 0.30        # 风格暴露偏离预算（zscore尺度，|·|≤0.3）
    min_weight: float = 0.0

    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


def shrink_cov(returns: pd.DataFrame, shrink: float = 0.10) -> pd.DataFrame:
    """样本协方差 + Ledoit-Wolf 式收缩到对角"""
    S = returns.cov()
    if S.empty:
        return S
    D = pd.DataFrame(np.diag(np.diag(S)), index=S.index, columns=S.columns)
    return (1 - shrink) * S + shrink * D


def solve_weights(alpha: pd.Series,
                  returns: pd.DataFrame,
                  style_exp: Optional[pd.DataFrame] = None,
                  bench_style: Optional[Dict[str, float]] = None,
                  prev_weights: Optional[Dict[str, float]] = None,
                  config: Optional[OptimizerConfig] = None) -> Dict[str, float]:
    """
    求解最优权重

    Args:
        alpha: 候选股 alpha 分数（选股域内的 symbol → score）
        returns: 候选股收益率历史（dates × candidates）
        style_exp: 候选股风格暴露（candidates × [size,momentum,volatility]）
        bench_style: 基准的风格暴露 {风格名: 值}（约束锚点）
        prev_weights: 上一期权重（换手惩罚）
        config: 优化参数

    Returns:
        {symbol: 权重}（和为1；求解失败回退等权）
    """
    cfg = config or OptimizerConfig()
    cands = [s for s in alpha.index if s in returns.columns]
    if not cands:
        return {}
    n = len(cands)
    if n < 5:
        w = min(1.0 / n, cfg.max_weight)
        return {s: w for s in cands}

    a = alpha[cands].fillna(alpha[cands].min()).values
    S = shrink_cov(returns[cands], cfg.shrinkage).values
    if np.isnan(S).any() or not np.isfinite(S).all():
        S = np.nan_to_num(S, nan=float(np.nanmean(S)) if np.isfinite(np.nanmean(S)) else 1e-4)

    w = cp.Variable(n)
    w_prev = np.array([prev_weights.get(s, 0.0) for s in cands]) if prev_weights else np.zeros(n)

    obj = cp.Maximize(a @ w
                      - cfg.risk_aversion * cp.quad_form(w, cp.psd_wrap(S))
                      - cfg.turnover_penalty * cp.norm1(w - w_prev))
    cons = [cp.sum(w) == 1, w >= cfg.min_weight, w <= cfg.max_weight]

    # 风格偏离约束（相对基准锚点）
    if style_exp is not None and bench_style:
        E = style_exp.reindex(cands)
        for f, anchor in bench_style.items():
            if f in E.columns and E[f].notna().sum() >= n * 0.8:
                e = E[f].fillna(E[f].median()).values
                cons.append(e @ w <= anchor + cfg.style_budget)
                cons.append(e @ w >= anchor - cfg.style_budget)

    prob = cp.Problem(obj, cons)
    try:
        prob.solve(solver=cp.ECOS)
    except Exception:
        try:
            prob.solve(solver=cp.SCS)
        except Exception as e:
            logger.warning(f"[optimizer] solve failed: {e}, fallback equal weight")
            w_eq = min(1.0 / n, cfg.max_weight)
            return {s: w_eq for s in cands}

    if w.value is None or prob.status not in ('optimal', 'optimal_inaccurate'):
        logger.warning(f"[optimizer] status={prob.status}, fallback equal weight")
        w_eq = min(1.0 / n, cfg.max_weight)
        return {s: w_eq for s in cands}

    weights = np.clip(w.value, 0, None)
    weights = weights / weights.sum()
    # 截尾再归一（数值稳定性）
    weights = np.minimum(weights, cfg.max_weight)
    weights = weights / weights.sum()
    return {s: float(x) for s, x in zip(cands, weights)}
