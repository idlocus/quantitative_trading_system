#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化 Barra 风险模型 V1 —— 风格中性化与因子收益估计

V1 范围（全部 PIT 干净）：市值(log) / 动量(60日收益) / 波动率(20日std)
行业分类暂缓（现有申万分类是当前快照，非 PIT；用于风险控制时引入并标注）

核心操作：
- neutralize(): 每日横截面回归 factor ~ 风格暴露，取残差 = 剔除风格暴露后的因子
- factor_returns(): 每日 r_i ~ 暴露_ik 回归 → 风格因子收益序列（归因/监控用）
"""

import numpy as np
import pandas as pd

from utils.logger import get_logger

logger = get_logger(__name__)


def style_exposures(close_mx: pd.DataFrame, mv_mx: pd.DataFrame,
                    momentum_window: int = 60, vol_window: int = 20) -> dict:
    """从价格/市值矩阵构建风格暴露（PIT：全部因果计算）"""
    size = np.log(mv_mx.where(mv_mx > 0))
    momentum = close_mx / close_mx.shift(momentum_window) - 1.0
    vol = close_mx.pct_change().rolling(vol_window).std()
    # 横截面 zscore 标准化（逐日，winsorize 3σ）
    return {'size': _zscore(size), 'momentum': _zscore(momentum), 'volatility': _zscore(vol)}


def _zscore(mx: pd.DataFrame, clip: float = 3.0) -> pd.DataFrame:
    def _row(s):
        v = s.dropna()
        if len(v) < 10:
            return s * np.nan
        m, sd = v.mean(), v.std()
        if sd == 0 or np.isnan(sd):
            return s * np.nan
        return ((v - m) / sd).clip(-clip, clip).reindex(s.index)
    return mx.apply(_row, axis=1)


def neutralize(factor_mx: pd.DataFrame, exposures: dict) -> pd.DataFrame:
    """
    每日横截面中性化：factor ~ style 残差（等权 OLS）

    Args:
        factor_mx: 因子矩阵（dates × symbols）
        exposures: {风格名: 暴露矩阵}（已标准化）
    Returns:
        残差矩阵（与 factor_mx 同形状；再标准化到可比量纲）
    """
    residual = factor_mx.copy() * np.nan
    exp_mats = [e.reindex(columns=factor_mx.columns) for e in exposures.values()]

    for t in factor_mx.index:
        y = factor_mx.loc[t]
        X = np.column_stack([e.loc[t].values for e in exp_mats])
        valid = ~(np.isnan(y.values) | np.isnan(X).any(axis=1))
        n, k = valid.sum(), X.shape[1]
        if n < max(30, (k + 1) * 8):
            continue
        yv, Xv = y.values[valid], X[valid]
        Xv = np.column_stack([np.ones(n), Xv])
        try:
            beta, *_ = np.linalg.lstsq(Xv, yv, rcond=None)
            resid = yv - Xv @ beta
            sd = resid.std()
            if sd == 0:
                continue
            vals = y.values.copy()
            vals[valid] = (resid - resid.mean()) / sd
            residual.loc[t] = vals
        except np.linalg.LinAlgError:
            continue
    return residual


def factor_returns(next_ret: pd.Series, exposures_at_t: dict) -> dict:
    """
    单日因子收益估计：r_i ~ 暴露（供归因/监控，逐日调用）

    Returns: {风格名: 当日因子收益}（截距为 residual_mean）
    """
    names = list(exposures_at_t)
    X = np.column_stack([exposures_at_t[n].values for n in names])
    y = next_ret.values
    valid = ~(np.isnan(y) | np.isnan(X).any(axis=1))
    if valid.sum() < 50:
        return {}
    Xv = np.column_stack([np.ones(valid.sum()), X[valid]])
    beta, *_ = np.linalg.lstsq(Xv, y[valid], rcond=None)
    return {'residual': float(beta[0]), **{n: float(b) for n, b in zip(names, beta[1:])}}
