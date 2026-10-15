#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
因子矩阵构建（数据层）—— 从缓存K线 + 指标函数构建 dates × symbols 截面矩阵

2026-10-15 依赖倒置修复（评审 §2.3）：本函数原在 research/factor_eval.py，
生产层（portfolio/construction）依赖研究层——研究重构会静默改变生产行为。
现下沉到数据层，research/factor_eval 反向引用（ backward compat）。
"""
import logging
from typing import Callable, Dict

import pandas as pd

logger = logging.getLogger(__name__)


def build_factor_matrix(cache: Dict[str, pd.DataFrame], factor_func: Callable) -> pd.DataFrame:
    """
    构建因子截面矩阵（dates × symbols）

    每只股票的指标在其完整历史K线上计算一次（rolling/ewm 只依赖历史，
    时点正确），再按日期对齐到统一矩阵。
    DataFrame 型指标自动取注册时声明的 primary_column。
    """
    data = {}
    for sym, df in cache.items():
        if df is None or df.empty:
            continue
        try:
            vals = factor_func(df)
            if isinstance(vals, pd.DataFrame):
                col = getattr(factor_func, '_primary_column', None)
                vals = vals[col] if col and col in vals.columns else vals.iloc[:, -1]
            vals = pd.Series(vals)
            vals = vals.astype(float)
            # 对齐日期索引
            if not isinstance(vals.index, pd.DatetimeIndex):
                if len(vals) == len(df):
                    vals.index = pd.to_datetime(df['date'])
                else:
                    continue
            # 统一时间分辨率（pandas 3 中 us/ns 混用会导致跨索引 .loc 失败）
            vals.index = pd.DatetimeIndex(vals.index).as_unit('ns')
            data[sym] = vals
        except Exception as e:
            logger.debug(f"[build_factor_matrix] {sym}: {e}")
            continue
    return pd.DataFrame(data)
