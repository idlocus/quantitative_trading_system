#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
资金流向指标 — 机构资金与游资分离

包含指标：
- institutional_ratio: 主力资金净流入占比 (N日平均)
- retail_ratio: 游资净流入占比 (N日平均)
- inst_retail_diff: 机构-游资资金差 (N日累计)
- inst_retail_ratio: 机构/游资比率 (>1=机构主导, <1=游资主导)

注意：这些指标的 data DataFrame 需要包含 '主力净流入' 和 '散户净流入' 列，
由 scanner 在调用前通过 FundFlowProvider 注入。

使用方式（在 strategies.yml 中引用）：
  - name: institutional_ratio
    operator: ">"
    threshold: 0.03
    weight: 1.5
"""

import pandas as pd
import numpy as np

from .registry import IndicatorRegistry


@IndicatorRegistry.register('institutional_ratio', category='volume')
def InstitutionalFlowRatio(data, period=5):
    """
    主力资金净流入占比 (N日平均)

    公式: 主力净流入 / 成交额 (N日平均)
    主力净流入 = 超大单净流入 + 大单净流入

    信号：
    - > 5%   = 机构强势买入（吸筹阶段）
    - > 0%   = 机构净买入
    - < 0%   = 机构净卖出（派发阶段）
    - < -5%  = 机构大幅卖出
    """
    # 主力净流入列由 scanner 注入
    inst_col = '主力净流入'
    amount_col = '成交额'
    close_col = 'close'

    if inst_col not in data.columns:
        # 没有资金流向数据，返回 NaN
        return pd.Series(np.nan, index=data.index, name='institutional_ratio')

    inst_flow = data[inst_col]

    # 如果有成交额列，用成交额；否则用 close * volume 估算
    if amount_col in data.columns:
        amount = data[amount_col].abs()
    else:
        amount = (data['close'] * data['volume']).abs()

    # 避免除以0
    ratio = inst_flow / amount.replace(0, np.nan)

    result = ratio.rolling(window=period).mean()
    return pd.Series(result.values, index=data.index, name='institutional_ratio')


@IndicatorRegistry.register('retail_ratio', category='volume')
def RetailFlowRatio(data, period=5):
    """
    游资净流入占比 (N日平均)

    公式: 散户净流入 / 成交额 (N日平均)
    游资净流入 = 中单净流入 + 小单净流入

    信号：
    - > 5%   = 游资强势买入（炒作阶段）
    - > 0%   = 游资净买入
    - < 0%   = 游资净卖出
    """
    retail_col = '散户净流入'
    amount_col = '成交额'

    if retail_col not in data.columns:
        return pd.Series(np.nan, index=data.index, name='retail_ratio')

    retail_flow = data[retail_col]

    if amount_col in data.columns:
        amount = data[amount_col].abs()
    else:
        amount = (data['close'] * data['volume']).abs()

    ratio = retail_flow / amount.replace(0, np.nan)
    result = ratio.rolling(window=period).mean()
    return pd.Series(result.values, index=data.index, name='retail_ratio')


@IndicatorRegistry.register('inst_retail_diff', category='volume')
def InstRetailDiff(data, period=5):
    """
    机构-游资资金差 (N日累计)

    公式: 主力净流入 - 散户净流入 (N日累计)
    > 0 = 机构主导（价值发现/长期上涨）
    < 0 = 游资主导（短线炒作）
    < -5亿 = 游资明显主导
    > 5亿 = 机构明显主导
    """
    inst_col = '主力净流入'
    retail_col = '散户净流入'

    if inst_col not in data.columns or retail_col not in data.columns:
        return pd.Series(np.nan, index=data.index, name='inst_retail_diff')

    inst_flow = data[inst_col].fillna(0)
    retail_flow = data[retail_col].fillna(0)

    diff = inst_flow - retail_flow
    result = diff.rolling(window=period).sum()
    return pd.Series(result.values, index=data.index, name='inst_retail_diff')


@IndicatorRegistry.register('inst_retail_ratio', category='volume')
def InstRetailRatio(data, period=5):
    """
    机构/游资比率 (N日平均)

    公式: 主力净流入 / 散户净流入 (N日平均)
    > 1.5 = 机构强烈主导（可能有长线资金）
    0.5~1.5 = 多空平衡
    < 0.5 = 游资主导（短线题材炒作）
    < 0 = 两者都在卖出（恐慌）

    注意：分母为0时返回NaN
    """
    inst_col = '主力净流入'
    retail_col = '散户净流入'

    if inst_col not in data.columns or retail_col not in data.columns:
        return pd.Series(np.nan, index=data.index, name='inst_retail_ratio')

    inst_flow = data[inst_col].fillna(0)
    retail_flow = data[retail_col].fillna(0)

    ratio = inst_flow / retail_flow.replace(0, np.nan)
    result = ratio.rolling(window=period).mean()
    return pd.Series(result.values, index=data.index, name='inst_retail_ratio')


@IndicatorRegistry.register('inst_dominant_days', category='volume')
def InstDominantDays(data, period=20):
    """
    机构主导天数占比 (N日内)

    统计最近N日中，机构资金净流入 > 游资资金净流入的天数占比
    > 60% = 机构持续主导
    < 40% = 游资持续主导
    """
    inst_col = '主力净流入'
    retail_col = '散户净流入'

    if inst_col not in data.columns or retail_col not in data.columns:
        return pd.Series(np.nan, index=data.index, name='inst_dominant_days')

    inst_flow = data[inst_col].fillna(0)
    retail_flow = data[retail_col].fillna(0)

    inst_dominant = (inst_flow > retail_flow).astype(float)
    result = inst_dominant.rolling(window=period).mean()
    return pd.Series(result.values, index=data.index, name='inst_dominant_days')