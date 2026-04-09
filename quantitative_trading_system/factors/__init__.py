#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
因子模块
提供基于技术指标的高级因子封装

因子是对指标的进一步封装和组合，用于生成综合评分和信号
"""

from .base import Factor, FactorResult, FactorRegistry
from .registry import FactorRegistry, factor_registry

# 导入所有因子（触发注册）
from .technical import (
    TrendFactor,
    MomentumFactor,
    VolatilityFactor,
    CompositeFactor
)

# 基本面因子
from .valuation import ValuationFactor
from .growth import GrowthFactor
from .profitability import ProfitabilityFactor

__all__ = [
    'Factor',
    'FactorResult',
    'FactorRegistry',
    'factor_registry',
    # 技术因子
    'TrendFactor',
    'MomentumFactor',
    'VolatilityFactor',
    'CompositeFactor',
    # 基本面因子
    'ValuationFactor',
    'GrowthFactor',
    'ProfitabilityFactor',
]
