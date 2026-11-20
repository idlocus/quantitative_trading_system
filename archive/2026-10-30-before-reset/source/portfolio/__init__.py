#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
组合模块：因子合成选股 + TopN 组合回测（向量化，全市场规模）
"""

from .construction import (composite_score, composite_from_matrices, select_portfolio,
                           equal_weights, inverse_vol_weights)
from .backtester import PortfolioBacktester, PortfolioConfig, PortfolioResult

__all__ = [
    'composite_score', 'composite_from_matrices', 'select_portfolio',
    'equal_weights', 'inverse_vol_weights',
    'PortfolioBacktester', 'PortfolioConfig', 'PortfolioResult',
]
