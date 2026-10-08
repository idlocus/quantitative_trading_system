#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
market_env - 宏观市场环境判断

提供市场宽度、A/D breadth、估值分位、宏观指标、领先指标等五大维度评分。
"""

from .base import IMarketEnvProvider
from .scorer import MarketEnvScore, MarketEnvScorer
from .macro_db import MacroDB, INDICATORS, CAT_LABELS
from .akshare_provider import AkShareMarketEnvProvider

__all__ = [
    'IMarketEnvProvider',
    'MarketEnvScore',
    'MarketEnvScorer',
    'MacroDB',
    'INDICATORS',
    'CAT_LABELS',
    'AkShareMarketEnvProvider',
]