#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据模块

数据源实现：
- WindDataSource:    IKline, IQuote, ISymbol (Wind Terminal API)
- AKShareDataSource: IKline, IQuote, ISymbol (akshare HTTP)
- WindDBDataSource:  IKline, IQuote, ISymbol, IIndustry (Wind Oracle DB)

使用 data.base 导入抽象层：
    from data.base import DataLoader, CachedDataSource, IKlineProvider, ...
"""

from data.base import (
    IKlineProvider,
    IQuoteProvider,
    ISymbolProvider,
    IIndustryProvider,
    DataLoader,
    CachedDataSource,
)
from data.wind import WindDataSource
from data.akshare import AKShareDataSource
from data.wind_db import WindDBDataSource

# 向后兼容别名
from data.wind_db import WindDBDataSource as IndustryDataSource

__all__ = [
    # 接口
    "IKlineProvider",
    "IQuoteProvider",
    "ISymbolProvider",
    "IIndustryProvider",
    # 加载器
    "DataLoader",
    "CachedDataSource",
    # 实现
    "WindDataSource",
    "AKShareDataSource",
    "WindDBDataSource",
    "IndustryDataSource",
]