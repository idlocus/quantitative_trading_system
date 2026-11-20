#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
data.base - 数据源抽象层

导出：
- 四个数据接口 (Protocol)
- DataLoader: 数据加载器
- CachedDataSource: 缓存封装
"""

from .interfaces import (
    IKlineProvider,
    IQuoteProvider,
    ISymbolProvider,
    IIndustryProvider,
)
from .data_loader import DataLoader, CachedDataSource

__all__ = [
    # 接口
    "IKlineProvider",
    "IQuoteProvider",
    "ISymbolProvider",
    "IIndustryProvider",
    # 加载器
    "DataLoader",
    "CachedDataSource",
]