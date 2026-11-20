#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sector - 行业轮动模块

提供行业动量数据、排名及配置权重。
"""

from .base import ISectorProvider
from .rotator import SectorAllocation, SectorRotator

__all__ = [
    'ISectorProvider',
    'SectorAllocation',
    'SectorRotator',
]