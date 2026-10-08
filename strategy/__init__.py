#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
策略框架
提供选股策略基类和实现
"""

from abc import ABC, abstractmethod
from typing import List

from selection.config import SelectionConfig, ScoredStock, ScanConstraint
from selection.scanner import StockScanner, load_strategy_from_yaml


class Strategy(ABC):
    """
    策略抽象基类

    定义选股策略的基本接口，用于每日选股
    """

    @abstractmethod
    def select(self, date, data, allowed_symbols=None, constraint=None) -> List[ScoredStock]:
        """
        每日选股

        Args:
            date: 选股日期
            data: 数据访问对象，需提供 get_tradable_symbols(date) 方法
            allowed_symbols: 可选的股票符号限制集合
            constraint: 可选的 ScanConstraint 约束

        Returns:
            按评分降序排列的股票列表
        """
        pass


from .momentum import MomentumStrategy
from .adaptive import AdaptiveStrategy
from .multi import MultiStrategy