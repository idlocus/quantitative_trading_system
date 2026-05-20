#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
策略框架
提供选股策略基类和实现
"""

from abc import ABC, abstractmethod
from typing import List

from selection.config import SelectionConfig, ScoredStock
from selection.scanner import StockScanner, load_strategy_from_yaml


class Strategy(ABC):
    """
    策略抽象基类

    定义选股策略的基本接口，用于每日选股
    """

    @abstractmethod
    def select(self, date, data) -> List[ScoredStock]:
        """
        每日选股

        Args:
            date: 选股日期
            data: 数据访问对象，需提供 get_tradable_symbols(date) 方法

        Returns:
            按评分降序排列的股票列表
        """
        pass


from .momentum import MomentumStrategy