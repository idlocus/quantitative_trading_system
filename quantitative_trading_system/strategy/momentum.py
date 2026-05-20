#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
动量策略实现
基于技术指标动量信号的选股策略
"""

from typing import List
import os

from selection.config import SelectionConfig, ScoredStock
from selection.scanner import StockScanner, load_strategy_from_yaml
from . import Strategy


class MomentumStrategy(Strategy):
    """
    动量策略

    基于技术指标动量信号进行选股
    """

    def __init__(self, config_name: str, data_source):
        """
        初始化动量策略

        Args:
            config_name: 策略配置名称
            data_source: 数据源，需提供 get_kline(symbol) 方法返回 OHLCV DataFrame
        """
        yaml_path = os.path.join(
            os.path.dirname(__file__),
            'config',
            'strategies.yaml'
        )
        self.config = load_strategy_from_yaml(config_name, yaml_path)
        self.data_source = data_source
        self.scanner = StockScanner(data_source)

    def select(self, date, data) -> List[ScoredStock]:
        """
        执行选股

        Args:
            date: 选股日期
            data: 数据访问对象，需提供 get_tradable_symbols(date) 方法

        Returns:
            按评分降序排列的股票列表
        """
        symbols = data.get_tradable_symbols(date)
        return self.scanner.scan(symbols, self.config)