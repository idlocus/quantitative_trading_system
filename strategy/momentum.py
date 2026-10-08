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

    def __init__(self, config_or_name, data_source):
        """
        初始化动量策略

        Args:
            config_or_name: 策略配置名称(str)或SelectionConfig对象
            data_source: 数据源，需提供 get_kline(symbol) 方法返回 OHLCV DataFrame
        """
        if isinstance(config_or_name, str):
            yaml_path = os.path.join(
                os.path.dirname(__file__),
                '..',
                'selection',
                'strategies.yml'
            )
            self.config = load_strategy_from_yaml(config_or_name, yaml_path)
        else:
            self.config = config_or_name
        self.data_source = data_source
        self.scanner = StockScanner(data_source)

    def select(self, date, data, allowed_symbols: set = None,
              constraint=None) -> List[ScoredStock]:
        """
        执行选股

        Args:
            date: 选股日期
            data: 数据访问对象，需提供 get_tradable_symbols(date) 方法
            allowed_symbols: 可选的股票符号限制集合
            constraint: 可选的 ScanConstraint 约束

        Returns:
            按评分降序排列的股票列表
        """
        symbols = data.get_tradable_symbols(date)
        return self.scanner.scan(symbols, self.config, ref_date=date,
                                allowed_symbols=allowed_symbols,
                                constraint=constraint)