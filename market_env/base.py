#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
市场环境数据提供者接口
"""

from typing import Protocol, Optional, Dict
import pandas as pd


class IMarketEnvProvider(Protocol):
    """市场环境数据提供者接口"""

    def get_market_breadth(self, end_date: str) -> Optional[pd.DataFrame]:
        """
        获取市场宽度数据（如 A/D breadth 上涨/下跌家数）。

        Args:
            end_date: 截止日期，格式 yyyyMMdd

        Returns:
            DataFrame with columns: [date, advance, decline], or None
        """
        ...

    def get_market_valuation(self, index_code: str, date: str) -> Optional[dict]:
        """
        获取指数估值分位（PE/PB 历史百分位）。

        Args:
            index_code: 指数代码，如 '000985.SH' (中证全指)
            date: 日期，格式 yyyyMMdd

        Returns:
            {'pe_percentile': float, 'pb_percentile': float} or None
        """
        ...

    def get_macro_indicators(self, end_date: str) -> Optional[Dict[str, float]]:
        """
        获取当前宏观指标值（用于诊断面板）。

        Args:
            end_date: 截止日期，格式 yyyyMMdd

        Returns:
            {'指标名': value, ...} or None
        """
        ...

    def get_leading_indicators(self, end_date: str) -> Optional[Dict[str, float]]:
        """
        获取当前领先指标值（用于诊断面板）。

        Args:
            end_date: 截止日期，格式 yyyyMMdd

        Returns:
            {'指标名': value, ...} or None
        """
        ...