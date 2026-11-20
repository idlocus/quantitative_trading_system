#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
行业数据提供者接口
"""

from typing import Protocol
import pandas as pd


class ISectorProvider(Protocol):
    """行业数据提供者接口"""

    def get_sector_momentum(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        获取行业动量数据（指定日期区间的行业收益率）。

        Args:
            start_date: 开始日期，格式 yyyyMMdd
            end_date: 结束日期，格式 yyyyMMdd

        Returns:
            DataFrame with columns: [sector_code, sector_name, return_1m, return_3m]
            return_1m: 近 1 月收益率
            return_3m: 近 3 月收益率
        """
        ...

    def get_sector_list(self, sector_type: str = 'sw') -> pd.DataFrame:
        """
        获取行业分类列表。

        Args:
            sector_type: 分类体系，'sw' 为申万行业

        Returns:
            DataFrame with columns: [sector_code, sector_name]
        """
        ...