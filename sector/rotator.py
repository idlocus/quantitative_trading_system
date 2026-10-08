#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
行业轮动器

基于行业动量数据，动态配置排名前列的行业。
"""

from dataclasses import dataclass
from typing import List, Optional
import pandas as pd

from utils.logger import get_logger
from .base import ISectorProvider

logger = get_logger(__name__)


@dataclass
class SectorAllocation:
    """行业配置"""
    sector_code: str
    sector_name: str
    weight: float          # 0.0-1.0
    momentum_rank: int     # 动量排名（1 为最强）


class SectorRotator:
    """
    行业轮动器

    根据动量排名动态选择配置行业。
    """

    def __init__(self, provider: ISectorProvider, top_n: int = 5):
        self.provider = provider
        self.top_n = top_n

    def get_top_sectors(self, date: str) -> List[SectorAllocation]:
        """
        返回指定日期评分时点，动量最强的 top_n 行业。

        Args:
            date: 评估日期，格式 yyyyMMdd

        Returns:
            List[SectorAllocation]，按 momentum_rank 升序
        """
        end = date
        start = _offset_date(date, 90)

        try:
            df = self.provider.get_sector_momentum(start, end)
        except Exception as e:
            logger.warning(f"Failed to get sector momentum: {e}")
            return []

        if df is None or df.empty:
            return []

        # 按近 1 月动量排序
        df = df.sort_values('return_1m', ascending=False).head(self.top_n)

        allocations = []
        total_return = df['return_1m'].sum()

        for rank, (_, row) in enumerate(df.iterrows(), start=1):
            # 权重按动量相对强度分配
            weight = float(row['return_1m'] / total_return) if total_return != 0 else 1.0 / len(df)
            allocations.append(SectorAllocation(
                sector_code=row['sector_code'],
                sector_name=row['sector_name'],
                weight=weight,
                momentum_rank=rank,
            ))

        return allocations


def _offset_date(date_str: str, days: int) -> str:
    """将 yyyyMMdd 格式日期往前推 days 天"""
    from datetime import datetime, timedelta
    dt = datetime.strptime(date_str, "%Y%m%d") - timedelta(days=days)
    return dt.strftime("%Y%m%d")