#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WindDB 行业轮动数据实现

通过 akshare 获取申万二级行业指数历史数据，计算行业动量。
"""

from typing import Optional, List
import pandas as pd

from data.wind_db import WindDBDataSource
from utils.logger import get_logger
from .base import ISectorProvider

logger = get_logger(__name__)


class WindSectorProvider:
    """
    基于 Wind Oracle + akshare 的行业动量数据提供方。

    数据来源：
    - Wind Oracle: AShareIndustriesClass（申万行业分类）
    - akshare: index_hist_sw()（申万行业指数日线）
    """

    def __init__(self, wind_db: WindDBDataSource):
        self._db = wind_db

    def get_sector_list(self, sector_type: str = 'sw') -> pd.DataFrame:
        """
        返回申万二级行业列表。

        Returns:
            DataFrame: [sector_code, sector_name]
        """
        try:
            sql = """
                SELECT DISTINCT wind_ind_code, wind_ind_name
                FROM AShareIndustriesClass
                WHERE wind_ind_code LIKE '80%'
                ORDER BY wind_ind_code
            """
            df = pd.read_sql(sql, self._db._conn)
            return df.rename(columns={'WIND_IND_CODE': 'sector_code',
                                      'WIND_IND_NAME': 'sector_name'})
        except Exception as e:
            logger.error(f"Failed to get sector list: {e}")
            return pd.DataFrame(columns=['sector_code', 'sector_name'])

    def get_sector_momentum(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        获取所有申万二级行业的动量数据。

        Returns:
            DataFrame: [sector_code, sector_name, return_1m, return_3m]
        """
        sectors = self.get_sector_list()
        if sectors.empty:
            return pd.DataFrame(columns=['sector_code', 'sector_name',
                                         'return_1m', 'return_3m'])

        results = []
        for _, row in sectors.iterrows():
            code = row['sector_code']
            name = row['sector_name']

            try:
                df = self._fetch_index(code, end_date)
            except Exception:
                continue

            if df is None or len(df) < 60:
                continue

            closes = df['close'].astype(float)
            end_price = closes.iloc[-1]
            price_1m = _price_at_date(df, start_date)
            price_3m = _price_at_date(df, _offset_date(end_date, 90))

            ret_1m = (end_price / price_1m - 1) if price_1m else 0
            ret_3m = (end_price / price_3m - 1) if price_3m else 0

            results.append({
                'sector_code': code,
                'sector_name': name,
                'return_1m': ret_1m,
                'return_3m': ret_3m,
            })

        if not results:
            return pd.DataFrame(columns=['sector_code', 'sector_name',
                                         'return_1m', 'return_3m'])
        return pd.DataFrame(results)

    def _fetch_index(self, sector_code: str, end_date: str) -> Optional[pd.DataFrame]:
        """通过 akshare 获取申万行业指数历史数据"""
        import akshare as ak
        try:
            df = ak.index_hist_sw(symbol=sector_code, period='day')
            if df is None or df.empty:
                return None
            df = df[df['日期'] <= end_date]
            return df.sort_values('日期')
        except Exception as e:
            logger.debug(f"Failed to fetch sector {sector_code}: {e}")
            return None


def _offset_date(date_str: str, days: int) -> str:
    """将 yyyyMMdd 格式往前推 days 天"""
    from datetime import datetime, timedelta
    dt = datetime.strptime(date_str, "%Y%m%d") - timedelta(days=days)
    return dt.strftime("%Y%m%d")


def _price_at_date(df: pd.DataFrame, date_str: str) -> float:
    """取最近 date_str 之前的收盘价"""
    dt = pd.to_datetime(date_str, format="%Y%m%d")
    subset = df[df['日期'] < dt]
    if subset.empty:
        return 0.0
    return float(subset.iloc[-1]['close'])