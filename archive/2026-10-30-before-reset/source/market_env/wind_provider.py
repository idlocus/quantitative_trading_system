#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WindDB 市场环境数据实现
"""

from typing import Optional
import pandas as pd

from data.wind_db import WindDBDataSource
from utils.logger import get_logger
from .base import IMarketEnvProvider

logger = get_logger(__name__)


class WindMarketEnvProvider:
    """
    基于 Wind Oracle DB 的市场环境数据提供方。

    - 市场宽度：通过每日涨跌家数统计计算 A/D ratio
    - 估值分位：通过 Wind Oracle AShareIndexValuation 表获取 PE/PB 历史分位
    - 宏观指标：通过 Wind MCP economic_data 查询 EDB
    """

    def __init__(self, wind_db: WindDBDataSource, mcp_client=None):
        self._db = wind_db
        self._mcp = mcp_client  # optional WindMCPClient for macro EDB

    def get_market_breadth(self, end_date: str) -> Optional[pd.DataFrame]:
        """
        获取市场宽度数据（近 N 日的每日上涨/下跌家数）。

        Returns:
            DataFrame with columns: [date, advance, decline], sorted by date asc.
        """
        try:
            sql = """
                SELECT
                    TRADE_DT                      AS date,
                    SUM(CASE WHEN S_DQ_CLOSE > S_DQ_PRECLOSE THEN 1 ELSE 0 END) AS advance,
                    SUM(CASE WHEN S_DQ_CLOSE < S_DQ_PRECLOSE THEN 1 ELSE 0 END) AS decline
                FROM AShareEODPrices
                WHERE TRADE_DT >= :start_dt
                  AND TRADE_DT <= :end_dt
                  AND (WIND_CODE LIKE '%.SH' OR WIND_CODE LIKE '%.SZ')
                GROUP BY TRADE_DT
                ORDER BY TRADE_DT
            """
            start_dt = _offset_date(end_date, 30)
            df = pd.read_sql(
                sql, self._db._conn,
                params={"start_dt": start_dt, "end_dt": end_date}
            )
            if df.empty:
                return None
            df['date'] = pd.to_datetime(df['date'], format="%Y%m%d")
            return df[['date', 'advance', 'decline']].sort_values('date')
        except Exception as e:
            logger.warning(f"Failed to get market breadth: {e}")
            return None

    def get_market_valuation(self, index_code: str, date: str) -> Optional[dict]:
        """
        获取指数估值分位。

        从 AShareIndexValuation 查询 PE_TTM 和 PB_MRQ 在历史中的百分位。
        """
        try:
            # 最新 PE/PB
            sql = """
                SELECT S_VAL_PE_TTM, S_VAL_PB_MRQ
                FROM AShareIndexValuation
                WHERE S_INFO_WINDCODE = :index_code
                  AND TRADE_DT = :date
                FETCH FIRST 1 ROWS ONLY
            """
            cursor = self._db._conn.cursor()
            cursor.execute(sql, {"index_code": index_code, "date": date})
            row = cursor.fetchone()
            cursor.close()

            if row is None:
                return None

            pe_ttm, pb_mrq = row
            if pe_ttm is None or pb_mrq is None:
                return None

            # 历史分位：用过去 5 年数据估算
            hist_sql = """
                SELECT
                    S_VAL_PE_TTM,
                    S_VAL_PB_MRQ
                FROM AShareIndexValuation
                WHERE S_INFO_WINDCODE = :index_code
                  AND TRADE_DT >= :start_dt
                  AND TRADE_DT <= :date
                  AND S_VAL_PE_TTM IS NOT NULL
                  AND S_VAL_PB_MRQ IS NOT NULL
                ORDER BY TRADE_DT
            """
            hist_df = pd.read_sql(
                hist_sql, self._db._conn,
                params={"index_code": index_code,
                        "start_dt": _offset_date(date, 365 * 5),
                        "date": date}
            )

            if hist_df.empty or len(hist_df) < 60:
                return {'pe_percentile': 50, 'pb_percentile': 50}

            pe_percentile = (hist_df['S_VAL_PE_TTM'] <= pe_ttm).mean() * 100
            pb_percentile = (hist_df['S_VAL_PB_MRQ'] <= pb_mrq).mean() * 100

            return {
                'pe_percentile': float(pe_percentile),
                'pb_percentile': float(pb_percentile),
            }
        except Exception as e:
            logger.warning(f"Failed to get market valuation for {index_code} on {date}: {e}")
            return None

    def get_macro_indicators(self, end_date: str):
        """通过 Wind MCP economic_data 查询宏观指标当前值"""
        if self._mcp is None:
            return None

        from .macro_db import MacroDB
        try:
            db = MacroDB(self._mcp)
            return db.get_current_values(end=end_date)
        except Exception as e:
            logger.warning(f"Failed to get macro indicators: {e}")
            return None

    def get_leading_indicators(self, end_date: str):
        """通过 Wind MCP economic_data 查询领先指标当前值"""
        if self._mcp is None:
            return None

        from .macro_db import INDICATORS, MacroDB
        try:
            db = MacroDB(self._mcp)
            result = {}
            for ind in INDICATORS:
                if ind["lead"] > 0:
                    s = db.fetch(ind["id"], end=end_date, freq=ind["freq"])
                    if len(s) > 0:
                        result[ind["id"]] = float(s.iloc[-1])
            return result
        except Exception as e:
            logger.warning(f"Failed to get leading indicators: {e}")
            return None


def _offset_date(date_str: str, days: int) -> str:
    """将 yyyyMMdd 格式日期往前推 days 天，返回新的 yyyyMMdd"""
    from datetime import datetime, timedelta
    dt = datetime.strptime(date_str, "%Y%m%d") - timedelta(days=days)
    return dt.strftime("%Y%m%d")