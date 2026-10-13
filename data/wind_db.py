#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Wind Oracle 数据库数据源

从 Wind Oracle 数据库读取所有数据：
- AShareDescription: 股票列表
- AShareEODPrices:   K线行情
- AShareIndustriesClass: 申万行业分类

实现 IKlineProvider, IQuoteProvider, ISymbolProvider, IIndustryProvider 四个接口。
行业指数历史数据通过 akshare 获取。
"""

import pandas as pd
from datetime import date
from typing import List, Optional

import oracledb
import akshare as ak

from data.base.interfaces import (
    IKlineProvider, IQuoteProvider, ISymbolProvider, IIndustryProvider
)
from utils.logger import get_logger

logger = get_logger(__name__)


class WindDBDataSource:
    """
    基于 Wind Oracle 数据库的统一数据源。

    从三张表读取数据：
    - AShareDescription:     股票列表
    - AShareEODPrices:      K线/日线行情
    - AShareIndustriesClass: 申万行业分类

    行业指数历史数据通过 akshare.index_hist_sw() 获取。
    """

    def __init__(self, oracle_dsn: str, user: str, password: str):
        oracledb.init_oracle_client()  # 必须用 thick mode，thin mode 有参数绑定 bug
        self._conn = oracledb.connect(user=user, password=password, dsn=oracle_dsn)

    def _cursor(self):
        """获取游标（每次请求创建新游标）"""
        return self._conn.cursor()

    # ---- ISymbolProvider ----

    def list_symbols(self, market: str = "A股") -> List[str]:
        """获取股票列表（仅主板，排除退市）"""
        try:
            sql = """
                SELECT S_INFO_WINDCODE
                FROM AShareDescription
                WHERE S_INFO_DELISTDATE IS NULL
                  AND (S_INFO_WINDCODE LIKE '%.SH' OR S_INFO_WINDCODE LIKE '%.SZ')
                ORDER BY S_INFO_WINDCODE
            """
            df = pd.read_sql(sql, self._conn)
            return df['S_INFO_WINDCODE'].tolist()
        except Exception as e:
            logger.error(f"Failed to list symbols: {e}")
            return []

    def get_tradable_symbols(self, trade_date: date) -> List[str]:
        """获取指定日期可交易的股票列表"""
        date_str = trade_date.strftime("%Y%m%d")
        try:
            sql = """
                SELECT DISTINCT S_INFO_WINDCODE
                FROM AShareEODPrices
                WHERE TRADE_DT = :trade_date
            """
            df = pd.read_sql(sql, self._conn, params={"trade_date": date_str})
            return df['S_INFO_WINDCODE'].tolist()
        except Exception as e:
            logger.warning(f"Failed to get tradable symbols for {date_str}: {e}")
            return self.list_symbols()

    # ---- IKlineProvider ----

    def get_kline(self, symbol: str, start: date, end: date) -> pd.DataFrame:
        """获取K线数据（复权），columns: date, open, high, low, close, volume"""
        # 先尝试 A 股表，再尝试港股表
        df = self._get_ashare_kline(symbol, start, end)
        if df is not None and not df.empty:
            return df
        # 港股 symbol 以 .HK 结尾，尝试 HKSHAREEODPRICES
        if symbol.endswith('.HK'):
            return self._get_hkshare_kline(symbol, start, end)
        return pd.DataFrame()

    def _get_ashare_kline(self, symbol: str, start: date, end: date) -> pd.DataFrame:
        """从 ASHAREEODPRICES 获取 K线（单位归一化与 LayeredDataSource 同口径：
        volume×100（手→股）、amount×1000（千元→元），见 trading/config 单位说明）"""
        try:
            sql = """
                SELECT
                    TRADE_DT                              AS dt,
                    CAST(S_DQ_ADJOPEN  AS FLOAT)          AS o,
                    CAST(S_DQ_ADJHIGH  AS FLOAT)          AS h,
                    CAST(S_DQ_ADJLOW   AS FLOAT)          AS l,
                    CAST(S_DQ_ADJCLOSE AS FLOAT)          AS c,
                    CAST(S_DQ_VOLUME   AS FLOAT) * 100    AS v,
                    CAST(S_DQ_AMOUNT   AS FLOAT) * 1000   AS amt
                FROM AShareEODPrices
                WHERE S_INFO_WINDCODE = :symbol
                  AND TRADE_DT >= :start_dt
                  AND TRADE_DT <= :end_dt
                ORDER BY TRADE_DT
            """
            cursor = self._cursor()
            cursor.execute(sql, {
                "symbol": symbol,
                "start_dt": start.strftime("%Y%m%d"),
                "end_dt": end.strftime("%Y%m%d"),
            })
            cols = [d[0].lower() for d in cursor.description]
            rows = cursor.fetchall()
            cursor.close()
            if not rows:
                return pd.DataFrame()
            df = pd.DataFrame(rows, columns=cols)
            df.rename(columns={'dt': 'date', 'o': 'open', 'h': 'high',
                               'l': 'low', 'c': 'close', 'v': 'volume',
                               'amt': 'amount'}, inplace=True)
            df['date'] = pd.to_datetime(df['date'], format="%Y%m%d")
            return df[["date", "open", "high", "low", "close", "volume", "amount"]]
        except Exception as e:
            logger.debug(f"ASHAREEODPRICES no data for {symbol}: {e}")
            return pd.DataFrame()

    def _get_hkshare_kline(self, symbol: str, start: date, end: date) -> pd.DataFrame:
        """从 HKSHAREEODPRICES 获取 K线（复权）。

        单位说明（2026-10-12）：港股不在主链（price_limit 返回 None、无流动性过滤），
        S_DQ_VOLUME/S_DQ_AMOUNT 的原表单位未验证——amount 原样直通不做换算，
        列结构与 A 股路径对齐（含 amount 列）。若未来接入港股主链，必须先实证
        单位并在 data/turnover.py 统一口径，禁止直接进入成交额计算。
        """
        try:
            sql = """
                SELECT
                    TRADE_DT                              AS dt,
                    CAST(S_DQ_ADJOPEN  AS FLOAT)          AS o,
                    CAST(S_DQ_ADJHIGH  AS FLOAT)          AS h,
                    CAST(S_DQ_ADJLOW   AS FLOAT)          AS l,
                    CAST(S_DQ_ADJCLOSE AS FLOAT)          AS c,
                    CAST(S_DQ_VOLUME   AS FLOAT)          AS v,
                    CAST(S_DQ_AMOUNT   AS FLOAT)          AS amt
                FROM HKSHAREEODPRICES
                WHERE S_INFO_WINDCODE = :symbol
                  AND TRADE_DT >= :start_dt
                  AND TRADE_DT <= :end_dt
                ORDER BY TRADE_DT
            """
            cursor = self._cursor()
            cursor.execute(sql, {
                "symbol": symbol,
                "start_dt": start.strftime("%Y%m%d"),
                "end_dt": end.strftime("%Y%m%d"),
            })
            cols = [d[0].lower() for d in cursor.description]
            rows = cursor.fetchall()
            cursor.close()
            if not rows:
                return pd.DataFrame()
            df = pd.DataFrame(rows, columns=cols)
            df.rename(columns={'dt': 'date', 'o': 'open', 'h': 'high',
                               'l': 'low', 'c': 'close', 'v': 'volume',
                               'amt': 'amount'}, inplace=True)
            df['date'] = pd.to_datetime(df['date'], format="%Y%m%d")
            return df[["date", "open", "high", "low", "close", "volume", "amount"]]
        except Exception as e:
            logger.error(f"Failed to fetch HK kline for {symbol}: {e}")
            return pd.DataFrame()

    # ---- IQuoteProvider ----

    def get_realtime(self, symbol: str) -> float:
        """Wind Oracle 不支持实时行情"""
        return 0.0

    def get_close(self, symbol: str, trading_date: date) -> float:
        """获取指定日期的收盘价"""
        kline = self.get_kline(symbol, trading_date, trading_date)
        if kline is not None and not kline.empty:
            return float(kline['close'].iloc[-1])
        return 0.0

    # ---- IIndustryProvider ----

    def get_sw_industry_constituents(self) -> List[dict]:
        """
        从 Oracle 查询申万二级行业成分股

        Returns:
            [{industry_code, industry_name, symbols}, ...]
        """
        try:
            sql = """
                SELECT s_info_windcode, wind_ind_code, wind_ind_name
                FROM AShareIndustriesClass
                WHERE wind_ind_code LIKE '80%'
            """
            cursor = self._cursor()
            cursor.execute(sql)
            rows = cursor.fetchall()
            cursor.close()

            industry_map = {}
            for wind_code, ind_code, ind_name in rows:
                if ind_code not in industry_map:
                    industry_map[ind_code] = {
                        'industry_code': ind_code,
                        'industry_name': ind_name,
                        'symbols': []
                    }
                industry_map[ind_code]['symbols'].append(wind_code)

            return list(industry_map.values())
        except Exception as e:
            logger.error(f"Failed to get industry constituents: {e}")
            return []

    def get_industry_momentum(self, industry_code: str) -> Optional[dict]:
        """
        获取板块动量指标（通过 akshare 获取申万二级行业指数历史数据）

        Returns:
            {return_5d, ma5, ma20, ma5_above_ma20, volume_ratio} or None
        """
        df = self._fetch_industry_index(industry_code)
        if df is None or len(df) < 25:
            logger.warning(f"Insufficient index data for {industry_code}")
            return None

        closes = df['close'].astype(float)
        volumes = df['volume'].astype(float) if 'volume' in df.columns else pd.Series([0] * len(df))

        return_5d = (closes.iloc[-1] / closes.iloc[-6] - 1) if len(closes) >= 6 else 0
        ma5 = closes.iloc[-5:].mean()
        ma20 = closes.iloc[-20:].mean() if len(closes) >= 20 else closes.mean()
        vol_ma20 = volumes.iloc[-20:].mean() if len(volumes) >= 20 else volumes.mean()

        return {
            'return_5d': return_5d,
            'ma5': ma5,
            'ma20': ma20,
            'ma5_above_ma20': ma5 > ma20,
            'volume_ratio': volumes.iloc[-1] / vol_ma20 if vol_ma20 > 0 else 0,
        }

    def _fetch_industry_index(self, industry_code: str) -> Optional[pd.DataFrame]:
        """通过 akshare 获取申万二级行业指数历史数据"""
        try:
            return ak.index_hist_sw(symbol=industry_code, period='day')
        except Exception as e:
            logger.error(f"Failed to fetch industry index for {industry_code}: {e}")
            return None

    # ---- FundFlow (资金流向) ----

    def get_fund_flow(self, symbol: str, start: date, end: date) -> pd.DataFrame:
        """
        获取个股资金流向（从 Wind Oracle AShareMoneyFlow）

        Returns DataFrame with columns:
        date, 主力净流入, 散户净流入 (按 date 排序)
        """
        try:
            sql = """
                SELECT
                    TRADE_DT                          AS dt,
                    VOLUME_DIFF_INSTITUTE + VOLUME_DIFF_LARGE_TRADER AS 主力净流入,
                    VOLUME_DIFF_SMALL_TRADER            AS 散户净流入,
                    S_MFD_INFLOW                       AS 净流入,
                    NET_INFLOW_RATE_VALUE             AS 净流入率
                FROM AShareMoneyFlow
                WHERE S_INFO_WINDCODE = :symbol
                  AND TRADE_DT >= :start_dt
                  AND TRADE_DT <= :end_dt
                ORDER BY TRADE_DT
            """
            cursor = self._cursor()
            cursor.execute(sql, {
                "symbol": symbol,
                "start_dt": start.strftime("%Y%m%d"),
                "end_dt": end.strftime("%Y%m%d"),
            })
            cols = [d[0].lower() for d in cursor.description]
            rows = cursor.fetchall()
            cursor.close()

            if not rows:
                return pd.DataFrame()

            df = pd.DataFrame(rows, columns=cols)
            df['date'] = pd.to_datetime(df['dt'], format="%Y%m%d")
            return df

        except Exception as e:
            logger.warning(f"get_fund_flow failed for {symbol}: {e}")
            return pd.DataFrame()

    # ---- 连接管理 ----

    def close(self):
        """关闭数据库连接"""
        self._conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
