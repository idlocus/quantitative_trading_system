#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
期货数据源 — 股指期货（中金所）+ 商品期货（上期所/大商所/郑商所）

数据来源：
- Wind Oracle CINDEXFUTURESEODPRICES（中金所股指期货，2006至今）
- Wind Oracle CCOMMODITYFUTURESEODPRICES（商品期货，1995-2017，仅历史）
- akshare（商品期货近期数据，akshare.futures.futures_daily_bar）

合约命名：
- IF00.CFE / IC00.CFE / IH00.CFE / IM00.CFE：主力连续
- IF2601.CFE：具体到期月合约
- AG1508.SHF：白银金属期货
- RU1605.RU：橡胶期货（.RU可能归属不同交易所）

使用方式：
    futures_ds = FuturesDataSource()
    df = futures_ds.get_kline('IF00.CFE', date(2026,1,1), date(2026,6,5))
    df = futures_ds.get_kline('AG.SHF', date(2026,1,1), date(2026,6,5))  # 商品期货
"""

import os
import pandas as pd
from datetime import date, timedelta
from typing import Optional

import oracledb

from data.base.interfaces import IKlineProvider, IQuoteProvider
from utils.logger import get_logger

logger = get_logger(__name__)


# 交易所后缀 → 期货数据表
_EXCHANGE_TABLE_MAP = {
    '.CFE': 'CINDEXFUTURESEODPRICES',   # 中金所（股指期货）
    '.SHF': 'CCOMMODITYFUTURESEODPRICES',  # 上期所（商品期货）
    '.DCE': 'CCOMMODITYFUTURESEODPRICES',  # 大商所
    '.CZC': 'CCOMMODITYFUTURESEODPRICES',  # 郑商所
    '.INE': 'CCOMMODITYFUTURESEODPRICES',  # 上期能源（原油）
}

# 商品期货品种代码前缀（用于akshare获取近期数据）
_COMMODITY_PREFIXES = {
    'AG': '贵金属',
    'AU': '贵金属',
    'CU': '有色金属',
    'AL': '有色金属',
    'ZN': '有色金属',
    'PB': '有色金属',
    'NI': '有色金属',
    'SN': '有色金属',
    'RU': '橡胶',
    'RB': '螺纹钢',
    'HC': '热卷',
    'I': '铁矿',
    'J': '焦炭',
    'JM': '焦煤',
    'ZC': '动力煤',
    'FG': '玻璃',
    'TA': 'PTA',
    'MA': '甲醇',
    'PP': '聚丙烯',
    'PE': '聚乙烯',
    'PVC': 'PVC',
    'L': '塑料',
    'V': 'PVC',
    '豆一': '农产品',
    '豆二': '农产品',
    '豆粕': '农产品',
    '豆油': '农产品',
    '棕榈': '农产品',
    '菜粕': '农产品',
    '菜油': '农产品',
    '玉米': '农产品',
    '鸡蛋': '农产品',
    '白糖': '农产品',
    '棉花': '农产品',
    '棉纱': '农产品',
    '菜籽': '农产品',
    '苹果': '农产品',
    '红枣': '农产品',
    '国债': '金融',
}


class FuturesDataSource:
    """
    期货数据源（支持中金所股指期货 + 商品期货）

    股指期货：中金所（CFE），合约如 IF00.CFE / IC2601.CFE
    商品期货：上期所（SHF）/ 大商所（DCE）/ 郑商所（CZC）

    实现 IKlineProvider, IQuoteProvider 接口。
    """

    _ORACLE_DSN = os.getenv('ORACLE_DSN', '')
    _ORACLE_USER = os.getenv('ORACLE_USER', '')
    _ORACLE_PASSWORD = os.getenv('ORACLE_PASSWORD', '')

    def __init__(self):
        self._conn = None
        self._ak_cache = {}

    def _ensure_conn(self):
        if self._conn is None:
            oracledb.init_oracle_client()
            self._conn = oracledb.connect(
                user=self._ORACLE_USER,
                password=self._ORACLE_PASSWORD,
                dsn=self._ORACLE_DSN
            )

    def _cursor(self):
        self._ensure_conn()
        return self._conn.cursor()

    def _get_table(self, symbol: str) -> Optional[str]:
        """根据合约后缀确定数据表"""
        for suffix, table in _EXCHANGE_TABLE_MAP.items():
            if symbol.endswith(suffix):
                return table
        return None

    def _parse_symbol(self, symbol: str) -> tuple[str, str]:
        """解析合约，返回 (raw_symbol, exchange_suffix)"""
        for suffix in _EXCHANGE_TABLE_MAP.keys():
            if symbol.endswith(suffix):
                return symbol, suffix
        return symbol, ''

    def get_kline(self, symbol: str, start: date, end: date) -> pd.DataFrame:
        """
        获取期货K线数据

        Args:
            symbol: 期货合约代码，如 'IF00.CFE'、'AG.SHF'、'CU.DCE'
            start: 开始日期
            end: 结束日期

        Returns:
            DataFrame with columns: date, open, high, low, close, volume, amount, oi
        """
        _, suffix = self._parse_symbol(symbol)

        if not suffix:
            logger.warning(f"Unknown futures exchange for {symbol}")
            return pd.DataFrame()

        # 中金所股指期货直接用Wind
        if suffix == '.CFE':
            return self._get_cfe_futures(symbol, start, end)

        # 商品期货：优先Wind，Wind无数据时用akshare
        wind_df = self._get_commodity_wind(symbol, start, end)
        if wind_df is not None and not wind_df.empty:
            return wind_df

        # akshare 回填（近期商品期货）
        return self._get_commodity_akshare(symbol, start, end)

    def _get_cfe_futures(self, symbol: str, start: date, end: date) -> pd.DataFrame:
        """从中金所股指期货表获取数据"""
        try:
            cursor = self._cursor()
            sql = """
                SELECT
                    TRADE_DT                             AS dt,
                    CAST(S_DQ_OPEN    AS FLOAT)          AS o,
                    CAST(S_DQ_HIGH    AS FLOAT)          AS h,
                    CAST(S_DQ_LOW     AS FLOAT)          AS l,
                    CAST(S_DQ_CLOSE   AS FLOAT)          AS c,
                    CAST(S_DQ_VOLUME  AS FLOAT)          AS v,
                    CAST(S_DQ_AMOUNT  AS FLOAT)          AS amt,
                    CAST(S_DQ_OI      AS FLOAT)          AS oi
                FROM CINDEXFUTURESEODPRICES
                WHERE S_INFO_WINDCODE = :symbol
                  AND TRADE_DT >= :start_dt
                  AND TRADE_DT <= :end_dt
                ORDER BY TRADE_DT
            """
            cursor.execute(sql, {
                'symbol': symbol,
                'start_dt': start.strftime('%Y%m%d'),
                'end_dt': end.strftime('%Y%m%d'),
            })
            cols = [d[0].lower() for d in cursor.description]
            rows = cursor.fetchall()
            cursor.close()

            if not rows:
                return pd.DataFrame()

            df = pd.DataFrame(rows, columns=cols)
            df['date'] = pd.to_datetime(df['dt'], format='%Y%m%d')
            return df[['date', 'o', 'h', 'l', 'c', 'v', 'amt', 'oi']]

        except Exception as e:
            logger.warning(f"CFE futures query failed for {symbol}: {e}")
            return pd.DataFrame()

    def _get_commodity_wind(self, symbol: str, start: date, end: date) -> Optional[pd.DataFrame]:
        """从Wind商品期货表获取数据"""
        try:
            cursor = self._cursor()
            sql = """
                SELECT
                    TRADE_DT                             AS dt,
                    CAST(S_DQ_OPEN    AS FLOAT)          AS o,
                    CAST(S_DQ_HIGH    AS FLOAT)          AS h,
                    CAST(S_DQ_LOW     AS FLOAT)          AS l,
                    CAST(S_DQ_CLOSE   AS FLOAT)          AS c,
                    CAST(S_DQ_VOLUME  AS FLOAT)          AS v,
                    CAST(S_DQ_AMOUNT  AS FLOAT)          AS amt,
                    CAST(S_DQ_OI      AS FLOAT)          AS oi
                FROM CCOMMODITYFUTURESEODPRICES
                WHERE S_INFO_WINDCODE = :symbol
                  AND TRADE_DT >= :start_dt
                  AND TRADE_DT <= :end_dt
                ORDER BY TRADE_DT
            """
            cursor.execute(sql, {
                'symbol': symbol,
                'start_dt': start.strftime('%Y%m%d'),
                'end_dt': end.strftime('%Y%m%d'),
            })
            cols = [d[0].lower() for d in cursor.description]
            rows = cursor.fetchall()
            cursor.close()

            if not rows:
                return None  # 返回None表示无数据，触发akshare回填

            df = pd.DataFrame(rows, columns=cols)
            df['date'] = pd.to_datetime(df['dt'], format='%Y%m%d')
            return df[['date', 'o', 'h', 'l', 'c', 'v', 'amt', 'oi']]

        except Exception as e:
            logger.debug(f"Commodity futures Wind query failed for {symbol}: {e}")
            return None

    def _get_commodity_akshare(self, symbol: str, start: date, end: date) -> pd.DataFrame:
        """
        通过akshare获取商品期货近期数据

        注意：akshare的期货接口返回的是特定品种的主力连续，
        需要自行按品种名查找对应接口。
        """
        try:
            import akshare.futures.futures_hist_em as fhe

            # 去掉交易所后缀得到品种代码
            base_symbol = symbol.split('.')[0] if '.' in symbol else symbol

            # 尝试主力连续
            try:
                df = fhe.futures_hist_em(
                    symbol=base_symbol,
                    period='daily',
                    start_date=start.strftime('%Y%m%d'),
                    end_date=end.strftime('%Y%m%d')
                )
                if df is not None and not df.empty:
                    df.columns = [c.lower() for c in df.columns]
                    return df
            except Exception as e:
                logger.debug(f"akshare futures_hist_em failed for {symbol}: {e}")

            return pd.DataFrame()

        except Exception as e:
            logger.debug(f"akshare commodity fallback failed for {symbol}: {e}")
            return pd.DataFrame()

    # ---- IQuoteProvider ----

    def get_realtime(self, symbol: str) -> float:
        """Wind Oracle 不支持期货实时行情"""
        return 0.0

    def get_close(self, symbol: str, trading_date: date) -> float:
        """获取指定日期的收盘价"""
        kline = self.get_kline(symbol, trading_date, trading_date)
        if kline is not None and not kline.empty:
            return float(kline['c'].iloc[-1])
        return 0.0

    # ---- 期货特定方法 ----

    def list_futures_contracts(self, exchange: str = 'CFE') -> list[str]:
        """
        获取指定交易所的期货合约列表

        Args:
            exchange: 'CFE'(中金所) / 'SHF'(上期所) / 'DCE'(大商所) / 'CZC'(郑商所)
        """
        suffix_map = {'CFE': '.CFE', 'SHF': '.SHF', 'DCE': '.DCE', 'CZC': '.CZC'}
        suffix = suffix_map.get(exchange)
        if not suffix:
            return []

        try:
            table = _EXCHANGE_TABLE_MAP.get(suffix)
            if not table:
                return []

            cursor = self._cursor()
            sql = f"""
                SELECT DISTINCT S_INFO_WINDCODE
                FROM {table}
                WHERE S_INFO_WINDCODE LIKE '%{suffix}'
                  AND TRADE_DT >= '20200101'
                ORDER BY S_INFO_WINDCODE
            """
            cursor.execute(sql)
            rows = cursor.fetchall()
            cursor.close()
            return [r[0] for r in rows]

        except Exception as e:
            logger.warning(f"list_futures_contracts failed for {exchange}: {e}")
            return []

    def get_main_contract(self, product: str, trade_date: date) -> Optional[str]:
        """
        获取某品种主力合约

        Args:
            product: 品种代码，如 'IF' / 'IC' / 'AG' / 'CU'
            trade_date: 交易日期

        Returns:
            主力合约代码，如 'IF00.CFE' 或 None
        """
        dt_str = trade_date.strftime('%Y%m%d')
        suffix_map = {
            'IF': '.CFE', 'IC': '.CFE', 'IH': '.CFE', 'IM': '.CFE',
            'AG': '.SHF', 'AU': '.SHF', 'CU': '.SHF', 'AL': '.SHF',
        }
        suffix = suffix_map.get(product)
        if not suffix:
            return None

        try:
            table = _EXCHANGE_TABLE_MAP.get(suffix)
            cursor = self._cursor()
            sql = f"""
                SELECT S_INFO_WINDCODE
                FROM {table}
                WHERE S_INFO_WINDCODE LIKE :pattern
                  AND TRADE_DT = :dt
                ORDER BY S_DQ_VOLUME DESC
            """
            cursor.execute(sql, {
                'pattern': f'{product}%{suffix}',
                'dt': dt_str
            })
            row = cursor.fetchone()
            cursor.close()
            return row[0] if row else None

        except Exception as e:
            logger.debug(f"get_main_contract failed for {product}: {e}")
            return None

    def get_continuous_futures(self, product: str, start: date, end: date) -> pd.DataFrame:
        """
        获取期货主力连续合约数据（自动拼接换月）

        通过持仓量判断每个月的主力合约，拼接成连续数据。
        适用于技术分析。

        Args:
            product: 品种代码，如 'IF' / 'CU' / 'AG'
            start: 开始日期
            end: 结束日期

        Returns:
            DataFrame with date, open, high, low, close, volume, amount, oi
        """
        suffix_map = {
            'IF': '.CFE', 'IC': '.CFE', 'IH': '.CFE', 'IM': '.CFE',
            'AG': '.SHF', 'AU': '.SHF', 'CU': '.SHF', 'AL': '.SHF',
            'ZN': '.SHF', 'PB': '.SHF', 'NI': '.SHF', 'SN': '.SHF',
            'RU': '.SHF', 'RB': '.SHF', 'HC': '.SHF', 'I': '.SHF',
            'J': '.DCE', 'JM': '.DCE', 'FG': '.CZC', 'TA': '.CZC',
            'MA': '.CZC', 'PP': '.DCE', 'PE': '.DCE', 'PVC': '.PVC',
        }
        suffix = suffix_map.get(product)
        if not suffix:
            logger.warning(f"Unknown product: {product}")
            return pd.DataFrame()

        table = _EXCHANGE_TABLE_MAP.get(suffix)
        if not table:
            return pd.DataFrame()

        try:
            cursor = self._cursor()
            sql = f"""
                SELECT
                    TRADE_DT                             AS dt,
                    S_INFO_WINDCODE                      AS sym,
                    CAST(S_DQ_OPEN    AS FLOAT)          AS o,
                    CAST(S_DQ_HIGH    AS FLOAT)          AS h,
                    CAST(S_DQ_LOW     AS FLOAT)          AS l,
                    CAST(S_DQ_CLOSE   AS FLOAT)          AS c,
                    CAST(S_DQ_VOLUME  AS FLOAT)          AS v,
                    CAST(S_DQ_AMOUNT  AS FLOAT)          AS amt,
                    CAST(S_DQ_OI      AS FLOAT)          AS oi
                FROM {table}
                WHERE S_INFO_WINDCODE LIKE :pattern
                  AND TRADE_DT >= :start_dt
                  AND TRADE_DT <= :end_dt
                ORDER BY TRADE_DT, S_DQ_OI DESC
            """
            cursor.execute(sql, {
                'pattern': f'{product}%{suffix}',
                'start_dt': start.strftime('%Y%m%d'),
                'end_dt': end.strftime('%Y%m%d'),
            })
            cols = [d[0].lower() for d in cursor.description]
            rows = cursor.fetchall()
            cursor.close()

            if not rows:
                return pd.DataFrame()

            raw_df = pd.DataFrame(rows, columns=cols)
            raw_df['date'] = pd.to_datetime(raw_df['dt'], format='%Y%m%d')

            # 按日选取持仓量最大的合约（主力合约）
            idx = raw_df.groupby('date')['oi'].idxmax()
            main_df = raw_df.loc[idx].copy()
            main_df = main_df.sort_values('date')

            return main_df[['date', 'o', 'h', 'l', 'c', 'v', 'amt', 'oi']]

        except Exception as e:
            logger.warning(f"get_continuous_futures failed for {product}: {e}")
            return pd.DataFrame()

    def close(self):
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def create_futures_data_source() -> FuturesDataSource:
    """工厂方法：创建期货数据源"""
    return FuturesDataSource()