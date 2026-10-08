#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AKShare 数据源实现

提供 K线、实时行情、股票列表
同时实现 IKlineProvider, IQuoteProvider, ISymbolProvider

K线获取策略：东财接口优先，失败时自动回退新浪接口（网络屏蔽某一数据源时仍可用）。
"""

import akshare as ak
import pandas as pd
from datetime import date
from typing import List

from data.base.interfaces import IKlineProvider, IQuoteProvider, ISymbolProvider
from utils.logger import get_logger

logger = get_logger(__name__)

# 常用指数代码（get_kline 自动路由到指数接口，供基准对比/市场状态计算用）
INDEX_SYMBOLS = {
    "000001.SH",  # 上证指数
    "000016.SH",  # 上证50
    "000300.SH",  # 沪深300
    "000905.SH",  # 中证500
    "000985.SH",  # 中证全指
    "399001.SZ",  # 深证成指
    "399006.SZ",  # 创业板指
}


class AKShareDataSource:
    """
    基于 AKShare 的行情数据源。

    实现 IKlineProvider, IQuoteProvider, ISymbolProvider 三个接口。
    指数代码（如 000300.SH）自动路由到指数接口。
    """

    # ---- IKlineProvider ----

    def get_kline(self, symbol: str, start: date, end: date) -> pd.DataFrame:
        """获取K线数据（指数走新浪指数接口；个股东财优先、新浪回退）"""
        if symbol in INDEX_SYMBOLS:
            return self._kline_index(symbol, start, end)
        df = self._kline_east(symbol, start, end)
        if df.empty:
            df = self._kline_sina(symbol, start, end)
        return df

    def _kline_index(self, symbol: str, start: date, end: date) -> pd.DataFrame:
        """指数日线（新浪接口）"""
        try:
            df = ak.stock_zh_index_daily(symbol=self._to_sina_symbol(symbol))
            if df is None or df.empty:
                return pd.DataFrame()
            # 新浪返回 date 列为 datetime.date 对象，先统一转 Timestamp 再过滤
            df = df.copy()
            df['date'] = pd.to_datetime(df['date'])
            mask = (df['date'] >= pd.Timestamp(start)) & (df['date'] <= pd.Timestamp(end))
            return df.loc[mask, ["date", "open", "high", "low", "close", "volume"]].reset_index(drop=True)
        except Exception as e:
            logger.warning(f"Index kline failed for {symbol}: {e}")
            return pd.DataFrame()

    def _kline_east(self, symbol: str, start: date, end: date) -> pd.DataFrame:
        """东财接口（前复权）"""
        try:
            df = ak.stock_zh_a_hist(
                symbol=symbol.replace(".SH", "").replace(".SZ", "").replace(".BJ", ""),
                start_date=start.strftime("%Y%m%d"),
                end_date=end.strftime("%Y%m%d"),
                adjust="qfq"
            )
            if df is None or df.empty:
                return pd.DataFrame()

            df = df.rename(columns={
                "日期": "date",
                "开盘": "open",
                "收盘": "close",
                "最高": "high",
                "最低": "low",
                "成交量": "volume"
            })
            return df[["date", "open", "high", "low", "close", "volume"]]
        except Exception as e:
            logger.debug(f"East kline failed for {symbol}: {e}")
            return pd.DataFrame()

    def _kline_sina(self, symbol: str, start: date, end: date) -> pd.DataFrame:
        """新浪接口（前复权，东财失败时的回退路径）"""
        try:
            df = ak.stock_zh_a_daily(
                symbol=self._to_sina_symbol(symbol),
                start_date=start.strftime("%Y%m%d"),
                end_date=end.strftime("%Y%m%d"),
                adjust="qfq"
            )
            if df is None or df.empty:
                logger.warning(f"No kline data for {symbol}")
                return pd.DataFrame()

            df = df.reset_index() if 'date' not in df.columns else df
            return df[["date", "open", "high", "low", "close", "volume"]]
        except Exception as e:
            logger.warning(f"Sina kline failed for {symbol}: {e}")
            return pd.DataFrame()

    @staticmethod
    def _to_sina_symbol(symbol: str) -> str:
        """标准代码转新浪代码：600519.SH → sh600519"""
        code = symbol.split(".")[0]
        suffix = symbol.split(".")[1].lower() if "." in symbol else ""
        if suffix:
            return f"{suffix}{code}"
        if code.startswith("6"):
            return f"sh{code}"
        if code.startswith(("4", "8")):
            return f"bj{code}"
        return f"sz{code}"

    # ---- IQuoteProvider ----

    def get_realtime(self, symbol: str) -> float:
        """获取实时价格（东财全市场快照优先，新浪回退）"""
        try:
            df = ak.stock_zh_a_spot_em()
            code_col = "代码" if "代码" in df.columns else "code"
            price_col = "最新价" if "最新价" in df.columns else "最新价"
            row = df[df[code_col] == symbol.replace(".SH", "").replace(".SZ", "")]
            if not row.empty:
                return float(row[price_col].values[0])
        except Exception as e:
            logger.debug(f"East spot failed for {symbol}: {e}")

        # 新浪回退：取最新一日K线收盘价
        try:
            df = self._kline_sina(symbol, date.today(), date.today())
            if df.empty:  # 当日无数据（节假日）则回看最近7天
                from datetime import timedelta
                df = self._kline_sina(symbol, date.today() - timedelta(days=7), date.today())
            if not df.empty:
                return float(df['close'].iloc[-1])
        except Exception as e:
            logger.warning(f"Failed to fetch realtime for {symbol}: {e}")
        return 0.0

    def get_close(self, symbol: str, trading_date: date) -> float:
        """获取指定日期的收盘价（复用K线接口，含新浪回退）"""
        try:
            df = self.get_kline(symbol, trading_date, trading_date)
            if df is not None and not df.empty and "close" in df.columns:
                return float(df["close"].iloc[-1])
        except Exception as e:
            logger.warning(f"Failed to get close for {symbol} on {trading_date}: {e}")
        return 0.0

    # ---- ISymbolProvider ----

    def list_symbols(self, market: str = "A股") -> List[str]:
        """获取股票列表"""
        try:
            df = ak.stock_info_a_code_name()
            code_col = "code" if "code" in df.columns else "代码"
            symbols = df[code_col].tolist()
            result = []
            for s in symbols:
                s = str(s)
                if s.startswith("6"):
                    result.append(s + ".SH")
                elif s.startswith(("4", "8")):
                    result.append(s + ".BJ")
                else:
                    result.append(s + ".SZ")
            return result
        except Exception as e:
            logger.error(f"Failed to list symbols: {e}")
            return []

    def get_tradable_symbols(self, date: date) -> List[str]:
        """获取当日可交易的股票列表"""
        return self.list_symbols()
