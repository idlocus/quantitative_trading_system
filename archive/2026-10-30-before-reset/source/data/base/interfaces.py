#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据源基类 - 抽象接口定义

提供四类数据接口：
- IKlineProvider: K线数据
- IQuoteProvider: 实时/收盘价
- ISymbolProvider: 股票列表
- IIndustryProvider: 申万行业数据
"""

from typing import List, Protocol, Optional
import pandas as pd
from datetime import date


class IKlineProvider(Protocol):
    """K线数据提供者"""

    def get_kline(self, symbol: str, start: date, end: date) -> pd.DataFrame:
        """获取K线数据，columns: date, open, high, low, close, volume"""


class IQuoteProvider(Protocol):
    """行情数据提供者"""

    def get_realtime(self, symbol: str) -> float:
        """获取实时价格"""

    def get_close(self, symbol: str, trading_date: date) -> float:
        """获取指定日期的收盘价"""


class ISymbolProvider(Protocol):
    """股票列表提供者"""

    def list_symbols(self, market: str = "A股") -> List[str]:
        """获取股票列表"""

    def get_tradable_symbols(self, date: date) -> List[str]:
        """获取当日可交易的股票列表"""


class IIndustryProvider(Protocol):
    """行业数据提供者"""

    def get_sw_industry_constituents(self) -> List[dict]:
        """
        获取申万二级行业成分股
        Returns: [{industry_code, industry_name, symbols}, ...]
        """

    def get_industry_momentum(self, industry_code: str) -> Optional[dict]:
        """
        计算板块动量指标
        Returns: {return_5d, ma5, ma20, ma5_above_ma20, volume_ratio} or None
        """