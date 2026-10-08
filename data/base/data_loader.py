#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据加载模块 - 统一管理数据源、K线缓存和预加载

提供 CachedDataSource 用于加速回测，缓存 K线数据在内存中。
DataLoader 负责创建数据源并预加载数据。
"""

import pandas as pd
from datetime import date
from typing import List, Optional, Dict

from .interfaces import IKlineProvider, IQuoteProvider, ISymbolProvider
from ..wind import WindDataSource
from ..akshare import AKShareDataSource
from ..wind_db import WindDBDataSource
from utils.logger import get_logger

logger = get_logger(__name__)


class CachedDataSource:
    """
    带内存缓存的数据源封装。

    预加载K线数据后，所有读取都从缓存进行，大幅提升回测速度。
    同时实现了 IKlineProvider、IQuoteProvider、ISymbolProvider 三个接口，
    可直接替换数据源注入到各组件。
    """

    def __init__(
        self,
        kline_source: IKlineProvider,
        quote_source: Optional[IQuoteProvider] = None,
        symbol_source: Optional[ISymbolProvider] = None,
        cache: Optional[Dict[str, pd.DataFrame]] = None,
    ):
        self._kline_source = kline_source
        self._quote_source = quote_source
        self._symbol_source = symbol_source
        self._cache: Dict[str, pd.DataFrame] = cache or {}

    # ---- IKlineProvider ----

    def get_kline(self, symbol: str, start: date, end: date) -> pd.DataFrame:
        """从缓存读取K线数据"""
        if symbol in self._cache and not self._cache[symbol].empty:
            df = self._cache[symbol]
            return df[(df['date'] >= pd.Timestamp(start)) & (df['date'] <= pd.Timestamp(end))]
        return pd.DataFrame()

    # ---- IQuoteProvider ----

    def get_realtime(self, symbol: str) -> float:
        """优先从缓存读取最新价，否则回退到 quote_source"""
        if symbol in self._cache and not self._cache[symbol].empty:
            return float(self._cache[symbol]['close'].iloc[-1])
        if self._quote_source:
            return self._quote_source.get_realtime(symbol)
        return 0.0

    def get_close(self, symbol: str, trading_date: date) -> float:
        """从缓存读取收盘价"""
        if symbol in self._cache and not self._cache[symbol].empty:
            df = self._cache[symbol]
            mask = df['date'] <= pd.Timestamp(trading_date)
            if mask.any():
                return float(df.loc[mask, 'close'].iloc[-1])
        return 0.0

    # ---- ISymbolProvider ----

    def list_symbols(self, market: str = "A股") -> List[str]:
        """委托给 symbol_source"""
        if self._symbol_source:
            return self._symbol_source.list_symbols(market)
        return list(self._cache.keys())

    def get_tradable_symbols(self, date: date) -> List[str]:
        """返回缓存中有数据的股票"""
        return list(self._cache.keys())

    def get_trading_dates(self, start: date, end: date) -> List[pd.Timestamp]:
        """
        从缓存K线的日期并集推导真实交易日历（自动排除节假日/停牌日）

        Args:
            start: 开始日期
            end: 结束日期

        Returns:
            排序后的交易日 Timestamp 列表（缓存为空时返回空列表）
        """
        all_dates = set()
        for df in self._cache.values():
            if not df.empty:
                all_dates.update(df['date'].tolist())
        return sorted(
            d for d in all_dates
            if pd.Timestamp(start) <= d <= pd.Timestamp(end)
        )

    # ---- FundFlow (机构/游资) ----

    def get_fund_flow(self, symbol: str, start: date, end: date) -> pd.DataFrame:
        """
        获取个股资金流向（机构/游资分离）

        优先从 fund_flow_provider 获取数据（akshare）；
        如果失败则返回空 DataFrame（不回测流程）。

        Args:
            symbol: 股票代码，如 '300433.SZ'
            start: 开始日期
            end: 结束日期

        Returns:
            DataFrame with columns: 主力净流入, 散户净流入, 超大单净流入,
            大单净流入, 成交额, date
        """
        if hasattr(self, '_fund_flow_provider') and self._fund_flow_provider is not None:
            return self._fund_flow_provider.get_fund_flow(symbol, start, end)
        elif hasattr(self, '_fund_flow_cache') and self._fund_flow_cache is not None:
            # 已有缓存
            cache_key = f"{symbol}_{start.isoformat()}_{end.isoformat()}"
            if cache_key in self._fund_flow_cache:
                return self._fund_flow_cache[cache_key].copy()
        return pd.DataFrame()

    def set_fund_flow_provider(self, provider):
        """注入资金流向Provider"""
        self._fund_flow_provider = provider
        self._fund_flow_cache = {}


class DataLoader:
    """
    数据加载器。

    根据配置创建数据源（Wind / AKShare / WindDB），支持预加载和缓存。
    """

    def __init__(self, data_source_type: str = 'wind'):
        self.data_source_type = data_source_type
        self.data_source = self._create_data_source(data_source_type)
        self._cache: Optional[CachedDataSource] = None
        logger.info(f"DataLoader initialized with {data_source_type}")

    def _create_data_source(self, data_source_type: str):
        if data_source_type == 'wind':
            try:
                return WindDataSource()
            except Exception as e:
                logger.warning(f"Wind not available: {e}, falling back to AKShare")
                return AKShareDataSource()
        elif data_source_type == 'layered':
            # 三层数据源：本地缓存 → Wind Oracle DB → Wind 客户端（省配额）
            from ..layered import LayeredDataSource
            return LayeredDataSource()
        elif data_source_type == 'akshare':
            return AKShareDataSource()
        elif data_source_type == 'winddb':
            raise ValueError(
                "DATA_SOURCE_TYPE=winddb requires oracle credentials. "
                "Use DataLoader.from_winddb(dsn, user, password) instead."
            )
        else:
            raise ValueError(f"Unknown data source type: {data_source_type}")

    @classmethod
    def from_winddb(cls, oracle_dsn: str, user: str, password: str) -> 'DataLoader':
        """直接使用 WindDB 数据源（绕过 DataLoader 构造函数）"""
        loader = cls.__new__(cls)
        loader.data_source_type = 'winddb'
        loader.data_source = WindDBDataSource(oracle_dsn, user, password)
        loader._cache = None
        logger.info("DataLoader initialized with WindDB")
        return loader

    def load_symbols(self, as_of: Optional[date] = None) -> List[str]:
        """加载股票列表（as_of 指定时返回该时点的 PIT 成分，含后来退市的股票）"""
        if as_of is not None and hasattr(self.data_source, 'get_tradable_symbols'):
            symbols = self.data_source.get_tradable_symbols(as_of)
        else:
            symbols = self.data_source.list_symbols()
        logger.info(f"Loaded {len(symbols)} symbols from {self.data_source_type}"
                    + (f" (PIT @ {as_of})" if as_of else ""))
        return symbols

    def preload_klines(
        self,
        symbols: List[str],
        start_date: date,
        end_date: date,
    ) -> CachedDataSource:
        """
        预加载所有K线数据到内存缓存。

        数据源支持批量时（Wind）按字段聚合调用，否则逐只加载。

        Args:
            symbols: 股票代码列表
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            CachedDataSource 实例（同时实现了 IKlineProvider、IQuoteProvider、ISymbolProvider）
        """
        logger.info(f"Pre-loading K-lines for {len(symbols)} symbols...")
        cache: Dict[str, pd.DataFrame] = {}

        if hasattr(self.data_source, 'preload_batch'):
            # 批量路径（Wind）：每字段聚合调用
            cache = self.data_source.preload_batch(symbols, start_date, end_date)
        else:
            for i, sym in enumerate(symbols):
                df = self.data_source.get_kline(sym, start_date, end_date)
                if not df.empty:
                    df = df.copy()
                    df['date'] = pd.to_datetime(df['date'])
                    cache[sym] = df

                if (i + 1) % 100 == 0:
                    logger.info(f"  Progress: {i+1}/{len(symbols)} ({100*(i+1)//len(symbols)}%) - {sym} loaded")

        logger.info(f"Pre-loading complete: {len(cache)} symbols cached")

        self._cache = CachedDataSource(
            kline_source=self.data_source,
            quote_source=self.data_source,
            symbol_source=self.data_source,
            cache=cache,
        )
        return self._cache

    def get_cached_source(self) -> Optional[CachedDataSource]:
        return self._cache

    @property
    def cache_ready(self) -> bool:
        return self._cache is not None