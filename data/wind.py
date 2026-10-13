#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Wind 数据源实现（WindPy 终端接口）

提供 K线、实时行情、股票列表，并支持：
- PIT（point-in-time）股票池：按历史日期取当日的全A成分（含后来退市的股票）
- 批量K线预加载：按字段批量 wsd，一次调用取50只，大幅减少终端往返

实现 IKlineProvider, IQuoteProvider, ISymbolProvider。
"""

import pandas as pd
from datetime import date
from typing import List, Optional, Dict

from data.base.interfaces import IKlineProvider, IQuoteProvider, ISymbolProvider
from utils.logger import get_logger

logger = get_logger(__name__)

# 万得全A板块ID（含当日存续的全部A股，包括后来退市的）
ALL_A_SECTOR_ID = "a001010100000000"

# 批量预加载参数
# K线字段（含成交额：wsd 'amt' 单位=元；volume 单位=股——与 DB 归一化口径一致。
# adjclose：wsd 默认返回不复权价，此处暂以 close 同值退化（Wind客户端源
# 不单独请求复权因子——实盘路径用不复权价是正确口径；回测优先走DB源取adjclose））
KLINE_FIELDS = ["open", "high", "low", "close", "volume", "amt"]
BATCH_CHUNK = 50  # 每次 wsd 的代码数


class WindDataSource:
    """
    基于 Wind 终端（WindPy）的行情数据源。

    - list_symbols(date)：PIT 全A股票池（消除幸存者偏差的关键能力）
    - preload_batch()：批量K线预加载（按字段聚合调用）
    """

    def __init__(self):
        try:
            from WindPy import w
            w.start()
            self._wind = w
        except ImportError:
            raise RuntimeError(
                "WindPy not installed. 运行: python <Wind目录>/bin/installWindPy.py <Wind目录>"
            )

    # ---- ISymbolProvider（PIT）----

    def list_symbols(self, market: str = "A股", as_of: Optional[date] = None) -> List[str]:
        """
        获取股票列表（PIT：as_of 日期当日的全A成分，含当时存续、后来退市的股票）

        Args:
            market: 保留兼容
            as_of: 时点日期（None = 今日）
        """
        d = (as_of or date.today()).strftime("%Y-%m-%d")
        try:
            result = self._wind.wset(
                "sectorconstituent", f"date={d};sectorid={ALL_A_SECTOR_ID}"
            )
            if result.ErrorCode == 0 and "wind_code" in result.Fields:
                codes = result.Data[result.Fields.index("wind_code")]
                logger.info(f"Wind PIT universe @ {d}: {len(codes)} symbols")
                return list(codes)
            logger.warning(f"Wind wset sectorconstituent failed: {result.ErrorCode}")
        except Exception as e:
            logger.error(f"Failed to list symbols: {e}")
        return []

    def get_tradable_symbols(self, trade_date: date) -> List[str]:
        """获取当日可交易的股票列表（PIT）"""
        return self.list_symbols(as_of=trade_date)

    # ---- IKlineProvider ----

    def get_kline(self, symbol: str, start: date, end: date) -> pd.DataFrame:
        """获取单只股票K线，columns: date, open, high, low, close, volume"""
        try:
            r = self._wind.wsd(
                symbol, ";".join(KLINE_FIELDS),
                start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"), ""
            )
            if r.ErrorCode != 0 or not r.Times or len(r.Data) < len(KLINE_FIELDS):
                logger.warning(f"Wind wsd failed for {symbol}: {r.ErrorCode}")
                return pd.DataFrame()
            # 单代码多字段结构：Data[field_idx] = 按 Times 的值列表
            times = pd.to_datetime(pd.Series(list(r.Times)))
            df = pd.DataFrame(
                {f: pd.Series(list(r.Data[fi]), index=times)
                 for fi, f in enumerate(KLINE_FIELDS)}
            ).dropna(subset=["open", "high", "low", "close", "volume"])  # 完整K线才可用（盘中残缺bar污染价格）
            if df.empty:
                return pd.DataFrame()
            df = df.reset_index().rename(columns={"index": "date", "amt": "amount"})
            for f in ["open", "high", "low", "close", "volume", "amount"]:
                if f in df.columns:
                    df[f] = pd.to_numeric(df[f], errors="coerce")
            return df
        except Exception as e:
            logger.error(f"Failed to fetch kline for {symbol}: {e}")
            return pd.DataFrame()

    def preload_batch(self, symbols: List[str], start: date, end: date,
                      progress_every: int = 10) -> Dict[str, pd.DataFrame]:
        """
        批量预加载K线（供 DataLoader.preload_klines 调用）

        本机 WindPy 限制：多代码只能搭配单字段，因此按字段聚合调用
        （每个字段一次 wsd 取 BATCH_CHUNK 只代码，共 5×chunks 次调用）。
        返回结构（经典模式）：Data[field_idx][code_idx][date]。
        """
        per_symbol: Dict[str, Dict[str, pd.Series]] = {s: {} for s in symbols}
        n_chunks = (len(symbols) + BATCH_CHUNK - 1) // BATCH_CHUNK

        for field in KLINE_FIELDS:
            for ci in range(n_chunks):
                chunk = symbols[ci * BATCH_CHUNK:(ci + 1) * BATCH_CHUNK]
                try:
                    r = self._wind.wsd(
                        ",".join(chunk), field,
                        start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"), ""
                    )
                    if r.ErrorCode != 0 or not r.Times:
                        logger.warning(f"Wind batch wsd({field}) chunk {ci}: {r.ErrorCode}")
                        continue
                    times = pd.to_datetime(pd.Series(list(r.Times)))
                    # 多代码单字段结构：Data[code_idx] = 该代码按 Times 的值列表
                    for code_idx, sym in enumerate(r.Codes):
                        try:
                            values = r.Data[code_idx]
                            if len(values) != len(times):
                                continue
                            per_symbol[sym][field] = pd.Series(
                                list(values), index=times
                            )
                        except Exception:
                            continue
                except Exception as e:
                    logger.warning(f"Wind batch wsd({field}) chunk {ci} failed: {e}")
                if progress_every and (ci + 1) % progress_every == 0:
                    logger.info(f"  preload {field}: {ci + 1}/{n_chunks} chunks")

        cache: Dict[str, pd.DataFrame] = {}
        for sym, parts in per_symbol.items():
            if "close" not in parts:
                continue
            try:
                df = pd.DataFrame(parts) if parts else pd.DataFrame()
            except Exception:
                continue
            if df.empty:
                continue
            df = df.reset_index().rename(columns={"index": "date", "amt": "amount"})
            # 行级完整性：残缺bar（如盘中只有close）整行剔除；amount允许缺失
            core = [f for f in ("open", "high", "low", "close", "volume") if f in df.columns]
            df = df.dropna(subset=core)
            for f in ("open", "high", "low", "close", "volume", "amount"):
                if f in df.columns:
                    df[f] = pd.to_numeric(df[f], errors="coerce")
            # adjclose=close 同值退化（wsd默认不复权；缓存格式统一）
            df['adjclose'] = df['close']
            cache[sym] = df

        logger.info(f"Wind preload_batch: {len(cache)}/{len(symbols)} symbols cached")
        return cache

    # ---- IQuoteProvider ----

    def get_realtime(self, symbol: str) -> float:
        """获取实时价格"""
        try:
            result = self._wind.wsq(symbol, "rt_last")
            if result.ErrorCode == 0:
                return float(result.Data[0][0])
            logger.warning(f"Wind wsq failed for {symbol}: {result.ErrorCode}")
        except Exception as e:
            logger.error(f"Failed to fetch realtime for {symbol}: {e}")
        return 0.0

    def get_close(self, symbol: str, trading_date: date) -> float:
        """获取指定日期的收盘价"""
        try:
            r = self._wind.wsd(symbol, "close",
                               trading_date.strftime("%Y-%m-%d"),
                               trading_date.strftime("%Y-%m-%d"), "")
            if r.ErrorCode == 0 and r.Data and r.Data[0] and r.Data[0][0]:
                return float(r.Data[0][0][-1])
        except Exception as e:
            logger.warning(f"Failed to get close for {symbol}: {e}")
        return 0.0
