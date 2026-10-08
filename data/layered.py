#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
三层分级数据源 —— 本地缓存 → Wind Oracle DB → Wind 客户端（配额）

逻辑：
  ① 本地 parquet 缓存已覆盖的区间直接使用（零成本）
  ② 缺失区间优先从 Wind Oracle DB 补（内网，无配额）
  ③ DB 仍缺（如最新未同步日）才走 WindPy 客户端（消耗配额，只取差额）
  取回后写回本地缓存，并记录三层用量统计。

一致性约定：K线统一不复权价（DB 用 S_DQ_OPEN/CLOSE 原始列，与 WindPy 默认一致）；
基本面估值字段（PE/PB/PS/总市值）DB 列直接映射，股息率无 DB 列走客户端。
"""

import os
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from utils.logger import get_logger

logger = get_logger(__name__)

# DB 估值表字段映射（ASHAREEODDERIVATIVEINDICATOR）
DB_FIELD_MAP = {
    'pe_ttm': 'S_VAL_PE_TTM',
    'pb_lf': 'S_VAL_PB_NEW',
    'ps_ttm': 'S_VAL_PS_TTM',
    'mkt_cap_ard': 'S_VAL_MV',
}
CLIENT_ONLY_FIELDS = {'dividendyield2'}

KLINE_FIELDS = ['open', 'high', 'low', 'close', 'volume']
_DB_KLINE_COLS = {'open': 'S_DQ_OPEN', 'high': 'S_DQ_HIGH', 'low': 'S_DQ_LOW',
                  'close': 'S_DQ_CLOSE', 'volume': 'S_DQ_VOLUME'}


class LayeredDataSource:
    """三层数据源（实现 preload_batch/list_symbols/get_kline，兼容 DataLoader）"""

    def __init__(self, cache_dir: str = 'data_cache'):
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            pass
        self.cache_dir = Path(cache_dir)
        self.kline_dir = self.cache_dir / 'kline'
        self.fund_dir = self.cache_dir / 'fundamental'
        self.kline_dir.mkdir(parents=True, exist_ok=True)
        self.fund_dir.mkdir(parents=True, exist_ok=True)
        self._client = None
        self._db = None
        self.stats = {'cache_rows': 0, 'db_rows': 0, 'client_rows': 0}

    # ---- 惰性底层源 ----

    def _client_src(self):
        if self._client is None:
            from data.wind import WindDataSource
            self._client = WindDataSource()
        return self._client

    def _db_src(self):
        if self._db is None:
            try:
                from data.wind_db import WindDBDataSource
                cfg = os.getenv
                self._db = WindDBDataSource(
                    oracle_dsn=cfg('ORACLE_DSN', ''),
                    user=cfg('ORACLE_USER', ''),
                    password=cfg('ORACLE_PASSWORD', ''),
                )
            except Exception as e:
                logger.warning(f"WindDB 不可用，历史数据将走客户端（消耗配额）: {e}")
                self._db = False
        return self._db or None

    # ================= K线 =================

    def preload_batch(self, symbols: List[str], start: date, end: date,
                      progress_every: int = 0) -> Dict[str, pd.DataFrame]:
        """三层取数 K线（返回 per-symbol OHLCV DataFrame）"""
        results: Dict[str, pd.DataFrame] = {}
        missing: Dict[str, List[tuple]] = {}

        # ① 本地缓存
        for sym in symbols:
            df, gaps = self._load_kline_cache(sym, start, end)
            if df is not None and not df.empty:
                results[sym] = df
            if gaps:
                missing[sym] = gaps
        self.stats['cache_rows'] += sum(len(df) for df in results.values())
        logger.info(f"[layered] ① 缓存命中 {len(results)}/{len(symbols)} 只，缺口 {len(missing)} 只")

        # ② Wind Oracle DB（批量，无配额；按缺口方向合并为至多两次批量查询）
        if missing and self._db_src():
            head_syms, tail_syms = [], []
            head_end, tail_start = None, None
            for sym, gaps in missing.items():
                for gs, ge in gaps:
                    if ge >= end:
                        tail_syms.append(sym)
                        tail_start = gs if tail_start is None else min(tail_start, gs)
                    else:
                        head_syms.append(sym)
                        head_end = ge if head_end is None else max(head_end, ge)
            fetched = {}
            if head_syms:
                part = self._db_kline_bulk(sorted(set(head_syms)), start, head_end)
                for sym, df in part.items():
                    fetched[sym] = self._merge_kline(fetched.get(sym), df)
            if tail_syms:
                part = self._db_kline_bulk(sorted(set(tail_syms)), tail_start, end)
                for sym, df in part.items():
                    fetched[sym] = self._merge_kline(fetched.get(sym), df)
            for sym, df in fetched.items():
                results[sym] = self._merge_kline(results.get(sym), df)
                self.stats['db_rows'] += len(df)
            missing = {s: g for s, g in self._missing_after(results, start, end).items() if g}
            logger.info(f"[layered] ② DB 补齐后仍缺 {len(missing)} 只")

        # ③ WindPy 客户端（仅差额，省配额）
        if missing:
            syms = list(missing.keys())
            min_gap_start = min(g[0][0] for g in missing.values() if g)
            fetched = self._client_src().preload_batch(
                syms, min_gap_start, end, progress_every=progress_every)
            for sym, df in fetched.items():
                results[sym] = self._merge_kline(results.get(sym), df)
                self.stats['client_rows'] += len(df)
            logger.info(f"[layered] ③ 客户端补 {len(fetched)} 只（配额消耗仅限差额区间）")

        # 写回缓存
        for sym, df in results.items():
            if not df.empty:
                self._save_kline_cache(sym, df)

        total = sum(len(df) for df in results.values())
        logger.info(f"[layered] K线完成: {len(results)}/{len(symbols)} 只, {total} 行 | "
                    f"缓存 {self.stats['cache_rows']} / DB {self.stats['db_rows']} / "
                    f"客户端 {self.stats['client_rows']}")
        return results

    def get_kline(self, symbol: str, start: date, end: date) -> pd.DataFrame:
        return self.preload_batch([symbol], start, end).get(symbol, pd.DataFrame())

    # ---- K线缓存读写 ----

    def _kline_path(self, sym: str) -> Path:
        return self.kline_dir / f"{sym.replace('.', '_')}.parquet"

    def _load_kline_cache(self, sym: str, start: date, end: date):
        """读缓存并计算缺口区间列表 [(s,e),...]"""
        path = self._kline_path(sym)
        if not path.exists():
            return None, [(start, end)]
        try:
            df = pd.read_parquet(path)
        except Exception:
            return None, [(start, end)]
        if df.empty:
            return None, [(start, end)]
        df = df[(df['date'] >= pd.Timestamp(start)) & (df['date'] <= pd.Timestamp(end))]
        if df.empty:
            return None, [(start, end)]
        gaps = []
        first, last = df['date'].iloc[0], df['date'].iloc[-1]
        if first > pd.Timestamp(start):
            gaps.append((start, (first - pd.Timedelta(days=1)).date()))
        if last < pd.Timestamp(end):
            gaps.append(((last + pd.Timedelta(days=1)).date(), end))
        return df, gaps

    def _save_kline_cache(self, sym: str, df: pd.DataFrame):
        path = self._kline_path(sym)
        if path.exists():
            try:
                old = pd.read_parquet(path)
                df = pd.concat([old, df]).drop_duplicates(subset='date').sort_values('date')
            except Exception:
                pass
        df.to_parquet(path, index=False)

    @staticmethod
    def _merge_kline(a: Optional[pd.DataFrame], b: Optional[pd.DataFrame]) -> pd.DataFrame:
        if a is None or a.empty:
            return b if b is not None else pd.DataFrame()
        if b is None or b.empty:
            return a
        merged = pd.concat([a, b]).drop_duplicates(subset='date').sort_values('date')
        return merged.reset_index(drop=True)

    def _missing_after(self, results, start, end) -> Dict[str, List[tuple]]:
        gaps_map = {}
        for sym, df in results.items():
            if df.empty:
                gaps_map[sym] = [(start, end)]
                continue
            gaps = []
            first, last = df['date'].iloc[0], df['date'].iloc[-1]
            if first > pd.Timestamp(start):
                gaps.append((start, (first - pd.Timedelta(days=1)).date()))
            if last < pd.Timestamp(end - timedelta(days=5)):
                # 尾部缺口超过5个自然日才视为真缺（容忍 DB 同步滞后）
                gaps.append(((last + pd.Timedelta(days=1)).date(), end))
            if gaps:
                gaps_map[sym] = gaps
        return gaps_map

    # ---- DB 批量 K线（不复权）----

    def _db_kline_bulk(self, symbols: List[str], start: date, end: date) -> Dict[str, pd.DataFrame]:
        db = self._db_src()
        out: Dict[str, pd.DataFrame] = {}
        s_dt, e_dt = start.strftime('%Y%m%d'), end.strftime('%Y%m%d')
        try:
            cur = db._cursor()
        except Exception as e:
            logger.warning(f"[layered] DB cursor 失败: {e}")
            return out
        try:
            for i in range(0, len(symbols), 500):
                chunk = symbols[i:i + 500]
                placeholders = ",".join(f":s{j}" for j in range(len(chunk)))
                binds = {f"s{j}": s for j, s in enumerate(chunk)}
                sql = f"""
                    SELECT S_INFO_WINDCODE AS sym, TRADE_DT AS dt,
                           CAST(S_DQ_OPEN AS FLOAT), CAST(S_DQ_HIGH AS FLOAT),
                           CAST(S_DQ_LOW AS FLOAT), CAST(S_DQ_CLOSE AS FLOAT),
                           CAST(S_DQ_VOLUME AS FLOAT)
                    FROM AShareEODPrices
                    WHERE S_INFO_WINDCODE IN ({placeholders})
                      AND TRADE_DT >= :start_dt AND TRADE_DT <= :end_dt
                    ORDER BY sym, dt
                """
                binds.update({'start_dt': s_dt, 'end_dt': e_dt})
                cur.execute(sql, binds)
                rows = cur.fetchall()
                cols = ['sym', 'date', 'open', 'high', 'low', 'close', 'volume']
                big = pd.DataFrame(rows, columns=cols)
                if big.empty:
                    continue
                for sym, g in big.groupby('sym'):
                    g = g.copy()
                    g['date'] = pd.to_datetime(g['date'], format='%Y%m%d')
                    for f in KLINE_FIELDS:
                        g[f] = pd.to_numeric(g[f], errors='coerce')
                    out[sym] = g[KLINE_COLS_ORDER].reset_index(drop=True)
                logger.info(f"[layered] DB K线 chunk {i//500 + 1}: 累计 {len(out)} 只")
            # 指数（如 000300.SH）不在个股表，从 AIndexEODPrices 补
            idx_syms = [s for s in symbols if s not in out]
            for i in range(0, len(idx_syms), 100):
                chunk = idx_syms[i:i + 100]
                placeholders = ",".join(f":s{j}" for j in range(len(chunk)))
                binds = {f"s{j}": s for j, s in enumerate(chunk)}
                sql = f"""
                    SELECT S_INFO_WINDCODE AS sym, TRADE_DT AS dt,
                           CAST(S_DQ_OPEN AS FLOAT), CAST(S_DQ_HIGH AS FLOAT),
                           CAST(S_DQ_LOW AS FLOAT), CAST(S_DQ_CLOSE AS FLOAT),
                           CAST(S_DQ_VOLUME AS FLOAT)
                    FROM AIndexEODPrices
                    WHERE S_INFO_WINDCODE IN ({placeholders})
                      AND TRADE_DT >= :start_dt AND TRADE_DT <= :end_dt
                """
                binds.update({'start_dt': s_dt, 'end_dt': e_dt})
                cur.execute(sql, binds)
                rows = cur.fetchall()
                if not rows:
                    continue
                big = pd.DataFrame(rows, columns=['sym', 'date'] + KLINE_FIELDS)
                for sym, g in big.groupby('sym'):
                    g = g.copy()
                    g['date'] = pd.to_datetime(g['date'], format='%Y%m%d')
                    for f in KLINE_FIELDS:
                        g[f] = pd.to_numeric(g[f], errors='coerce')
                    out[sym] = g[['date'] + KLINE_FIELDS].reset_index(drop=True)
            if idx_syms:
                logger.info(f"[layered] DB 指数补齐 {len([s for s in idx_syms if s in out])}/{len(idx_syms)}")
        except Exception as e:
            logger.warning(f"[layered] DB K线批量失败: {e}")
        finally:
            try:
                cur.close()
            except Exception:
                pass
        return out

    # ================= 基本面字段矩阵 =================

    def get_field_matrix(self, field: str, symbols: List[str],
                         start: date, end: date) -> pd.DataFrame:
        """三层取数基本面字段（dates × symbols 宽矩阵）"""
        path = self.fund_dir / f"{field}.parquet"
        cached = pd.DataFrame()
        if path.exists():
            try:
                cached = pd.read_parquet(path)
            except Exception:
                cached = pd.DataFrame()

        need_db = field in DB_FIELD_MAP
        matrices = []
        if not cached.empty:
            matrices.append(cached)

        # 计算矩阵缺口（首末日期）
        def matrix_gaps(mx):
            if mx.empty:
                return (start, end)
            gaps = []
            if mx.index.min() > pd.Timestamp(start):
                gaps.append((start, (mx.index.min() - pd.Timedelta(days=1)).date()))
            if mx.index.max() < pd.Timestamp(end - timedelta(days=5)):
                gaps.append(((mx.index.max() + pd.Timedelta(days=1)).date(), end))
            return gaps or None

        gaps = matrix_gaps(cached)
        db_part = pd.DataFrame()
        client_part = pd.DataFrame()

        if gaps and need_db and self._db_src():
            db_part = self._db_fundamental_bulk(field, symbols, start, end)
        if gaps and db_part.empty and gaps:
            # DB 无该列或失败 → 客户端补差额
            gs = min(g[0] for g in gaps) if isinstance(gaps, list) else gaps[0]
            from data.wind import WindDataSource  # noqa: F401
            client_matrix = self._client_src_matrix(field, symbols, gs, end)
            client_part = client_matrix

        merged = self._merge_matrices([m for m in (cached, db_part, client_part) if not m.empty])
        if not merged.empty:
            merged.to_parquet(path)
        stats_line = (f"[layered] 字段 {field}: 缓存{len(cached)}行 DB{len(db_part)}行 "
                      f"客户端{len(client_part)}行 → 合并{len(merged)}行")
        logger.info(stats_line)
        self.stats['db_rows'] += len(db_part)
        self.stats['client_rows'] += len(client_part)
        self.stats['cache_rows'] += len(cached)
        # 只返回请求区间
        if not merged.empty:
            merged = merged.loc[(merged.index >= pd.Timestamp(start)) & (merged.index <= pd.Timestamp(end))]
        return merged

    def _client_src_matrix(self, field, symbols, start, end):
        """客户端字段矩阵（复用 WindFundamentalProvider 的拉取逻辑）"""
        from data.fundamentals import WindFundamentalProvider
        provider = WindFundamentalProvider(self._client_src())
        return provider.get_field_matrix(field, symbols, start, end)

    def _db_fundamental_bulk(self, field: str, symbols: List[str],
                             start: date, end: date) -> pd.DataFrame:
        col = DB_FIELD_MAP[field]
        db = self._db_src()
        frames = []
        try:
            cur = db._cursor()
            for i in range(0, len(symbols), 500):
                chunk = symbols[i:i + 500]
                placeholders = ",".join(f":s{j}" for j in range(len(chunk)))
                binds = {f"s{j}": s for j, s in enumerate(chunk)}
                sql = f"""
                    SELECT S_INFO_WINDCODE AS sym, TRADE_DT AS dt,
                           CAST({col} AS FLOAT) AS val
                    FROM ASHAREEODDERIVATIVEINDICATOR
                    WHERE S_INFO_WINDCODE IN ({placeholders})
                      AND TRADE_DT >= :start_dt AND TRADE_DT <= :end_dt
                """
                binds.update({'start_dt': start.strftime('%Y%m%d'), 'end_dt': end.strftime('%Y%m%d')})
                cur.execute(sql, binds)
                rows = cur.fetchall()
                if rows:
                    frames.append(pd.DataFrame(rows, columns=['sym', 'dt', 'val']))
            cur.close()
        except Exception as e:
            logger.warning(f"[layered] DB 字段 {field} 失败: {e}")
            return pd.DataFrame()
        if not frames:
            return pd.DataFrame()
        long_df = pd.concat(frames)
        long_df['dt'] = pd.to_datetime(long_df['dt'], format='%Y%m%d')
        wide = long_df.pivot(index='dt', columns='sym', values='val').sort_index()
        wide.index = pd.DatetimeIndex(wide.index).as_unit('ns')
        return wide

    @staticmethod
    def _merge_matrices(mats):
        if not mats:
            return pd.DataFrame()
        out = mats[0]
        for m in mats[1:]:
            out = out.combine_first(m)
        return out.sort_index()

    # ================= PIT 股票池 =================

    def list_symbols(self, market: str = "A股", as_of: Optional[date] = None) -> List[str]:
        """PIT 全A（优先 DB：当日有行情记录的股票；DB 不可用回退客户端 wset）"""
        d = (as_of or date.today())
        if self._db_src():
            try:
                cur = self._db_src()._cursor()
                cur.execute(
                    "SELECT DISTINCT S_INFO_WINDCODE FROM AShareEODPrices WHERE TRADE_DT = :dt",
                    {'dt': d.strftime('%Y%m%d')})
                rows = [r[0] for r in cur.fetchall()]
                cur.close()
                if rows:
                    logger.info(f"[layered] PIT universe (DB) @{d}: {len(rows)}")
                    return sorted(rows)
            except Exception as e:
                logger.warning(f"[layered] DB PIT 失败，回退客户端: {e}")
        return self._client_src().list_symbols(as_of=as_of or date.today())

    def get_tradable_symbols(self, trade_date: date) -> List[str]:
        return self.list_symbols(as_of=trade_date)

    def get_kline_calendar(self, symbols, start: date, end: date):
        """从本地缓存取交易日历（优先指数，其次个股）"""
        for sym in list(symbols) + ['000300.SH']:
            path = self._kline_path(sym)
            if path.exists():
                try:
                    df = pd.read_parquet(path)
                    sel = df[(df['date'] >= pd.Timestamp(start)) & (df['date'] <= pd.Timestamp(end))]
                    if not sel.empty:
                        return pd.DatetimeIndex(sel['date'])
                except Exception:
                    continue
        return None

    def get_index_members(self, index_code: str, as_of: Optional[date] = None) -> List[str]:
        """PIT 指数成分（AINDEXMEMBERS：入场/剔除日期判定）"""
        d = (as_of or date.today()).strftime('%Y%m%d')
        if self._db_src():
            try:
                cur = self._db_src()._cursor()
                cur.execute("""
                    SELECT S_CON_WINDCODE FROM AINDEXMEMBERS
                    WHERE S_INFO_WINDCODE = :idx
                      AND S_CON_INDATE <= :dt
                      AND (S_CON_OUTDATE IS NULL OR S_CON_OUTDATE > :dt)
                """, {'idx': index_code, 'dt': d})
                rows = sorted(r[0] for r in cur.fetchall())
                cur.close()
                if rows:
                    logger.info(f"[layered] {index_code} PIT成员 @{d}: {len(rows)}")
                    return rows
            except Exception as e:
                logger.warning(f"[layered] 指数成员查询失败: {e}")
        return []


KLINE_COLS_ORDER = ['date', 'open', 'high', 'low', 'close', 'volume']
