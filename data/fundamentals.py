#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Wind 基本面数据提供者 —— 日频估值/市值字段矩阵

与 K线不同，基本面因子直接以 (dates × symbols) 矩阵形式提供，
复用 WindPy 多代码单字段批量调用（50只/次）。

可用字段（已探测）：pe_ttm / pb_lf / ps_ttm / dividendyield2 / mkt_cap_ard
（ROE 等报告期字段需公告日 PIT 处理，暂不支持）

PIT 说明：Wind 日频估值字段按"当日已知最新财报"计算并保留历史档案，
Wind 自身存在少量回填风险，报告时需注明此 caveat。
"""

from datetime import date
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from utils.logger import get_logger

logger = get_logger(__name__)

# 已验证可用的日频基本面字段
AVAILABLE_FIELDS = ['pe_ttm', 'pb_lf', 'ps_ttm', 'dividendyield2', 'mkt_cap_ard']

BATCH_CHUNK = 50


class WindFundamentalProvider:
    """Wind 日频基本面字段矩阵（批量 + 内存缓存）"""

    def __init__(self, wind_data_source=None):
        """
        Args:
            wind_data_source: WindDataSource 或 LayeredDataSource（三层取数时委托）
        """
        if wind_data_source is None:
            from data.wind import WindDataSource
            wind_data_source = WindDataSource()
        self._wind = (wind_data_source if hasattr(wind_data_source, 'get_field_matrix')
                      else getattr(wind_data_source, '_wind', wind_data_source))
        self._cache: Dict[tuple, pd.DataFrame] = {}

    def get_field_matrix(self, field: str, symbols: List[str],
                         start: date, end: date) -> pd.DataFrame:
        """
        获取字段的全量矩阵（dates × symbols）

        若底层数据源是 LayeredDataSource（三层取数），直接委托；
        否则走客户端批量 wsd。
        """
        # 三层源委托（缓存 → DB → 客户端）
        if hasattr(self._wind, 'get_field_matrix'):
            return self._wind.get_field_matrix(field, symbols, start, end)

        # 审计 2026-11-19 四-b：缓存键必须含**完整证券集合与日期范围**——
        # 此前只有 field+数量+首尾代码：同首尾不同中间证券、或相同证券
        # 不同年份，都会直接取回上一次的矩阵。
        cache_key = (field, tuple(symbols), start, end)
        if cache_key in self._cache:
            return self._cache[cache_key].copy()

        columns: Dict[str, pd.Series] = {}
        failed_chunks = 0
        n_chunks = (len(symbols) + BATCH_CHUNK - 1) // BATCH_CHUNK
        for ci in range(n_chunks):
            chunk = symbols[ci * BATCH_CHUNK:(ci + 1) * BATCH_CHUNK]
            try:
                r = self._wind.wsd(
                    ",".join(chunk), field,
                    start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"), ""
                )
                if r.ErrorCode != 0 or not r.Times:
                    logger.warning(f"fundamental wsd({field}) chunk {ci}: {r.ErrorCode}")
                    failed_chunks += 1
                    continue
                times = pd.DatetimeIndex(pd.to_datetime(pd.Series(list(r.Times))))
                for code_idx, sym in enumerate(r.Codes):
                    try:
                        values = r.Data[code_idx]
                        if len(values) != len(times):
                            continue
                        columns[sym] = pd.Series(
                            pd.to_numeric(pd.Series(list(values)), errors='coerce').values,
                            index=times,
                        )
                    except Exception:
                        continue
            except Exception as e:
                failed_chunks += 1
                logger.warning(f"fundamental wsd({field}) chunk {ci} failed: {e}")

        mx = pd.DataFrame(columns).sort_index().replace([np.inf, -np.inf], np.nan)
        mx.index = pd.DatetimeIndex(mx.index).as_unit('ns')
        if failed_chunks:
            # 审计 2026-11-19 四-b：分批拉取有失败 → 结果残缺，不得缓存
            # （下次同请求会重拉而非复用残缺矩阵）
            logger.warning(f"fundamental wsd({field}): {failed_chunks}/{n_chunks} "
                           f"批次失败，残缺结果不入缓存")
            return mx
        missing = [s for s in symbols if s not in mx.columns]
        if missing:
            logger.info(f"fundamental wsd({field}): {len(missing)} 只证券无数据列"
                        f"（首例 {missing[0]}，属数据级缺失）")
            return mx
        if mx.empty or mx.isna().any().any():
            return mx
        self._cache[cache_key] = mx.copy()
        logger.info(f"fundamental field '{field}': {mx.shape[0]} 天 × {mx.shape[1]} 只")
        return mx


# =============================================================================
# 基本面因子定义（field → 因子矩阵，方向由调用方指定）
# =============================================================================

def factor_ep(pe_mx: pd.DataFrame) -> pd.DataFrame:
    """盈利收益率 E/P = 1/PE_TTM（价值因子，越高越便宜）"""
    return 1.0 / pe_mx.replace(0, np.nan)


def factor_bp(pb_mx: pd.DataFrame) -> pd.DataFrame:
    """账面市值比 B/P = 1/PB_LF（价值因子）"""
    return 1.0 / pb_mx.replace(0, np.nan)


def factor_sp(ps_mx: pd.DataFrame) -> pd.DataFrame:
    """销售收益率 S/P = 1/PS_TTM（价值因子）"""
    return 1.0 / ps_mx.replace(0, np.nan)


def factor_dp(div_mx: pd.DataFrame) -> pd.DataFrame:
    """股息率（dividendyield2，%）——直接使用，高股息为正向"""
    return div_mx / 100.0


def factor_size(mv_mx: pd.DataFrame) -> pd.DataFrame:
    """市值因子 = log(总市值)（小市值溢价方向由数据检验决定）"""
    return np.log(mv_mx.where(mv_mx > 0))


# 因子注册表：名字 → (依赖字段, 变换函数, 默认方向, 说明)
FUNDAMENTAL_FACTORS = {
    'ep':   ('pe_ttm', factor_ep, +1.0, '盈利收益率 1/PE（价值）'),
    'bp':   ('pb_lf', factor_bp, +1.0, '账面市值比 1/PB（价值）'),
    'sp':   ('ps_ttm', factor_sp, +1.0, '销售收益率 1/PS（价值）'),
    'dp':   ('dividendyield2', factor_dp, +1.0, '股息率（红利）'),
    'size': ('mkt_cap_ard', factor_size, -1.0, 'log总市值（小市值）'),
}


def build_fundamental_matrix(provider: WindFundamentalProvider,
                             factor_name: str, symbols: List[str],
                             start: date, end: date) -> pd.DataFrame:
    """构建单个基本面因子矩阵"""
    if factor_name not in FUNDAMENTAL_FACTORS:
        raise ValueError(f"未知基本面因子 '{factor_name}'. 可用: {list(FUNDAMENTAL_FACTORS)}")
    field, transform, _, _ = FUNDAMENTAL_FACTORS[factor_name]
    raw = provider.get_field_matrix(field, symbols, start, end)
    if raw.empty:
        return pd.DataFrame()
    return transform(raw)
