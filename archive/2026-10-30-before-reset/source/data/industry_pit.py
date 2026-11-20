#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PIT 行业分类（万得行业层级，经 Oracle AShareIndustriesClass）

层级：前4位=L1(11大类) / 前6位=L2(25) / 前8位=L3(66) / 10位=L4(155)。
PIT 语义：ENTRY_DT ≤ as_of < REMOVE_DT（REMOVE_DT 空=现行）；行业变更
约 20% 股票有历史（1,973 行已移除记录）——按"当时"分类，避免幸存者口径。

名称经 CCSTOCKINDUSTRIESMEMBERS.WIND_IND_NAME 映射（同库）。
"""
import logging
from typing import Dict, Optional

import pandas as pd

logger = logging.getLogger(__name__)

_L2_PAD = '0000'   # L2 代码在名称表中的形态：6 位前缀 + 4 个零


def industry_classification(ds, symbols, as_of) -> Dict[str, str]:
    """股票 → L2 行业代码（PIT：as_of 时点的生效分类）。

    Returns: {symbol: L2代码（6位）}；无分类的股票不在结果中。
    """
    cur = ds._db_src()._cursor()
    syms = list(symbols)
    out = {}
    for i in range(0, len(syms), 500):
        chunk = syms[i:i + 500]
        ph = ','.join(f':s{j}' for j in range(len(chunk)))
        cur.execute(
            f"SELECT S_INFO_WINDCODE, WIND_IND_CODE, ENTRY_DT FROM AShareIndustriesClass "
            f"WHERE S_INFO_WINDCODE IN ({ph}) "
            f"AND ENTRY_DT <= :dt AND (REMOVE_DT IS NULL OR REMOVE_DT > :dt) "
            f"ORDER BY S_INFO_WINDCODE, ENTRY_DT",
            {**{f's{j}': s for j, s in enumerate(chunk)},
             'dt': as_of.strftime('%Y%m%d')})
        seen = set()
        for sym, code, _entry in cur.fetchall():
            if sym in seen:
                continue          # 同日多条（罕见）：取 ENTRY_DT 最新（ORDER BY 保证）
            seen.add(sym)
            out[sym] = str(code)[:6]
    return out


def industry_names(ds, level: int = 2) -> Dict[str, str]:
    """行业代码 → 名称（level=1/2/3；经 CCSTOCKINDUSTRIESMEMBERS 映射）。

    名称表直接携带的多为 L1/L4 形态；L2/L3 名称缺失时用该前缀下
    最短代码的名称回推（L2 624010 → 子代"多元化银行"=银行类，标签
    够用；归因的真实键是代码，名称仅展示）。"""
    n = {1: 4, 2: 6, 3: 8}[level]
    pad = '0' * (10 - n)
    cur = ds._db_src()._cursor()
    cur.execute(
        "SELECT DISTINCT WIND_IND_CODE, WIND_IND_NAME FROM CCSTOCKINDUSTRIESMEMBERS")
    best: Dict[str, tuple] = {}     # 前缀 → (代码长度, 名称)
    for code, name in cur.fetchall():
        code = str(code)
        if len(code) != 10:
            continue
        pre = code[:n]
        cand = (len(code.rstrip('0')), str(name))
        # 优先：本级零填充形态（精确名称）；否则取最长零填充/最短实质代码
        if code[n:] == pad:
            best[pre] = (0, str(name))       # 精确本级名，最高优先
        elif pre not in best or (best[pre][0] > 0 and cand[0] < best[pre][0]):
            if best.get(pre, (1,))[0] != 0:  # 不覆盖精确名
                best[pre] = cand
    return {k: v[1] for k, v in best.items()}


def industry_matrix(ds, symbols, dates: pd.DatetimeIndex,
                    level: int = 2) -> pd.DataFrame:
    """dates × symbols 的 L2 行业代码矩阵（月频分类查询 + 向前填充）。

    在每月首个日期查询 PIT 分类，月内沿用（行业变更频率远低于月）。
    """
    n = {1: 4, 2: 6, 3: 8}[level]
    # 月度查询点
    month_starts, seen = [], set()
    for d in dates:
        m = (d.year, d.month)
        if m not in seen:
            seen.add(m)
            month_starts.append(d)
    rows = {}
    for md in month_starts:
        cls = industry_classification(ds, symbols, md.date())
        rows[md] = pd.Series(cls)
    ser = pd.DataFrame(rows).T.sort_index()
    ser = ser.reindex(dates).ffill()
    ser.index.name = 'date'
    return ser
