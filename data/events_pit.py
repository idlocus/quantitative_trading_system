#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PIT 财报事件层 —— 业绩预告（AShareProfitNotice）事件提取

事件方向由数值字段直接定义（不依赖库内码表，自文档化且稳健）：
  +1 预增      CHANGEMIN ≥ +30（净利润增幅下限≥30%）
  -1 预减      CHANGEMAX ≤ -30 或 NETPROFITMAX < 0（预亏）
   0 模糊      其余（不入选研究）
PIT 语义：公告日（FIRSTANNDATE 优先）收盘后才可见，T+1 才可交易。
"""
import logging
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


def profit_notice_events(ds, d0, d1) -> pd.DataFrame:
    """业绩预告事件（公告日 ∈ [d0, d1]）。

    Returns: DataFrame[symbol, ann_date(Timestamp), period, direction,
              change_min, change_max, net_min, net_max]
    """
    cur = ds._db_src()._cursor()
    cur.execute(
        "SELECT S_INFO_WINDCODE, NVL(S_PROFITNOTICE_FIRSTANNDATE, S_PROFITNOTICE_DATE), "
        "S_PROFITNOTICE_PERIOD, S_PROFITNOTICE_CHANGEMIN, S_PROFITNOTICE_CHANGEMAX, "
        "S_PROFITNOTICE_NETPROFITMIN, S_PROFITNOTICE_NETPROFITMAX "
        "FROM AShareProfitNotice "
        "WHERE NVL(S_PROFITNOTICE_FIRSTANNDATE, S_PROFITNOTICE_DATE) >= :d0 "
        "AND NVL(S_PROFITNOTICE_FIRSTANNDATE, S_PROFITNOTICE_DATE) <= :d1",
        {'d0': d0.strftime('%Y%m%d'), 'd1': d1.strftime('%Y%m%d')})
    rows = []
    for sym, dt, period, cmin, cmax, nmin, nmax in cur.fetchall():
        if not dt or len(str(dt)) != 8:
            continue
        cmin = float(cmin) if cmin is not None else None
        cmax = float(cmax) if cmax is not None else None
        nmax = float(nmax) if nmax is not None else None
        if cmin is not None and cmin >= 30:
            direction = 1
        elif (cmax is not None and cmax <= -30) or \
                (nmax is not None and nmax < 0):
            direction = -1
        else:
            continue                      # 模糊预告不入选
        rows.append({'symbol': sym, 'ann_date': pd.Timestamp(str(dt)),
                     'period': str(period), 'direction': direction,
                     'change_min': cmin, 'change_max': cmax})
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(['symbol', 'ann_date'])
        # 同股同日多条（更正公告等）去重取最后一条
        df = df.drop_duplicates(['symbol', 'ann_date'], keep='last')
    logger.info(f"[events] 业绩预告事件 {d0}~{d1}: {len(df)} 条"
                f"（+{int((df.direction > 0).sum())}/-{int((df.direction < 0).sum())}）"
                if not df.empty else f"[events] 业绩预告事件 {d0}~{d1}: 0 条")
    return df
