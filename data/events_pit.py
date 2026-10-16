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


def sue_events(ds, d0, d1, min_abs_est_wan=1000.0) -> pd.DataFrame:
    """一致预期超预期（SUE）事件：实际归母净利 vs 公告前最后一致预期。

    口径（实测 2026-10-16）：
      - AShareIncome.NET_PROFIT_EXCL_MIN_INT_INC 单位=元；同报告期多行
        （多报表口径+更正公告）→ 取首次公告日、该日最大值（合并归母）
      - AShareConsensusData.NET_PROFIT_AVG 单位=万元 → 统一换算到万元
      - 预期取 EST_DT < 公告日 的最后一条非空快照
    surprise = (实际-预期)/|预期|，clip ±300%；|预期|<min_abs_est_wan 剔除
    （微小分母会产生虚假巨大 SUE）。

    Returns: DataFrame[symbol, ann_date, period, surprise, actual_wan, est_wan]
    """
    cur = ds._db_src()._cursor()
    cur.execute(
        "SELECT S_INFO_WINDCODE, REPORT_PERIOD, ANN_DT, NET_PROFIT_EXCL_MIN_INT_INC "
        "FROM AShareIncome WHERE ANN_DT >= :d0 AND ANN_DT <= :d1 "
        "AND NET_PROFIT_EXCL_MIN_INT_INC IS NOT NULL",
        {'d0': d0.strftime('%Y%m%d'), 'd1': d1.strftime('%Y%m%d')})
    inc = pd.DataFrame(cur.fetchall(),
                       columns=['symbol', 'period', 'ann_dt', 'np'])
    if inc.empty:
        return pd.DataFrame()
    inc['ann_dt'] = inc['ann_dt'].astype(str)
    # 首次公告日（同报告期多次披露/更正取最早）
    first_ann = inc.groupby(['symbol', 'period'])['ann_dt'].min()
    inc = inc.set_index(['symbol', 'period']).join(first_ann.rename('first'), on=None)
    inc = inc.reset_index()
    inc = inc[inc.ann_dt == inc['first']]
    # 同日多口径行 → 取最大值（合并归母；茅台实测 74,734M 为最大=归母）
    actual = inc.groupby(['symbol', 'period', 'ann_dt'])['np'].max().reset_index()
    actual['actual_wan'] = actual['np'] / 1e4           # 元 → 万元
    actual = actual.drop(columns='np')

    # 一致预期（EST_REPORT_DT 对应报告期；仅取所需列压内存）
    periods = sorted(set(actual.period))
    est_rows = []
    B = 400
    for i in range(0, len(periods), B):
        ph = ','.join(f':p{j}' for j in range(min(B, len(periods) - i)))
        cur.execute(
            f"SELECT S_INFO_WINDCODE, EST_REPORT_DT, EST_DT, NET_PROFIT_AVG "
            f"FROM AShareConsensusData WHERE EST_REPORT_DT IN ({ph}) "
            f"AND NET_PROFIT_AVG IS NOT NULL",
            {f'p{j}': p for j, p in enumerate(periods[i:i + B])})
        est_rows.extend(cur.fetchall())
    est = pd.DataFrame(est_rows, columns=['symbol', 'period', 'est_dt', 'est_wan'])
    if est.empty:
        return pd.DataFrame()
    est['est_dt'] = est['est_dt'].astype(str)

    # 公告前最后快照
    m = actual.merge(est, on=['symbol', 'period'], how='left')
    m = m[(m.est_dt.notna()) & (m.est_dt < m.ann_dt)]
    m = m.sort_values(['symbol', 'period', 'ann_date' if 'ann_date' in m.columns else 'ann_dt', 'est_dt'])
    last_est = m.groupby(['symbol', 'period', 'ann_dt']).tail(1)

    last_est['surprise'] = (
        (last_est.actual_wan - last_est.est_wan) / last_est.est_wan.abs()
    ).clip(-3.0, 3.0)
    out = last_est[last_est.est_wan.abs() >= min_abs_est_wan].copy()
    out['ann_date'] = pd.to_datetime(out.ann_dt, format='%Y%m%d', errors='coerce')
    out = out[out.ann_date.notna()]
    logger.info(f"[events] SUE事件 {d0}~{d1}: {len(out)} 条")
    return out[['symbol', 'ann_date', 'period', 'surprise',
                'actual_wan', 'est_wan']]
