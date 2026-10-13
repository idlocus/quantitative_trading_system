#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
指数缓存重建 —— 单一数据源（DB AIndexEODPrices，查询已修复）端到端重建，
消除多源拼接的点位断层（2024-09-09 曾出现 -35% 假跳变），并做连续性校验：
|单日涨跌|>20% 即判不连续（真实指数单日极限远低于此）。
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import date, timedelta

import pandas as pd

from data.layered import LayeredDataSource

INDEXES = ['000300.SH', '000905.SH', '000852.SH']


def rebuild(ds, sym, d0, d1):
    db = ds._db_src()
    cur = db._cursor()
    cur.execute(
        "SELECT TRADE_DT, CAST(S_DQ_OPEN AS FLOAT), CAST(S_DQ_HIGH AS FLOAT), "
        "CAST(S_DQ_LOW AS FLOAT), CAST(S_DQ_CLOSE AS FLOAT), "
        "CAST(S_DQ_VOLUME AS FLOAT)*100, CAST(S_DQ_AMOUNT AS FLOAT)*1000 "
        "FROM AIndexEODPrices WHERE S_INFO_WINDCODE = :1 "
        "AND TRADE_DT BETWEEN :2 AND :3 ORDER BY TRADE_DT",
        {'1': sym, '2': d0.strftime('%Y%m%d'), '3': d1.strftime('%Y%m%d')})
    rows = cur.fetchall()
    cur.close()
    if not rows:
        print(f"{sym}: DB无数据")
        return None
    df = pd.DataFrame(rows, columns=['dt', 'open', 'high', 'low', 'close',
                                     'volume', 'amount'])
    df['date'] = pd.to_datetime(df['dt'], format='%Y%m%d')
    df = df[['date', 'open', 'high', 'low', 'close', 'volume', 'amount']]
    df = df.dropna(subset=['open', 'high', 'low', 'close', 'volume'])
    return df


def main():
    ds = LayeredDataSource()
    d0, d1 = date(2016, 1, 1), date.today()
    for sym in INDEXES:
        path = ds._kline_path(sym)
        df = rebuild(ds, sym, d0, d1)
        if df is None:
            continue
        chg = df['close'].pct_change().abs()
        jumps = df.loc[chg > 0.20, ['date', 'close']]
        n_gap = int((df['date'].diff() > pd.Timedelta(days=15)).sum())
        if len(jumps) or n_gap:
            print(f"{sym}: ⚠️ 不连续！跳变{len(jumps)}处 长缺口{n_gap}处")
            print(jumps.to_string())
            continue
        df.to_parquet(path, index=False)
        print(f"{sym}: 重建 {len(df)} 行 {df.date.min().date()}~{df.date.max().date()} "
              f"最大单日|涨跌| {chg.max():.2%} ✓连续")


if __name__ == '__main__':
    main()
