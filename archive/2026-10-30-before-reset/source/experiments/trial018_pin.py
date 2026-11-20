#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""试炼 018：插针策略可行性（股票 T+1 版 + 股指期货 T+0 版）"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import numpy as np
import pandas as pd

START, END = '2015-01-01', '2026-10-23'
ERA_SPLIT = pd.Timestamp('2021-01-01')
XS = (0.03, 0.05, 0.08)
COST = 0.0015

print('== 股指期货 ==', flush=True)
fut = {}
try:
    import akshare as ak
    import time
    for code in ('IF0', 'IC0', 'IH0', 'IM0'):
        try:
            df = ak.futures_main_sina(symbol=code, start_date='20150101', end_date='20261027')
            df['日期'] = pd.to_datetime(df['日期'])
            fut[code] = df.set_index('日期')[['开盘价', '最高价', '最低价', '收盘价']].astype(float)
            print(f'{code}: {len(df)} 日 | {df["日期"].min().date()} ~ {df["日期"].max().date()}', flush=True)
            time.sleep(0.5)
        except Exception as e:
            print(f'{code}: 失败 {str(e)[:60]}', flush=True)
except Exception as e:
    print('akshare 不可用:', str(e)[:80], flush=True)

for code, df in fut.items():
    df = df[(df.index >= START) & (df.index <= END)]
    if len(df) < 500:
        continue
    for x in XS:
        L = df['收盘价'].shift(1) * (1 - x)
        filled = df['最低价'] <= L
        fill_px = pd.Series(np.minimum(df['开盘价'].values, L.values), index=df.index)
        r = ((df['收盘价'] / fill_px - 1)[filled] - COST).dropna()
        if len(r) < 10:
            print(f'{code} x={x:.0%}: 成交仅{len(r)}次', flush=True)
            continue
        e1 = r[r.index < ERA_SPLIT]; e2 = r[r.index >= ERA_SPLIT]
        freq = len(r) / 11.8
        m1 = f'{e1.mean():+.2%}' if len(e1) else 'n/a'
        m2 = f'{e2.mean():+.2%}' if len(e2) else 'n/a'
        print(f'{code} x={x:.0%}: n={len(r):>4} ({freq:.0f}次/年) | 均值 {r.mean():+.2%} 中位 {r.median():+.2%} '
              f'P10 {r.quantile(0.1):+.1%} | 年化贡献 {r.mean()*freq:+.1%} | '
              f'e1 {m1}(n={len(e1)}) e2 {m2}(n={len(e2)})', flush=True)

print('\n== 股票（T+1）==', flush=True)
cache = {}
for fn in os.listdir('data_cache/kline'):
    if not fn.endswith('.parquet'):
        continue
    code, suf = fn[:-8].split('_')
    if (suf == 'SH' and code[0] == '6') or (suf == 'SZ' and code[0] in '03'):
        s = code + '.' + suf
        df = pd.read_parquet(f'data_cache/kline/{fn}')
        if {'open', 'high', 'low', 'close', 'date'} <= set(df.columns) and len(df) > 260:
            df['date'] = pd.to_datetime(df['date'])
            df = df[(df['date'] >= START) & (df['date'] <= END)].reset_index(drop=True)
            if len(df) > 260:
                cache[s] = df
print(f'股票: {len(cache)}', flush=True)

from trading import rules
rows = []
for sym, df in cache.items():
    o, h, l, c = (df[k].values for k in ('open', 'high', 'low', 'close'))
    dts = df['date'].values
    n = len(df)
    ratio = rules.price_limit_ratio(sym)
    limit_dn = ratio - 0.003
    prev = np.roll(c, 1); prev[0] = np.nan
    chg_close = c / prev - 1
    for x in XS:
        L = prev * (1 - x)
        filled = (l <= L) & ~np.isnan(L)
        fill_px = np.minimum(o, np.where(np.isnan(L), np.inf, L))
        for i in np.where(filled)[0]:
            if i + 5 >= n:
                continue
            rows.append({'sym': sym, 'date': pd.Timestamp(dts[i]), 'x': x,
                         'fill': fill_px[i],
                         'r_nextopen': o[i+1] / fill_px[i] - 1,
                         'r_nextclose': c[i+1] / fill_px[i] - 1,
                         'r_5d': c[i+5] / fill_px[i] - 1,
                         'sealed': bool(chg_close[i] <= -limit_dn),
                         'era': '2015-2020' if pd.Timestamp(dts[i]) < ERA_SPLIT else '2021-2026'})
ev = pd.DataFrame(rows)
ev.to_csv('ai-created/trial018_stock_events.csv', index=False, encoding='utf-8-sig')
print(f'股票事件: {len(ev)}', flush=True)

for x in XS:
    sub = ev[ev.x == x]
    if not len(sub):
        continue
    print(f'\n--- 股票 x={x:.0%}（成交 {len(sub)} 次）---', flush=True)
    for tag, m in (('全部', sub), ('剔除封死跌停', sub[~sub.sealed]), ('仅封死跌停', sub[sub.sealed])):
        if len(m) < 30:
            continue
        for col in ('r_nextopen', 'r_nextclose', 'r_5d'):
            r = m[col] - COST
            e1 = r[m.era == '2015-2020']; e2 = r[m.era == '2021-2026']
            m1 = f'{e1.mean():+.2%}' if len(e1) else 'n/a'
            m2 = f'{e2.mean():+.2%}' if len(e2) else 'n/a'
            print(f'  {tag} {col}: 均值 {r.mean():+.2%} 中位 {r.median():+.2%} P10 {r.quantile(0.1):+.1%} '
                  f'| e1 {m1}(n={len(e1)}) e2 {m2}(n={len(e2)})', flush=True)

from datetime import datetime
reg_p = 'results/data/trial_registry.json'
reg = json.load(open(reg_p, encoding='utf-8'))
reg['trials'].append({'label': '018-插针可行性', 'ran_at': datetime.now().isoformat(timespec='seconds'),
                      'verdict': '见日志'})
reg['count'] = len(reg['trials'])
tmp = reg_p + '.tmp'
with open(tmp, 'w', encoding='utf-8') as f:
    json.dump(reg, f, ensure_ascii=False, indent=1)
os.replace(tmp, reg_p)
print(f'\n登记册: {reg["count"]} 条 | 明细: ai-created/trial018_stock_events.csv')
