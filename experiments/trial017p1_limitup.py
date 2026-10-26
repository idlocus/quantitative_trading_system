#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""试炼 017-P1：涨停板次日溢价（首板/二板/三板+，regime拆分）"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import numpy as np
import pandas as pd

from trading import rules

START, END = '2015-01-01', '2026-10-23'
ERA_SPLIT = '2021-01-01'
COST = 0.0015   # 单边

# 加载日线 OHLC
cache = {}
for fn in os.listdir('data_cache/kline'):
    if not fn.endswith('.parquet'):
        continue
    code, suf = fn[:-8].split('_')
    if (suf == 'SH' and code[0] == '6') or (suf == 'SZ' and code[0] in '03'):
        sym = code + '.' + suf
        try:
            df = pd.read_parquet(f'data_cache/kline/{fn}')
            if {'open', 'high', 'low', 'close', 'date'} <= set(df.columns) and len(df):
                df['date'] = pd.to_datetime(df['date'])
                df = df[(df['date'] >= START) & (df['date'] <= END)]
                if len(df) > 250:
                    cache[sym] = df.reset_index(drop=True)
        except Exception:
            continue
print(f'股票: {len(cache)}', flush=True)

idx = pd.read_parquet('data_cache/kline/000300_SH.parquet')
idx['date'] = pd.to_datetime(idx['date'])
c300 = idx.set_index('date')['close']
c300 = c300[(c300.index >= START) & (c300.index <= END)]
ma60 = c300.rolling(60).mean()
regime = (c300 > ma60)   # True=强市

cal = c300.index
calpos = {d: i for i, d in enumerate(cal)}

rows = []
for sym, df in cache.items():
    o, h, c = df['open'].values, df['high'].values, df['close'].values
    dates = df['date'].values
    n = len(df)
    if n < 260:
        continue
    ratio = rules.price_limit_ratio(sym)
    limit_up = ratio - 0.003        # 收盘涨停判定容差
    prev = np.roll(c, 1); prev[0] = np.nan
    chg = c / prev - 1
    is_zt = chg >= limit_up
    # 连板数（截止 T 日）
    streak = 0
    streaks = np.zeros(n, dtype=int)
    for i in range(n):
        streak = streak + 1 if is_zt[i] else 0
        streaks[i] = streak
    for i in range(1, n - 6):
        if not is_zt[i] or streaks[i] > 5:
            continue
        t1 = i + 1
        if t1 >= n:
            continue
        # 可买性：T+1 开盘未一字涨停（open < 当日涨停价近似：open涨幅<limit-0.005）
        chg_o = o[t1] / c[i] - 1
        if chg_o >= ratio - 0.005:
            continue
        r1 = c[t1] / o[t1] - 1
        r5 = c[min(i + 5, n - 1)] / o[t1] - 1
        d = pd.Timestamp(dates[i])
        rows.append({'sym': sym, 'date': d, 'board': min(int(streaks[i]), 3),
                     'era': '2015-2020' if d < pd.Timestamp(ERA_SPLIT) else '2021-2026',
                     'regime': bool(regime.reindex([d], method='ffill').iloc[0]) if d in regime.index or regime.index[0] < d else True,
                     'r1': r1, 'r5': r5})
ev = pd.DataFrame(rows)
ev.to_csv('ai-created/trial017p1_events.csv', index=False, encoding='utf-8-sig')
print(f'事件: {len(ev)}（已剔 T+1 开盘涨停不可买）', flush=True)

def grp(df, tag):
    for col in ('r1', 'r5'):
        x = df[col]
        net1 = x - COST
        yr = len(df) / 11.8
        print(f'  {tag} {col}: n={len(df):>6} | 均值 {x.mean():+.2%} 中位 {x.median():+.2%} '
              f'| 净均值(扣单边) {net1.mean():+.2%} | P25 {x.quantile(0.25):+.1%} P75 {x.quantile(0.75):+.1%}')

print('\n== 按板档 ==')
for b in (1, 2, 3):
    grp(ev[ev.board == b], f'{b}板')
print('\n== 板档 × 年代 ==')
for b in (1, 2, 3):
    for era in ('2015-2020', '2021-2026'):
        grp(ev[(ev.board == b) & (ev.era == era)], f'{b}板 {era}')
print('\n== 板档 × 市场状态（300>MA60）==')
for b in (1, 2, 3):
    for rg, tag in ((True, '强市'), (False, '弱市')):
        grp(ev[(ev.board == b) & (ev.regime == rg)], f'{b}板 {tag}')

# 按年频率×净溢价 = 年化金额（每万元）
print('\n== 年化金额估算（净均值×年频率，每万元本金满仓滚动）==')
for b in (1, 2, 3):
    sub = ev[ev.board == b]
    net = (sub['r1'] - COST).mean()
    freq = len(sub) / 11.8
    print(f'  {b}板: {net:+.2%} × {freq:.0f} 次/年 = {net*freq:+.0%}（理论上限：单事件单仓位，未计资金占用重叠）')

from datetime import datetime
reg_p = 'results/data/trial_registry.json'
reg = json.load(open(reg_p, encoding='utf-8'))
reg['trials'].append({'label': '017-P1涨停次日溢价', 'ran_at': datetime.now().isoformat(timespec='seconds'),
                      'n_events': len(ev), 'verdict': '见日志'})
reg['count'] = len(reg['trials'])
tmp = reg_p + '.tmp'
with open(tmp, 'w', encoding='utf-8') as f:
    json.dump(reg, f, ensure_ascii=False, indent=1)
os.replace(tmp, reg_p)
print(f'\n登记册: {reg["count"]} 条 | 明细: ai-created/trial017p1_events.csv')
