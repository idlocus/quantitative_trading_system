#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""试炼 016：011×006 并行分仓（预注册见日志；006多头=20日反转低分位等权月度组合）"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import numpy as np
import pandas as pd

from data.layered import LayeredDataSource
from data.st_provider import STProvider

START, END = '2015-06-01', '2026-10-23'
HOLD_START = pd.Period('2017-01', 'M')
ERA1_END = pd.Period('2021-12', 'M')
COST = 0.0025

cache = {}
for fn in os.listdir('data_cache/kline'):
    if not fn.endswith('.parquet'):
        continue
    code, suf = fn[:-8].split('_')
    if (suf == 'SH' and code[0] == '6') or (suf == 'SZ' and code[0] in '03'):
        s = code + '.' + suf
        df = pd.read_parquet(f'data_cache/kline/{fn}')
        if 'adjclose' in df.columns and len(df):
            df['date'] = pd.to_datetime(df['date'])
            cache[s] = df
px = pd.DataFrame({s: d.set_index('date')['adjclose'].astype(float)
                   for s, d in cache.items()}).sort_index()
px = px[(px.index >= START) & (px.index <= END)]
me = px.groupby(px.index.to_period('M')).last()
me_ff = px.ffill().groupby(px.index.to_period('M')).last()
fwd = me_ff.shift(-1) / me - 1
listed = me.notna() & me.shift(12).notna()
rev20 = (px / px.shift(20) - 1).groupby(px.index.to_period('M')).last()

ds = LayeredDataSource()
st = STProvider(ds).st_matrix(pd.DatetimeIndex(px.index), list(px.columns))
st_m = st.groupby(st.index.to_period('M')).max()

# 006 多头（20日反转低分位，等权，月度，扣换手成本）
long_r, idx, prev = [], [], set()
for m in me.index:
    if not (HOLD_START <= m <= pd.Period('2026-09', 'M')):
        continue
    f = rev20.loc[m] if m in rev20.index else None
    if f is None:
        continue
    r = fwd.loc[m]
    uni = f.index[f.notna() & listed.loc[m].reindex(f.index).fillna(False)
                  & me.loc[m].reindex(f.index).notna().values]
    if m in st_m.index:
        stt = st_m.loc[m].reindex(uni).fillna(False)
        uni = uni[~stt]
    if len(uni) < 500:
        continue
    q_lo = set(f[uni].rank(pct=True)[lambda x: x <= 0.1].index)
    g = float(r[list(q_lo)].mean())
    t_lo = 1 - len(q_lo & prev) / len(q_lo) if prev else 1.0
    prev = q_lo
    long_r.append(g - COST * t_lo)
    idx.append(m)
sleeve = pd.Series(long_r, index=idx)
print(f'006多头sleeve: {len(sleeve)} 月 | 年化 {float((1+sleeve).prod()**(12/len(sleeve))-1):+.2%}')
sleeve.to_csv('ai-created/trial016_sleeve_monthly.csv', encoding='utf-8-sig')

# 011 C 曲线 → 月度
eq = pd.read_csv('ai-created/trial011_equity_c.csv', index_col=0, parse_dates=True).iloc[:, 0]
base_m = eq.groupby(eq.index.to_period('M')).last()
base_ret = base_m.pct_change().dropna()

def blend(w):
    common = base_ret.index.intersection(sleeve.index)
    b, s = base_ret[common], sleeve[common]
    # 月度再平衡混合
    mixed = (1 - w) * b + w * s
    return mixed

def ann(x):
    return float((1 + x).prod() ** (12 / len(x)) - 1)

def sharpe(x):
    return float(x.mean() / x.std() * np.sqrt(12)) if x.std() > 0 else np.nan

def mdd(x):
    c = (1 + x).cumprod()
    return float((c / c.cummax() - 1).min())

print(f'\n基线011: 年化 {ann(base_ret):+.2%} 夏普 {sharpe(base_ret):.2f} 回撤 {mdd(base_ret):.1%}')
best = None
for w in (0.1, 0.2, 0.3):
    m = blend(w)
    e1 = m[(m.index >= HOLD_START) & (m.index <= ERA1_END)]
    e2 = m[m.index > ERA1_END]
    corr = float(base_ret.corr(sleeve))
    print(f'w={w:.0%}: 年化 {ann(m):+.2%} 夏普 {sharpe(m):.2f} 回撤 {mdd(m):.1%} '
          f'| e1 {ann(e1):+.2%} e2 {ann(e2):+.2%}')
    if best is None or sharpe(m) > best[1]:
        best = (w, sharpe(m), m)
w, sh, m = best
e1 = m[(m.index >= HOLD_START) & (m.index <= ERA1_END)]
e2 = m[m.index > ERA1_END]
ok = sharpe(m) > 0.51 and ann(e1) > 0 and ann(e2) > 0 and mdd(m) >= -0.31
print(f'\n相关性(011×006): {corr:+.2f}')
print(f'判据(最优w={w:.0%}: 夏普>0.51 两期正 回撤≤31%): {"PASS" if ok else "FAIL"}')

from datetime import datetime
reg_p = 'results/data/trial_registry.json'
reg = json.load(open(reg_p, encoding='utf-8'))
reg['trials'].append({'label': '016-并行分仓(011×006)', 'ran_at': datetime.now().isoformat(timespec='seconds'),
                      'best_w': w, 'sharpe': sh, 'corr': corr, 'verdict': 'PASS' if ok else 'FAIL'})
reg['count'] = len(reg['trials'])
tmp = reg_p + '.tmp'
with open(tmp, 'w', encoding='utf-8') as f:
    json.dump(reg, f, ensure_ascii=False, indent=1)
os.replace(tmp, reg_p)
print(f'登记册: {reg["count"]} 条')
