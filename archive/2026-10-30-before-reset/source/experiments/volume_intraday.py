#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ETF 放量研究 · 第2步：分钟级验证（2026-10-16 预注册，见 docs/events研究方案.md）

数据分工（防复权污染）：JQ 5分钟线→日内特征（量/VWAP）；Wind adjclose→收益。
信号：放量日（日频稳健 volz>2）且 收盘强于日内VWAP（"有效承接"）。
交易：T+1 收盘进 → T+5 收盘出，往返成本 0.1%。
判据：信号组5日净收益 − 全样本基线 > +0.3%，信号组 t>2，两期同号。
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import date

import numpy as np
import pandas as pd

from data.layered import LayeredDataSource

START, END = date(2015, 1, 1), date(2026, 10, 9)
ETFS = {'510300.SH': '510300.XSHG', '510500.SH': '510500.XSHG',
        '511010.SH': '511010.XSHG', '518880.SH': '518880.XSHG'}

ds = LayeredDataSource()
cache = ds.preload_batch(list(ETFS), START, END)

rows = []
for wind_sym, jq_sym in ETFS.items():
    df = cache[wind_sym]
    idx = pd.DatetimeIndex(pd.to_datetime(df['date']))
    adj = pd.Series(pd.to_numeric(df['adjclose'], errors='coerce').values, index=idx)
    amt = pd.Series(pd.to_numeric(df['amount'], errors='coerce').values, index=idx)

    # 日频量能 z（同第1步口径）
    lamt = np.log(amt.replace(0, np.nan))
    volz = (lamt - lamt.rolling(20).median()) / \
        ((lamt - lamt.rolling(20).median()).abs().rolling(20).median() * 1.4826 + 1e-9)

    # JQ 5分钟线 → 日内特征
    jq = pd.read_parquet(f'data_cache/jq/{jq_sym.replace(".", "_")}_5m.parquet')
    jq['typ'] = (jq['open'] + jq['high'] + jq['low'] + jq['close']) / 4
    jq['pv'] = jq['typ'] * jq['volume']
    day_grp = jq.groupby(jq.index.date)
    vwap = day_grp['pv'].sum() / day_grp['volume'].sum()          # 日内VWAP
    close_d = day_grp['close'].last()
    tail_share = (jq['volume'].groupby([jq.index.date, jq.index.hour >= 14])
                  .sum().unstack())
    tail30 = tail_share[True] / (tail_share.sum(axis=1))           # 14:00后量占比

    feats = pd.DataFrame({'vwap': vwap, 'close': close_d, 'tail': tail30})
    feats['f2'] = (feats.close - feats.vwap) / feats.vwap          # 收盘 vs VWAP
    feats.index = pd.to_datetime(feats.index)
    tailz = (feats['tail'] - feats['tail'].rolling(20).mean()) / \
        (feats['tail'].rolling(20).std() + 1e-9)

    common = adj.index.intersection(feats.index)
    f5 = adj.shift(-5) / adj - 1                                    # T→T+5（口径同持有期）
    f5_entry = adj.shift(-5) / adj.shift(-1) - 1                    # T+1→T+5（真实成交窗口）
    for t in common:
        rows.append({
            'sym': wind_sym, 'date': t,
            'volz': volz.get(t, np.nan), 'f2': feats.loc[t, 'f2'],
            'tailz': tailz.get(t, np.nan),
            'f5_entry': f5_entry.get(t, np.nan),
            'era': '17-21' if t < pd.Timestamp('2022-01-01') else '22-26'})

d = pd.DataFrame(rows).dropna(subset=['volz', 'f2', 'f5_entry'])
d = d[d.date >= '2016-01-01']
print(f"样本: {len(d)} ETF日", flush=True)

sig = d[(d.volz > 2) & (d.f2 > 0)]
base = d.f5_entry.mean()
COST = 0.001

def stats(g, label):
    r = g.f5_entry - COST
    t_ = r.mean() / (r.std() / np.sqrt(len(r))) if len(r) > 5 and r.std() > 0 else 0
    print(f"  {label}: N={len(r)} 净5日 {r.mean():+.3%} (t={t_:.1f})")
    return r.mean(), t_

print(f"\n全样本基线(毛): {base:+.3%} | 信号组（放量+收强于VWAP）:")
print(f"{'=' * 68}")
for era in ('all', '17-21', '22-26'):
    dd = d if era == 'all' else d[d.era == era]
    ss = sig if era == 'all' else sig[sig.era == era]
    print(f"── {era} ── 基线 {dd.f5_entry.mean():+.3%}")
    stats(ss, '信号(量z>2 & VWAP上方)')
    stats(dd[(dd.volz > 2) & (dd.f2 <= 0)], '对照(量z>2 & VWAP下方)')

m_all, t_all = stats(sig, '全期信号')
m1, _ = stats(sig[sig.era == '17-21'], '信号17-21')
m2, _ = stats(sig[sig.era == '22-26'], '信号22-26')
c_diff = m_all - (d.f5_entry.mean() - COST) > 0.003
c_t = t_all > 2
c_era = m1 > 0 and m2 > 0
print(f"""
{'=' * 68}
预注册判定（净5日 vs 基线差>+0.3% 且 t>2 且 两期同号）
  差额判据: {'✅' if c_diff else '❌'} | 显著性: {'✅' if c_t else '❌'} (t={t_all:.1f})
  | 两期: {'✅' if c_era else '❌'} ({m1:+.2%}/{m2:+.2%})
  综合: {'✅ 分钟级信号存活——设计执行验证' if all([c_diff, c_t, c_era]) else
         '❌ 放量线按预注册整体关闭'}""")
