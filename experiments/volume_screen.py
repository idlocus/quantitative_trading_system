#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ETF 放量研究 · 第1步：日频量能筛选（2026-10-16 预注册）

问题：量能信息在日频粒度是否携带前瞻信息（为分钟级研究定先验）。

设计（预注册）：
  量能异常 volz = (log(amount) − 20日log中位数) / 20日MAD（稳健z）
  趋势状态 trend = 126日复权收益 > 0（与趋势引擎同规则）
  A 条件表：trend∈{上,下} × volz∈{高≥1,低<1} 的前瞻5/20日收益（4ETF×日样本）
  B 放量事件：volz>2 的日子按当日涨跌分组，次日/次5日收益
  判据：
    A) 趋势向上组内，高量 vs 低量的前瞻收益差 >0 且两期同号 → 日频量能有信息
    B) 放量上涨日的次日收益显著为正（t>2）且两期同号 → 事件方向性存在
  分流（预注册）：日频通过→分钟级精细确认；日频不通过→分钟级仍做一次
  （时段内结构是分钟数据独有信息），两者皆负→放量线整体关闭。
数据：Wind 日频缓存（adjclose 收益口径 / amount 量能），复用已验证管道。
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import date

import numpy as np
import pandas as pd

from data.layered import LayeredDataSource

START, END = date(2015, 1, 1), date(2026, 10, 15)
ETFS = {'510300.SH': '沪深300', '510500.SH': '中证500',
        '511010.SH': '国债', '518880.SH': '黄金'}

ds = LayeredDataSource()
cache = ds.preload_batch(list(ETFS), START, END)

frames = []
for sym in ETFS:
    df = cache[sym].copy()
    idx = pd.DatetimeIndex(pd.to_datetime(df['date']))
    px = pd.Series(pd.to_numeric(df['adjclose'], errors='coerce').values, index=idx)
    amt = pd.Series(pd.to_numeric(df['amount'], errors='coerce').values, index=idx)
    lamt = np.log(amt.replace(0, np.nan))
    med = lamt.rolling(20).median()
    mad = (lamt - med).abs().rolling(20).median()
    volz = (lamt - med) / (mad * 1.4826 + 1e-9)        # 稳健z
    trend = (px / px.shift(126) - 1) > 0
    f5 = px.shift(-5) / px - 1
    f20 = px.shift(-20) / px - 1
    f1 = px.shift(-1) / px - 1
    frames.append(pd.DataFrame({
        'sym': sym, 'px': px, 'ret1': px / px.shift(1) - 1,
        'f1': f1, 'f5': f5, 'f20': f20,
        'volz': volz, 'trend': trend}).dropna(subset=['volz', 'trend']))
d = pd.concat(frames)
d = d[d.index >= '2016-01-01']                          # 126日预热后
d['era'] = np.where(d.index < '2022-01-01', '17-21', '22-26')
d['hi_vol'] = d.volz >= 1.0
print(f"样本: {len(d)} ETF日 | 放量事件(volz>2): {int((d.volz > 2).sum())} 天", flush=True)

def cell(g, col):
    return g[col].mean(), len(g)

print(f"""
{'=' * 74}
A. 条件表：trend × volz 的前瞻收益（均值 | N）
{'=' * 74}""")
for era in ('all', '17-21', '22-26'):
    dd = d if era == 'all' else d[d.era == era]
    print(f"\n── {era} ──  （5日前瞻 / 20日前瞻）")
    for tr in (True, False):
        for hv in (True, False):
            g = dd[(dd.trend == tr) & (dd.hi_vol == hv)]
            m5, n = cell(g, 'f5')
            m20, _ = cell(g, 'f20')
            print(f"  趋势{'↑' if tr else '↓'} 量{'高' if hv else '低'}: "
                  f"{m5:+.2%}/{m20:+.2%}  N={n}")

print(f"""
{'=' * 74}
B. 放量事件（volz>2）按当日方向：次日 / 次5日收益
{'=' * 74}""")
for era in ('all', '17-21', '22-26'):
    dd = d if era == 'all' else d[d.era == era]
    ev = dd[dd.volz > 2]
    up = ev[ev.ret1 > 0]
    dn = ev[ev.ret1 <= 0]
    base = dd.f1.mean()
    print(f"\n── {era} ── 基线次日 {base:+.3%}")
    for tag, g in (('放量上涨', up), ('放量下跌', dn)):
        if len(g) < 30:
            print(f"  {tag}: N={len(g)} 样本不足")
            continue
        m1 = g.f1.mean()
        m5 = g.f5.mean()
        t1 = m1 / (g.f1.std() / np.sqrt(len(g))) if g.f1.std() > 0 else 0
        print(f"  {tag}: 次日 {m1:+.3%}(t={t1:.1f}) 次5日 {m5:+.2%}  N={len(g)}")

# ── 预注册判据 ──
def spread_hivol(dd, tr=True, col='f5'):
    a = dd[(dd.trend == tr) & dd.hi_vol][col]
    b = dd[(dd.trend == tr) & ~dd.hi_vol][col]
    return a.mean() - b.mean()

e1 = d[d.era == '17-21']
e2 = d[d.era == '22-26']
sA_all, sA_1, sA_2 = spread_hivol(d), spread_hivol(e1), spread_hivol(e2)
up_all = d[(d.volz > 2) & (d.ret1 > 0)]
upv1 = up_all.f1.mean()
upv1_1 = up_all[up_all.era == '17-21'].f1.mean() if len(up_all[up_all.era == '17-21']) > 30 else float('nan')
upv1_2 = up_all[up_all.era == '22-26'].f1.mean() if len(up_all[up_all.era == '22-26']) > 30 else float('nan')
cA = sA_all > 0 and sA_1 > 0 and sA_2 > 0
cB = upv1 > 0 and upv1_1 > 0 and upv1_2 > 0
print(f"""
{'=' * 74}
预注册判定
{'=' * 74}
A 趋势↑内高量−低量（5日前瞻差）: 全期 {sA_all:+.3%} | {sA_1:+.3%} / {sA_2:+.3%}
  → {'✅ 两期同号为正' if cA else '❌'}
B 放量上涨日次日收益: 全期 {upv1:+.3%} | {upv1_1:+.3%} / {upv1_2:+.3%}
  → {'✅ 两期同号为正' if cB else '❌'}
综合: {'✅ 日频量能携带信息——进入分钟级精细验证' if (cA or cB) else
       '❌ 日频无信号——按预注册：分钟级仍验证一次（时段内结构为分钟独有信息），双负则放量线关闭'}""")
