#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
财报事件研究 P2：一致预期超预期（SUE）后的行业内超额（docs/events研究方案.md）

预注册判据：SUE 多空组（月队列内 Q5−Q1）20 日行业内差 > +0.5% 且两期同号。
方法学与 P1 一致：T+1 收盘入场、PIT 万得 L2 行业内等权基准（全宇宙）、
连续行业日收益（跨月）、时代分层、同股 60 日重叠取最新。
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import date

import numpy as np
import pandas as pd

from data.layered import LayeredDataSource
from data.base.data_loader import DataLoader
from data.semantics import close_matrix
from data.industry_pit import industry_classification
from data.events_pit import sue_events

START, END = date(2016, 6, 1), date(2026, 10, 9)
WINDOWS = (5, 10, 20, 60)

ds = LayeredDataSource()
events = sue_events(ds, date(2017, 1, 1), END)
events = events[events.ann_date <= pd.Timestamp(END)]
print(f"SUE 事件: {len(events)} 条 | surprise 分布: "
      f"p10={events.surprise.quantile(0.1):.2f} 中位={events.surprise.median():.2f} "
      f"p90={events.surprise.quantile(0.9):.2f}", flush=True)

# 公告月队列内五分位（截面排名——SUE 的时变基线不跨期可比）
events['ym'] = events.ann_date.dt.to_period('M')
events['sue_q'] = events.groupby('ym')['surprise'].transform(
    lambda s: pd.qcut(s.rank(method='first'), 5, labels=False) + 1
    if len(s) >= 50 else None)
events = events[events.sue_q.notna()]
events['sue_q'] = events.sue_q.astype(int)
print(f"分位后事件: {len(events)}（Q5={int((events.sue_q == 5).sum())} "
      f"Q1={int((events.sue_q == 1).sum())}）", flush=True)

# ── 价格与行业（与 P1 同口径） ──
print("加载全市场价格 …", flush=True)
loader = DataLoader.__new__(DataLoader)
loader.data_source_type = 'layered'
loader.data_source = ds
loader._cache = None
cache = loader.preload_klines(sorted(ds.list_symbols(as_of=date(2026, 10, 9))),
                              START, END)._cache
px = close_matrix(cache, basis='adjusted')
dates = px.index
print(f"价格矩阵: {px.shape}", flush=True)

print("PIT 行业分类（按月）…", flush=True)
all_syms = sorted(set(px.columns) | set(events.symbol))
month_keys = sorted(set((d.year, d.month) for d in events.ann_date))
cls_by_month, ind_ret_by_month = {}, {}
for (y, m) in month_keys:
    first = min(d for d in dates if (d.year, d.month) == (y, m))
    cls = industry_classification(ds, all_syms, first.date())
    cls_by_month[(y, m)] = cls
    cols = [s for s in cls if s in px.columns]
    sub = px[cols]
    sub = sub.loc[(sub.index.year == y) & (sub.index.month == m)]
    if len(sub) < 5:
        continue
    groups = {}
    for s in cols:
        groups.setdefault(cls[s], []).append(s)
    rets = sub.pct_change()
    ind_ret_by_month[(y, m)] = pd.DataFrame(
        {ind: rets[c].mean(axis=1) for ind, c in groups.items()})
ind_ret_all = pd.concat([ind_ret_by_month[k] for k in sorted(ind_ret_by_month)],
                        axis=0).sort_index().fillna(0.0)

# ── 事件主循环（与 P1 修正版同语义） ──
rows = []
last_event = {}
for _, ev in events.sort_values('ann_date').iterrows():
    sym, t = ev.symbol, ev.ann_date
    prev = last_event.get(sym)
    if prev is not None and (t - prev).days < 60:
        continue
    last_event[sym] = t
    i = dates.searchsorted(t, side='right')
    if i <= 0 or i >= len(dates):
        continue
    entry = dates[i]
    cls = cls_by_month.get((entry.year, entry.month), {})
    ind = cls.get(sym)
    if ind is None or ind not in ind_ret_all.columns:
        continue
    s = px[sym].loc[entry:] if sym in px.columns else None
    ir = ind_ret_all[ind].loc[entry:]
    if s is None or ir is None or len(s) < WINDOWS[-1] + 1 or len(ir) < WINDOWS[-1] + 1:
        continue
    stock_cum = s / s.iloc[0]
    ind_cum = (1 + ir).cumprod()
    for w in WINDOWS:
        rows.append({'symbol': sym, 'ann_date': t, 'sue_q': ev.sue_q,
                     'surprise': ev.surprise, 'window': w,
                     'excess': float(stock_cum.iloc[w]) - float(ind_cum.iloc[w]),
                     'era': '2017-21' if t < pd.Timestamp('2022-01-01') else '22-26'})

df = pd.DataFrame(rows)
df.to_csv('ai-created/event_study_p2_detail.csv', encoding='utf-8', index=False)
print(f"有效事件×窗口: {len(df)}", flush=True)

print(f"""
{'=' * 70}
P2 事件筛选：SUE 五分位组的行业内超额（T+1 入场，行业内等权基准）
{'=' * 70}""")
for era in ('all', '2017-21', '22-26'):
    d = df if era == 'all' else df[df.era == era]
    if d.empty:
        continue
    print(f"\n── 时代 {era}（20日窗口事件数 {d[d.window == 20].shape[0]}）──")
    print(f"{'组':>4s} {'窗口':>4s} {'均值':>8s} {'中位':>8s} {'t值':>6s} {'N':>6s}")
    for q in (5, 3, 1):
        for w in WINDOWS:
            x = d[(d.sue_q == q) & (d.window == w)]['excess']
            if len(x) < 30:
                continue
            tval = x.mean() / (x.std() / np.sqrt(len(x))) if x.std() > 0 else 0
            print(f"Q{q:<3d} {w:>4d}日 {x.mean():>+8.2%} {x.median():>+8.2%} "
                  f"{tval:>6.1f} {len(x):>6d}")

# 预注册判据：Q5−Q1 的 20 日行业内差
def spread(d):
    a = d[(d.sue_q == 5) & (d.window == 20)]['excess']
    b = d[(d.sue_q == 1) & (d.window == 20)]['excess']
    return a.mean() - b.mean() if len(a) > 30 and len(b) > 30 else float('nan')

sp_all = spread(df)
sp_e1 = spread(df[df.era == '2017-21'])
sp_e2 = spread(df[df.era == '22-26'])
c_pass = sp_all > 0.005
c_consist = sp_e1 > 0 and sp_e2 > 0
print(f"""
{'=' * 70}
预注册判据（Q5−Q1 的 20 日行业内差）:
  全期: {sp_all:+.2%}（要求 > +0.5%） → {'✅' if c_pass else '❌'}
  两期: {sp_e1:+.2%} / {sp_e2:+.2%} → {'✅ 同号' if c_consist else '❌ 不同号'}
  综合: {'✅ H2 存活，进入 P3 因子化' if (c_pass and c_consist) else
         '❌ 未达标——按预注册规则：事件线关闭，重心回工程/ETF'}
{'=' * 70}""")
