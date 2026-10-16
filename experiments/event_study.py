#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
财报事件研究 P1：业绩预告公告后的行业内超额收益（docs/events研究方案.md）

预注册判据：正向事件组合 20 日行业内超额 > +0.5% 且负向对称成立 → H1 存活。
方法学：T+1 收盘入场（公告日收盘后才可见）；基准=PIT 万得 L2 行业内等权
（全宇宙，非事件股）；时代分层 2017-21 / 2022-26；同股 60 日窗口内重叠取最新。
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
from data.events_pit import profit_notice_events

START, END = date(2016, 6, 1), date(2026, 10, 9)
WINDOWS = (5, 10, 20, 60)          # 交易日前瞻窗口

ds = LayeredDataSource()

# ── 事件（2017 起统计；2016 下半年留给窗口起算） ──
events = profit_notice_events(ds, date(2017, 1, 1), END)
events = events[events.ann_date <= pd.Timestamp(END)]
print(f"事件: {len(events)} 条", flush=True)

# ── 价格矩阵（全缓存股票——行业内基准需要全部成员） ──
print("加载全市场价格 …", flush=True)
loader = DataLoader.__new__(DataLoader)
loader.data_source_type = 'layered'
loader.data_source = ds
loader._cache = None
cache = loader.preload_klines(sorted(ds.list_symbols(as_of=date(2026, 10, 9))),
                              START, END)._cache
px = close_matrix(cache, basis='adjusted')
print(f"价格矩阵: {px.shape}", flush=True)
dates = px.index

# ── 月度 PIT 行业分类（事件月查询一次） ──
print("PIT 行业分类（按月）…", flush=True)
all_syms = sorted(set(px.columns) | set(events.symbol))
month_keys = sorted(set((d.year, d.month) for d in events.ann_date))
cls_by_month = {}
for (y, m) in month_keys:
    first = min(d for d in dates if (d.year, d.month) == (y, m))
    cls_by_month[(y, m)] = industry_classification(ds, all_syms, first.date())

# 行业日收益（成员等权，全宇宙）
print("行业日收益矩阵 …", flush=True)
ind_ret_by_month = {}
for (y, m), cls in cls_by_month.items():
    cols = [s for s in cls if s in px.columns]
    sub = px[cols]
    m_mask = (sub.index.year == y) & (sub.index.month == m)
    sub = sub.loc[m_mask]
    if len(sub) < 5:
        continue
    rets = sub.pct_change()
    ser = {s: cls[s] for s in cols if s in rets.columns}
    # 行业×日 等权收益（pandas 3 无 groupby(axis=1)，逐组 mean 更快）
    groups = {}
    for s, ind_code in ser.items():
        groups.setdefault(ind_code, []).append(s)
    ind_ret = pd.DataFrame({ind: rets[cols_].mean(axis=1)
                            for ind, cols_ in groups.items()})
    ind_ret_by_month[(y, m)] = ind_ret

# 连续行业日收益（跨月拼接——事件前瞻窗口跨月，按月切片会把 20/60 日
# 窗口截断成"仅月初入场事件"，判据窗口曾因此建立在有偏子样本上）
ind_ret_all = pd.concat(
    [ind_ret_by_month[k] for k in sorted(ind_ret_by_month)],
    axis=0).sort_index()
ind_ret_all = ind_ret_all.fillna(0.0)   # 行业缺日=按0收益续链（等权重构近似）

# ── 事件研究主循环 ──
def industry_at(dt):
    key = (dt.year, dt.month)
    return cls_by_month.get(key, {})

rows = []
last_event = {}   # symbol → ann_date（60 日重叠剔除）
for _, ev in events.sort_values('ann_date').iterrows():
    sym, t = ev.symbol, ev.ann_date
    prev = last_event.get(sym)
    if prev is not None and (t - prev).days < 60:
        continue                     # 重叠：取最新（旧的跳过）
    last_event[sym] = t

    i = dates.searchsorted(t, side='right')        # T+1 入场（次日）
    if i <= 0 or i >= len(dates):
        continue
    entry = dates[i]
    cls = industry_at(entry)
    ind = cls.get(sym)
    if ind is None:
        continue
    ind_ret = ind_ret_by_month.get((entry.year, entry.month))
    if ind_ret is None or ind not in ind_ret.columns:
        continue

    s = px[sym].loc[entry:] if sym in px.columns else None
    if s is None or len(s) < WINDOWS[-1] + 1:
        continue
    stock_cum = s / s.iloc[0]
    ir = ind_ret_all[ind].loc[entry:] if ind in ind_ret_all.columns else None
    if ir is None or len(ir) < WINDOWS[-1] + 1:
        continue
    ind_cum = (1 + ir).cumprod()
    for w in WINDOWS:
        r_stock = float(stock_cum.iloc[w]) - 1
        r_ind = float(ind_cum.iloc[w]) - 1
        rows.append({'symbol': sym, 'ann_date': t, 'entry': entry,
                     'direction': ev.direction, 'window': w,
                     'excess': r_stock - r_ind,
                     'era': '2017-21' if t < pd.Timestamp('2022-01-01') else '22-26'})

df = pd.DataFrame(rows)
df.to_csv('ai-created/event_study_p1_detail.csv', encoding='utf-8', index=False)
print(f"有效事件×窗口: {len(df)}", flush=True)

# ── 汇总（预注册判据表） ──
print(f"""
{'=' * 70}
P1 事件筛选：业绩预告后的行业内超额（T+1 入场，行业内等权基准）
{'=' * 70}""")
for era in ('all', '2017-21', '22-26'):
    d = df if era == 'all' else df[df.era == era]
    if d.empty:
        continue
    print(f"\n── 时代 {era}（事件数 {d[d.window == 20].shape[0]}）──")
    print(f"{'方向':>4s} {'窗口':>4s} {'均值':>8s} {'中位':>8s} {'t值':>6s} {'N':>6s}")
    for direction, tag in ((1, '预增'), (-1, '预减')):
        for w in WINDOWS:
            x = d[(d.direction == direction) & (d.window == w)]['excess']
            if len(x) < 30:
                continue
            tval = x.mean() / (x.std() / np.sqrt(len(x))) if x.std() > 0 else 0
            print(f"{tag:>4s} {w:>4d}日 {x.mean():>+8.2%} {x.median():>+8.2%} "
                  f"{tval:>6.1f} {len(x):>6d}")

# 预注册判据
d20 = df[(df.window == 20) & (df.direction == 1)]
pos_all = d20.excess.mean()
neg_all = df[(df.window == 20) & (df.direction == -1)].excess.mean()
pos_e1 = d20[d20.era == '2017-21'].excess.mean()
pos_e2 = d20[d20.era == '22-26'].excess.mean()
neg_e1 = df[(df.window == 20) & (df.direction == -1) & (df.era == '2017-21')].excess.mean()
neg_e2 = df[(df.window == 20) & (df.direction == -1) & (df.era == '22-26')].excess.mean()
c_pass = pos_all > 0.005 and neg_all < -0.005
c_consist = pos_e1 > 0 and pos_e2 > 0 and neg_e1 < 0 and neg_e2 < 0
print(f"""
{'=' * 70}
预注册判据（20 日行业内超额）:
  全期: 预增 {pos_all:+.2%}（要求>+0.5%） 预减 {neg_all:+.2%}（要求<-0.5%）
    → {'✅' if c_pass else '❌'} 幅度判据
  两期同号（方法学红线）: 预增 {pos_e1:+.2%}/{pos_e2:+.2%} 预减 {neg_e1:+.2%}/{neg_e2:+.2%}
    → {'✅' if c_consist else '❌'} 一致性判据
  综合判定: {'✅ H1 存活，进入 P2' if (c_pass and c_consist) else
             '⚠️ 幅度过但一致性不过——信号有年代依赖，谨慎（结合 P2 再判）' if c_pass else
             '❌ 未达标——按预注册规则处理'}
{'=' * 70}""")
