#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
行业龙头溢价检验（用户问题：追行业龙头曾经赚钱，现在还存在吗？）

设计（预注册）：
  月度，T-1 收盘市值在 PIT 万得 L2 行业内排名：
    龙头=行业内市值第1名；头部=行业内市值前10%分位；尾部=后50%
  度量：T+1 入场，20/60 日收益 − 行业内等权（与 P1/P2 同口径的行业内超额）
  分期：2017-21 / 2022-26 / 2024-26（"现在的市场"）
  判定：龙头/头部行业内超额显著>0 且分期不翻号 → 溢价存在
池：全部缓存股票（含已退市，幸存者偏差最小化；2017-2026 间上市即退市的
  小盘缺失，只影响尾部组的精度，不影响龙头侧结论）。
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

START, END = date(2016, 12, 1), date(2026, 10, 9)
WINDOWS = (20, 60)

ds = LayeredDataSource()

# ── 全缓存股票价格 ──
syms_all = sorted(f[:-8].replace('_', '.')
                  for f in os.listdir('data_cache/kline') if f.endswith('.parquet'))
print(f"缓存股票: {len(syms_all)} 只（含已退市）", flush=True)
loader = DataLoader.__new__(DataLoader)
loader.data_source_type = 'layered'
loader.data_source = ds
loader._cache = None
cache = loader.preload_klines(syms_all, START, END)._cache
px = close_matrix(cache, basis='adjusted')
dates = px.index
print(f"价格矩阵: {px.shape}", flush=True)

month_starts, seen = [], set()
for d in dates:
    m = (d.year, d.month)
    if m not in seen:
        seen.add(m)
        month_starts.append(d)
month_starts = [d for d in month_starts
                if d >= pd.Timestamp('2017-01-01')
                and dates.searchsorted(d) + WINDOWS[-1] + 1 < len(dates)]

# ── 月度市值（T-1 日）──
print("拉取月度市值 …", flush=True)
cur = ds._db_src()._cursor()
cap_rows = []
tds = [dates[dates.searchsorted(t, side='left') - 1].strftime('%Y%m%d')
       for t in month_starts]        # T-1 收盘市值（无前视）
B = 60
for i in range(0, len(tds), B):
    ph = ','.join(f':d{j}' for j in range(min(B, len(tds) - i)))
    cur.execute(
        f"SELECT S_INFO_WINDCODE, TRADE_DT, S_VAL_MV FROM AShareEODDerivativeIndicator "
        f"WHERE TRADE_DT IN ({ph}) AND S_VAL_MV IS NOT NULL",
        {f'd{j}': d for j, d in enumerate(tds[i:i + B])})
    cap_rows.extend(cur.fetchall())
cap = pd.DataFrame(cap_rows, columns=['symbol', 'td', 'mv'])
cap_map = {(r.symbol, r.td): r.mv for r in cap.itertuples()}
print(f"市值记录: {len(cap)}", flush=True)

# ── 行业分类与行业收益（P2 同款） ──
print("PIT 行业分类与行业收益 …", flush=True)
cls_by_month, ind_ret_by_month = {}, {}
for t in month_starts:
    key = (t.year, t.month)
    if key in cls_by_month:
        continue
    cls = industry_classification(ds, list(px.columns), t.date())
    cls_by_month[key] = cls
    cols = [s for s in cls if s in px.columns]
    sub = px[cols]
    sub = sub.loc[(sub.index.year == t.year) & (sub.index.month == t.month)]
    if len(sub) < 5:
        continue
    groups = {}
    for s in cols:
        groups.setdefault(cls[s], []).append(s)
    rets = sub.pct_change()
    ind_ret_by_month[key] = pd.DataFrame(
        {ind: rets[c].mean(axis=1) for ind, c in groups.items()})
ind_ret_all = pd.concat([ind_ret_by_month[k] for k in sorted(ind_ret_by_month)],
                        axis=0).sort_index().fillna(0.0)

# ── 主循环 ──
rows = []
for t in month_starts:
    td = tds[month_starts.index(t)]
    cls = cls_by_month.get((t.year, t.month), {})
    # 行业内市值排名
    by_ind = {}
    for s, ind in cls.items():
        mv = cap_map.get((s, td))
        if mv:
            by_ind.setdefault(ind, []).append((s, float(mv)))
    if not by_ind:
        continue
    i = dates.searchsorted(t, side='right')
    if i <= 0 or i + WINDOWS[-1] >= len(dates):
        continue
    entry = dates[i]
    ir_all = ind_ret_all
    for ind, members in by_ind.items():
        if len(members) < 8:           # 行业太小不评龙头
            continue
        if ind not in ir_all.columns:
            continue
        ir = ir_all[ind].loc[entry:]
        if len(ir) < WINDOWS[-1] + 1:
            continue
        ind_cum = (1 + ir).cumprod()
        members.sort(key=lambda x: -x[1])
        n = len(members)
        groups = {'龙头(第1名)': members[:1],
                  '头部(前10%)': members[:max(1, n // 10)],
                  '尾部(后50%)': members[n // 2:]}
        for gname, mem in groups.items():
            rets = []
            for s, _mv in mem:
                ser = px[s].loc[entry:] if s in px.columns else None
                if ser is not None and len(ser) > WINDOWS[-1]:
                    rets.append(float(ser.iloc[WINDOWS[-1]] / ser.iloc[0]) - 1)
            if len(rets) < max(1, len(mem)) * 0.6:
                continue
            for w in WINDOWS:
                rs = []
                for s, _mv in mem:
                    ser = px[s].loc[entry:] if s in px.columns else None
                    if ser is not None and len(ser) > w:
                        rs.append(float(ser.iloc[w] / ser.iloc[0]) - 1)
                if not rs:
                    continue
                r_ind = float(ind_cum.iloc[w]) - 1
                rows.append({'month': t, 'group': gname, 'window': w,
                             'excess': float(np.mean(rs)) - r_ind,
                             'era': ('2017-21' if t < pd.Timestamp('2022-01-01')
                                     else '22-26'),
                             'recent': t >= pd.Timestamp('2024-01-01')})

df = pd.DataFrame(rows)
df.to_csv('ai-created/leader_premium_detail.csv', encoding='utf-8', index=False)
print(f"有效观察: {len(df)}", flush=True)

print(f"""
{'=' * 70}
行业龙头溢价检验（行业内超额 = 组合收益 − 行业内等权，T+1 入场）
{'=' * 70}""")
for era, d in (('全期 2017-2026', df), ('2017-21', df[df.era == '2017-21']),
               ('2022-26', df[df.era == '22-26']),
               ('近两年 2024-26', df[df.recent])):
    if d.empty:
        continue
    print(f"\n── {era} ──")
    print(f"{'组':<12s} {'窗口':>4s} {'均值':>8s} {'中位':>8s} {'t值':>6s} {'N':>5s}")
    for g in ('龙头(第1名)', '头部(前10%)', '尾部(后50%)'):
        for w in WINDOWS:
            x = d[(d.group == g) & (d.window == w)]['excess']
            if len(x) < 20:
                continue
            tval = x.mean() / (x.std() / np.sqrt(len(x))) if x.std() > 0 else 0
            print(f"{g:<12s} {w:>4d}日 {x.mean():>+8.2%} {x.median():>+8.2%} "
                  f"{tval:>6.1f} {len(x):>5d}")
