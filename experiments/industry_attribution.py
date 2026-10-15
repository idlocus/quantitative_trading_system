#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
行业归因仪器（进化路线第1步）：剥离行业配置后，策略还剩多少选股 alpha。

数据：既有十年工件（A 口径=定稿重入协议，2017-2026）的 daily_holdings
+ PIT 万得 L2 行业（data/industry_pit.py，30 行业，月频分类）。
方法：月度 Brinson-Fachler，基准=PIT 宇宙等权（与三口径对照的等权同族；
不叠加流动性过滤，归因内部自洽即可）。现金月（重入冷却）按 CASH 伪行业
（收益0）处理。
分解：超额 = Σ(wp_i−wb_i)(rb_i−rb) [配置] + Σ wp_i(rp_i−rb_i) [行业内选股]
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import numpy as np
import pandas as pd

from datetime import date
from data.layered import LayeredDataSource
from data.semantics import close_matrix
from data.industry_pit import industry_classification, industry_names

ART = 'results/data/portfolio_composite_v5_full_2017_2026_5因子生.json'
START, END = date(2016, 6, 1), date(2026, 10, 9)

with open(ART, encoding='utf-8') as f:
    art = json.load(f)
dh = {pd.Timestamp(k): {s: v for s, v in rows} for k, rows in art['daily_holdings'].items()}
eq = pd.Series(art['equity']['values'],
               index=pd.to_datetime(art['equity']['dates']))
print(f"工件: A口径十年 | 净值点 {len(eq)} | 持仓日 {len(dh)}", flush=True)

ds = LayeredDataSource()
pool = sorted(ds.list_symbols(as_of=date(2017, 1, 1)))
print(f"池: {len(pool)} 只", flush=True)

from data.base.data_loader import DataLoader
loader = DataLoader.__new__(DataLoader)
loader.data_source_type = 'layered'
loader.data_source = ds
loader._cache = None
cache = loader.preload_klines(pool + ['000300.SH'], START, END)._cache
px = close_matrix(cache, basis='adjusted')          # R2: 声明复权（收益用）
print(f"价格矩阵: {px.shape[0]}日 × {px.shape[1]}只", flush=True)

names = industry_names(ds, level=2)
dates = eq.index
month_starts, seen = [], set()
for d in dates:
    m = (d.year, d.month)
    if m not in seen:
        seen.add(m)
        month_starts.append(d)
month_starts = [d for d in month_starts if d + pd.offsets.BMonthEnd(1) <= dates[-1]]
print(f"归因月份: {len(month_starts)}（{month_starts[0].date()} ~ {month_starts[-1].date()}）", flush=True)

rows = []
for t in month_starts:
    t_next = dates[dates.searchsorted(t, side='right')]
    # 下月首个交易日（区间终点=下下月初）——月收益区间 [t, next_next)
    i_next = month_starts[month_starts.index(t) + 1] if t != month_starts[-1] else None
    if i_next is None:
        continue
    end_d = dates[dates.searchsorted(i_next, side='left') - 1] if i_next in dates else dates[-1]
    # 宇宙：t 日有价格的池内股票
    px_t = px.loc[t].dropna()
    universe = [s for s in px_t.index if s in set(pool)]
    r_univ = px.loc[end_d, universe] / px.loc[t, universe] - 1      # 月收益
    r_univ = r_univ.dropna()

    cls = industry_classification(ds, universe, t.date())
    ind = pd.Series(cls).reindex(r_univ.index)
    has_ind = ind.notna()
    r_univ, ind = r_univ[has_ind], ind[has_ind]

    # 行业收益（宇宙内等权）
    rb_i = r_univ.groupby(ind).mean()
    wb = ind.value_counts(normalize=True)
    rb = float((wb * rb_i.reindex(wb.index)).sum())

    # 策略持仓 → 行业权重（市值权重按行业归并）；现金=伪行业
    hold = dh.get(t, {})
    if hold:
        hv = pd.Series(hold).astype(float)
        hv = hv[hv > 0]
        exposure = min(1.0, hv.sum() / eq.loc[t])   # 权益仓位（现金月<1）
        hold_ind = pd.Series({s: cls.get(s) for s in hv.index})
        hv_ind = hv.groupby(hold_ind).sum()          # 行业→持仓市值
        uncov_val = float(hv_ind.pop(None)) if None in hv_ind.index else 0.0  # 无分类持仓（极少）
        eq_val = hv.sum()
        wp = hv_ind / eq_val * exposure              # 各行业权益权重
        wp_cash = 1.0 - exposure + uncov_val / eq_val * exposure
    else:
        wp = pd.Series(dtype=float)
        exposure = 0.0
        wp_cash = 1.0

    # 策略行业内收益（策略等权 → 持仓内等权；月内止损/调仓计入残差，注明）
    rp_i = {}
    for i_code in wp.index:
        members = ind.index[ind == i_code]
        held = [s for s in members if s in hold]
        if held:
            rp_i[i_code] = float(r_univ.reindex(held).mean())
    rp_i = pd.Series(rp_i)

    # Brinson-Fachler（现金伪行业 rb=0）
    codes = wb.index.union(wp.index)
    Wp = wp.reindex(codes).fillna(0.0)
    Wp['CASH'] = wp_cash
    Wb = wb.reindex(codes).fillna(0.0)
    Wb['CASH'] = 0.0
    Rb = rb_i.reindex(codes).fillna(0.0)
    Rb['CASH'] = 0.0
    Rp = rp_i.reindex(codes).fillna(0.0)
    Rp['CASH'] = 0.0

    alloc = (Wp - Wb) * (Rb - rb)
    selec = Wp * (Rp - Rb)
    rows.append({'month': t, 'rp': float((Wp * Rp).sum()), 'rb': rb,
                 'alloc': float(alloc.sum()), 'selec': float(selec.sum()),
                 'excess': float(alloc.sum() + selec.sum()),
                 'exposure': exposure,
                 **{f'a_{c}': alloc.get(c, 0.0) for c in wb.index}})

df = pd.DataFrame(rows).set_index('month')
total = (1 + df['rp']).prod() - 1, (1 + df['rb']).prod() - 1
cum_a = (1 + df['alloc']).prod() - 1
cum_s = (1 + df['selec']).prod() - 1

print(f"""
{'='*72}
行业归因（A口径十年 2017-01 ~ 2026-{df.index[-1].month:02d}，月度 Brinson）
{'='*72}
  策略月度复合收益:   {total[0]:+.1%}
  宇宙等权收益:       {total[1]:+.1%}
  超额:               {total[0]-total[1]:+.1%}
  ├─ 行业配置（累计）: {cum_a:+.1%}  （月均 {df['alloc'].mean():+.2%}，正贡献月占比 {(df['alloc']>0).mean():.0%}）
  └─ 行业内选股（累计）: {cum_s:+.1%}  （月均 {df['selec'].mean():+.2%}，正贡献月占比 {(df['selec']>0).mean():.0%}）
  平均权益仓位:       {df['exposure'].mean():.0%}
""", flush=True)

a_cols = [c for c in df.columns if c.startswith('a_')]
alloc_by_ind = df[a_cols].sum().sort_values()
print("行业配置贡献 Top5（正）:")
for c in alloc_by_ind.index[-5:][::-1]:
    code = c[2:]
    print(f"  {names.get(code, code):<14s} {df[c].sum():+.2%}")
print("行业配置贡献 Top5（负）:")
for c in alloc_by_ind.index[:5]:
    code = c[2:]
    print(f"  {names.get(code, code):<14s} {df[c].sum():+.2%}")

df.to_csv('ai-created/industry_attribution_monthly.csv', encoding='utf-8')
print("\n月度明细: ai-created/industry_attribution_monthly.csv")
