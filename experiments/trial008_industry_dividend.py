#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""试炼 008：行业层红利（股息率前5行业轮动）。框架同 003（成本/对照/口径）。"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import numpy as np
import pandas as pd

OUT = 'data_cache/sw_industry'
COST_TRADE, FEE_YEAR = 0.0015, 0.006
ERA1_END = pd.Period('2021-12', 'M')

px = {}
info = json.load(open(f'{OUT}/industries.json', encoding='utf-8'))
for c in info['codes']:
    df = pd.read_parquet(f'{OUT}/price_{c}.parquet')
    df['日期'] = pd.to_datetime(df['日期'])
    px[c] = df.set_index('日期')['收盘'].astype(float)
px = pd.DataFrame(px).sort_index()
px = px[(px.index >= '2011-01-01') & (px.index <= '2026-10-23')]
me = px.groupby(px.index.to_period('M')).last()

val = pd.read_parquet(f'{OUT}/valuation.parquet')
val['发布日期'] = pd.to_datetime(val['发布日期'])
dv = val.pivot_table(index='发布日期', columns='指数代码', values='股息率').sort_index()
dv = dv[(dv.index >= '2011-01-01') & (dv.index <= '2026-10-23')]
dv_m = dv.groupby(dv.index.to_period('M')).last()

hold_months = [m for m in me.index if pd.Period('2017-01', 'M') <= m <= pd.Period('2026-10', 'M')]

def run(topn):
    net, bench, idx, prev_w, tos = [], [], [], None, []
    for m in hold_months:
        i = me.index.get_loc(m)
        nxt = me.index[i + 1] if i + 1 < len(me) else None
        if nxt is None:
            break
        row_c, row_n = me.iloc[i], me.iloc[i + 1]
        if m not in dv_m.index:
            continue
        d = dv_m.loc[m].dropna()
        d = d[[c for c in d.index if c in me.columns and pd.notna(row_c.get(c))]]
        if len(d) < 10:
            continue
        sel = list(d.sort_values(ascending=False).index[:topn])
        r_sel = row_n[sel] / row_c[sel] - 1
        g = float(r_sel.mean())
        w = pd.Series(0.0, index=me.columns)
        w[sel] = 1.0 / len(sel)
        is_first = prev_w is None
        traded = 1.0 if is_first else float((w - prev_w.reindex(w.index).fillna(0)).abs().sum())
        cost = COST_TRADE * traded + FEE_YEAR / 12
        prev_w = w
        bench_univ = row_n.dropna().index
        rb = float((row_n[bench_univ] / row_c[bench_univ] - 1).mean())
        net.append(g - cost); bench.append(rb - FEE_YEAR / 12 - (COST_TRADE if is_first else 0))
        tos.append(traded); idx.append(nxt)
    return pd.Series(net, index=idx), pd.Series(bench, index=idx), float(np.mean(tos))

def ann(x):
    return float((1 + x).prod() ** (12 / len(x)) - 1) if len(x) else np.nan

res = {}
for topn in (5, 8):
    n, b, to = run(topn)
    e = {'full': ann(n) - ann(b), 'to': to}
    for era, a, z in (('e1', pd.Period('2017-01', 'M'), ERA1_END),
                      ('e2', pd.Period('2022-01', 'M'), n.index[-1])):
        seg = n[(n.index >= a) & (n.index <= z)]
        segb = b[(b.index >= a) & (b.index <= z)]
        e[era] = ann(seg) - ann(segb)
    res[topn] = e
    print(f'前{topn}: 全期净超额 {e["full"]:+.2%} | e1 {e["e1"]:+.2%} e2 {e["e2"]:+.2%} | 月均换手 {to:.0%}')

ok = all(v['e1'] > 0 and v['e2'] > 0 for v in res.values())
print(f'判据（前5/前8 两期超额均>0）: {"PASS" if ok else "FAIL"}')

from datetime import datetime
reg_p = 'results/data/trial_registry.json'
reg = json.load(open(reg_p, encoding='utf-8'))
reg['trials'].append({'label': '008-行业红利(前5/前8)',
                      'ran_at': datetime.now().isoformat(timespec='seconds'),
                      'e1_excess': res[5]['e1'], 'e2_excess': res[5]['e2'],
                      'verdict': 'PASS' if ok else 'FAIL'})
reg['count'] = len(reg['trials'])
tmp = reg_p + '.tmp'
with open(tmp, 'w', encoding='utf-8') as f:
    json.dump(reg, f, ensure_ascii=False, indent=1)
os.replace(tmp, reg_p)
print(f'登记册: {reg["count"]} 条')
