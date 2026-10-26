#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
试炼 003：行业轮动配置（3a 动量版 + 3b 均值回归版）
预注册见 docs/研究日志/003_行业轮动配置.md

规则（预注册，只实现不更改）：
  3a：每月末按过去 M 月收益排序，持有前 5 行业等权，月调仓（M=6 主判据，
      4/8 敏感性）
  3b：每月末按 PE 的 5 年滚动分位排序，持有分位最低 5 行业，月调仓
      （窗 5 年主判据，4/6 年敏感性；PE≤0 视为最贵垫底；估值史≥2年
      方可参与，用可得窗口）
  资格：3a 需 M 月价格史；行业清单按当月实际可得（申万标准 2014/2021
      演进为现实成本，官方指数连续链，跨标准动量/分位照常计算并声明）
  成本（策略与对照同口径）：双边合计 0.15%×Σ|Δw| + ETF 管理托管费拖累
      0.6%/年（月度 0.05%）；对照的月度再平衡换手成本可忽略（近似，声明）
  对照：当月全部有价格行业等权（无轮动能力）
  区间：持有月 2017-01 ~ 2026-10（末月为不完整月，至数据终点 2026-10-23）
  判据：两期（2017-2021 / 2022-2026）净年化超额 vs 对照均 >0（同号）；
      3a 的 M∈{4,6,8} 三档同过；3b 的窗∈{4,5,6} 年三档同过
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import numpy as np
import pandas as pd

OUT = 'data_cache/sw_industry'
INFO = json.load(open(f'{OUT}/industries.json', encoding='utf-8'))
CODES, NAMES = INFO['codes'], INFO['names']

COST_TRADE, FEE_YEAR = 0.0015, 0.006
TOPN = 5
ERA1_END = pd.Period('2021-12', 'M')

# ── 1. 数据 ──
px = {}
for c in CODES:
    df = pd.read_parquet(f'{OUT}/price_{c}.parquet')
    df['日期'] = pd.to_datetime(df['日期'])
    px[c] = df.set_index('日期')['收盘'].astype(float)
px = pd.DataFrame(px).sort_index()
px = px[(px.index >= '2011-01-01') & (px.index <= '2026-10-23')]

val = pd.read_parquet(f'{OUT}/valuation.parquet')
val['发布日期'] = pd.to_datetime(val['发布日期'])
pe = val.pivot_table(index='发布日期', columns='指数代码', values='市盈率').sort_index()
pe = pe[(pe.index >= '2011-01-01') & (pe.index <= '2026-10-23')]
print(f'价格: {px.shape} | PE: {pe.shape}', flush=True)

me = px.groupby(px.index.to_period('M')).last()          # 月末收盘
pe_m = pe.groupby(pe.index.to_period('M')).last()        # 月末PE
pe_p = pe.index.to_period('M')                           # 预计算（性能）

hold_months = [m for m in me.index if pd.Period('2017-01', 'M') <= m <= pd.Period('2026-10', 'M')]
print(f'持有月: {len(hold_months)} 个（{hold_months[0]} ~ {hold_months[-1]}）', flush=True)

def percentile_rank(hist, cur):
    h = hist.dropna().values
    if len(h) < 2:
        return np.nan
    return float((h <= cur).mean())

# ── 2. 月度选行业 ──
def run_version(kind, param):
    """kind='mom' param=M月；kind='val' param=W年。返回月度净收益序列与明细"""
    gross, net, bench_g, bench_n, tos, n_elig, picks = [], [], [], [], [], [], []
    prev_w = None
    for m in hold_months:
        i = me.index.get_loc(m)
        nxt = me.index[i + 1] if i + 1 < len(me) else None
        if nxt is None:
            break  # 末月无下一月末收益（2026-10持仓至终点不计）
        row_c, row_n = me.iloc[i], me.iloc[i + 1]
        if kind == 'mom':
            if i - param < 0:
                continue
            past = me.iloc[i - param]
            mom = row_c / past - 1
            elig = mom.dropna()
            score = elig.sort_values(ascending=False)
            sel = list(score.index[:TOPN])
            n_e = len(elig)
        else:
            lo = m - param * 12
            cur = pe_m.loc[m].dropna() if m in pe_m.index else pd.Series(dtype=float)
            sel, scores = [], {}
            for c in cur.index:
                if c not in me.columns or pd.isna(row_c.get(c)):
                    continue
                hist = pe[c][(pe_p > lo) & (pe_p <= m)]
                hist = hist.dropna()
                if hist.empty:
                    continue
                n_years = (m - hist.index.to_period('M').min()).n / 12
                if n_years < 2:
                    continue
                pe_now = cur[c]
                scores[c] = 10.0 if (pd.isna(pe_now) or pe_now <= 0) else percentile_rank(hist, pe_now)
            score = pd.Series(scores).dropna().sort_values()   # 分位低=便宜
            sel = list(score.index[:TOPN])
            n_e = len(scores)
        if len(sel) < TOPN:
            continue
        # 策略毛收益（下月）
        r_sel = row_n[sel] / row_c[sel] - 1
        g = float(r_sel.mean())
        # 成本
        w = pd.Series(0.0, index=me.columns)
        w[sel] = 1.0 / len(sel)
        is_first = prev_w is None
        traded = 1.0 if is_first else float((w - prev_w.reindex(w.index).fillna(0)).abs().sum())
        cost = COST_TRADE * traded + FEE_YEAR / 12
        prev_w = w
        # 对照（等权全部有价格行业；同口径费率拖累+建仓成本，月度再平衡
        # 换手成本可忽略已声明为近似）
        bench_univ = row_n.dropna().index
        rb = float((row_n[bench_univ] / row_c[bench_univ] - 1).mean())
        gross.append(g); net.append(g - cost)
        bench_g.append(rb)
        bench_n.append(rb - FEE_YEAR / 12 - (COST_TRADE if is_first else 0))
        tos.append(traded); n_elig.append(n_e)
        picks.append((str(nxt), sel))
    idx = [pd.Period(p[0], 'M') for p in picks]
    return (pd.Series(net, index=idx), pd.Series(bench_n, index=idx),
            pd.Series(gross, index=idx), np.mean(tos), np.mean(n_elig), picks)

def ann(r):
    if not len(r):
        return np.nan
    total = float((1 + r).prod())
    return total ** (12 / len(r)) - 1

def evaluate(label, kind, param, verbose=True):
    net, bench, gross, to, ne, picks = run_version(kind, param)
    res = {'label': f'{label}({param})', 'full_net': ann(net), 'full_bench': ann(bench)}
    for era, lo, hi in (('era1', None, ERA1_END), ('era2', pd.Period('2022-01', 'M'), None)):
        seg = net[(net.index >= (lo or net.index[0])) & (net.index <= (hi or net.index[-1]))]
        segb = bench[(bench.index >= (lo or bench.index[0])) & (bench.index <= (hi or bench.index[-1]))]
        res[f'{era}_net'] = ann(seg)
        res[f'{era}_excess'] = ann(seg) - ann(segb)
    if verbose:
        print(f'\n== {label} param={param} | 月均换手 {to:.1%} | 平均参与行业 {ne:.0f} ==')
        print(f"  全期净年化 {res['full_net']:+.2%} vs 对照 {res['full_bench']:+.2%} "
              f"(超额 {res['full_net']-res['full_bench']:+.2%})")
        print(f"  2017-2021: 净 {res['era1_net']:+.2%} 超额 {res['era1_excess']:+.2%} | "
              f"2022-2026: 净 {res['era2_net']:+.2%} 超额 {res['era2_excess']:+.2%}")
    return res, net, bench

print('\n' + '=' * 72)
all_res = {}
for kind, label, params in (('mom', '3a 动量', (4, 6, 8)), ('val', '3b 均值回归', (4, 5, 6))):
    for p in params:
        r, net, bench = evaluate(label, kind, p)
        all_res[(kind, p)] = r
        if (kind, p) == ('mom', 6):
            net6, bench6 = net, bench
        if (kind, p) == ('val', 5):
            net5, bench5 = net, bench

# 分年度（主参数）
for label, net, bench in (('3a(M=6)', net6, bench6), ('3b(W=5y)', net5, bench5)):
    yr = pd.DataFrame({'net': net.groupby(net.index.year).apply(lambda x: (1 + x).prod() - 1),
                       'bench': bench.groupby(bench.index.year).apply(lambda x: (1 + x).prod() - 1)})
    yr['excess'] = yr.net - yr.bench
    print(f'\n{label} 分年度:')
    print(yr.round(4).to_string())

# 判据
def pass_v(kind, params):
    ok = all(all_res[(kind, p)][f'{e}_excess'] > 0 for p in params for e in ('era1', 'era2'))
    detail = {p: (round(all_res[(kind, p)]['era1_excess'], 4), round(all_res[(kind, p)]['era2_excess'], 4))
              for p in params}
    return ok, detail

ok_a, d_a = pass_v('mom', (4, 6, 8))
ok_b, d_b = pass_v('val', (4, 5, 6))
print(f'\n判据 3a（M=4/6/8 两期超额均>0）: {"PASS" if ok_a else "FAIL"} {d_a}')
print(f'判据 3b（窗4/5/6年 两期超额均>0）: {"PASS" if ok_b else "FAIL"} {d_b}')

# ── 3. 明细落盘 + 登记册 ──
pd.DataFrame({'net_3a': net6, 'bench': bench6, 'net_3b': net5}).to_csv(
    'ai-created/trial003_monthly.csv', encoding='utf-8-sig')

from datetime import datetime
reg_p = 'results/data/trial_registry.json'
reg = json.load(open(reg_p, encoding='utf-8'))
reg['trials'].append({'label': '003-行业轮动(3a动量+3b均值回归)',
                      'ran_at': datetime.now().isoformat(timespec='seconds'),
                      '3a_era1_excess': all_res[('mom', 6)]['era1_excess'],
                      '3a_era2_excess': all_res[('mom', 6)]['era2_excess'],
                      '3b_era1_excess': all_res[('val', 5)]['era1_excess'],
                      '3b_era2_excess': all_res[('val', 5)]['era2_excess'],
                      '3a_verdict': 'PASS' if ok_a else 'FAIL',
                      '3b_verdict': 'PASS' if ok_b else 'FAIL'})
reg['count'] = len(reg['trials'])
tmp = reg_p + '.tmp'
with open(tmp, 'w', encoding='utf-8') as f:
    json.dump(reg, f, ensure_ascii=False, indent=1)
os.replace(tmp, reg_p)
print(f'\n登记册: {reg["count"]} 条 | 明细: ai-created/trial003_monthly.csv')
