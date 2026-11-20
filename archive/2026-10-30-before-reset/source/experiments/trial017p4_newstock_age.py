#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""试炼 017-P4：次新股上市时长维度（预注册见 017 文档 P4 节）

月末 age 桶等权月调仓 vs 全市场等权；主判据桶 B2=61-250 上市交易日。
2015 缓存起点前上市的股票 age=∞（防缓存截断造假次新）。
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import numpy as np
import pandas as pd
from scipy import stats

START, END = '2015-01-01', '2026-10-23'
ERA_SPLIT = '2021-01-01'
HOLD_START = '2016-01-01'      # 1年 warmup 后起算
COST = 0.0010                  # 单边（主链费率+滑点口径）

# ---------- 载入（宇宙同 P1：SH 6xx 含科创 / SZ 0/3 开头） ----------
closes, first_dates = {}, {}
for fn in os.listdir('data_cache/kline'):
    if not fn.endswith('.parquet'):
        continue
    code, suf = fn[:-8].split('_')
    if (suf == 'SH' and code[0] == '6') or (suf == 'SZ' and code[0] in '03'):
        sym = code + '.' + suf
        try:
            df = pd.read_parquet(f'data_cache/kline/{fn}')
            if {'close', 'date'} <= set(df.columns) and len(df):
                df['date'] = pd.to_datetime(df['date'])
                df = df[(df['date'] >= START) & (df['date'] <= END)]
                if len(df) > 60:
                    closes[sym] = pd.Series(
                        df['close'].astype(float).values,
                        index=pd.DatetimeIndex(df['date'])).groupby(level=0).last()
                    first_dates[sym] = df['date'].iloc[0]
        except Exception:
            continue
print(f'股票: {len(closes)}', flush=True)

close_mx = pd.DataFrame(closes).sort_index()
# 缓存起点前上市 → age=∞：首个交易日 <= 缓存起点+10日 视为"起点前"
cache_origin = close_mx.index[0]
pre_existing = {s for s, d in first_dates.items() if d <= cache_origin + pd.Timedelta(days=10)}
print(f'缓存起点({cache_origin.date()})前已上市（age=∞）: {len(pre_existing)} 只', flush=True)

age_mx = close_mx.notna().cumsum()
old_cols = [s for s in pre_existing if s in age_mx.columns]
age_mx[old_cols] = np.inf

# 月末矩阵
me = close_mx.groupby(close_mx.index.to_period('M')).last()
age_me = age_mx.groupby(age_mx.index.to_period('M')).last()
months = [m for m in me.index if pd.Period('2016-01', 'M') <= m <= pd.Period('2026-10', 'M')]

BUCKETS = {'B1(1-60)': (1, 60), 'B2(61-250)': (61, 250),
           'B3(251-500)': (251, 500), 'B4(>500)': (501, np.inf)}

rows = []
prev_members = {k: set() for k in BUCKETS}
for i, m in enumerate(months):
    if i + 1 >= len(me):
        break
    nxt = months[i + 1] if months[i + 1] in me.index else me.index[me.index.get_loc(m) + 1]
    r_now, r_nxt = me.loc[m], me.loc[nxt]
    ret = (r_nxt / r_now - 1).dropna()
    age_now = age_me.loc[m].reindex(ret.index)
    univ = ret.index
    bench = float(ret.mean())
    row = {'month': nxt, 'bench': bench, 'n_univ': len(univ)}
    for tag, (lo, hi) in BUCKETS.items():
        members = set(age_now[(age_now >= lo) & (age_now <= hi)].index)
        sub = ret.reindex(sorted(members)).dropna()
        row[f'{tag}_ret'] = float(sub.mean()) if len(sub) >= 10 else np.nan
        row[f'{tag}_n'] = len(sub)
        pm = prev_members[tag]
        if len(pm | members):
            row[f'{tag}_migr'] = 1 - len(pm & members) / len(pm | members)
        prev_members[tag] = members
    rows.append(row)
df = pd.DataFrame(rows).set_index('month')
print(f'持有月: {len(df)} | 全市场平均宽度 {int(df.n_univ.mean())} 只', flush=True)

era = pd.Series(np.where(df.index <= pd.Period('2020-12', 'M'),
                         '2016-2020', '2021-2026'), index=df.index)

# ---------- 报告 ----------
print('\n== 各桶 月均净超额 vs 全市场等权（扣单边0.1%）==')
results = {}
for tag in BUCKETS:
    ex = df[f'{tag}_ret'] - df['bench']
    ex_net = ex - COST
    ok = ex_net.dropna()
    t, p = stats.ttest_1samp(ok, 0)
    e1 = ex_net[era == '2016-2020']
    e2 = ex_net[era == '2021-2026']
    ann_b = float((1 + df[f'{tag}_ret'].dropna()).prod() ** (12 / df[f'{tag}_ret'].dropna().shape[0]) - 1)
    ann_c = float((1 + df['bench']).prod() ** (12 / len(df)) - 1)
    results[tag] = {'mean_net': float(ok.mean()), 'p': float(p),
                    'e1': float(e1.mean()), 'e2': float(e2.mean()),
                    'ann': ann_b, 'bench_ann': ann_c,
                    'migr': float(df[f'{tag}_migr'].mean()),
                    'n_med': int(df[f'{tag}_n'].median())}
    r = results[tag]
    print(f"  {tag:<12} n中位{r['n_med']:>4} | 月均净超额 {r['mean_net']:+.2%} "
          f"(t={t:+.2f} p={p:.4f}) | 年化 {r['ann']:+.1%} vs 对照 {r['bench_ann']:+.1%} "
          f"| 两期 {r['e1']:+.2%}/{r['e2']:+.2%} | 月迁移率 {r['migr']:.0%}")

# 主判据桶 B2 的分年代净年化（实战口径）
b2ret = df['B2(61-250)_ret'].dropna()
for tag_era, lo, hi in (('2016-2020', None, '2020-12'), ('2021-2026', '2021-01', None)):
    seg = b2ret[(b2ret.index >= pd.Period(lo or '2016-01', 'M')) &
                (b2ret.index <= pd.Period(hi or '2026-10', 'M'))]
    segb = df['bench'].reindex(seg.index)
    print(f'  B2 {tag_era}: 净年化 {float((1+seg).prod()**(12/len(seg))-1):+.1%} '
          f'vs 对照 {float((1+segb).prod()**(12/len(segb))-1):+.1%}')

# ---------- 主判据（B2） ----------
print('\n== P4 主判据判定（B2=61-250日次新稳态期）==')
r = results['B2(61-250)']
c1 = r['mean_net'] > 0 and r['p'] < 0.05
c2 = (r['e1'] > 0) and (r['e2'] > 0)
c3 = r['ann'] > r['bench_ann'] + 0.02
print(f"  1) 月均净超额>0且p<0.05: {'PASS' if c1 else 'FAIL'}（{r['mean_net']:+.2%}, p={r['p']:.4f}）")
print(f"  2) 两年代同号:           {'PASS' if c2 else 'FAIL'}（{r['e1']:+.2%} / {r['e2']:+.2%}）")
print(f"  3) 年化>对照+2pp:        {'PASS' if c3 else 'FAIL'}（{r['ann']:+.1%} vs {r['bench_ann']:+.1%}）")
verdict = c1 and c2 and c3
print(f'  ==> P4（次新上市时长）: {"PASS" if verdict else "FAIL"}')

df.to_csv('ai-created/trial017p4_monthly.csv', encoding='utf-8-sig')
from datetime import datetime
reg_p = 'results/data/trial_registry.json'
reg = json.load(open(reg_p, encoding='utf-8'))
reg['trials'].append({'label': '017-P4次新上市时长',
                      'ran_at': datetime.now().isoformat(timespec='seconds'),
                      'B2_mean_net': r['mean_net'], 'B2_p': r['p'],
                      'verdict': 'PASS' if verdict else 'FAIL'})
reg['count'] = len(reg['trials'])
tmp = reg_p + '.tmp'
with open(tmp, 'w', encoding='utf-8') as f:
    json.dump(reg, f, ensure_ascii=False, indent=1)
os.replace(tmp, reg_p)
print(f'\n登记册: {reg["count"]} 条 | 明细: ai-created/trial017p4_monthly.csv')
