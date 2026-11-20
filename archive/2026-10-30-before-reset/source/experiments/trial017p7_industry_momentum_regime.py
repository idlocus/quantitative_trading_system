#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""试炼 017-P7：行业动量 regime 复活（预注册见 017 文档 P7 节）

R1 轮动速度 = 相邻调仓时点动量Top5重合度（overlap≥3/5 慢轮动，≤2/5 快轮动）。
信号/成本/对照与 003a 完全同口径（M=6 主档）。统计单位=月。
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import numpy as np
import pandas as pd
from scipy import stats

OUT = 'data_cache/sw_industry'
INFO = json.load(open(f'{OUT}/industries.json', encoding='utf-8'))
CODES, NAMES = INFO['codes'], INFO['names']

COST_TRADE, FEE_YEAR = 0.0015, 0.006
TOPN = 5
ERA1_END = pd.Period('2021-12', 'M')

# ── 数据（003 同款加载） ──
px = {}
for c in CODES:
    df = pd.read_parquet(f'{OUT}/price_{c}.parquet')
    df['日期'] = pd.to_datetime(df['日期'])
    px[c] = df.set_index('日期')['收盘'].astype(float)
px = pd.DataFrame(px).sort_index()
px = px[(px.index >= '2011-01-01') & (px.index <= '2026-10-23')]
me = px.groupby(px.index.to_period('M')).last()

c300 = pd.read_parquet('data_cache/kline/000300_SH.parquet')
c300['date'] = pd.to_datetime(c300['date'])
s300 = c300.set_index('date')['close'].astype(float).sort_index()
s300 = s300[(s300.index >= '2011-01-01') & (s300.index <= '2026-10-23')]
ma200_m = s300.rolling(200).mean().groupby(s300.index.to_period('M')).last()

hold_months = [m for m in me.index
               if pd.Period('2017-01', 'M') <= m <= pd.Period('2026-10', 'M')]
print(f'行业: {len(CODES)} | 持有月: {len(hold_months)}', flush=True)


def run_momentum(param):
    """003a 同口径动量组合，附逐月 overlap / R2 / R3 / R4。
    返回 DataFrame(index=持有月)：net, bench_net, gross, overlap, slow, r2, r3, r4"""
    rows = []
    prev_sel, prev_w = None, None
    for m in hold_months:
        i = me.index.get_loc(m)
        nxt = me.index[i + 1] if i + 1 < len(me) else None
        if nxt is None or i - param < 0:
            prev_sel = None
            continue
        row_c, row_n = me.iloc[i], me.iloc[i + 1]
        mom = (row_c / me.iloc[i - param] - 1).dropna()
        sel = list(mom.sort_values(ascending=False).index[:TOPN])
        if len(sel) < TOPN:
            prev_sel = None
            continue
        overlap = (len(set(sel) & set(prev_sel)) / TOPN) if prev_sel else np.nan
        # 策略净收益（下月）
        g = float((row_n[sel] / row_c[sel] - 1).mean())
        w = pd.Series(0.0, index=me.columns)
        w[sel] = 1.0 / TOPN
        traded = 1.0 if prev_w is None else float((w - prev_w.reindex(w.index).fillna(0)).abs().sum())
        net = g - COST_TRADE * traded - FEE_YEAR / 12
        # 对照（当月全行业等权，003 同口径）
        univ = row_n.dropna().index
        rb = float((row_n[univ] / row_c[univ] - 1).mean())
        bench_net = rb - FEE_YEAR / 12
        # regime 变量（全部时点 m 已知）
        r2 = bool(s300.groupby(s300.index.to_period('M')).last()[m] >
                  ma200_m.get(m, np.nan))
        r3 = float(mom.std())
        r4 = rows[-1]['gross'] if rows else np.nan   # 上月动量组合当月毛收益
        rows.append({'month': nxt, 'net': net, 'bench_net': bench_net,
                     'gross': g, 'excess': net - bench_net,
                     'overlap': overlap, 'slow': (overlap >= 3 / TOPN) if overlap == overlap else None,
                     'r2': r2, 'r3': r3, 'r4': r4, 'sel': sel})
        prev_sel, prev_w = sel, w
    df = pd.DataFrame(rows).set_index('month')
    return df


def ann(r):
    if not len(r):
        return np.nan
    return float((1 + r).prod()) ** (12 / len(r)) - 1


def report(df, tag):
    has = df.dropna(subset=['overlap'])
    slow = has[has.slow == True]    # noqa: E712
    fast = has[has.slow == False]   # noqa: E712
    t_s = stats.ttest_1samp(slow['excess'], 0)
    t_d = stats.ttest_ind(slow['excess'], fast['excess'], equal_var=False)
    print(f'\n== {tag} | regime 可判月 {len(has)}/{len(df)} '
          f'(慢 {len(slow)} / 快 {len(fast)}) ==')
    for sub, name in ((slow, '慢轮动'), (fast, '快轮动'), (has.dropna(), '全部')):
        if len(sub):
            print(f'  {name}: 月均超额 {sub.excess.mean():+.2%} '
                  f'(t={stats.ttest_1samp(sub.excess, 0).statistic:+.2f} '
                  f'p={stats.ttest_1samp(sub.excess, 0).pvalue:.4f}) | '
                  f'净年化 {ann(sub.net):+.2%} vs 对照 {ann(sub.bench_net):+.2%}')
    print(f'  慢-快差: {slow.excess.mean()-fast.excess.mean():+.2%}/月 '
          f'(Welch t={t_d.statistic:+.2f} p={t_d.pvalue:.4f})')
    # 分年代（慢态）
    for era, lo, hi in (('2017-21', None, ERA1_END),
                        ('2022-26', pd.Period('2022-01', 'M'), None)):
        seg = slow[(slow.index >= (lo or slow.index[0])) & (slow.index <= (hi or slow.index[-1]))]
        if len(seg):
            print(f'  慢态·{era}: 月均超额 {seg.excess.mean():+.2%}（n={len(seg)}）')
    return slow, fast, t_s, t_d


# ── 主档 M=6 ──
df6 = run_momentum(6)
slow6, fast6, t_s6, t_d6 = report(df6, 'M=6（主判据）')

# 条件化策略：慢轮动月持动量组合、快轮动月持行业等权。
# 收益=按状态取该月动量净收益或对照净收益；状态切换月扣切换成本
# 0.15%×平均权重变化（动量5行业↔全行业等权，近似 |Δw|≈0.8-1.0，取实际
# 上下界声明为近似 0.15%×0.9）
has6 = df6.dropna(subset=['overlap'])
all_months = list(has6.index)
states = has6['slow'].astype(bool)
cond_net = [df6.loc[m, 'net'] if states[m] else df6.loc[m, 'bench_net']
            for m in all_months]
cond = pd.Series(cond_net, index=all_months)
switches = int((states != states.shift(1)).sum()) - 1   # 首次建仓不计
SWITCH_COST = COST_TRADE * 0.9                            # 近似切换成本
switch_total = switches * SWITCH_COST
print(f'\n（条件化状态切换 {switches} 次，近似切换成本合计 {switch_total:.2%}）')
uncond = df6.loc[all_months, 'net']
bench_all = df6.loc[all_months, 'bench_net']
n_years = len(cond) / 12
ann_cond = ann(cond) - switch_total / n_years   # 切换成本按年均摊

print('\n== 条件化策略 vs 无条件动量 vs 行业等权（M=6）==')
print(f'  条件化净年化   {ann_cond:+.2%}')
print(f'  无条件动量净年化 {ann(uncond):+.2%}')
print(f'  行业等权净年化   {ann(bench_all):+.2%}')

# ── 主判据 ──
print('\n== P7 主判据判定 ==')
c1 = (slow6.excess.mean() > 0) and (t_s6.pvalue < 0.05)
c2 = (slow6.excess.mean() - fast6.excess.mean() > 0) and (t_d6.pvalue < 0.05)
c3 = (ann_cond > ann(uncond)) and (ann_cond > ann(bench_all))
print(f'  1) 慢态月均超额>0且p<0.05: {"PASS" if c1 else "FAIL"}'
      f'（{slow6.excess.mean():+.2%}, p={t_s6.pvalue:.4f}）')
print(f'  2) 慢-快差>0且p<0.05:     {"PASS" if c2 else "FAIL"}'
      f'（{slow6.excess.mean()-fast6.excess.mean():+.2%}, p={t_d6.pvalue:.4f}）')
print(f'  3) 条件化>无条件且>对照:   {"PASS" if c3 else "FAIL"}'
      f'（{ann_cond:+.2%} vs {ann(uncond):+.2%} vs {ann(bench_all):+.2%}）')
verdict = c1 and c2 and c3
print(f'  ==> P7（行业动量×轮动速度regime）: {"PASS" if verdict else "FAIL"}')

# ── 稳健性 M=4/8 慢态同号 ──
print('\n== 稳健性（降级）：M=4/8 慢态月均超额 ==')
for p in (4, 8):
    dfp = run_momentum(p)
    hp = dfp.dropna(subset=['overlap'])
    sp = hp[hp.slow == True]   # noqa: E712
    print(f'  M={p}: 慢态 {sp.excess.mean():+.2%}/月（n={len(sp)}，'
          f'净年化 {ann(sp.net):+.2%}）')

# ── 辅助发现性 R2/R3/R4 ──
print('\n== 辅助（不进判定）：R2趋势 / R3离散度 / R4动量上月收益 二分 ==')
med3, med4 = has6.r3.median(), has6.r4.median()
for name, mask in (('R2·趋势上', has6.r2 == True),
                   ('R2·趋势下', has6.r2 == False),
                   ('R3·离散高', has6.r3 >= med3), ('R3·离散低', has6.r3 < med3),
                   ('R4·上月赚', has6.r4 > 0), ('R4·上月亏', has6.r4 <= 0)):
    sub = has6[mask]
    if len(sub) > 10:
        print(f'  {name}: 月均超额 {sub.excess.mean():+.2%}（n={len(sub)}）')

# ── 落盘 + 登记 ──
df6.drop(columns=['sel']).to_csv('ai-created/trial017p7_monthly.csv', encoding='utf-8-sig')
from datetime import datetime
reg_p = 'results/data/trial_registry.json'
reg = json.load(open(reg_p, encoding='utf-8'))
reg['trials'].append({'label': '017-P7行业动量×轮动regime',
                      'ran_at': datetime.now().isoformat(timespec='seconds'),
                      'slow_excess_m': float(slow6.excess.mean()),
                      'fast_excess_m': float(fast6.excess.mean()),
                      'cond_ann': float(ann_cond),
                      'verdict': 'PASS' if verdict else 'FAIL'})
reg['count'] = len(reg['trials'])
tmp = reg_p + '.tmp'
with open(tmp, 'w', encoding='utf-8') as f:
    json.dump(reg, f, ensure_ascii=False, indent=1)
os.replace(tmp, reg_p)
print(f'\n登记册: {reg["count"]} 条 | 明细: ai-created/trial017p7_monthly.csv')
