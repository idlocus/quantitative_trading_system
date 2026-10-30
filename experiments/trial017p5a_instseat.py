#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""试炼 017-P5a：机构席位 × 涨停事件溢价（预注册见 017 文档 P5a 节）

事件=P1/P2 同款涨停事件 ∩ 当日LHB上榜；分组=机构净买/无机构/机构净卖。
收益自算 r1=c[T+1]/o[T+1]-1（可成交口径，与P1/P2同构）。
统计纪律：日级聚合 Welch t（同日事件横截面相关）。
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import numpy as np
import pandas as pd
from scipy import stats

from trading import rules

START, END = '2015-01-01', '2026-10-23'
ERA_SPLIT = '2021-01-01'
COST = 0.0015

# ---------- K线缓存（宇宙同 P1/P2） ----------
cache = {}
for fn in os.listdir('data_cache/kline'):
    if not fn.endswith('.parquet'):
        continue
    code, suf = fn[:-8].split('_')
    if (suf == 'SH' and code[0] == '6') or (suf == 'SZ' and code[0] in '03'):
        sym = code + '.' + suf
        try:
            df = pd.read_parquet(f'data_cache/kline/{fn}')
            if {'open', 'high', 'low', 'close', 'date'} <= set(df.columns) and len(df):
                df['date'] = pd.to_datetime(df['date'])
                df = df[(df['date'] >= START) & (df['date'] <= END)]
                if len(df) > 250:
                    cache[sym] = df.reset_index(drop=True)
        except Exception:
            continue
print(f'股票: {len(cache)}', flush=True)

# ---------- LHB 表 ----------
det = pd.read_parquet('data_cache/lhb/detail.parquet')
det['date'] = pd.to_datetime(det['上榜日'])
det['sym6'] = det['代码'].astype(str).str.zfill(6)
lhb_keys = set(zip(det['sym6'], det['date']))
inst = pd.read_parquet('data_cache/lhb/inst.parquet')
inst['date'] = pd.to_datetime(inst['上榜日期'])
inst['sym6'] = inst['代码'].astype(str).str.zfill(6)
# 机构净买额（元）；同日同代码取首条（汇总级无重复口径）
inst_net = {(r.sym6, r.date): r.机构买入净额 for r in inst.itertuples()}
print(f'LHB detail: {len(det)} 行 | inst: {len(inst)} 行', flush=True)

# ---------- 事件（P1/P2 同逻辑，只保留当日上榜的涨停事件） ----------
events = []
for sym, df in cache.items():
    o, h, c = df['open'].values, df['high'].values, df['close'].values
    dates = df['date'].values
    n = len(df)
    if n < 260:
        continue
    ratio = rules.price_limit_ratio(sym)
    limit_up = ratio - 0.003
    prev = np.roll(c, 1); prev[0] = np.nan
    is_zt = (c / prev - 1) >= limit_up
    streak = 0
    streaks = np.zeros(n, dtype=int)
    for i in range(n):
        streak = streak + 1 if is_zt[i] else 0
        streaks[i] = streak
    d_ts = [pd.Timestamp(x) for x in dates]
    for i in range(1, n - 6):
        if not is_zt[i] or streaks[i] > 5:
            continue
        d = d_ts[i]
        if (sym.split('.')[0], d) not in lhb_keys:   # ∩ 当日上榜
            continue
        t1 = i + 1
        if o[t1] / c[i] - 1 >= ratio - 0.005:        # T+1 开盘涨停不可买
            continue
        net = inst_net.get((sym.split('.')[0], d), np.nan)
        events.append({'sym': sym, 'date': d,
                       'board': min(int(streaks[i]), 3),
                       'era': '2015-2020' if d < pd.Timestamp(ERA_SPLIT) else '2021-2026',
                       'r1': c[t1] / o[t1] - 1,
                       'r5': c[min(i + 5, n - 1)] / o[t1] - 1,
                       'amt': float(df['amount'].iloc[t1]) if 'amount' in df else np.nan,
                       'inst_net': net})
ev = pd.DataFrame(events)
n_all = len(ev)
ev['group'] = np.where(ev.inst_net.isna(), 'G2无机构',
               np.where(ev.inst_net > 0, 'G1机构净买', 'G3机构净卖'))
print(f'涨停∩上榜事件: {n_all}（G1 {sum(ev.group=="G1机构净买")} / '
      f'G2 {sum(ev.group=="G2无机构")} / G3 {sum(ev.group=="G3机构净卖")}）', flush=True)
ev.to_csv('ai-created/trial017p5a_events.csv', index=False, encoding='utf-8-sig')

# ---------- 报告 ----------
def grp(df, tag, col='r1'):
    x = df[col]
    print(f'  {tag:<14} n={len(df):>6} | 均值 {x.mean():+.2%} 中位 {x.median():+.2%} '
          f'| 净 {x.mean()-COST:+.2%} | P25 {x.quantile(0.25):+.1%} '
          f'P75 {x.quantile(0.75):+.1%}', flush=True)

def daily_ttest(a, b, col='r1'):
    da = a.groupby('date')[col].mean()
    db = b.groupby('date')[col].mean()
    t, p = stats.ttest_ind(da, db, equal_var=False)
    return len(da), len(db), t, p

g1 = ev[ev.group == 'G1机构净买']
g2 = ev[ev.group == 'G2无机构']
g3 = ev[ev.group == 'G3机构净卖']

print('\n== 分组 × r1（净=扣单边0.15%）==')
for df, tag in ((g1, 'G1机构净买'), (g2, 'G2无机构'), (g3, 'G3机构净卖')):
    grp(df, tag)
na, nb, t, p = daily_ttest(g1, g2)
print(f'  G1vsG2（日级聚合 {na}/{nb} 日）: 差 {g1.r1.mean()-g2.r1.mean():+.2%} '
      f'| Welch t={t:.2f} p={p:.4f}', flush=True)

print('\n== 板档 × 分组（r1）==')
for b in (1, 2, 3):
    for g, tag in (('G1机构净买', 'G1'), ('G2无机构', 'G2'), ('G3机构净卖', 'G3')):
        grp(ev[(ev.board == b) & (ev.group == g)], f'{b}板·{tag}')

print('\n== 5日延续（r5）==')
for df, tag in ((g1, 'G1机构净买'), (g2, 'G2无机构'), (g3, 'G3机构净卖')):
    grp(df, tag, col='r5')

print('\n== 稳健性（降级）：两年代同号? ==')
for era in ('2015-2020', '2021-2026'):
    for df, tag in ((g1, 'G1'), (g2, 'G2')):
        grp(df[df.era == era], f'{tag}·{era}')

print('\n== 实战口径：年化金额（每万元满仓滚动理论上限）==')
for df, tag in ((g1, 'G1'), (g2, 'G2'), (g3, 'G3')):
    net = df.r1.mean() - COST
    print(f'  {tag}: {net:+.2%} × {len(df)/10.6:.0f} 次/年 = {net*len(df)/10.6:+.0%}')
print(f'  G1 事件日成交额: 中位 {g1.amt.median()/1e8:.1f}亿 P25 {g1.amt.quantile(0.25)/1e8:.1f}亿')

# ---------- 主判据 ----------
print('\n== P5a 主判据判定 ==')
c1 = (g1.r1.mean() - g2.r1.mean()) > 0 and p < 0.05
c2 = (g1.r1.mean() - COST) > 0
c3 = (g1.r1.mean() - COST) * (len(g1) / 10.6) > 0
print(f'  1) G1vsG2差>0且p<0.05: {"PASS" if c1 else "FAIL"}（差{g1.r1.mean()-g2.r1.mean():+.2%}, p={p:.4f}）')
print(f'  2) G1净溢价>0:         {"PASS" if c2 else "FAIL"}（{g1.r1.mean()-COST:+.2%}）')
print(f'  3) 年化金额>0:         {"PASS" if c3 else "FAIL"}')
print(f'  ==> P5a（机构席位×涨停事件）: {"PASS" if (c1 and c2 and c3) else "FAIL"}')

# 交叉验证（sanity check，放判定后；同key多行时dict取后写=末条）
sub = det[['sym6', 'date', '上榜后1日']].dropna(subset=['上榜后1日'])
first1 = dict(zip(zip(sub['sym6'], sub['date']), sub['上榜后1日']))
m = ev.assign(builtin1=[first1.get((s.split('.')[0], d), np.nan)
                        for s, d in zip(ev.sym, ev.date)]).dropna(subset=['builtin1'])
if len(m) > 100:
    print(f'口径交叉验证: 自算r1 vs 内置上榜后1日相关 {m.r1.corr(m.builtin1):.3f}（n={len(m)}）')

from datetime import datetime
reg_p = 'results/data/trial_registry.json'
reg = json.load(open(reg_p, encoding='utf-8'))
reg['trials'].append({'label': '017-P5a机构席位×涨停溢价', 'ran_at': datetime.now().isoformat(timespec='seconds'),
                      'n_events': int(len(ev)), 'verdict': 'PASS' if (c1 and c2 and c3) else 'FAIL'})
reg['count'] = len(reg['trials'])
tmp = reg_p + '.tmp'
with open(tmp, 'w', encoding='utf-8') as f:
    json.dump(reg, f, ensure_ascii=False, indent=1)
os.replace(tmp, reg_p)
print(f'\n登记册: {reg["count"]} 条 | 明细: ai-created/trial017p5a_events.csv')
