#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
试炼 004：ETF 场内份额激增（大资金申购痕迹）
预注册见 docs/研究日志/004_ETF场内份额激增.md

信号（预注册，只实现不更改）：
  单只 ETF 份额 5 个交易日净增长率 ≥ 5%（基期=5 交易日前），池内
  ≥1 只触发 → 事件日。份额 T 日盘后公布，信号从 T+1 起算——
  T+1 开盘买 000300，N∈{20,60} 交易日后收盘（与 001 完全同口径）。
判据：事件段（10 日聚类）≥15；两期（2015-2020 / 2021-2026）20/60 日
  同号为正；合并 Welch t≥2（60 日，vs 无条件全样本）。
池：6 只上交所宽基（510300/510310/510330/510050/510500/512100；
159919 深交所无历史接口，排除并声明）。
数据：data_cache/etf_share/sse.parquet（逐日抓取自上交所），
      000300 本地缓存。
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import numpy as np
import pandas as pd

END = pd.Timestamp('2026-10-15')
K_GROWTH, W = 0.05, 5
WINDOWS = (20, 60)
ERA_SPLIT = pd.Timestamp('2021-01-01')
CLUSTER = 10
POOL = ['510300', '510310', '510330', '510050', '510500', '512100']

# ── 1. 数据 ──
sh = pd.read_parquet('data_cache/etf_share/sse.parquet')
sh['date'] = pd.to_datetime(sh['date'], format='%Y%m%d')
sh = sh[sh['date'] <= END]
print(f'份额数据: {len(sh)} 行 | {sh["date"].min().date()} ~ {sh["date"].max().date()}', flush=True)

idx = pd.read_parquet('data_cache/kline/000300_SH.parquet')
idx['date'] = pd.to_datetime(idx['date'])
idx = idx[(idx['date'] >= '2015-01-01') & (idx['date'] <= END)].set_index('date')
cal = idx.index
pos_cal = {d: i for i, d in enumerate(cal)}
print(f'主日历: {cal[0].date()} ~ {cal[-1].date()} ({len(cal)} 交易日)', flush=True)

# ── 2. 前向收益（同001：次日起开盘入场，N交易日后收盘）──
opens, closes = idx['open'], idx['close']
n = len(cal)
rows = []
for i in range(n - 1):
    j = i + 1
    r = {'date': cal[i]}
    for w in WINDOWS:
        r[f'r{w}'] = (closes.iloc[j + w] / opens.iloc[j] - 1) if j + w < n else np.nan
    rows.append(r)
fwd = pd.DataFrame(rows).set_index('date')

# ── 3. 信号：5日份额净增 ≥5%，≥1只 ──
def etf_close_series(code):
    """ETF 日线收盘（拆分护栏用；宽基ETF单日涨跌不可能超±15%，
    份额拆分会造成价格同比例跳变——识别并剔除该类假触发）"""
    p = f"data_cache/kline/{code}_SH.parquet"
    if not os.path.exists(p):
        return None
    df = pd.read_parquet(p)
    df['date'] = pd.to_datetime(df['date'])
    s = df.set_index('date')['close']
    return s[(s.index >= '2015-01-01') & (s.index <= END)]

trig = {}
for code in POOL:
    s = sh[sh.code == code].set_index('date')['share'].reindex(cal)
    base = s.shift(W)
    t_ = ((s / base - 1) >= K_GROWTH) & base.notna() & s.notna()
    # 拆分护栏：触发日该ETF自身价格单日跳变>±15% → 判为份额折算假信号；
    # 拆分后 W 日内基期仍含折算前份额（比率持续虚高），整窗压制
    px_s = etf_close_series(code)
    if px_s is not None:
        pct = px_s / px_s.shift(1) - 1
        split_days = [d for d in pct[abs(pct) > 0.15].index if d in pos_cal]
        if split_days:
            n_before = int(t_.sum())
            for d in split_days:
                for k in range(W + 1):
                    dd = cal[pos_cal[d] + k] if pos_cal[d] + k < len(cal) else None
                    if dd is not None and dd in t_.index:
                        t_.loc[dd] = False
            print(f'{code}: ⚠️ 份额折算疑似日 {[str(d.date()) for d in split_days]}，'
                  f'整窗压制触发 {n_before}→{int(t_.sum())}', flush=True)
    n_t = int(t_.sum())
    trig[code] = t_
    print(f'{code}: 触发 {n_t} 日 | 覆盖 {s.notna().sum()} 日', flush=True)
tM = pd.DataFrame(trig)
event = tM[tM.any(axis=1)].index
print(f'\n原始触发日: {len(event)}', flush=True)

# ── 4. 事件段聚类与统计（同001结构）──
pos = {d: i for i, d in enumerate(cal)}

def to_episodes(days, cluster):
    eps = []
    for d in sorted(days):
        if d not in pos:
            continue
        if eps and pos[d] - pos[eps[-1]] <= cluster:
            continue
        eps.append(d)
    return eps

def era_of(d):
    return '2015-2020' if d < ERA_SPLIT else '2021-2026'

eps = to_episodes(event, CLUSTER)
sub = fwd.loc[fwd.index.isin(eps)]
base_all = fwd.dropna(subset=['r60'])
print(f'独立事件段(10日聚类): {len(eps)}')
print(f'事件段按年: {dict(pd.Series(eps).groupby(pd.DatetimeIndex(eps).year).size())}')

out = {}
for w in WINDOWS:
    ev = sub[f'r{w}'].dropna()
    if not len(ev):
        print(f'  {w}日: 无可用事件')
        continue
    t1 = ev.mean() / (ev.std() / np.sqrt(len(ev))) if ev.std() > 0 else 0.0
    nv, nb = len(ev), len(base_all)
    sp = ((nv - 1) * ev.var() + (nb - 1) * base_all[f'r{w}'].var()) / (nv + nb - 2)
    tw = (ev.mean() - base_all[f'r{w}'].mean()) / np.sqrt(sp * (1 / nv + 1 / nb)) if sp > 0 else 0.0
    print(f'  {w}日: 事件段均值 {ev.mean():+.2%} (n={nv}) | 无条件均值 {base_all[f"r{w}"].mean():+.2%} '
          f'| 超额 {ev.mean()-base_all[f"r{w}"].mean():+.2%} | 单样本t={t1:.2f} Welch t={tw:.2f}')
    out[f'mean_{w}'] = float(ev.mean())
    out[f'excess_{w}'] = float(ev.mean() - base_all[f'r{w}'].mean())
    out[f'welch_t_{w}'] = float(tw)

era_means = {}
for era in ('2015-2020', '2021-2026'):
    s_ = sub[[era_of(d) == era for d in sub.index]]
    line = f'  {era}: n={len(s_)}'
    for w in WINDOWS:
        ev = s_[f'r{w}'].dropna()
        m = float(ev.mean()) if len(ev) else np.nan
        era_means[(era, w)] = m
        line += f' | {w}日 {m:+.2%}(n={len(ev)})' if len(ev) else f' | {w}日 无事件'
    print(line)

def _pos(x):
    return (not bool(np.isnan(x))) and x > 0

two_era = all(_pos(era_means[('2015-2020', w)]) and _pos(era_means[('2021-2026', w)])
              for w in WINDOWS)
enough = len(eps) >= 15
t60 = out.get('welch_t_60', 0) >= 2
verdict = 'PASS' if (two_era and enough and t60) else 'FAIL'
print(f'判据: 事件段≥15 {enough} | 两期同号为正 {two_era} | Welch t60≥2 {t60} → {verdict}')
print(f'聚类敏感性(5/10/20): { {c: len(to_episodes(event, c)) for c in (5, 10, 20)} }')

# ── 5. 明细落盘 + 登记册 ──
det = [{'episode': d.date(), 'era': era_of(d),
        'trigger_etfs': [c for c in POOL if d in tM.index and tM.loc[d, c]],
        'r20': None if d not in fwd.index else fwd.loc[d].get('r20'),
        'r60': None if d not in fwd.index else fwd.loc[d].get('r60')}
       for d in eps]
pd.DataFrame(det).to_csv('ai-created/trial004_events.csv', index=False, encoding='utf-8-sig')

from datetime import datetime
reg_p = 'results/data/trial_registry.json'
reg = json.load(open(reg_p, encoding='utf-8'))
reg['trials'].append({'label': '004-ETF场内份额激增',
                      'ran_at': datetime.now().isoformat(timespec='seconds'),
                      'n_events': len(eps), **out,
                      'two_era_positive': bool(two_era), 'verdict': verdict})
reg['count'] = len(reg['trials'])
tmp = reg_p + '.tmp'
with open(tmp, 'w', encoding='utf-8') as f:
    json.dump(reg, f, ensure_ascii=False, indent=1)
os.replace(tmp, reg_p)
print(f'\n登记册: {reg["count"]} 条 | 明细: ai-created/trial004_events.csv')
