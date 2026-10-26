#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
试炼 001：国家队 ETF 放量跟随（预注册见 docs/研究日志/001_国家队ETF放量跟随.md）

预注册要点（跑之前已写死，本脚本只实现不更改）：
  信号A：池内 ETF 全日成交额 ≥ 自身前20日均×3，同日 ≥2 只触发 → 事件日
  信号B：510300/510500 尾盘30分钟（14:35~15:00 六根5分bar）金额 ≥ 自身
        前20日同时段均×4，≥1只触发 → 事件日
  事件窗：信号日次日开盘买入 000300，N∈{20,60} 交易日后收盘的绝对收益
  判据：两期（2015-2020 / 2021-2026）同号为正 + 合并 t≥2（60日窗）
        + 事件数 ≥15
  事件聚类（分析时固定声明）：10 个交易日内多次触发合并为一个独立事件段
        ——国家队护盘天然连续多日，重叠事件窗会虚假放大 t 值；主口径用
        事件段，同时报告原始日计数与 5/20 日聚类敏感性

数据口径（R2 声明）：
  510300 日线：data_cache/kline/510300_SH.parquet（amount，元；layered 对该
    代码返回空（DB/WindPy 补齐报错 -40521008），绕行直读缓存）
  其余 6 只 + 000300：layered get_kline（amount 列；layered 失败时回退
    data_cache/kline parquet。各源单位可能不同，但信号为自身相对比率，
    不跨标的比较绝对值）
  5分钟线：data_cache/jq/{510300,510500}_XSHG_5m.parquet（money，元）
  共同终点 2026-10-15（JQ 缓存终点）
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import date
import numpy as np
import pandas as pd

from data.layered import LayeredDataSource

END = pd.Timestamp('2026-10-15')
K3, K4, W = 3.0, 4.0, 20
WINDOWS = (20, 60)
ERA_SPLIT = pd.Timestamp('2021-01-01')
CLUSTER = 10

# ── 1. 数据 ──
ds = LayeredDataSource()

def layered_daily(sym):
    df = ds.get_kline(sym, date(2015, 1, 1), date(2026, 10, 15))
    if df is None or len(df) == 0:
        df = pd.read_parquet(f"data_cache/kline/{sym.replace('.', '_')}.parquet")
    s = df.set_index('date')['amount'].astype(float)
    return s[(s.index >= '2015-01-01') & (s.index <= END)]

amounts = {}
p = pd.read_parquet('data_cache/kline/510300_SH.parquet')
s = p.set_index('date')['amount'].astype(float)
amounts['510300.SH'] = s[(s.index >= '2015-01-01') & (s.index <= END)]
for sym in ('510310.SH', '510330.SH', '159919.SZ', '510050.SH',
            '510500.SH', '512100.SH'):
    amounts[sym] = layered_daily(sym)

idx_df = ds.get_kline('000300.SH', date(2015, 1, 1), date(2026, 10, 15))
idx = idx_df.set_index('date')[['open', 'close']].astype(float)
idx = idx[idx.index <= END]
cal = idx.index
print(f'主日历: {cal[0].date()} ~ {cal[-1].date()} ({len(cal)} 交易日)', flush=True)
print('池覆盖:', {k: f'{v.index.min().date()}~{v.index.max().date()}' for k, v in amounts.items()}, flush=True)

# ── 2. 前向收益（次日起开盘入场，N交易日后收盘）──
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

# ── 3. 信号A（全日成交额 ≥3×前20日均，同日≥2只）──
trigA = {}
for sym, s in amounts.items():
    a = s.reindex(cal)
    ma = a.shift(1).rolling(W).mean()
    trigA[sym] = ((a >= K3 * ma) & ma.notna()).fillna(False)
nA = pd.DataFrame(trigA).sum(axis=1)
eventA = nA[nA >= 2].index
print(f'\n信号A 原始触发日: {len(eventA)}', flush=True)

# ── 4. 信号B（尾盘30分钟 ≥4×前20日同时段均，≥1只）──
trigB = {}
for sym, fn in (('510300.SH', '510300_XSHG_5m.parquet'),
                ('510500.SH', '510500_XSHG_5m.parquet')):
    j = pd.read_parquet(f'data_cache/jq/{fn}')
    t = j.index
    tail_mask = ((t.hour == 14) & (t.minute >= 35)) | ((t.hour == 15) & (t.minute == 0))
    tj = j[tail_mask]
    tail = tj.groupby(tj.index.date).money.sum()
    tail.index = pd.to_datetime(tail.index)
    tail = tail.reindex(cal)
    ma = tail.shift(1).rolling(W).mean()
    trigB[sym] = ((tail >= K4 * ma) & ma.notna()).fillna(False)
tB = pd.DataFrame(trigB)
eventB = tB[tB.any(axis=1)].index
print(f'信号B 原始触发日: {len(eventB)}', flush=True)

# ── 5. 事件段聚类与统计 ──
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

def stats(name, days):
    eps = to_episodes(days, CLUSTER)
    sub = fwd.loc[fwd.index.isin(eps)]
    base = fwd.dropna(subset=['r60'])
    print(f'\n{"="*72}\n{name}')
    print(f'原始触发日 {len(days)} | 独立事件段(10日聚类) {len(eps)}')
    print(f'事件段按年: {dict(pd.Series(eps).groupby(pd.DatetimeIndex(eps).year).size())}')
    out = {'label': name, 'n_days': len(days), 'n_eps': len(eps)}
    for w in WINDOWS:
        ev = sub[f'r{w}'].dropna()
        if len(ev) == 0:
            print(f'  {w}日: 无可用事件')
            continue
        t1 = ev.mean() / (ev.std() / np.sqrt(len(ev))) if ev.std() > 0 else 0.0
        nv, nb = len(ev), len(base)
        sp = ((nv - 1) * ev.var() + (nb - 1) * base[f'r{w}'].var()) / (nv + nb - 2)
        tw = (ev.mean() - base[f'r{w}'].mean()) / np.sqrt(sp * (1 / nv + 1 / nb)) if sp > 0 else 0.0
        print(f'  {w}日: 事件段均值 {ev.mean():+.2%} (n={nv}) | '
              f'无条件均值 {base[f"r{w}"].mean():+.2%} | 超额 {ev.mean()-base[f"r{w}"].mean():+.2%} '
              f'| 单样本t={t1:.2f} Welch t={tw:.2f}')
        out[f'mean_{w}'] = float(ev.mean())
        out[f'excess_{w}'] = float(ev.mean() - base[f'r{w}'].mean())
        out[f'welch_t_{w}'] = float(tw)
    same_sign = {}
    for era in ('2015-2020', '2021-2026'):
        s_ = sub[[era_of(d) == era for d in sub.index]]
        line = f'  {era}: n={len(s_)}'
        for w in WINDOWS:
            ev = s_[f'r{w}'].dropna()
            m = float(ev.mean()) if len(ev) else np.nan
            same_sign[(era, w)] = m
            line += f' | {w}日 {m:+.2%}(n={len(ev)})' if len(ev) else f' | {w}日 无事件'
        print(line)
    def _pos(x):
        return (not bool(np.isnan(x))) and x > 0

    out['two_era_positive'] = all(
        _pos(same_sign[('2015-2020', w)]) and _pos(same_sign[('2021-2026', w)])
        for w in WINDOWS)
    out['enough_events'] = len(eps) >= 15
    out['t60_ge2'] = out.get('welch_t_60', 0) >= 2
    out['verdict'] = 'PASS' if (out['two_era_positive'] and out['enough_events']
                                and out['t60_ge2']) else 'FAIL'
    print(f"  判据: 事件段≥15 {out['enough_events']} | 两期同号为正 {out['two_era_positive']} "
          f"| Welch t60≥2 {out['t60_ge2']}  →  {out['verdict']}")
    # 聚类敏感性
    sens = {c: len(to_episodes(days, c)) for c in (5, 10, 20)}
    print(f'  聚类敏感性(事件段数 5/10/20日): {sens}')
    return eps, out

epsA, outA = stats('信号A（全日放量≥3×，≥2只联动）', eventA)
epsB, outB = stats('信号B（尾盘30分钟≥4×，≥1只）', eventB)
overlap = len(set(epsA) & set(epsB))
print(f'\nA/B 事件段重叠: {overlap}/{max(len(epsA), len(epsB), 1)}')

# ── 6. 明细落盘 + 登记册 ──
det = []
for sig, eps in (('A', epsA), ('B', epsB)):
    for d in eps:
        row = fwd.loc[d] if d in fwd.index else None
        det.append({'signal': sig, 'episode': d.date(),
                    'era': era_of(d),
                    'r20': None if row is None else row.get('r20'),
                    'r60': None if row is None else row.get('r60')})
pd.DataFrame(det).to_csv('ai-created/trial001_events.csv', index=False,
                         encoding='utf-8-sig')

from datetime import datetime
reg_p = 'results/data/trial_registry.json'
reg = json.load(open(reg_p, encoding='utf-8'))
for o in (outA, outB):
    o['ran_at'] = datetime.now().isoformat(timespec='seconds')
    reg['trials'].append(o)
reg['count'] = len(reg['trials'])
tmp = reg_p + '.tmp'
with open(tmp, 'w', encoding='utf-8') as f:
    json.dump(reg, f, ensure_ascii=False, indent=1)
os.replace(tmp, reg_p)
print(f'\n登记册: {reg["count"]} 条 | 明细: ai-created/trial001_events.csv')
