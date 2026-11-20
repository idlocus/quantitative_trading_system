#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
试炼 013：006×011 组合（预注册见 docs/研究日志/013_反转与价值组合.md）
- 13a 负向过滤（按注册精确实现）：Top100 候选中剔除近W日涨幅截面前10%，
  顺位递补（score 置 NaN）
- 13b 反转倾斜（实现偏差声明：注册写"升序加权"，等权引擎不支持自定义
  权重——以选择倾斜等价实现：score - 0.15×z(rev)，把近期跑输者排进Top100）
- 对照与判据：011 原策略（夏普 0.51 为预注册门槛）；两期年化>0 且
  夏普>0.51；敏感性 W∈{10,20,40} 至少两档同向改善
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from datetime import date
import numpy as np
import pandas as pd

from data.layered import LayeredDataSource
from data.fundamentals import WindFundamentalProvider, build_fundamental_matrix
from data.fundamental_pit import PITFinancialProvider
from data.semantics import close_matrix
from data.st_provider import STProvider
from data.corporate_actions import load_corporate_actions
from portfolio.construction import composite_from_matrices
from portfolio.daily_replay import DailyReplayBacktester, ReplayConfig
from trading import config as chain
from trading.strategy import _build_momentum_matrix

START, END = date(2016, 6, 1), date(2026, 10, 15)
SIGNAL_START, ERA_SPLIT = '2017-01-03', '2021-12-31'
BAR = 0.51   # 011 预注册夏普门槛

cache = {}
for fn in os.listdir('data_cache/kline'):
    if not fn.endswith('.parquet'):
        continue
    code, suf = fn[:-8].split('_')
    if (suf == 'SH' and code[0] == '6') or (suf == 'SZ' and code[0] in '03'):
        sym = code + '.' + suf
    else:
        continue
    try:
        df = pd.read_parquet(f'data_cache/kline/{fn}')
        if 'adjclose' in df.columns and len(df):
            df['date'] = pd.to_datetime(df['date'])
            df = df[(df['date'] >= pd.Timestamp(START)) & (df['date'] <= pd.Timestamp(END))]
            if len(df):
                cache[sym] = df
    except Exception:
        continue
pool = sorted(cache)
print(f'宇宙: {len(pool)}', flush=True)
px = close_matrix(cache, basis='adjusted')

ds = LayeredDataSource()
prov = WindFundamentalProvider(ds)
mats = {f: build_fundamental_matrix(prov, f, pool, START, END)
        for f in chain.PRODUCTION_FACTORS}
mats = {k: v for k, v in mats.items() if not v.empty}
pit = PITFinancialProvider(ds)
value = composite_from_matrices(mats, directions=chain.PRODUCTION_FACTORS)
value = value[value.columns.intersection(cache.keys())]
roe = pit.get_factor_matrix('roe', pool, START, END)
lev = pit.get_factor_matrix('leverage', pool, START, END)
snv = value.copy()
for mx_, thr, is_min in ((roe, chain.QUALITY_FILTER['roe_min'], True),
                         (lev, chain.QUALITY_FILTER['leverage_max'], False)):
    m = mx_.reindex(index=snv.index, columns=snv.columns)
    snv = snv.mask(m < thr) if is_min else snv.mask(m > thr)

def zrow(mx):
    return mx.sub(mx.mean(axis=1), axis=0).div(mx.std(axis=1) + 1e-12, axis=0)

v_z = zrow(snv)
mom = _build_momentum_matrix(cache, START, END)
mom_z = zrow(mom.reindex_like(v_z))
score = zrow((1 - chain.MOMENTUM_WEIGHT) * v_z + chain.MOMENTUM_WEIGHT * mom_z)

st_matrix = None
try:
    idx_dates = pd.DatetimeIndex(sorted({d for df in cache.values() for d in df['date']}))
    st_matrix = STProvider(ds).st_matrix(idx_dates, pool)
except Exception:
    pass
ca = load_corporate_actions(ds, pool, start='20160101', end='20261231')
cfg = ReplayConfig(top_n=chain.TOP_N, st_matrix=st_matrix, band=chain.BUFFER_BAND,
                   use_risk_gate=False, st_data_required=False)
bt = DailyReplayBacktester(cache, cfg, corporate_actions=ca)

# 反转矩阵（对齐到分数行日期，asof 前向取值）
rev_al = {}
union = score.index.union(px.index)
for w in (10, 20, 40):
    r_ = (px / px.shift(w) - 1)
    rev_al[w] = r_.reindex(union).ffill().reindex(score.index)

def stats(r):
    eq = r.equity_curve
    def ann(seg):
        yrs = (seg.index[-1] - seg.index[0]).days / 365.25
        return float(seg.iloc[-1] / seg.iloc[0]) ** (1 / yrs) - 1 if yrs > 0.3 else np.nan
    e1, e2 = ann(eq[eq.index <= ERA_SPLIT]), ann(eq[eq.index > ERA_SPLIT])
    return r.sharpe, e1, e2

print('\n臂 × 窗口运行（7 个重放）...', flush=True)
rows = []
for w in (10, 20, 40):
    rev = rev_al[w]
    # 13a：剔除截面涨幅前10%（NaN不动）
    hi = rev.ge(rev.quantile(0.9, axis=1), axis=0)
    s13a = score.mask(hi.reindex_like(score).fillna(False))
    for tag, s in ((f'13a过滤W={w}', s13a),
                   (f'13b倾斜W={w}', zrow(score - 0.15 * zrow(rev.reindex_like(score))))):
        r = bt.run(s, start_date=SIGNAL_START, end_date=str(END))
        sh, e1, e2 = stats(r)
        rows.append({'arm': tag, 'sharpe': sh, 'e1': e1, 'e2': e2})
        print(f'{tag}: 夏普 {sh:.3f}（门槛{BAR}）| e1 {e1:+.2%} e2 {e2:+.2%}', flush=True)

d = pd.DataFrame(rows)
d.to_csv('ai-created/trial013_summary.csv', index=False, encoding='utf-8-sig')
ok13a = all(d[d.arm.str.startswith('13a')][['e1', 'e2']].iloc[0] > 0) and any(
    d[d.arm.str.startswith('13a')].sharpe > BAR) and (d[d.arm.str.startswith('13a')].sharpe > BAR).sum() >= 2
ok13b = all(d[d.arm.str.startswith('13b')][['e1', 'e2']].iloc[0] > 0) and any(
    d[d.arm.str.startswith('13b')].sharpe > BAR) and (d[d.arm.str.startswith('13b')].sharpe > BAR).sum() >= 2
print(f'\n判据（两期>0 且 夏普>0.51，三档≥2档达标）: 13a {"PASS" if ok13a else "FAIL"} | 13b {"PASS" if ok13b else "FAIL"}')

from datetime import datetime
reg_p = 'results/data/trial_registry.json'
reg = json.load(open(reg_p, encoding='utf-8'))
reg['trials'].append({'label': '013-006×011组合(13a过滤/13b倾斜)',
                      'ran_at': datetime.now().isoformat(timespec='seconds'),
                      'results': rows, '13a_verdict': 'PASS' if ok13a else 'FAIL',
                      '13b_verdict': 'PASS' if ok13b else 'FAIL'})
reg['count'] = len(reg['trials'])
tmp = reg_p + '.tmp'
with open(tmp, 'w', encoding='utf-8') as f:
    json.dump(reg, f, ensure_ascii=False, indent=1)
os.replace(tmp, reg_p)
print(f'登记册: {reg["count"]} 条')
