#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
试炼 005/006/007：截面因子共享检验框架（独立信号，从零检验）
预注册见 docs/研究日志/005/006/007

因子（月末计算，全市场 PIT 池：主板+创业板+科创板，非ST，非北交所）：
  005 动量：P[t-21]/P[t-273]-1（12-1）| 敏感性 6-1 / 12-1 / 18-1
  006 反转：P[t]/P[t-20]-1（负向）| 敏感性 10 / 20 / 40 日
  007 低波：60日日收益标准差（负向）| 敏感性 60 / 120 / 250 日
检验：十分位等权组合 → 下月收益；价差=Q_high−Q_low（方向按各假设）；
  成本=单边0.25%×(多头换手+空头换手) 计入价差；IC=月末因子rank×下月收益rank
判据：两期（2017-21/2022-26）年化净价差同号为正；合并 IC t≥2；
  敏感性三档同号。诊断：多头净 vs 全池等权。
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import numpy as np
import pandas as pd

from data.layered import LayeredDataSource
from data.st_provider import STProvider

START, END = '2015-06-01', '2026-10-23'
HOLD_START = pd.Period('2017-01', 'M')
ERA1_END = pd.Period('2021-12', 'M')
COST = 0.0025

# ── 1. 数据 ──
print('加载全缓存股票价格...', flush=True)
cache = {}
for fn in os.listdir('data_cache/kline'):
    if not fn.endswith('.parquet'):
        continue
    stem = fn[:-8]
    code, suf = stem.split('_')
    if suf == 'SH' and code[0] == '6':
        sym = code + '.SH'
    elif suf == 'SZ' and code[0] in '03':
        sym = code + '.SZ'
    else:
        continue
    try:
        df = pd.read_parquet(f'data_cache/kline/{fn}')
        if 'adjclose' in df.columns and len(df):
            df['date'] = pd.to_datetime(df['date'])
            cache[sym] = df
    except Exception:
        continue
print(f'股票数: {len(cache)}', flush=True)

px = pd.DataFrame({s: d.set_index('date')['adjclose'].astype(float)
                   for s, d in cache.items()}).sort_index()
px = px[(px.index >= START) & (px.index <= END)]
me = px.groupby(px.index.to_period('M')).last()
fwd = me.shift(-1) / me - 1          # 下月收益（信号月行）

# ST 矩阵（月末采样）
print('ST 矩阵...', flush=True)
ds = LayeredDataSource()
st = STProvider(ds).st_matrix(pd.DatetimeIndex(px.index), list(px.columns))
st_m = st.groupby(st.index.to_period('M')).max() if st is not None else None
print(f'ST矩阵: {None if st is None else st.shape}', flush=True)

# 流动性/上市过滤：当月有价且 12 个月前有价（上市≥1年）
listed = me.notna() & me.shift(12).notna()

def decile_stats(name, factor_fn, direction, sens_params):
    """factor_fn(m, param) -> Series 因子；direction=+1 高分做多 / -1 低分做多"""
    print(f'\n{"="*72}\n{name}')
    results = {}
    for tag, param in sens_params:
        spreads_g, spreads_n, ics, longs, benchs, idx = [], [], [], [], [], []
        prev_hi, prev_lo = set(), set()
        for m in me.index:
            if m < HOLD_START or m > pd.Period('2026-09', 'M'):
                continue
            f = factor_fn(m, param)
            if f is None:
                continue
            r = fwd.loc[m]
            uni = f.index[f.notna() & r.notna() & listed.loc[m].reindex(f.index).fillna(False)]
            if st_m is not None and m in st_m.index:
                stt = st_m.loc[m].reindex(uni).fillna(False)
                uni = uni[~stt]
            if len(uni) < 500:
                continue
            fv = f[uni].rank(pct=True)
            rv = r[uni]
            ic = fv.corr(rv, method='spearman')
            if not np.isnan(ic):
                ics.append(ic)
            q_hi = set(fv[fv >= fv.quantile(0.9)].index)
            q_lo = set(fv[fv <= fv.quantile(0.1)].index)
            g_hi = float(rv[list(q_hi)].mean()); g_lo = float(rv[list(q_lo)].mean())
            # 换手（成员变化率，首月=1）
            t_hi = 1 - len(q_hi & prev_hi) / len(q_hi) if prev_hi else 1.0
            t_lo = 1 - len(q_lo & prev_lo) / len(q_lo) if prev_lo else 1.0
            prev_hi, prev_lo = q_hi, q_lo
            cost = COST * (t_hi + t_lo)          # 多空两腿各计单边
            spreads_g.append((g_hi - g_lo) * direction)
            spreads_n.append((g_hi - g_lo) * direction - cost)
            longs.append(g_hi - COST * t_hi)
            benchs.append(float(rv.mean()))
            idx.append(m)
        sg, sn = pd.Series(spreads_g, index=idx), pd.Series(spreads_n, index=idx)
        lo_n = pd.Series(longs, index=idx); be = pd.Series(benchs, index=idx)
        ic_s = pd.Series(ics, index=idx[:len(ics)])

        def ann(x):
            return float((1 + x).prod() ** (12 / len(x)) - 1) if len(x) else np.nan

        def seg(x, lo, hi):
            return x[(x.index >= lo) & (x.index <= hi)]

        e = {}
        e['full_net'] = ann(sn)
        for era, a, b in (('e1', HOLD_START, ERA1_END), ('e2', pd.Period('2022-01', 'M'), idx[-1])):
            e[f'{era}_net'] = ann(seg(sn, a, b))
            e[f'{era}_gross'] = ann(seg(sg, a, b))
        ic_t = ic_s.mean() / (ic_s.std() / np.sqrt(len(ic_s))) * direction if len(ic_s) > 2 else np.nan
        e['ic_mean'] = float(ic_s.mean() * direction)
        e['ic_t'] = float(ic_t)
        e['long_net'] = ann(lo_n) - ann(be)
        results[tag] = e
        print(f"  [{tag}] 全期净价差 {e['full_net']:+.2%} | e1 {e['e1_net']:+.2%} e2 {e['e2_net']:+.2%}"
              f" | IC {e['ic_mean']:+.3f}(t={e['ic_t']:.1f}) | 多头净超额 {e['long_net']:+.2%}")
    return results

# 005 动量（月度采样）：P[m-1月]/P[m-1-L月]-1（跳过最近1个月，回看L月）
def mom_fn(m, L):
    i = me.index.get_loc(m)
    if i - 1 - L < 0:
        return None
    return me.iloc[i - 1] / me.iloc[i - 1 - L] - 1

# 006 反转（日频计算取月末行）：P[t]/P[t-w日]-1
ret_d = px.pct_change()
rev_m = {w: (px / px.shift(w) - 1).groupby(px.index.to_period('M')).last()
         for w in (10, 20, 40)}

def rev_fn(m, w):
    if m not in rev_m[w].index:
        return None
    return rev_m[w].loc[m]

# 007 低波：日收益 rolling std 取月末行
vol_m = {w: ret_d.rolling(w).std().groupby(px.index.to_period('M')).last()
         for w in (60, 120, 250)}

def vol_fn(m, w):
    if m not in vol_m[w].index:
        return None
    return vol_m[w].loc[m]

r5 = decile_stats('005 截面动量（方向+）', mom_fn, +1,
                  [('6-1', 6), ('12-1', 12), ('18-1', 18)])
r6 = decile_stats('006 短期反转（方向-）', rev_fn, -1,
                  [('10d', 10), ('20d', 20), ('40d', 40)])
r7 = decile_stats('007 低波动（方向-）', vol_fn, -1,
                  [('60d', 60), ('120d', 120), ('250d', 250)])

def verdict(rs):
    main = rs[list(rs)[1]] if len(rs) > 2 else rs[list(rs)[0]]
    # 主判据用中间参数档
    ok = (main['e1_net'] > 0 and main['e2_net'] > 0
          and abs(main['ic_t']) >= 2 and main['ic_mean'] > 0)
    robust = all(v['e1_net'] > 0 and v['e2_net'] > 0 for v in rs.values())
    return ('PASS' if (ok and robust) else 'FAIL'), robust

for label, rs in (('005', r5), ('006', r6), ('007', r7)):
    v, rb = verdict(rs)
    print(f'\n{label} 判据: {v}（敏感性三档同号: {rb}）')

# ── 2. 登记册 ──
from datetime import datetime
reg_p = 'results/data/trial_registry.json'
reg = json.load(open(reg_p, encoding='utf-8'))
for label, rs in (('005-截面动量', r5), ('006-短期反转', r6), ('007-低波动', r7)):
    v, _ = verdict(rs)
    main = list(rs.values())[1] if len(rs) > 2 else list(rs.values())[0]
    reg['trials'].append({'label': label, 'ran_at': datetime.now().isoformat(timespec='seconds'),
                          'e1_net_spread': main['e1_net'], 'e2_net_spread': main['e2_net'],
                          'ic_mean': main['ic_mean'], 'ic_t': main['ic_t'],
                          'long_excess': main['long_net'], 'verdict': v})
reg['count'] = len(reg['trials'])
tmp = reg_p + '.tmp'
with open(tmp, 'w', encoding='utf-8') as f:
    json.dump(reg, f, ensure_ascii=False, indent=1)
os.replace(tmp, reg_p)
print(f'\n登记册: {reg["count"]} 条')
