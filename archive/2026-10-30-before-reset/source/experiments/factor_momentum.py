#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
因子动量研究（2026-10-23 预注册）

假设：因子截面 IC 的近期表现具有持续性——过去3个月IC高的因子在接下来
继续跑赢（Arnott et al. 2022）。用动态权重替代固定的等权 ep/bp/sp。

方法：
  每月计算各因子(ep/bp/sp)过去3个月(63日)的日频截面IC均值 →
  IC>0 因子权重增大（按IC的softmax或线性映射），IC<0 减小 →
  与固定等权基线对比。
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import date
import numpy as np
import pandas as pd

from data.layered import LayeredDataSource
from data.base.data_loader import DataLoader
from data.semantics import close_matrix
from data.fundamentals import WindFundamentalProvider, build_fundamental_matrix
from portfolio.construction import composite_from_matrices
from portfolio.daily_replay import DailyReplayBacktester, ReplayConfig
from trading import config as chain

START, END = date(2016, 6, 1), date(2026, 10, 9)
SIGNAL_START = '2017-01-03'

ds = LayeredDataSource()
pool = sorted(ds.list_symbols(as_of=date(2017, 1, 3)))
print(f"池: {len(pool)}", flush=True)

loader = DataLoader.__new__(DataLoader)
loader.data_source_type = 'layered'
loader.data_source = ds
loader._cache = None
cache = loader.preload_klines(pool + ['000300.SH'], START, END)._cache
px = close_matrix(cache, basis='adjusted')

# 因子矩阵
prov = WindFundamentalProvider(ds)
mats = {}
for f in chain.PRODUCTION_FACTORS:
    fm = build_fundamental_matrix(prov, f, pool, START, END)
    if not fm.empty:
        mats[f] = fm
print(f"因子矩阵: {sorted(mats)}", flush=True)

# 前向收益（计算因子IC用）
fwd_1m = px.pct_change(21).shift(-21)  # 21日前向收益

# ── 计算各因子每日截面IC ──
print("计算因子IC...", flush=True)
factors = sorted(mats)
ic_data = {f: [] for f in factors}
common_dates = px.index

for f in factors:
    fm = mats[f]
    for dt in common_dates:
        if dt not in fm.index or dt not in fwd_1m.index:
            continue
        f_row = fm.loc[dt]
        r_row = fwd_1m.loc[dt]
        common = f_row.notna() & r_row.notna()
        if common.sum() < 200:
            continue
        # 截面rank IC（Spearman）
        ic = f_row[common].rank().corr(r_row[common].rank())
        ic_data[f].append({'date': dt, 'ic': ic})

ic_dfs = {f: pd.DataFrame(v).set_index('date')['ic'] for f, v in ic_data.items()}
ic_mx = pd.DataFrame(ic_dfs).loc[SIGNAL_START:]
print(f"IC矩阵: {ic_mx.shape} | 各因子IC均值:", flush=True)
for f in factors:
    print(f"  {f}: {ic_mx[f].mean():+.4f}", flush=True)

# ── 构建因子动量权重 ──
print("\n构建动态权重...", flush=True)
IC_LOOKBACK = 63  # 3个月IC回看

# 每月第一个交易日更新权重
month_starts = []
seen = set()
for d in ic_mx.index:
    m = (d.year, d.month)
    if m not in seen:
        seen.add(m)
        month_starts.append(d)

weight_rows = []
for ms in month_starts:
    lookback = ic_mx.loc[:ms].tail(IC_LOOKBACK)
    if len(lookback) < 30:
        continue
    recent_ic = lookback.mean()
    # 方法1：线性映射（IC>0的因子按IC比例分配权重，IC<0权重为0）
    pos_ic = recent_ic.clip(lower=0)
    if pos_ic.sum() > 0:
        w_linear = pos_ic / pos_ic.sum()
    else:
        w_linear = pd.Series(1/len(factors), index=factors)
    weight_rows.append({'date': ms, **{f'{f}_w': w_linear[f] for f in factors}})

weights = pd.DataFrame(weight_rows).set_index('date')
print(f"动态权重: {len(weights)} 个月度更新", flush=True)
print("权重分布:")
for f in factors:
    print(f"  {f}: 均值{weights[f'{f}_w'].mean():.3f} "
          f"范围[{weights[f'{f}_w'].min():.3f}, {weights[f'{f}_w'].max():.3f}]")

# ── 构建动态权重分数并重放 ──
print("\n构建动态分数并重放（C口径）...", flush=True)

# 基线分数（固定等权）— 用与RPS研究相同的重建管道
from data.fundamental_pit import PITFinancialProvider
pit = PITFinancialProvider(ds)
value = composite_from_matrices(mats, directions=chain.PRODUCTION_FACTORS)
value = value[value.columns.intersection(cache.keys())]
roe = pit.get_factor_matrix('roe', pool, START, END)
lev = pit.get_factor_matrix('leverage', pool, START, END)
snv = value.copy()
for mx_, thr, is_min in ((roe, 0.08, True), (lev, 65.0, False)):
    m = mx_.reindex(index=snv.index, columns=snv.columns)
    snv = snv.mask(m < thr) if is_min else snv.mask(m > thr)

def zrow(mx):
    return mx.sub(mx.mean(axis=1), axis=0).div(mx.std(axis=1) + 1e-12, axis=0)

v_z = zrow(snv)

# 动态权重分数
from trading.strategy import _build_momentum_matrix
mom = _build_momentum_matrix(cache, START, END)
mom_z = zrow(mom.reindex_like(v_z).fillna(0))

dyn_score = pd.DataFrame(index=v_z.index, columns=v_z.columns, dtype=float)
for _, wrow in weights.iterrows():
    # 该月所有日期用同一组权重
    mask = (v_z.index.to_period('M') == wrow.name.to_period('M'))
    if not mask.any():
        continue
    month_dates = v_z.index[mask]
    composite = None
    for f in factors:
        fz = zrow(mats[f].reindex(index=month_dates, columns=v_z.columns))
        w = wrow[f'{f}_w']
        composite = (w * fz) if composite is None else (composite + w * fz)
    # 归一化权重合成（每行除以因子覆盖数的调和）
    dyn_score.loc[month_dates] = composite.values

dyn_score = dyn_score.astype(float)
# 动态分数 + 0.2动量 → 最终分数
dyn_final = zrow(0.8 * dyn_score + 0.2 * mom_z)

# 基线最终分数（固定等权 + 0.2动量）
base_final = zrow(0.8 * v_z + 0.2 * mom_z)

st_matrix = None
try:
    from data.st_provider import STProvider
    idx = pd.DatetimeIndex(pd.to_datetime(next(iter(cache.values()))['date']))
    st_matrix = STProvider(ds).st_matrix(idx, pool)
except Exception:
    pass

cfg = ReplayConfig(top_n=chain.TOP_N, st_matrix=st_matrix, band=chain.BUFFER_BAND,
                   use_risk_gate=False, st_data_required=False)

def run_c(sc):
    return DailyReplayBacktester(cache, cfg).run(sc, start_date=SIGNAL_START,
                                                  end_date=str(END))

base_r = run_c(base_final)
dyn_r = run_c(dyn_final)

def era_annual(eq):
    out = {}
    for a, b, tag in (('2017', '2021', 'e1'), ('2022', '2026', 'e2')):
        seg = eq.loc[f'{a}':f'{b}']
        if len(seg) > 2:
            yrs = (seg.index[-1] - seg.index[0]).days / 365.25
            out[tag] = float(seg.iloc[-1] / seg.iloc[0]) ** (1 / yrs) - 1
    return out

be = era_annual(base_r.equity_curve)
de = era_annual(dyn_r.equity_curve)

print(f"\n{'='*70}")
print(f"因子动量 vs 固定等权（C口径）")
print(f"{'='*70}")
print(f"固定等权: {base_r.total_return:+.1%} 夏普{base_r.sharpe:.2f} | era {be['e1']:+.1%}/{be['e2']:+.1%}")
print(f"因子动量: {dyn_r.total_return:+.1%} 夏普{dyn_r.sharpe:.2f} | era {de['e1']:+.1%}/{de['e2']:+.1%}")
diff = (dyn_r.total_return - base_r.total_return) * 100
print(f"增量: {diff:+.1f}pp")
