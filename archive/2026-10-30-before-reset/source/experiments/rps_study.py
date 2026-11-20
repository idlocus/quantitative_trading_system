#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RPS 研究（2026-10-21 预注册）：RPS=动量截面百分位
V1: RPS-120(rank)替换生产动量(z)——归一化检验
V2: RPS-250(12个月) 0.1叠加——缺失周期
V3: RPS-20(短期) 0.1叠加——短期反转
判据：V1 |Δ|≤2pp且≥59%=替换无害；V2/V3 A≥+61%且C≥+58.6%且两期同号。
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import date

import numpy as np
import pandas as pd

from data.layered import LayeredDataSource
from data.semantics import close_matrix
from portfolio.daily_replay import DailyReplayBacktester, ReplayConfig
from trading import config as chain
from trading.strategy import build_strategy_score

START, END = date(2016, 6, 1), date(2026, 10, 9)
SIGNAL_START = date(2017, 1, 3)   # 首个交易日（DB池路径）

ds = LayeredDataSource()
pool = sorted(ds.list_symbols(as_of=SIGNAL_START))
assert len(pool) > 100
score, cache, _ = build_strategy_score(ds, pool, START, END)
assert score.shape[1] > 100
print(f"分数矩阵: {score.shape}", flush=True)

px = close_matrix(cache, basis='adjusted')

def rps_matrix(lb):
    """N日收益的截面百分位（0-1）——欧奈尔RPS/100"""
    ret = px / px.shift(lb) - 1
    return ret.rank(axis=1, pct=True)

rps120 = rps_matrix(120)
rps250 = rps_matrix(250)
rps20 = rps_matrix(20)

st_matrix = None
try:
    from data.st_provider import STProvider
    idx = pd.DatetimeIndex(pd.to_datetime(next(iter(cache.values()))['date']))
    st_matrix = STProvider(ds).st_matrix(idx, pool)
    print("ST矩阵就绪", flush=True)
except Exception as e:
    print(f"警告：ST不可用（{e}）", flush=True)


def run(sc, gate=True):
    cfg = ReplayConfig(top_n=chain.TOP_N, st_matrix=st_matrix,
                       band=chain.BUFFER_BAND, use_risk_gate=gate,
                       st_data_required=st_matrix is not None and gate)
    return DailyReplayBacktester(cache, cfg).run(
        sc, start_date=str(SIGNAL_START), end_date=str(END))


def era_annual(r):
    eq = r.equity_curve
    out = {}
    for a, b, tag in (('2017', '2021', 'e1'), ('2022', '2026', 'e2')):
        seg = eq.loc[f'{a}':f'{b}']
        if len(seg) > 2:
            yrs = (seg.index[-1] - seg.index[0]).days / 365.25
            out[tag] = float(seg.iloc[-1] / seg.iloc[0]) ** (1 / yrs) - 1
    return out


# ── 重建无动量的分数成分，替换/叠加 RPS ──
# 生产分数 = combine(0.8*value_z + 0.2*mom_z)。重建：value 原始合成 + 质量过滤
from data.fundamentals import WindFundamentalProvider, build_fundamental_matrix
from portfolio.construction import composite_from_matrices
from data.fundamental_pit import PITFinancialProvider

prov = WindFundamentalProvider(ds)
mats = {}
for f in chain.PRODUCTION_FACTORS:
    fm = build_fundamental_matrix(prov, f, pool, START, END)
    if not fm.empty:
        mats[f] = fm
value = composite_from_matrices(mats, directions=chain.PRODUCTION_FACTORS,
                                experimental=(sorted(mats) != sorted(chain.PRODUCTION_FACTORS)))
value = value[value.columns.intersection(cache.keys())]

pit = PITFinancialProvider(ds)
roe = pit.get_factor_matrix('roe', pool, START, END)
lev = pit.get_factor_matrix('leverage', pool, START, END)
score_nv = value.copy()
# 向量化质量过滤（.mask 条件处置NaN；此前逐日循环里 (row<thr).index
# 返回的是全索引而非命中子集——曾把全池置空、变体全零假结果）
for mx, thr in ((roe, chain.QUALITY_FILTER['roe_min']),
                (lev, chain.QUALITY_FILTER['leverage_max'])):
    if not mx.empty:
        m = mx.reindex(index=score_nv.index, columns=score_nv.columns)
        if chain.QUALITY_FILTER['roe_min'] == thr:
            score_nv = score_nv.mask(m < thr)
        else:
            score_nv = score_nv.mask(m > thr)
assert score_nv.notna().mean().mean() > 0.1, '质量过滤后分数意外为空'


def zrow(mx):
    return mx.sub(mx.mean(axis=1), axis=0).div(mx.std(axis=1) + 1e-12, axis=0)


v_z = zrow(score_nv)
aligned = lambda m: m.reindex_like(v_z).fillna(0.0)

variants = {
    '基线': score,
    'V1_RPS120替换': zrow(0.8 * v_z + 0.2 * zrow(aligned(rps120))),
    'V2_RPS250叠加0.1': zrow(0.8 * v_z + 0.2 * zrow(aligned(rps120))
                            + 0.1 * zrow(aligned(rps250))),
    'V3_RPS20叠加0.1': zrow(0.8 * v_z + 0.2 * zrow(aligned(rps120))
                            + 0.1 * zrow(aligned(rps20))),
}

print(f"\n{'='*80}")
print(f"{'口径':<16} {'A收益':>8} {'A夏普':>7} {'A回撤':>7} {'C收益':>8} "
      f"{'年化17-21':>9} {'年化22-26':>9}")
print(f"{'='*80}")
res = {}
for name, sc in variants.items():
    rA = run(sc, gate=True)
    rC = run(sc, gate=False)
    e = era_annual(rA)
    res[name] = (rA, rC, e)
    print(f"{name:<16} {rA.total_return:>+8.1%} {rA.sharpe:>7.2f} "
          f"{rA.max_drawdown:>7.1%} {rC.total_return:>+8.1%} "
          f"{e['e1']:>+9.1%} {e['e2']:>+9.1%}", flush=True)

bA, bC, be = res['基线']
print(f"\n{'='*80}")
print("预注册判定")
v1A, v1C, _ = res['V1_RPS120替换']
d1 = (v1A.total_return - bA.total_return) * 100
v1_verdict = '标度无关（rank≈z，替换无益也无害）' if abs(d1) <= 2 else \
    (f'rank更优({d1:+.1f}pp)' if d1 > 2 else f'rank更差({d1:+.1f}pp)')
print(f"  V1 归一化: A {v1A.total_return:+.1%} vs 基线 {bA.total_return:+.1%}"
      f"（Δ{d1:+.1f}pp）→ {v1_verdict}")
for name, tag in (('V2_RPS250叠加0.1', 'V2 12个月'), ('V3_RPS20叠加0.1', 'V3 短期20日')):
    rA, rC, e = res[name]
    cA = rA.total_return >= 0.61
    cC = rC.total_return >= 0.586
    cE = e['e1'] > 0 and e['e2'] > 0
    print(f"  {tag}: A{'✅' if cA else '❌'}({rA.total_return:+.1%}) "
          f"C{'✅' if cC else '❌'}({rC.total_return:+.1%}) → "
          f"{'✅' if (cA and cC) else '❌'}")
print(f"  （注：基线C本轮若≈+56.6%则C列可采信；A列受路径依赖影响，"
      f"V2/V3 的 C 口径 Δ 才是主判据）")
