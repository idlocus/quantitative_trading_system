#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
改进③质量连续评分（预注册见 docs/events研究方案.md 2026-10-19 节）

变体：score2 = 0.8×价值z + 0.2×动量z + wq×质量z（质量z=ROE_z−杠杆_z，
clip±3，缺失=0），替代 ROE/杠杆硬过滤。
对照：生产 build_strategy_score（硬过滤）A/C 双口径。
判据：A≥57.0%且回撤≤-27%；C≥54.6%；三档 wq 全过；两期年化不塌方。
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import date

import numpy as np
import pandas as pd

from data.layered import LayeredDataSource
from data.fundamental_pit import PITFinancialProvider
from portfolio.construction import composite_from_matrices
from portfolio.daily_replay import DailyReplayBacktester, ReplayConfig
from trading import config as chain
from trading.strategy import build_strategy_score, _build_momentum_matrix, _combine_value_momentum

START, END = date(2016, 6, 1), date(2026, 10, 9)
SIGNAL_START = date(2017, 1, 1)
WQ = (0.1, 0.2, 0.3)

ds = LayeredDataSource()
pool = sorted(ds.list_symbols(as_of=SIGNAL_START))
print(f"池: {len(pool)} 只", flush=True)

# 基线（唯一实现，含硬过滤）
print("基线分数（唯一实现）…", flush=True)
base_score, cache, mats = build_strategy_score(ds, pool, START, END)

# 变体分数：价值+动量（无硬过滤）+ 质量z
print("质量矩阵（PIT roe/leverage）…", flush=True)
pit = PITFinancialProvider(ds)
roe = pit.get_factor_matrix('roe', pool, START, END)
lev = pit.get_factor_matrix('leverage', pool, START, END)
value = composite_from_matrices(mats, directions=chain.PRODUCTION_FACTORS)
value = value[value.columns.intersection(cache.keys())]
momentum = _build_momentum_matrix(cache, START, END)
vm = _combine_value_momentum(value, momentum, chain.MOMENTUM_WEIGHT)  # 0.8vz+0.2mz


def zrow(mx, clip=3.0):
    z = mx.sub(mx.mean(axis=1), axis=0).div(mx.std(axis=1) + 1e-12, axis=0)
    return z.clip(-clip, clip)


qz = zrow(roe) - zrow(lev)          # 质量z（杠杆负向）
qz = qz.reindex_like(vm).fillna(0.0)
vm_z = zrow(vm)                     # 与生产同口径：先合成再截面z

# ST 矩阵
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
                       use_risk_gate=gate,
                       st_data_required=st_matrix is not None and gate)
    return DailyReplayBacktester(cache, cfg).run(
        sc, start_date=str(SIGNAL_START), end_date=str(END))


def era_annual(r):
    eq = r.equity_curve
    out = {}
    for a, b, tag in (('2017', '2021', '17-21'), ('2022', '2026', '22-26')):
        seg = eq.loc[f'{a}':f'{b}']
        if len(seg) > 2:
            yrs = (seg.index[-1] - seg.index[0]).days / 365.25
            out[tag] = float(seg.iloc[-1] / seg.iloc[0]) ** (1 / yrs) - 1
    return out


print("\nA口径：基线 …", flush=True)
bA = run(base_score)
eb = era_annual(bA)
print(f"  A基线: {bA.total_return:+.1%} 夏普{bA.sharpe:.2f} 回撤{bA.max_drawdown:.1%} "
      f"| 年化 {eb['17-21']:+.1%}/{eb['22-26']:+.1%}", flush=True)

results = {}
for wq in WQ:
    sc = vm_z + wq * qz
    print(f"A口径：wq={wq} …", flush=True)
    r = run(sc)
    e = era_annual(r)
    results[wq] = (r, e)
    print(f"  wq={wq}: {r.total_return:+.1%} 夏普{r.sharpe:.2f} 回撤{r.max_drawdown:.1%} "
          f"| 年化 {e['17-21']:+.1%}/{e['22-26']:+.1%}", flush=True)

# C口径交叉验证：基线 vs 最接近判据的 wq（全部三档更稳妥，成本可控就全跑）
print("\nC口径（无闸门，防路径运气）…", flush=True)
bC = run(base_score, gate=False)
print(f"  C基线: {bC.total_return:+.1%} 回撤{bC.max_drawdown:.1%}", flush=True)
c_res = {}
for wq in WQ:
    r = run(vm_z + wq * qz, gate=False)
    c_res[wq] = r
    print(f"  C wq={wq}: {r.total_return:+.1%} 回撤{r.max_drawdown:.1%}", flush=True)

# 判定
print(f"""
{'=' * 72}
③ 预注册判定（判据见 docs/events研究方案.md）
{'=' * 72}""")
ok_wq = []
for wq in WQ:
    r, e = results[wq]
    c1 = r.total_return >= 0.57 and r.max_drawdown >= -0.27
    c2 = c_res[wq].total_return >= 0.546
    c4 = (e['17-21'] >= eb['17-21'] - 0.02) and (e['22-26'] >= eb['22-26'] - 0.02)
    ok = c1 and c2 and c4
    ok_wq.append(ok)
    print(f"  wq={wq}: A{'✅' if c1 else '❌'}({r.total_return:+.1%}/{r.max_drawdown:.1%}) "
          f"C{'✅' if c2 else '❌'}({c_res[wq].total_return:+.1%}) "
          f"分期{'✅' if c4 else '❌'} → {'过' if ok else '不过'}")
c3 = all(ok_wq)
verdict = (f"✅ 通过——质量连续评分进入五关候选（最优wq待band敏感性复核）" if c3
           else ("⚠️ 部分wq过——不足以采纳（三档全过才可信）" if any(ok_wq)
                 else "❌ 未达——维持硬过滤（现行生产语义不变）"))
print(f"  三档全过: {'✅' if c3 else '❌'}\n  综合: {verdict}")
