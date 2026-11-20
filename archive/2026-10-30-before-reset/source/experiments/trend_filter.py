#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
趋势过滤研究（2026-10-21 预注册）：过滤 ≠ 叠加——保价值排序，只剔除下跌趋势股
F1 硬过滤(6m动量≤0) / F2 均线(<MA120) / F3 软过滤(动量底部20%)
判据：A≥+61% 且 C≥+58.6% 且两期不塌方；顺周期检查=禁买天数+2019反弹参与度。
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
from trading.strategy import build_strategy_score, _build_momentum_matrix

START, END = date(2016, 6, 1), date(2026, 10, 9)
SIGNAL_START = date(2017, 1, 3)   # 首个交易日（非假期）——DB路径可用；
                                   # 旧实验用1月1日（假期）静默走了WindPy回退，
                                   # 对终端形成隐藏依赖（2026-10-21发现）

ds = LayeredDataSource()
pool = sorted(ds.list_symbols(as_of=SIGNAL_START))
score, cache, _ = build_strategy_score(ds, pool, START, END)
assert score.shape[1] > 100, f'池异常: {score.shape}'   # 空池硬门禁
print(f"分数矩阵: {score.shape}", flush=True)

# 动量与均线（复权口径）
px = close_matrix(cache, basis='adjusted')
mom = _build_momentum_matrix(cache, START, END)          # 6m剔1m（与生产同）
ma120 = px.rolling(120).mean()
px_al = px.reindex(mom.index)

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
    # 2019 反弹参与度
    seg = eq.loc['2019-01':'2019-04']
    out['rebound19'] = float(seg.iloc[-1] / seg.iloc[0] - 1) if len(seg) > 2 else np.nan
    return out


# 三种过滤
f1 = score.copy()
f1[mom.reindex_like(score) <= 0] = np.nan                # 硬过滤
f2 = score.copy()
f2[px_al.reindex_like(score) < ma120.reindex_like(score)] = np.nan   # 均线过滤
f3 = score.copy()
mq = mom.reindex_like(score)
q20 = mq.quantile(0.2, axis=1)                           # 软过滤：底部20%
f3[mq.lt(q20, axis=0)] = np.nan

VARIANTS = {'基线': score, 'F1硬过滤': f1, 'F2均线MA120': f2, 'F3软过滤(底20%)': f3}

print(f"\n{'='*84}")
print(f"{'口径':<14} {'A收益':>8} {'A夏普':>7} {'A回撤':>7} {'C收益':>8} "
      f"{'年化17-21':>9} {'年化22-26':>9} {'19年反弹':>9}")
print(f"{'='*84}")
res = {}
for name, sc in VARIANTS.items():
    rA = run(sc, gate=True)
    rC = run(sc, gate=False)
    e = era_annual(rA)
    res[name] = (rA, rC, e)
    print(f"{name:<14} {rA.total_return:>+8.1%} {rA.sharpe:>7.2f} {rA.max_drawdown:>7.1%} "
          f"{rC.total_return:>+8.1%} {e['e1']:>+9.1%} {e['e2']:>+9.1%} "
          f"{e['rebound19']:>+9.1%}", flush=True)

bA, bC, be = res['基线']
print(f"\n{'='*84}")
print("预注册判定（A≥+61% 且 C≥+58.6% 且 两期不塌方 且 反弹不缺席）")
for name in ('F1硬过滤', 'F2均线MA120', 'F3软过滤(底20%)'):
    rA, rC, e = res[name]
    c1 = rA.total_return >= 0.61
    c2 = rC.total_return >= 0.586
    c3 = e['e1'] >= be['e1'] - 0.02 and e['e2'] >= be['e2'] - 0.02
    c4 = e['rebound19'] >= be['rebound19'] - 0.01
    ok = c1 and c2 and c3 and c4
    print(f"  {name}: A{'✅' if c1 else '❌'}({rA.total_return:+.1%}) "
          f"C{'✅' if c2 else '❌'}({rC.total_return:+.1%}) "
          f"分期{'✅' if c3 else '❌'} 反弹{'✅' if c4 else '❌'}"
          f"({e['rebound19']:+.1%} vs 基线{be['rebound19']:+.1%}) → "
          f"{'✅ 候选' if ok else '❌'}")
