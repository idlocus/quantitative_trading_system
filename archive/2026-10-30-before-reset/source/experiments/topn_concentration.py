#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
持仓集中度实验（2026-10-20 预注册）：TopN 30/50/70 vs 基线100

预注册判据（防路径运气——②的教训）：
  A口径十年 ≥ 基线(+59.0%)+3pp 且 C口径(无闸门) ≥ 基线(+56.6%)+3pp
  → 才算改进候选；同时报告回撤/换手/两期年化（风险画像）。
先验：100≥150≥200 单调（sweep 实测），方向有利；但 30-50 只的
特质性风险与止损冲击上升，回撤可能加深。
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import date

import numpy as np
import pandas as pd

from data.layered import LayeredDataSource
from portfolio.daily_replay import DailyReplayBacktester, ReplayConfig
from trading import config as chain
from trading.strategy import build_strategy_score

START, END = date(2016, 6, 1), date(2026, 10, 9)
SIGNAL_START = date(2017, 1, 1)
TOPNS = (30, 50, 70)

ds = LayeredDataSource()
pool = sorted(ds.list_symbols(as_of=SIGNAL_START))
print(f"池: {len(pool)} 只", flush=True)
score, cache, _ = build_strategy_score(ds, pool, START, END)
print(f"分数矩阵: {score.shape}", flush=True)

st_matrix = None
try:
    from data.st_provider import STProvider
    idx = pd.DatetimeIndex(pd.to_datetime(next(iter(cache.values()))['date']))
    st_matrix = STProvider(ds).st_matrix(idx, pool)
    print("ST矩阵就绪", flush=True)
except Exception as e:
    print(f"警告：ST不可用（{e}）", flush=True)


def run(top_n, gate=True):
    cfg = ReplayConfig(top_n=top_n, st_matrix=st_matrix, band=chain.BUFFER_BAND,
                       use_risk_gate=gate,
                       st_data_required=st_matrix is not None and gate)
    return DailyReplayBacktester(cache, cfg).run(
        score, start_date=str(SIGNAL_START), end_date=str(END))


def era_annual(r):
    eq = r.equity_curve
    out = {}
    for a, b, tag in (('2017', '2021', '17-21'), ('2022', '2026', '22-26')):
        seg = eq.loc[f'{a}':f'{b}']
        if len(seg) > 2:
            yrs = (seg.index[-1] - seg.index[0]).days / 365.25
            out[tag] = float(seg.iloc[-1] / seg.iloc[0]) ** (1 / yrs) - 1
    return out


print(f"\n{'='*76}")
print(f"{'TopN':>5} {'A收益':>8} {'A夏普':>7} {'A回撤':>7} {'A止损':>6} "
      f"{'C收益':>8} {'C回撤':>8} {'年化17-21':>10} {'年化22-26':>10}")
print(f"{'='*76}")
rows = []
for top_n in TOPNS + (100,):        # 100 = 当轮基线复现（须 +59.0%/+56.6%）
    rA = run(top_n, gate=True)
    rC = run(top_n, gate=False)
    e = era_annual(rA)
    print(f"{top_n:>5} {rA.total_return:>+8.1%} {rA.sharpe:>7.2f} {rA.max_drawdown:>7.1%} "
          f"{rA.n_stop_sells:>6d} {rC.total_return:>+8.1%} {rC.max_drawdown:>8.1%} "
          f"{e['17-21']:>+10.1%} {e['22-26']:>+10.1%}", flush=True)
    rows.append({'top_n': top_n, 'A': rA.total_return, 'A_dd': rA.max_drawdown,
                 'C': rC.total_return, 'C_dd': rC.max_drawdown,
                 'e1': e['17-21'], 'e2': e['22-26']})

d = pd.DataFrame(rows).set_index('top_n')
base = d.loc[100]
print(f"""
{'='*76}
预注册判定（A≥+62% 且 C≥+59.6% 才算改进候选）""")
for tn in TOPNS:
    row = d.loc[tn]
    ok = row.A >= base.A + 0.03 and row.C >= base.C + 0.03
    print(f"  Top{tn}: A{row.A:+.1%}(基线{row.A-base.A:+.1%}) "
          f"C{row.C:+.1%}({row.C-base.C:+.1%}) 回撤{row.A_dd:.1%} → "
          f"{'✅ 候选' if ok else '❌'}")
pd.DataFrame(rows).to_csv('ai-created/topn_concentration.csv',
                          index=False, encoding='utf-8')
