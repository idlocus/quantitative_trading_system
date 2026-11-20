#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
持仓集中度实验·右半边：TopN 120/150/200（2026-10-20）

问题：既然"个股噪音该分散"，持仓越多越好吗？
锚点：Top100=+56.6%(C)；理论极限 N=全宇宙≈可交易等权 -14.4%。
预期：右半边单调下降（稀释行业暴露→向宇宙 beta 收敛），
与左半边（30-70 劣化）合成驼峰曲线，100 附近是肘部。
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import date

import pandas as pd

from data.layered import LayeredDataSource
from portfolio.daily_replay import DailyReplayBacktester, ReplayConfig
from trading import config as chain
from trading.strategy import build_strategy_score

START, END = date(2016, 6, 1), date(2026, 10, 9)
SIGNAL_START = date(2017, 1, 1)
TOPNS = (120, 150, 200)

ds = LayeredDataSource()
pool = sorted(ds.list_symbols(as_of=SIGNAL_START))
score, cache, _ = build_strategy_score(ds, pool, START, END)
print(f"分数矩阵: {score.shape}", flush=True)

st_matrix = None
try:
    from data.st_provider import STProvider
    idx = pd.DatetimeIndex(pd.to_datetime(next(iter(cache.values()))['date']))
    st_matrix = STProvider(ds).st_matrix(idx, pool)
except Exception as e:
    print(f"警告：ST不可用（{e}）", flush=True)


def run(top_n, gate=True):
    cfg = ReplayConfig(top_n=top_n, st_matrix=st_matrix, band=chain.BUFFER_BAND,
                       use_risk_gate=gate,
                       st_data_required=st_matrix is not None and gate)
    return DailyReplayBacktester(cache, cfg).run(
        score, start_date=str(SIGNAL_START), end_date=str(END))


# ── 可交易等权极限（N=全宇宙）作右端锚点 ──
from portfolio.backtester import PortfolioConfig
from portfolio.tradable_benchmark import tradable_equal_weight
ctrl_cfg = PortfolioConfig(top_n=chain.TOP_N, rebalance_days=20,
                           st_matrix=st_matrix)
_, rEW, _ = tradable_equal_weight(cache, ctrl_cfg, start_date=str(SIGNAL_START),
                                  end_date=str(END))
print(f"极限锚点（可交易等权≈N→∞）: {rEW.total_return:+.1%} "
      f"夏普{rEW.sharpe:.2f} 回撤{rEW.max_drawdown:.1%}", flush=True)

print(f"\n{'TopN':>5} {'A收益':>8} {'A夏普':>7} {'A回撤':>7} {'C收益':>8} {'C夏普':>7}")
rows = []
for top_n in TOPNS + (100,):        # 100 当轮复现
    rA = run(top_n, gate=True)
    rC = run(top_n, gate=False)
    print(f"{top_n:>5} {rA.total_return:>+8.1%} {rA.sharpe:>7.2f} "
          f"{rA.max_drawdown:>7.1%} {rC.total_return:>+8.1%} {rC.sharpe:>7.2f}",
          flush=True)
    rows.append({'top_n': top_n, 'A': rA.total_return,
                 'A_dd': rA.max_drawdown, 'C': rC.total_return,
                 'C_sharpe': rC.sharpe})
pd.DataFrame(rows).to_csv('ai-created/topn_right.csv', index=False,
                          encoding='utf-8')
print("\n左右合成的完整曲线（C口径·干净信号）:")
print("  30:+18.9% → 50:+32.6% → 70:+54.8% → 100:+56.6% → "
      + " → ".join(f"{r['top_n']}:{r['C']:+.1%}" for r in rows)
      + f" → ∞:{rEW.total_return:+.1%}")
