#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ETF 多资产趋势策略验证 —— 经由唯一实现（trading/etf_strategy.py）

2026-10-15 集成版：真实 ETF 代码（510300/510500/511010/518880，2015 起、
含黄金共 4 资产），替代原型（指数、3 资产、2021-10 起）。
预注册淘汰标准（与原型一致，一条不满足即不集成）：
  1. 趋势版本净收益 > 0
  2. 最大回撤 < 全股票持有（300ETF）的 50%
  3. 夏普 > 60/40 固定比例
  4. 参数 N=3/6/9/12 个月全正
对照：300ETF 持有 / 60/40 股债 / 等权月再平衡。
信号时点：月度调仓日以 T-1 收盘判定趋势（无前视），成交=调仓日收盘。
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import date

import numpy as np
import pandas as pd

from data.layered import LayeredDataSource
from data.semantics import close_matrix
from trading import config as chain
from trading.etf_strategy import build_etf_signal

START = date(2015, 1, 1)
END = date(2026, 10, 9)
COST = chain.ETF_COST_RATE

ds = LayeredDataSource()
codes = list(chain.ETF_ASSETS)
print(f"资产: {codes} | {START} ~ {END}", flush=True)
loader_cache = ds.preload_batch(codes, START, END)
px = close_matrix(loader_cache, basis='adjusted')
print(f"价格矩阵: {px.shape[0]}日 × {px.shape[1]}只"
      f"（{px.index[0].date()} ~ {px.index[-1].date()}）", flush=True)

dates = px.index
month_starts, seen = [], set()
for d in dates:
    m = (d.year, d.month)
    if m not in seen:
        seen.add(m)
        month_starts.append(d)


def backtest(lookback: int):
    nav, holdings, costs = 1.0, {}, 0.0
    curve, cur = {}, holdings
    for i, t in enumerate(month_starts):
        # T-1 信号：t 的目标基于 ≤ 前一交易日的数据
        if i > lookback // 21 + 1:
            prev_day = dates[dates.searchsorted(t, side='left') - 1]
            targets, _, _ = build_etf_signal(ds, prev_day.date(), lookback)
        else:
            targets = {}
        # t 日收盘执行调仓（换手计成本）
        old = set(holdings)
        new = set(targets)
        turnover = len(old ^ new) / max(len(old | new), 1)
        costs += turnover * COST
        holdings = dict(targets)
        # 持有期收益：t → 下月首日
        if i + 1 < len(month_starts):
            t2 = month_starts[i + 1]
            if holdings:
                r = float(np.mean([px.loc[t2, c] / px.loc[t, c] - 1
                                   for c in holdings]))
            else:
                r = 0.0
            nav *= (1 + r)
        curve[t] = nav - costs
    s = pd.Series(curve).sort_index()
    rets = s.pct_change().dropna()
    total = float(s.iloc[-1] / s.iloc[0] - 1)
    years = (s.index[-1] - s.index[0]).days / 365.25
    ann = (1 + total) ** (1 / years) - 1
    sharpe = float(rets.mean() / rets.std() * np.sqrt(12)) if rets.std() > 0 else 0
    dd = float((s / s.cummax() - 1).min())
    return {'total': total, 'ann': ann, 'sharpe': sharpe, 'dd': dd,
            'cost': costs, 'curve': s}


def buy_hold(sym):
    s = px[sym].dropna()
    s = s / s.iloc[0]
    rets = s.resample('ME').last().pct_change().dropna()
    dd = float((s / s.cummax() - 1).min())
    return {'total': float(s.iloc[-1] - 1),
            'sharpe': float(rets.mean() / rets.std() * np.sqrt(12)) if rets.std() > 0 else 0,
            'dd': dd}


def fixed_ratio():
    w = {'510300.SH': 0.6, '511010.SH': 0.4}
    s = sum(px[c].dropna() / px[c].dropna().iloc[0] * v for c, v in w.items())
    rets = s.resample('ME').last().pct_change().dropna()
    return {'total': float(s.iloc[-1] - 1),
            'sharpe': float(rets.mean() / rets.std() * np.sqrt(12)) if rets.std() > 0 else 0,
            'dd': float((s / s.cummax() - 1).min())}


print("\n参数扫描（经由唯一实现 build_etf_signal）:", flush=True)
results = {}
for lb in (63, 126, 189, 252):
    results[f'趋势{lb // 21}月'] = backtest(lb)
    r = results[f'趋势{lb // 21}月']
    print(f"  趋势{lb // 21}月: 净{r['total']:+.1%} 年化{r['ann']:+.1%} "
          f"夏普{r['sharpe']:.2f} 回撤{r['dd']:.1%} 成本{r['cost']:.1%}", flush=True)

bh300 = buy_hold('510300.SH')
bh500 = buy_hold('510500.SH')
bhgld = buy_hold('518880.SH')
fx = fixed_ratio()
print(f"\n--- 对照（同窗口 ETF 代码） ---")
print(f"  持有300ETF:  净{bh300['total']:+.1%} 夏普{bh300['sharpe']:.2f} 回撤{bh300['dd']:.1%}")
print(f"  持有500ETF:  净{bh500['total']:+.1%} 夏普{bh500['sharpe']:.2f} 回撤{bh500['dd']:.1%}")
print(f"  持有黄金ETF: 净{bhgld['total']:+.1%} 夏普{bhgld['sharpe']:.2f} 回撤{bhgld['dd']:.1%}")
print(f"  60/40股债:   净{fx['total']:+.1%} 夏普{fx['sharpe']:.2f} 回撤{fx['dd']:.1%}")

# ---- 预注册标准判定 ----
base = results['趋势6月']
c1 = base['total'] > 0
c2 = base['dd'] > bh300['dd'] / 2
c3 = base['sharpe'] > fx['sharpe']
c4 = all(results[f'趋势{lb // 21}月']['total'] > 0 for lb in (63, 126, 189, 252))
print(f"""
{'=' * 64}
预注册标准判定（趋势6月为基准）
{'=' * 64}
  1. 净收益>0:              {'✅' if c1 else '❌'} ({base['total']:+.1%})
  2. 回撤<300持有50%:       {'✅' if c2 else '❌'} ({base['dd']:.1%} vs {bh300['dd'] / 2:.1%})
  3. 夏普>60/40:            {'✅' if c3 else '❌'} ({base['sharpe']:.2f} vs {fx['sharpe']:.2f})
  4. 参数3/6/9/12全正:      {'✅' if c4 else '❌'}
  → {'全部通过：可集成' if all([c1, c2, c3, c4]) else '未通过：不集成（诚实报告）'}
""")

base['curve'].to_csv('ai-created/etf_trend_validation_nav.csv', encoding='utf-8')
print("净值曲线: ai-created/etf_trend_validation_nav.csv")
