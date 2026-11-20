#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ETF sleeve 配比敏感性（2026-10-20，评审P2-1采纳为研究项）

用两条已验证的十年收益流（股票A口径 + ETF趋势6月基准）按不同权重组合，
月度再平衡到目标配比。纯研究——不改生产（生产变更走A3窗口）。
判读：组合夏普/回撤/两期一致性；注意ETF引擎2015-2026的超额大部分来自
债+金牛市，高配比=对"该趋势延续"下注（年代依赖要在两期分层数字里显形）。
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
import json

# 股票 sleeve：A口径十年净值（工件 equity）
with open('results/data/portfolio_composite_v5_full_2017_2026_5因子生.json',
          encoding='utf-8') as f:
    art = json.load(f)
stock = pd.Series(art['equity']['values'],
                  index=pd.to_datetime(art['equity']['dates']))
stock = stock / stock.iloc[0]

# ETF sleeve：趋势6月基准净值曲线（重验证工件CSV）
etf = pd.read_csv('ai-created/etf_trend_validation_nav.csv',
                  index_col=0, parse_dates=True).iloc[:, 0]
etf = etf / etf.iloc[0]

common = stock.index.intersection(etf.index)
common = common[common >= '2017-01-03']
stock, etf = stock.loc[common], etf.loc[common]
rets = pd.DataFrame({'stock': stock.pct_change().fillna(0),
                     'etf': etf.pct_change().fillna(0)})

# 月度再平衡组合
month_ends = rets.index.to_series().groupby(rets.index.to_period('M')).tail(1)
print(f"共同样本: {common[0].date()} ~ {common[-1].date()}（{len(common)}日）")
print(f"{'配比ETF':>7} {'总收益':>8} {'年化':>7} {'夏普':>6} {'回撤':>7} "
      f"{'17-21年化':>9} {'22-26年化':>9}")
rows = []
for w in (0.0, 0.10, 0.30, 0.50, 0.70, 1.0):
    nav, cur = [], 1.0
    ws = {'stock': 1 - w, 'etf': w}
    for d, r in rets.iterrows():
        cur *= 1 + (ws['stock'] * r.stock + ws['etf'] * r.etf)
        nav.append(cur)
        if d in month_ends.values:            # 月度再平衡
            ws = {'stock': 1 - w, 'etf': w}
    s = pd.Series(nav, index=rets.index)
    yrs = (s.index[-1] - s.index[0]).days / 365.25
    total = s.iloc[-1] - 1
    ann = (1 + total) ** (1 / yrs) - 1
    r_ = s.pct_change().dropna()
    sharpe = r_.mean() / r_.std() * np.sqrt(252) if r_.std() > 0 else 0
    dd = (s / s.cummax() - 1).min()
    e = {}
    for a, b, tag in (('2017', '2021', 'e1'), ('2022', '2026', 'e2')):
        seg = s.loc[f'{a}':f'{b}']
        y2 = (seg.index[-1] - seg.index[0]).days / 365.25
        e[tag] = float(seg.iloc[-1] / seg.iloc[0]) ** (1 / y2) - 1
    print(f"{w:>7.0%} {total:>+8.1%} {ann:>+7.1%} {sharpe:>6.2f} {dd:>7.1%} "
          f"{e['e1']:>+9.1%} {e['e2']:>+9.1%}")
    rows.append({'w_etf': w, 'total': total, 'ann': ann, 'sharpe': sharpe,
                 'dd': dd, **e})
pd.DataFrame(rows).to_csv('ai-created/etf_ratio_sensitivity.csv',
                          index=False, encoding='utf-8')
print("\n判读要点：夏普峰值配比 vs 回撤代价；两期年化是否同号稳健；"
      "ETF高配比的两期数字揭示其对债金牛市的依赖度。决策走A3变更窗口。")
