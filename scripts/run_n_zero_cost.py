#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""N 系：毛收益上限探索（0bp 诊断，预注册见 N 系节）。"""
import sys, os
if __name__ == '__main__':
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from pathlib import Path
import numpy as np
import pandas as pd

from data.preparation import atomic_json
from research import technical_signals as ts
from research import portfolio_metrics as pm
from research.factor_screen2 import PROTOCOL, load_ohlc_panel

DEST = Path('ai-created/research/n_zero_cost_2026-11-11')


def n2_limitup_events(panel, price, eligible):
    """017-P1 口径 + 条件分组变量。返回事件表。"""
    from trading.rules import price_limit_ratio
    close, open_px, vol = panel['close'], panel['open'], panel['volume']
    prev = close.shift(1)
    rows = []
    syms_ratio = {s: price_limit_ratio(s) for s in price.columns}
    dates = price.index
    for i in range(1, len(dates) - 2):
        d = dates[i]
        d1, d2 = dates[i + 1], dates[i + 2]
        for s in price.columns:
            c0, c1 = close.at[d, s], prev.at[d, s]
            o1 = open_px.at[d1, s]
            if not (np.isfinite(c0) and np.isfinite(c1) and np.isfinite(o1)):
                continue
            ratio = syms_ratio[s]
            if c0 / c1 - 1 < ratio - 0.003:
                continue                       # 非涨停
            if o1 / c0 - 1 >= ratio - 0.005:
                continue                       # T+1 一字开盘不可买
            o2 = open_px.at[d2, s]
            if not np.isfinite(o2):
                continue
            ret = o2 / o1 - 1                  # 1日持有毛收益
            # 分组变量（全部 T 日可知）
            hi52 = close.at[d, s] / (close[s].loc[:d].rolling(250, min_periods=60).max().iloc[-1])
            vr = vol.at[d, s] / vol[s].loc[:d].rolling(20, min_periods=20).mean().iloc[-1]
            rows.append({'date': d, 'symbol': s, 'ret': ret,
                         'px': c0, 'hi52': hi52, 'vr': vr})
    return pd.DataFrame(rows)


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    panel, base, evidence = load_ohlc_panel(PROTOCOL)
    price = base['price']
    eligible = (base['usable']
                & price.notna().rolling(PROTOCOL['minimum_history_sessions']).sum()
                .eq(PROTOCOL['minimum_history_sessions'])
                & base['amount'].rolling(PROTOCOL['liquidity_window'], min_periods=PROTOCOL['liquidity_window'])
                .median().ge(PROTOCOL['minimum_median_daily_amount_local_currency']))
    atomic_json(DEST / 'inputs.json', evidence)

    # N1: 多头最强合成 0bp 组合（诊断口径）
    from research.composite_study import PORTFOLIO_PROTOCOL, portfolio_layer
    signals = ts.build_all_signals(panel)
    d16 = signals['D16_rel_atr']
    n1 = (signals['A5_overnight_share'].rank(axis=1, pct=True)
          + signals['A6_vol_term'].rank(axis=1, pct=True)
          + signals['D15_psy12'].rank(axis=1, pct=True)
          + signals['D16_rel_atr'].rank(axis=1, pct=True)
          + signals['A1_risk_adj_reversal'].rank(axis=1, pct=True)) / 5.0
    n1 = n1.where(eligible)
    p1 = dict(PORTFOLIO_PROTOCOL, study='n1_zero_cost_top', risk_variants=['uncontrolled'],
              rebalance_sessions=1, one_way_cost_scenarios=[0.0])
    rep1 = portfolio_layer({'N1_top5_0bp': n1}, eligible, d16, base, evidence, p1,
                           dest=DEST / 'N1_0bp', cost=0.0)
    for name, full in rep1.items():
        if 'benchmark' in name:
            continue
        m = full['metrics']
        va = full['splits']['validation']['geometric_monthly_return']
        nav = pd.read_csv(DEST / 'N1_0bp' / f'{name}_uncontrolled_0bps_nav.csv')
        tr = pd.read_csv(DEST / 'N1_0bp' / f'{name}_uncontrolled_0bps_trades.csv')
        to = pm.amount_turnover_per_month(tr, nav)
        print(f"N1 {name}: 毛月收益 {m['geometric_monthly_return']:+.2%} | "
              f"毛验证月 {va:+.2%} | 换手 {to:.1f}x | maxDD {abs(m['max_drawdown']):.1%}")
    # N2: 涨停事件条件分组（先跑统计——快）
    print('N2 涨停事件扫描中...', flush=True)
    ev = n2_limitup_events(panel, price, eligible)
    ev.to_csv(DEST / 'limitup_events.csv', index=False, encoding='utf-8-sig')
    print(f'事件总数: {len(ev)}（0bp，1日持有）')
    print('\n== 全体与条件分组（日均毛收益 bp；候选线=+15bp）==')
    print(f"全体: {ev.ret.mean()*10000:+.1f}bp | 中位 {ev.ret.median()*10000:+.1f}bp | P75 {ev.ret.quantile(0.75)*10000:+.1f}bp")
    ev['hi52_q'] = pd.qcut(ev.hi52, 3, labels=['低位', '中位', '高位'])
    ev['vr_q'] = pd.qcut(ev.vr, 3, labels=['缩量', '常量', '放量'])
    for col in ('hi52_q', 'vr_q'):
        for g, sub in ev.groupby(col):
            if len(sub) > 100:
                print(f"  {col}={g}: n={len(sub):>6} 均值 {sub.ret.mean()*10000:+.1f}bp 中位 {sub.ret.median()*10000:+.1f}bp")
    # 交叉
    for hi in ('低位', '高位'):
        for vr in ('缩量', '放量'):
            sub = ev[(ev.hi52_q == hi) & (ev.vr_q == vr)]
            if len(sub) > 100:
                print(f"  {hi}×{vr}: n={len(sub):>6} 均值 {sub.ret.mean()*10000:+.1f}bp 中位 {sub.ret.median()*10000:+.1f}bp")
    from research.factor_screen2 import register
    register(f'N系0bp: 涨停事件{len(ev)}条件分组', label='n1_zero_cost_top')
    print(f'\n证据: {DEST}')


if __name__ == '__main__':
    main()
