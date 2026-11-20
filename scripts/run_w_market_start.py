#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""W 系：市场启动日指标叠加 G1（预注册见 W 系节）。"""
import sys, os
if __name__ == '__main__':
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from pathlib import Path
import pandas as pd

from data.preparation import atomic_json
from research import technical_signals as ts
from research.composite_study import PORTFOLIO_PROTOCOL, portfolio_layer
from research import portfolio_metrics as pm
from research.factor_screen2 import PROTOCOL, load_ohlc_panel

DEST = Path('ai-created/research/w_market_start_2026-11-19')  # 审计2026-11-19重跑(候选深度修复:K200/K300真实生效)


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    panel, base, evidence = load_ohlc_panel(PROTOCOL)
    price = base['price']
    eligible = (base['usable']
                & price.notna().rolling(PROTOCOL['minimum_history_sessions']).sum()
                .eq(PROTOCOL['minimum_history_sessions'])
                & base['amount'].rolling(PROTOCOL['liquidity_window'], min_periods=PROTOCOL['liquidity_window'])
                .median().ge(PROTOCOL['minimum_median_daily_amount_local_currency']))
    signals = ts.build_all_signals(panel)
    d16 = signals['D16_rel_atr']
    fund = Path('data_cache/fundamental')
    pb = pd.read_parquet(fund / 'pb_lf.parquet')
    pe = pd.read_parquet(fund / 'pe_ttm.parquet')
    bp_rank = (1.0 / pb.where(pb > 0)).reindex(
        index=price.index, columns=price.columns).rank(axis=1, pct=True)
    loss = pe.reindex(index=price.index, columns=price.columns).isna()
    d7_rank = signals['D7_expma_gap'].rank(axis=1, pct=True)
    g1 = (((d16.rank(axis=1, pct=True) + bp_rank) / 2.0)
          .where(~loss).where(d7_rank < 0.90))
    atomic_json(DEST / 'inputs.json', evidence)

    # 市场启动日指标（PIT: T日收盘判定 → T+1日生效）。
    # 审计 2026-11-19 一-3：覆盖键=**决策日**（启动日当日），仿真器内部
    # 平移到下一交易日生效——此前脚本自行映射到 T+1 而仿真器无平移，
    # 与权重/热卖等口径不一致。
    mkt = pd.read_csv('runtime/market_start_indicator.csv')
    mkt['date'] = pd.to_datetime(mkt['date'])
    start_dates = set(mkt.loc[mkt['market_start'] == True, 'date'])
    print(f'启动日(决策日): {len(start_dates)}（于下一交易日生效）', flush=True)

    # 构造日度覆盖：{decision_date: K}
    # W1: 启动日次日 K=99999（几乎不卖）
    # W2: 启动日次日 K=200
    # W3: 启动日次日 K=300
    w1_overrides = {d: 99999 for d in start_dates}
    w2_overrides = {d: 200 for d in start_dates}
    w3_overrides = {d: 300 for d in start_dates}

    composite = {'W1_freeze': g1, 'W2_k200': g1, 'W3_k300': g1}
    protocol = dict(PORTFOLIO_PROTOCOL, study='w_market_start_enhance',
                    risk_variants=['budget_guard'], rebalance_sessions=1,
                    hold_buffer_ranks={k: 100 for k in composite},
                    daily_buffer_overrides={'W1_freeze': w1_overrides,
                                           'W2_k200': w2_overrides,
                                           'W3_k300': w3_overrides})
    reports = portfolio_layer(composite, eligible, d16, base, evidence,
                              protocol, dest=DEST)

    print('\n== W 系：市场启动日叠加 G1（基线 G1: +0.06%/验证+0.08%/11.8%）==')
    for name, full in reports.items():
        if 'benchmark' in name:
            continue
        m = full['metrics']
        va = full['splits']['validation']['geometric_monthly_return']
        nav = pd.read_csv(DEST / f'{name}_nav.csv')
        tr = pd.read_csv(DEST / f'{name}_trades.csv')
        to = pm.amount_turnover_per_month(tr, nav)
        print(f"{name:<16} 月收益 {m['geometric_monthly_return']:+.2%} | "
              f"验证月 {va:+.2%} | maxDD {abs(m['max_drawdown']):.1%} | "
              f"换手 {to:.1f}x | qualifies={full.get('qualifies_research_candidate')}")

    from research.factor_screen2 import register
    register('W系市场启动日叠加3run: 见w_market_start证据', label='W系市场启动日叠加')
    print(f'\n证据: {DEST}')


if __name__ == '__main__':
    main()
