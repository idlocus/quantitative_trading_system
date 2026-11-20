#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""H 系：G1 基底（日频缓冲带K100）上的因子增强（预注册见 FACTOR_PREREG H 系节）。"""
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

DEST = Path('ai-created/research/h_trials_2026-11-10')


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
    excl = ~loss & (d7_rank < 0.90)
    d16_rank = d16.rank(axis=1, pct=True)
    oversold = (signals['D3_kdj_j'].rank(axis=1, pct=True)
                + signals['D4_boll_pctb'].rank(axis=1, pct=True)
                + signals['D13_wr14'].rank(axis=1, pct=True)) / 3.0
    composite = {
        'H1_oversold3': ((d16_rank + bp_rank + oversold.rank(axis=1, pct=True)) / 3.0).where(excl),
        'H2_pctb': ((d16_rank + bp_rank + signals['D4_boll_pctb'].rank(axis=1, pct=True)) / 3.0).where(excl),
        'H3_idio_rev': ((d16_rank + bp_rank + signals['A2_idio_reversal'].rank(axis=1, pct=True)) / 3.0).where(excl),
    }
    atomic_json(DEST / 'inputs.json', evidence)

    protocol = dict(PORTFOLIO_PROTOCOL, study='h_enhance_g1',
                    risk_variants=['budget_guard'], rebalance_sessions=1,
                    hold_buffer_ranks={k: 100 for k in composite})
    reports = portfolio_layer(composite, eligible, d16, base, evidence,
                              protocol, dest=DEST)

    print('\n== H 系：G1 基底因子增强（基线 G1: +0.06%/验证+0.08%/12.7%/2.0x）==')
    for name, full in reports.items():
        if 'benchmark' in name:
            continue
        m = full['metrics']
        va = full['splits']['validation']['geometric_monthly_return']
        nav = pd.read_csv(DEST / f'{name}_nav.csv')
        tr = pd.read_csv(DEST / f'{name}_trades.csv')
        to = pm.amount_turnover_per_month(tr, nav)
        print(f"{name:<22} 月收益 {m['geometric_monthly_return']:+.2%} | "
              f"验证月 {va:+.2%} | maxDD {abs(m['max_drawdown']):.1%} | "
              f"月换手 {to:.1f}x | qualifies={full.get('qualifies_research_candidate')}")
    from research.factor_screen2 import register
    register('H系G1增强3run: 见h_trials证据', label='h_enhance_g1')
    print(f"\n证据: {DEST}")


if __name__ == '__main__':
    main()
