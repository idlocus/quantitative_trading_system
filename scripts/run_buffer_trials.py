#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""G 系：缓冲带持有型日频（预注册见 FACTOR_PREREG G 系节）。"""
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

DEST = Path('ai-created/research/buffer_trial_k80_2026-11-10')


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
    p3a1 = (((d16.rank(axis=1, pct=True) + bp_rank) / 2.0)
            .where(~loss).where(d7_rank < 0.90))
    atomic_json(DEST / 'inputs.json', evidence)

    composite = {'G0_buffer80': p3a1}
    protocol = dict(PORTFOLIO_PROTOCOL, study='hold_buffer_daily',
                    risk_variants=['budget_guard'], rebalance_sessions=1,
                    hold_buffer_ranks={'G0_buffer80': 80})
    reports = portfolio_layer(composite, eligible, d16, base, evidence,
                              protocol, dest=DEST)

    print('\n== G 系：缓冲带持有型日频（对照 F1 日频 -0.03%/2.6x，月频基线 +0.06%）==')
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
    register('G系K80补测: 见buffer_trial_k80证据', label='hold_buffer_daily')
    print(f"\n证据: {DEST}")


if __name__ == '__main__':
    main()
