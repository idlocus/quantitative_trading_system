#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K 系：强势股策略族 vs G1 正面对比（预注册见 FACTOR_PREREG K 系节）。"""
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
from research.factor_screen2 import PROTOCOL, load_ohlc_panel

DEST = Path('ai-created/research/k_trials_2026-11-10')


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
    pe = pd.read_parquet(fund / 'pe_ttm.parquet')
    loss = pe.reindex(index=price.index, columns=price.columns).isna()

    rps120 = signals['RPS_120']
    d7 = signals['D7_expma_gap']
    d12_rank = signals['D12_ma_align'].rank(axis=1, pct=True)
    d16_rank = d16.rank(axis=1, pct=True)
    composite = {
        'M1_rps120': rps120.where(~loss),
        'M2_expma': d7.where(~loss),
        'M3_trend_comp': ((rps120.rank(axis=1, pct=True) + d12_rank) / 2.0).where(~loss),
        'M4_rps_lowvol': ((rps120.rank(axis=1, pct=True) + d16_rank) / 2.0).where(~loss),
    }
    atomic_json(DEST / 'inputs.json', evidence)
    protocol = dict(PORTFOLIO_PROTOCOL, study='k_momentum_vs_g1',
                    risk_variants=['budget_guard'], rebalance_sessions=1,
                    hold_buffer_ranks={k: 100 for k in composite})
    reports = portfolio_layer(composite, eligible, d16, base, evidence,
                              protocol, dest=DEST)

    print('\n== K 系：强势股策略 vs G1（同形态公平对比）==')
    print('G1 基线: +0.06%/验证+0.08%/maxDD12.7%/换手2.0x')
    for name, full in reports.items():
        if 'benchmark' in name:
            continue
        m = full['metrics']
        va = full['splits']['validation']['geometric_monthly_return']
        tr_va = full['splits']['train']['geometric_monthly_return']
        nav = pd.read_csv(DEST / f'{name}_nav.csv')
        tr_csv = pd.read_csv(DEST / f'{name}_trades.csv')
        from research.portfolio_metrics import amount_turnover_per_month
        to = amount_turnover_per_month(tr_csv, nav)
        print(f"{name:<20} 月收益 {m['geometric_monthly_return']:+.2%} | "
              f"训练月 {tr_va:+.2%} | 验证月 {va:+.2%} | "
              f"maxDD {abs(m['max_drawdown']):.1%} | 换手 {to:.1f}x | "
              f"qualifies={full.get('qualifies_research_candidate')}")
    from research.factor_screen2 import register
    register('K系强势vs G1对比4run: 见k_trials证据', label='k_momentum_vs_g1')
    print(f"\n证据: {DEST}")


if __name__ == '__main__':
    main()
