#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""I 系：热卖出规则（因子作出口信号，预注册见 FACTOR_PREREG I 系节）。"""
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

DEST = Path('ai-created/research/i_trials_2026-11-19')  # 审计2026-11-19重跑(热卖滞后生效+不回补)


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
    g1_score = ((d16.rank(axis=1, pct=True) + bp_rank) / 2.0).where(excl)

    d6_rank = signals['D6_rsi14'].rank(axis=1, pct=True)
    hot90 = d7_rank >= 0.90
    hot80 = d7_rank >= 0.80
    hot_double = hot90 | (d6_rank >= 0.95)

    def to_sets(mask):
        return {day: set(mask.columns[mask.loc[day].fillna(False)])
                for day in mask.index}

    composite = {'I1_hot90': g1_score, 'I2_hot80': g1_score, 'I3_double': g1_score}
    protocol = dict(PORTFOLIO_PROTOCOL, study='i_hot_exit',
                    risk_variants=['budget_guard'], rebalance_sessions=1,
                    hold_buffer_ranks={k: 100 for k in composite},
                    hot_exit_sets={'I1_hot90': to_sets(hot90),
                                   'I2_hot80': to_sets(hot80),
                                   'I3_double': to_sets(hot_double)})
    atomic_json(DEST / 'inputs.json', evidence)
    print(f"热态日均股票数: hot90={int((hot90&eligible).sum(axis=1).mean())} "
          f"hot80={int((hot80&eligible).sum(axis=1).mean())} double={int((hot_double&eligible).sum(axis=1).mean())}", flush=True)
    reports = portfolio_layer(composite, eligible, d16, base, evidence,
                              protocol, dest=DEST)

    print('\n== I 系：热卖出规则（基线 G1: +0.06%/验证+0.08%/12.7%/2.0x）==')
    for name, full in reports.items():
        if 'benchmark' in name:
            continue
        m = full['metrics']
        va = full['splits']['validation']['geometric_monthly_return']
        nav = pd.read_csv(DEST / f'{name}_nav.csv')
        tr = pd.read_csv(DEST / f'{name}_trades.csv')
        to = pm.amount_turnover_per_month(tr, nav)
        print(f"{name:<20} 月收益 {m['geometric_monthly_return']:+.2%} | "
              f"验证月 {va:+.2%} | maxDD {abs(m['max_drawdown']):.1%} | "
              f"月换手 {to:.1f}x | qualifies={full.get('qualifies_research_candidate')}")
    from research.factor_screen2 import register
    register('I系热卖出3run: 见i_trials证据', label='I系热卖出')
    print(f"\n证据: {DEST}")


if __name__ == '__main__':
    main()
