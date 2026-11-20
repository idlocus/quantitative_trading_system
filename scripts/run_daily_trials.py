#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""F 系：日持有测量（预注册见 FACTOR_PREREG F 系节）。"""
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

DEST = Path('ai-created/research/daily_trials_2026-11-10')


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
    excl = ~loss & (d7_rank < 0.90)                # P3+A1 的两个排除
    p3a1 = ((d16.rank(axis=1, pct=True) + bp_rank) / 2.0).where(excl)
    oversold = (signals['D3_kdj_j'].rank(axis=1, pct=True)
                + signals['D4_boll_pctb'].rank(axis=1, pct=True)
                + signals['D13_wr14'].rank(axis=1, pct=True)) / 3.0
    f2a = oversold.where(excl)
    f2b = ((d16.rank(axis=1, pct=True) + signals['D4_boll_pctb'].rank(axis=1, pct=True)) / 2.0).where(excl)
    atomic_json(DEST / 'inputs.json', evidence)

    def run(study, scores, rebalance):
        p = dict(PORTFOLIO_PROTOCOL, study=study, risk_variants=['budget_guard'],
                 rebalance_sessions=rebalance)
        return portfolio_layer(scores, eligible, d16, base, evidence, p,
                               dest=DEST / study)

    reports = {}
    reports.update(run('F1_daily_p3a1', {'F1_reb1': p3a1}, 1))
    reports.update(run('F1_weekly_p3a1', {'F1_reb5': p3a1}, 5))
    reports.update(run('F2a_oversold_daily', {'F2a': f2a}, 1))
    reports.update(run('F2b_lowvol_oversold', {'F2b': f2b}, 1))

    print('\n== F 系：日持有测量（基线 P3+A1 月频: +0.06%/验证+0.09%/11.8%）==')
    for name, full in reports.items():
        if 'benchmark' in name:
            continue
        m = full['metrics']
        va = full['splits']['validation']['geometric_monthly_return']
        nav = pd.read_csv(next((DEST).rglob(f'{name}_nav.csv')))
        tr = pd.read_csv(next((DEST).rglob(f'{name}_trades.csv')))
        n_months = len(nav) / 21.0
        from research import portfolio_metrics as pm
        monthly_turnover = pm.amount_turnover_per_month(tr, nav)  # 双边金额换手(二轮复审#19)
        print(f"{name:<24} 月收益 {m['geometric_monthly_return']:+.2%} | "
              f"验证月 {va:+.2%} | maxDD {abs(m['max_drawdown']):.1%} | "
              f"月换手 {monthly_turnover:.1f}x | "
              f"qualifies={full.get('qualifies_research_candidate')}")
    from research.factor_screen2 import register
    register('F系日持有4run: 见daily_trials证据', label='daily_trials')
    print(f"\n证据: {DEST}")


if __name__ == '__main__':
    main()
