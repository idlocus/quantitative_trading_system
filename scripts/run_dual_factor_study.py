#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""D16×BP 双慢变量月度组合试验（预注册见 FACTOR_PREREG 双因子组合试验节）。"""
import sys, os
if __name__ == '__main__':
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import json
from pathlib import Path

import numpy as np
import pandas as pd

from data.preparation import atomic_json
from research import technical_signals as ts
from research.composite_study import PORTFOLIO_PROTOCOL, portfolio_layer
from research.factor_screen2 import PROTOCOL, load_ohlc_panel

DEST = Path('ai-created/research/dual_factor_2026-11-09')


def build_bp(price: pd.DataFrame) -> pd.DataFrame:
    pb = pd.read_parquet('data_cache/fundamental/pb_lf.parquet')
    bp = 1.0 / pb.where(pb > 0)
    return bp.reindex(index=price.index, columns=price.columns)


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
    bp = build_bp(price)
    d16_rank = d16.rank(axis=1, pct=True)
    bp_rank = bp.rank(axis=1, pct=True)
    composite = {
        'V1_bp_pool_d16': d16.where(bp_rank >= 0.5),
        'V2_rank_mean': (d16_rank + bp_rank) / 2.0,
        'BP_benchmark': bp,
    }
    atomic_json(DEST / 'inputs.json', evidence)
    print(f"V1 池日均 {int(composite['V1_bp_pool_d16'].notna().sum(axis=1).mean())} 只", flush=True)

    reports = portfolio_layer(composite, eligible, d16, base, evidence,
                              dict(PORTFOLIO_PROTOCOL, study='dual_factor_d16_bp'),
                              dest=DEST)

    print('\n== 组合层（N=50 月调仓 25bp）==')
    verdict = {}
    for name in list(composite) + ['D16_benchmark']:
        for variant in ('budget_guard', 'uncontrolled'):
            key = f'{name}_{variant}_25bps'
            full = reports[key]
            r = full['metrics']
            verdict[key] = {'monthly': r['geometric_monthly_return'],
                            'max_dd': r['max_drawdown'],
                            'qualifies': full.get('qualifies_research_candidate')}
            print(f"{key:<38} 月收益 {r['geometric_monthly_return']:+.2%} | "
                  f"maxDD {abs(r['max_drawdown']):.1%} | "
                  f"qualifies={full.get('qualifies_research_candidate')}")

    guard = {k: v for k, v in verdict.items() if 'budget_guard' in k}
    bench_best = max(verdict['BP_benchmark_budget_guard_25bps']['monthly'],
                     verdict['D16_benchmark_budget_guard_25bps']['monthly'])
    winners = [k for k, v in guard.items()
               if v['monthly'] > 0 and v['monthly'] >= bench_best and v['qualifies']]
    atomic_json(DEST / 'verdict.json', verdict)
    from research.factor_screen2 import register
    register(f'D16×BP双因子: {winners or "无达标"}', label='dual_factor_d16_bp')
    print(f"\n判据（>0 且 ≥max(单因子) 且双15%）: {winners or 'FAIL'}")
    print(f"证据: {DEST}")


if __name__ == '__main__':
    main()
