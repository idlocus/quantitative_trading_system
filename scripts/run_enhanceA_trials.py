#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""增厚路径A：趋势过热排除（预注册见 FACTOR_PREREG 增厚路径A节）。

P3 冻结基底 + 三档趋势排除；基线引用 E0。研究期 2016-2023。
"""
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

DEST = Path('ai-created/research/enhanceA_2026-11-10')


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
    d16_rank = d16.rank(axis=1, pct=True)
    p3 = ((d16_rank + bp_rank) / 2.0).where(~loss)

    d7_rank = signals['D7_expma_gap'].rank(axis=1, pct=True)   # 高=趋势过热
    d12_full = signals['D12_ma_align'] >= 4                    # 满排列
    composite = {
        'A1_excl_hot10': p3.where(d7_rank < 0.90),
        'A2_excl_hot20': p3.where(d7_rank < 0.80),
        'A3_excl_hot_or_align': p3.where((d7_rank < 0.85) & ~d12_full),
    }
    atomic_json(DEST / 'inputs.json', evidence)
    for name, sc in composite.items():
        n = sc.where(eligible).notna().sum(axis=1).mean()
        print(f"{name}: 池均 {int(n)} 只", flush=True)

    protocol = dict(PORTFOLIO_PROTOCOL, study='enhanceA_trend_exclusion',
                    risk_variants=['budget_guard'])
    reports = portfolio_layer(composite, eligible, d16, base, evidence,
                              protocol, dest=DEST)

    print('\n== 路径A：趋势过热排除（vs E0 基线 +0.05%/验证+0.09%/12.1%）==')
    verdict = {}
    for name, full in reports.items():
        r = full['metrics']
        va = full['splits']['validation']['geometric_monthly_return']
        verdict[name] = {'monthly': r['geometric_monthly_return'],
                         'validation_monthly': va,
                         'max_dd': r['max_drawdown'],
                         'qualifies': full.get('qualifies_research_candidate')}
        print(f"{name:<30} 月收益 {r['geometric_monthly_return']:+.2%} | "
              f"验证月 {va:+.2%} | maxDD {abs(r['max_drawdown']):.1%} | "
              f"qualifies={full.get('qualifies_research_candidate')}")
    atomic_json(DEST / 'verdict.json', verdict)
    from research.factor_screen2 import register
    winners = [k for k, v in verdict.items()
               if v['monthly'] > 0.0005 and v['validation_monthly'] >= 0.0008
               and v['qualifies']]
    register(f'增厚A趋势排除3run: {winners or "无达标"}', label='enhanceA_trend_exclusion')
    print(f"\n判据: {winners or 'FAIL'} | 证据: {DEST}")


if __name__ == '__main__':
    main()
