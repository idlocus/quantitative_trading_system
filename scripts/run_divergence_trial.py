#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""增厚③：MACD 顶背离排除（预注册见 FACTOR_PREREG 增厚③节）。"""
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

DEST = Path('ai-created/research/divergence_trial_2026-11-10')


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
    # 顶背离：价格≥60日高 且 MACD柱 < 60日柱峰值×0.8
    hist = signals['D1_macd_level'] * price                      # 柱值(价格单位)
    px_high = price >= price.rolling(60, min_periods=60).max()
    hist_peak = hist.rolling(60, min_periods=60).max()
    top_div = px_high & (hist < hist_peak * 0.8)
    n_div = (top_div & eligible).sum(axis=1).mean()
    composite = {'M1_p3a1_excl_topdiv': p3a1.where(~top_div)}
    atomic_json(DEST / 'inputs.json', evidence)
    print(f"日均顶背离(eligible池内): {int(n_div)} 只", flush=True)

    protocol = dict(PORTFOLIO_PROTOCOL, study='divergence_exclusion',
                    risk_variants=['budget_guard'])
    reports = portfolio_layer(composite, eligible, d16, base, evidence,
                              protocol, dest=DEST)
    for name, full in reports.items():
        r = full['metrics']
        va = full['splits']['validation']['geometric_monthly_return']
        print(f"{name:<32} 月收益 {r['geometric_monthly_return']:+.2%} | "
              f"验证月 {va:+.2%} | maxDD {abs(r['max_drawdown']):.1%} | "
              f"qualifies={full.get('qualifies_research_candidate')}")
    from research.factor_screen2 import register
    r0 = next(iter(reports.values()))
    ok = (r0['metrics']['geometric_monthly_return'] > 0.0007
          and r0['splits']['validation']['geometric_monthly_return'] >= 0.0008
          and r0.get('qualifies_research_candidate'))
    register(f'增厚③背离排除: {"PASS" if ok else "FAIL"}', label='divergence_exclusion')
    print(f"判据(>基线0.06%+1bp): {'PASS' if ok else 'FAIL'} | 证据: {DEST}")


if __name__ == '__main__':
    main()
