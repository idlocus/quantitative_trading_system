#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""增厚④负债率排除 + ⑤N敏感性（预注册见 FACTOR_PREREG 增厚④⑤节）。"""
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

DEST = Path('ai-created/research/debt_n_trials_2026-11-10')


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
    low_debt_neg = pd.read_parquet(
        'ai-created/research/quality_screen_2026-11-10/pit_low_debt.parquet')
    debt = (-low_debt_neg).reindex(index=price.index, columns=price.columns)
    composite = {'L1_excl_debt70': p3a1.where(~(debt > 70)),
                 'N75': p3a1, 'N100': p3a1}
    atomic_json(DEST / 'inputs.json', evidence)
    n_excl = int(((debt > 70) & eligible).sum(axis=1).mean())
    print(f"负债率>70% 日均剔除(eligible内): {n_excl} 只", flush=True)

    # ④ 与 ⑤ 需不同 N → 分两次调用
    r = {}
    p4 = dict(PORTFOLIO_PROTOCOL, study='debt_excl', risk_variants=['budget_guard'])
    r.update(portfolio_layer({'L1_excl_debt70': composite['L1_excl_debt70']},
                             eligible, d16, base, evidence, p4, dest=DEST / 'L1'))
    p5 = dict(PORTFOLIO_PROTOCOL, study='n_sensitivity', risk_variants=['budget_guard'],
              positions=75)
    r.update(portfolio_layer({'N75': p3a1}, eligible, d16, base, evidence,
                             dict(p5, positions=75), dest=DEST / 'N75'))
    r.update(portfolio_layer({'N100': p3a1}, eligible, d16, base, evidence,
                             dict(p5, positions=100, study='n_sensitivity_100'),
                             dest=DEST / 'N100'))

    print('\n== 增厚④⑤（基线 P3+A1: +0.06%/验证+0.09%/11.8%）==')
    verdict = {}
    for name, full in r.items():
        m = full['metrics']
        va = full['splits']['validation']['geometric_monthly_return']
        verdict[name] = {'monthly': m['geometric_monthly_return'], 'validation_monthly': va,
                         'max_dd': m['max_drawdown'],
                         'qualifies': full.get('qualifies_research_candidate')}
        print(f"{name:<28} 月收益 {m['geometric_monthly_return']:+.2%} | "
              f"验证月 {va:+.2%} | maxDD {abs(m['max_drawdown']):.1%} | "
              f"qualifies={full.get('qualifies_research_candidate')}")
    atomic_json(DEST / 'verdict.json', verdict)
    from research.factor_screen2 import register
    register('增厚④⑤: 负债排除+N敏感性3run', label='debt_excl')


if __name__ == '__main__':
    main()
