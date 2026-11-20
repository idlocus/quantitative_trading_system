#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P3 复跑验收（数据修复后）+ V3 多因子版（预注册见 FACTOR_PREREG 修复与V3节）。"""
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

DEST = Path('ai-created/research/qualify_v3b_2026-11-09')  # 代码修订后新目录（协议锁定）


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
    ps = pd.read_parquet(fund / 'ps_ttm.parquet')

    def align(df):
        return df.reindex(index=price.index, columns=price.columns)

    bp_rank = (1.0 / pb.where(pb > 0)).pipe(align).rank(axis=1, pct=True)
    sp_rank = (1.0 / ps.where(ps > 0)).pipe(align).rank(axis=1, pct=True)
    ep_rank = (1.0 / pe.where(pe > 0)).pipe(align).rank(axis=1, pct=True)
    loss = align(pe).isna()                       # 试验D'排除
    d16_rank = d16.rank(axis=1, pct=True)
    v2 = (d16_rank + bp_rank) / 2.0
    value = (bp_rank + sp_rank + ep_rank) / 3.0   # 价值族合成
    v3 = (value + d16_rank) / 2.0
    composite = {
        'P3_v2_excl_guard': v2.where(~loss),
        'V3_multifactor_excl': v3.where(~loss),
    }
    atomic_json(DEST / 'inputs.json', evidence)

    protocol = dict(PORTFOLIO_PROTOCOL, study='qualify_and_v3',
                    risk_variants=['budget_guard'])
    reports = portfolio_layer(composite, eligible, d16, base, evidence,
                              protocol, dest=DEST)
    print('\n== 验收跑（N=50 月调仓 25bp guard）==')
    verdict = {}
    for name, full in reports.items():
        r = full['metrics']
        va = full['splits']['validation']['geometric_monthly_return']
        tr = full['splits']['train']['geometric_monthly_return']
        verdict[name] = {'monthly': r['geometric_monthly_return'],
                         'train_monthly': tr, 'validation_monthly': va,
                         'max_dd': r['max_drawdown'],
                         'qualifies': full.get('qualifies_research_candidate'),
                         'accounting_valid': full.get('accounting_valid')}
        print(f"{name:<26} 月收益 {r['geometric_monthly_return']:+.2%} | "
              f"训练月 {tr:+.2%} | 验证月 {va:+.2%} | maxDD {abs(r['max_drawdown']):.1%} | "
              f"qualifies={full.get('qualifies_research_candidate')} | "
              f"账务={full.get('accounting_valid')}")
    atomic_json(DEST / 'verdict.json', verdict)
    from research.factor_screen2 import register
    first = [k for k, v in verdict.items() if v['qualifies']]
    register(f"验收跑: 首个qualifies={first or '无'}", label='qualify_and_v3')
    print(f"\nqualifies: {first or '仍无'} | 证据: {DEST}")


if __name__ == '__main__':
    main()
