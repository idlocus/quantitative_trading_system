#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""终极检验：P3/V3 在 2024+ 封存区间（预注册见 FACTOR_PREREG 终极检验节）。

面板 2023-06 起（暖机+guard 初始化），全部指标自 2024-01-01 切算。
判据：月收益>0 且双15%不破 且账务无 critical。
"""
import sys, os
if __name__ == '__main__':
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from pathlib import Path

import pandas as pd

from data.preparation import atomic_json
from research.portfolio_study import publish_study
from research import technical_signals as ts
from research.composite_study import PORTFOLIO_PROTOCOL, portfolio_layer
from research.factor_screen2 import PROTOCOL, load_ohlc_panel, valuation_input

DEST = Path('ai-created/research/holdout_pool_v3')  # 审计2026-11-19重跑(仿真器语义修复+指标口径更正)
HOLDOUT_START = pd.Timestamp('2024-01-01')


def _compute(dest):
    dest.mkdir(parents=True, exist_ok=True)
    protocol = dict(PROTOCOL, start='2023-06-01', validation_end='2026-10-30',
                    train_end='2023-12-31')   # 2024+ 全部落入 validation 段
    panel, base, evidence = load_ohlc_panel(protocol)
    price = base['price']
    eligible = (base['usable']
                & price.notna().rolling(protocol['minimum_history_sessions']).sum()
                .eq(protocol['minimum_history_sessions'])
                & base['amount'].rolling(protocol['liquidity_window'], min_periods=protocol['liquidity_window'])
                .median().ge(protocol['minimum_median_daily_amount_local_currency']))
    d16 = ts.d16_rel_atr(panel['high'], panel['low'], panel['close'])

    fund = Path('data_cache/fundamental')
    pb = valuation_input(fund / 'pb_lf.parquet', evidence)
    pe = valuation_input(fund / 'pe_ttm.parquet', evidence)
    ps = valuation_input(fund / 'ps_ttm.parquet', evidence)

    def align(df):
        return df.reindex(index=price.index, columns=price.columns)

    bp_rank = (1.0 / pb.where(pb > 0)).pipe(align).where(eligible).rank(axis=1, pct=True)
    sp_rank = (1.0 / ps.where(ps > 0)).pipe(align).where(eligible).rank(axis=1, pct=True)
    ep_rank = (1.0 / pe.where(pe > 0)).pipe(align).where(eligible).rank(axis=1, pct=True)
    loss = align(pe).isna()
    d16_rank = d16.where(eligible).rank(axis=1, pct=True)
    v2 = (d16_rank + bp_rank) / 2.0
    value = (bp_rank + sp_rank + ep_rank) / 3.0
    composite = {
        'HOLDOUT_P3': v2.where(~loss),
        'HOLDOUT_V3': ((value + d16_rank) / 2.0).where(~loss),
    }
    sim_protocol = dict(PORTFOLIO_PROTOCOL, study='final_holdout',
                        start='2023-06-01', validation_end='2026-10-30',
                        train_end='2023-12-31', risk_variants=['budget_guard'])
    availability = {'HOLDOUT_P3': v2.notna().any(axis=1),
                    'HOLDOUT_V3': (value.notna() & d16_rank.notna()).any(axis=1)}
    reports = portfolio_layer(composite, eligible, d16, base, evidence,
                              sim_protocol, dest=dest, availability=availability)

    print('\n== 终极检验（2024-01-01 起切算；N=50 月调仓 25bp guard）==')
    from research.portfolio_metrics import performance
    verdict = {}
    for name, full in reports.items():
        # 自 2024-01-01 切算全部指标（暖机段不计数）。
        # 审计 2026-11-19 一-1：月度指标统一走 performance()（首月基准=
        # 上年最后交易日净值、自然月数、峰值含 initial），旧口径"本月首日
        # 收盘为基准 + 交易日数/21 月数"漏每月首日盈亏，已废弃。
        rows = pd.read_csv(dest / f'{name}_nav.csv')
        rows['date'] = pd.to_datetime(rows['date'])
        pre = rows[rows['date'] < HOLDOUT_START]
        seg_nav = rows[rows['date'] >= HOLDOUT_START].set_index('date')['nav']
        initial = float(pre['nav'].iloc[-1])
        perf = performance(seg_nav, initial=initial)
        gmr = perf['geometric_monthly_return']
        max_dd = perf['max_drawdown']
        worst_month = perf['worst_calendar_month']
        worst_mtd = perf['worst_daily_month_to_date']
        segment = rows.loc[rows['date'] >= HOLDOUT_START]
        issues = pd.read_csv(dest / f'{name}_issues.csv')
        critical = issues.loc[(pd.to_datetime(issues['date']) >= HOLDOUT_START)
                              & ~issues['issue'].isin({'unknown_cash_payment_locked', 'unknown_bonus_listing_locked',
                                                       'documented_suspension_valuation_carry'})]
        crit = critical.empty
        risk_ok = not segment[['opening_breach', 'closing_breach']].any().any() and segment['drawdown'].min() >= -0.15
        passed = gmr > 0 and risk_ok and max_dd >= -0.15 and worst_mtd >= -0.15 and worst_month >= -0.15 and crit
        verdict[name] = {'monthly': gmr, 'max_dd': max_dd,
                         'worst_month': worst_month, 'worst_mtd': worst_mtd,
                         'months': perf['months'], 'holdout_initial_nav': initial,
                         'opening_or_closing_breach': bool(segment[['opening_breach', 'closing_breach']].any().any()),
                         'lifetime_drawdown_in_holdout': float(segment['drawdown'].min()), 'accounting_valid': crit, 'PASSED': bool(passed)}
        print(f"{name:<12} 月收益 {gmr:+.2%} | maxDD {abs(max_dd):.1%} | "
              f"最差月 {worst_month:+.1%} | 月内最差 {worst_mtd:+.1%} | "
              f"账务={crit} | **{'PASSED' if passed else 'FAILED'}**")
    atomic_json(dest / 'verdict.json', verdict)
    return verdict


def main(register_trial=True):
    with publish_study(DEST, 'holdout_manifest.json') as stage:
        verdict = _compute(stage)
    if not register_trial:
        return verdict
    from research.factor_screen2 import register
    winners = [k for k, v in verdict.items() if v['PASSED']]
    register(f'终极检验2024+（2026-11-19更正口径）: {winners or "全部未通过"}',
             label='P3/V3终极检验')
    print(f"\n终极判定: {winners or 'FAIL'} | 证据: {DEST}")


if __name__ == '__main__':
    main()
