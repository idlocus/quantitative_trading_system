#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""X 系：市场启动日买入侧增强（预注册见 X 系节）。"""
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

DEST = Path('ai-created/research/x_buy_enhance_2026-11-19')  # 审计2026-11-19重跑(深度+分母修复)


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
    g1 = (((d16.rank(axis=1, pct=True) + bp_rank) / 2.0)
          .where(~loss).where(d7_rank < 0.90))
    atomic_json(DEST / 'inputs.json', evidence)

    mkt = pd.read_csv('runtime/market_start_indicator.csv')
    mkt['date'] = pd.to_datetime(mkt['date'])
    start_dates = set(mkt.loc[mkt['market_start'] == True, 'date'])
    # 审计 2026-11-19 一-3：覆盖键=决策日（启动日当日），仿真器内部平移
    # 到下一交易日生效
    print(f'启动日(决策日): {len(start_dates)}（于下一交易日生效）', flush=True)

    composite = {'X1_n70': g1, 'X2_k70': g1, 'X3_n70k70': g1}
    n70 = {d: 70 for d in start_dates}
    k70 = {d: 70 for d in start_dates}
    protocol = dict(PORTFOLIO_PROTOCOL, study='x_buy_enhance',
                    risk_variants=['budget_guard'], rebalance_sessions=1,
                    hold_buffer_ranks={k: 100 for k in composite},
                    daily_position_overrides={'X1_n70': n70, 'X3_n70k70': n70},
                    daily_buffer_overrides={'X2_k70': k70, 'X3_n70k70': k70})
    reports = portfolio_layer(composite, eligible, d16, base, evidence,
                             protocol, dest=DEST)

    print('\n== X 系：启动日买入增强（基线 G1: +0.06%/验证+0.08%/11.8%/2.0x）==')
    for name, full in reports.items():
        if 'benchmark' in name:
            continue
        m = full['metrics']
        va = full['splits']['validation']['geometric_monthly_return']
        print(f"{name:<14} 月收益 {m['geometric_monthly_return']:+.2%} | "
              f"验证月 {va:+.2%} | maxDD {abs(m['max_drawdown']):.1%} | "
              f"qualifies={full.get('qualifies_research_candidate')}")

    from research.factor_screen2 import register
    register('X系启动日买入增强3run: 见x_buy_enhance证据', label='X系启动日买入增强')
    print(f'\n证据: {DEST}')


if __name__ == '__main__':
    main()
