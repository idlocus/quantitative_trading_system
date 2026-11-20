#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""处方试验A：排除过滤器 + 中段选股（预注册见 FACTOR_PREREG 处方试验A节）。

R1 = 排除 + D16rank[0.5,0.9]池 + A2选50（完整结构）
R2 = 排除 + D16 直接Top50（分离排除贡献）
R3 = 无排除 + 中段池 + A2（分离中段贡献）
组合形态与阶段二完全一致（N=50 月调仓 25bp budget_guard）。
"""
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

DEST = Path('ai-created/research/midband_study_2026-11-09')
POS = ['D16_rel_atr', 'D5_boll_bandwidth', 'A2_idio_reversal',
       'D4_boll_pctb', 'A6_vol_term']


def build_structures(signals: dict) -> dict:
    d16 = signals['D16_rel_atr']
    a2 = signals['A2_idio_reversal']
    # 排除：5 因子中 ≥2 个处于自身看空端 10%（因子内 rank≤0.1）
    votes = None
    for name in POS:
        bot = signals[name].rank(axis=1, pct=True) <= 0.10
        votes = bot.astype(float) if votes is None else votes.add(bot.astype(float))
    excluded = votes >= 2
    d16_rank = d16.rank(axis=1, pct=True)
    midband = (d16_rank >= 0.50) & (d16_rank <= 0.90)
    return {
        'R1_excl_mid_a2': a2.where(midband & ~excluded),
        'R2_excl_d16top': d16.where(~excluded),
        'R3_mid_a2_noexcl': a2.where(midband),
    }


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    protocol = PROTOCOL
    panel, base, evidence = load_ohlc_panel(protocol)
    price = base['price']
    eligible = (base['usable']
                & price.notna().rolling(protocol['minimum_history_sessions']).sum()
                .eq(protocol['minimum_history_sessions'])
                & base['amount'].rolling(protocol['liquidity_window'], min_periods=protocol['liquidity_window'])
                .median().ge(protocol['minimum_median_daily_amount_local_currency']))
    signals = ts.build_all_signals(panel)
    needed = set(POS) | {'D16_rel_atr', 'A2_idio_reversal'}
    structures = build_structures(signals)
    d16_score = signals['D16_rel_atr']
    atomic_json(DEST / 'inputs.json', evidence)

    print('结构池规模（日均候选）:')
    for name, score in structures.items():
        n_daily = score.notna().sum(axis=1)
        print(f'  {name}: {int(n_daily.mean())} 只/日')

    reports = portfolio_layer(structures, eligible, d16_score, base, evidence,
                              dict(PORTFOLIO_PROTOCOL, study='midband_exclusion_phase'),
                              dest=DEST)

    print('\n== 组合层（N=50 月调仓 25bp budget_guard）==')
    verdict = {}
    for name in structures:
        full = reports[f'{name}_budget_guard_25bps']
        r = full['metrics']
        verdict[name] = {'monthly': r['geometric_monthly_return'],
                         'max_dd': r['max_drawdown'],
                         'worst_month': r['worst_calendar_month'],
                         'qualifies': full.get('qualifies_research_candidate')}
        print(f"{name:<20} 月收益 {r['geometric_monthly_return']:+.2%} | "
              f"maxDD {abs(r['max_drawdown']):.1%} | 最差月 {r['worst_calendar_month']:+.1%} | "
              f"qualifies={full.get('qualifies_research_candidate')}")
    atomic_json(DEST / 'verdict.json', verdict)
    from research.factor_screen2 import register
    r1 = verdict.get('R1_excl_mid_a2', {})
    register(f"处方A排除+中段: R1月收益{r1.get('monthly', float('nan')):+.2%}", label='midband_exclusion_phase')
    print(f'\n证据: {DEST}')


if __name__ == '__main__':
    main()
