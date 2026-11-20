#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""增厚②：组合内权重（W2 信号加权 / W3 逆波动率，预注册见 FACTOR_PREREG 增厚路径②节）。"""
import sys, os
if __name__ == '__main__':
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from pathlib import Path

import numpy as np
import pandas as pd

from data.preparation import atomic_json
from research import technical_signals as ts
from research.composite_study import PORTFOLIO_PROTOCOL, portfolio_layer
from research.factor_screen2 import PROTOCOL, load_ohlc_panel

DEST = Path('ai-created/research/weight_trials_2026-11-19')  # 审计2026-11-19重跑(前视/缓冲带/分母修复)


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
            .where(~loss).where(d7_rank < 0.90))          # P3+A1 基底
    atomic_json(DEST / 'inputs.json', evidence)

    # 逐日权重（对全部日期计算，仿真只在调仓日用）
    z = p3a1.sub(p3a1.mean(axis=1), axis=0).div(p3a1.std(axis=1) + 1e-12, axis=0)
    w_signal = (z.clip(-2, 2) + 2.5)                       # W2
    ret = price.pct_change(fill_method=None)
    vol60 = (ret.rolling(60, min_periods=60).std() * np.sqrt(252))
    w_invvol = 1.0 / vol60.where(vol60 > 0)                # W3

    def to_daily(wmat):
        return {day: {s: float(v) for s, v in row.dropna().items()}
                for day, row in wmat.iterrows()}

    composite = {'W2_signal_w': p3a1, 'W3_inv_vol_w': p3a1, 'W1_equal': p3a1}
    protocol = dict(PORTFOLIO_PROTOCOL, study='weight_trials',
                    risk_variants=['budget_guard'],
                    daily_weights={'W2_signal_w': to_daily(w_signal),
                                   'W3_inv_vol_w': to_daily(w_invvol)})
    reports = portfolio_layer(composite, eligible, d16, base, evidence,
                              protocol, dest=DEST)

    print('\n== 增厚②：组合内权重（W1 等权基线随跑对照）==')
    verdict = {}
    for name, full in reports.items():
        r = full['metrics']
        va = full['splits']['validation']['geometric_monthly_return']
        verdict[name] = {'monthly': r['geometric_monthly_return'],
                         'validation_monthly': va, 'max_dd': r['max_drawdown'],
                         'qualifies': full.get('qualifies_research_candidate')}
        print(f"{name:<28} 月收益 {r['geometric_monthly_return']:+.2%} | "
              f"验证月 {va:+.2%} | maxDD {abs(r['max_drawdown']):.1%} | "
              f"qualifies={full.get('qualifies_research_candidate')}")
    atomic_json(DEST / 'verdict.json', verdict)
    w1 = verdict.get('W1_equal_budget_guard_25bps', {})
    winners = [k for k, v in verdict.items()
               if k.startswith(('W2', 'W3'))
               and v['monthly'] > w1.get('monthly', 9e9) + 0.0001
               and v['validation_monthly'] >= w1.get('validation_monthly', 9e9) - 0.0001
               and v['qualifies']]
    from research.factor_screen2 import register
    register(f'增厚②权重2run: {winners or "无达标"}', label='W系组合内权重')
    print(f"\n判据: {winners or 'FAIL'} | 证据: {DEST}")


if __name__ == '__main__':
    main()
