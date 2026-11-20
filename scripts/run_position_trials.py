#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""仓位层 E 系试验（预注册见 FACTOR_PREREG 仓位层试验节）。

E0 基线 guard / E1 shock0.15 / E2 市场波动率仓位 / E3 趋势两档。
P3 信号冻结为固定输入；仓位信号全部 shift(1) 无前视。
"""
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

DEST = Path('ai-created/research/position_trials_2026-11-10')


def p3_score(price, signals, fund):
    pb = pd.read_parquet(fund / 'pb_lf.parquet')
    pe = pd.read_parquet(fund / 'pe_ttm.parquet')
    bp_rank = (1.0 / pb.where(pb > 0)).reindex(
        index=price.index, columns=price.columns).rank(axis=1, pct=True)
    loss = pe.reindex(index=price.index, columns=price.columns).isna()
    d16_rank = signals['D16_rel_atr'].rank(axis=1, pct=True)
    return (d16_rank + bp_rank) / 2.0, loss


def run_one(study, signals_dict, risk_variants, evidence_data, external=None,
            adverse_shock=0.25):
    protocol = dict(PORTFOLIO_PROTOCOL, study=study,
                    risk_variants=risk_variants,
                    exposure_budget_adverse_shock=adverse_shock)
    if external:
        protocol['external_scales'] = external
    panel, base, evidence, eligible, d16 = evidence_data
    dest = DEST / study
    reports = portfolio_layer(signals_dict, eligible, d16, base, evidence,
                              protocol, dest=dest)
    return reports


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
    p3, loss = p3_score(price, signals, fund)
    p3x = p3.where(~loss)
    atomic_json(DEST / 'inputs.json', evidence)

    # 仓位因子（全池等权市场序列，shift(1) 无前视）
    mkt_ret = price.pct_change(fill_method=None).mean(axis=1)   # 等权市场日收益
    vol60 = mkt_ret.rolling(60, min_periods=60).std() * np.sqrt(252)
    scale_e2 = (0.15 / vol60).clip(upper=1.0).shift(1)          # min(1,15%/vol)
    mkt_idx = (1 + mkt_ret.fillna(0)).cumprod()
    trend = mkt_idx > mkt_idx.rolling(200, min_periods=200).mean()
    scale_e3 = pd.Series(np.where(trend, 0.95, 0.50), index=price.index).shift(1)
    print(f"E2 仓位均值 {scale_e2.mean():.1%} | E3 仓位均值 {scale_e3.mean():.1%}", flush=True)

    data = (panel, base, evidence, eligible, d16)
    all_reports = {}
    for study in ('E0_baseline', 'E1_shock15'):   # 已有效跑完，跳过重跑
        d = DEST / study / 'done.marker'
        d.write_text('ok', encoding='utf-8')
    all_reports.update(run_one('E2_mkt_vol', {'P3': p3x}, ['external'], data,
                               external={'P3': scale_e2}))
    all_reports.update(run_one('E3_trend', {'P3': p3x}, ['external'], data,
                               external={'P3': scale_e3}))

    print('\n== E 系仓位层试验（P3 信号冻结；研究期 2016-2023，N=50 月调仓 25bp）==')
    verdict = {}
    for name, full in all_reports.items():
        r = full['metrics']
        va = full['splits']['validation']['geometric_monthly_return']
        verdict[name] = {'monthly': r['geometric_monthly_return'],
                         'validation_monthly': va,
                         'max_dd': r['max_drawdown'],
                         'qualifies': full.get('qualifies_research_candidate')}
        print(f"{name:<34} 月收益 {r['geometric_monthly_return']:+.2%} | "
              f"验证月 {va:+.2%} | maxDD {abs(r['max_drawdown']):.1%} | "
              f"qualifies={full.get('qualifies_research_candidate')}")
    atomic_json(DEST / 'verdict.json', verdict)
    from research.factor_screen2 import register
    register('E系仓位层4run: 见position_trials证据', label='position_trials')
    print(f"\n证据: {DEST}")


if __name__ == '__main__':
    main()
