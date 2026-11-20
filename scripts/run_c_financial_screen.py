#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""C 组财务因子信号筛选（预注册见 FACTOR_PREREG_2026-11-06.md 处方试验B节）。

数据：data_cache/fundamental/ 日频估值矩阵（当日即知，无前视）。
流程与 32 检验同管线：Spearman IC + HAC + train/val + 族内 BH-FDR。
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
from scipy import stats as sps

from data.preparation import atomic_json
from research.factor_screen2 import PROTOCOL, load_ohlc_panel, run_partitions
from research.signal_screen import daily_statistics, forward_labels

DEST = Path('ai-created/research/c_financial_screen_2026-11-09')


def build_financial_signals() -> dict[str, pd.DataFrame]:
    base = Path('data_cache/fundamental')
    pe = pd.read_parquet(base / 'pe_ttm.parquet')
    pb = pd.read_parquet(base / 'pb_lf.parquet')
    ps = pd.read_parquet(base / 'ps_ttm.parquet')
    mv = pd.read_parquet(base / 'mkt_cap_ard.parquet')
    out = {}
    out['C1_ep'] = 1.0 / pe.where(pe > 0)               # 亏损/负值 PE → NaN（保守）
    out['C2_bp'] = 1.0 / pb.where(pb > 0)
    out['C3_sp'] = 1.0 / ps.where(ps > 0)
    out['C4_size'] = -np.log(mv.where(mv > 0))
    return out


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    protocol = PROTOCOL
    # 面板只用于 eligible/labels 与 oracle 口径对齐（价格用 adjclose）
    panel, base, evidence = load_ohlc_panel(protocol)
    price = base['price']
    eligible = (base['usable']
                & price.notna().rolling(protocol['minimum_history_sessions']).sum()
                .eq(protocol['minimum_history_sessions'])
                & base['amount'].rolling(protocol['liquidity_window'], min_periods=protocol['liquidity_window'])
                .median().ge(protocol['minimum_median_daily_amount_local_currency']))
    labels = forward_labels(price, protocol['horizon_sessions'])
    fin = build_financial_signals()
    atomic_json(DEST / 'inputs.json', evidence)

    results, pvals = {}, {}
    for name, sig in fin.items():
        aligned = sig.reindex(index=price.index, columns=price.columns)
        daily = daily_statistics(aligned, labels, eligible, protocol['minimum_cross_section'])
        daily.to_csv(DEST / f'{name}_daily.csv', index_label='signal_date')
        parts = run_partitions(daily, protocol)
        results[name] = parts
        t = parts['validation']['hac_t_stat']
        pvals[name] = 1.0 if t is None else float(2 * sps.norm.sf(abs(t)))
        print(f"{name}: train IC {parts['train']['mean_ic']} | "
              f"val IC {parts['validation']['mean_ic']} | "
              f"t {t} | p {pvals[name]:.4f}")

    names_sorted = sorted(pvals, key=pvals.get)
    p = np.array([pvals[n] for n in names_sorted])
    from scipy.stats import false_discovery_control
    rejected = false_discovery_control(p, method='bh') <= 0.10
    fdr = {n: bool(r) for n, r in zip(names_sorted, rejected)}
    survivors = [n for n, ok in fdr.items() if ok]
    print(f"\nFDR(q=0.10) 存活: {survivors or '无'}")

    atomic_json(DEST / 'summary.json', {'signals': results, 'p_values': pvals,
                                        'fdr': fdr, 'survivors': survivors})
    from research.factor_screen2 import register
    register(f'C组财务4因子: 存活{survivors or "无"}', label='c_financial_screen')
    print(f"证据: {DEST}")


if __name__ == '__main__':
    main()
