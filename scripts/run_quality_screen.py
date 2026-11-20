#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""增厚路径B·信号层：PIT 质量因子筛选（预注册见 FACTOR_PREREG 增厚路径B节）。

PIT：每股每日取公告日≤T 的最新报告期（同公告日多期取报告期最大）。
"""
import sys, os
if __name__ == '__main__':
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from pathlib import Path

import pandas as pd
from scipy import stats as sps

from data.preparation import atomic_json
from research.factor_screen2 import PROTOCOL, load_ohlc_panel, run_partitions
from research.signal_screen import daily_statistics, forward_labels

DEST = Path('ai-created/research/quality_screen_2026-11-10')


def pit_quality(price: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """公告日 PIT 的 ROE / 负债率矩阵（dates×symbols）"""
    fin = pd.read_parquet('data_cache/preparation/oracle/financials/ashare_financial_indicator.parquet')
    fin['ann_date'] = pd.to_datetime(fin['ann_date'], format='%Y%m%d')
    fin['report_period'] = pd.to_datetime(fin['report_period'], format='%Y%m%d', errors='coerce')
    # 同公告日多期 → 取报告期最大；同 (symbol, ann) 完全重复 → 去重
    fin = fin.sort_values(['symbol', 'ann_date', 'report_period'])
    fin = fin.drop_duplicates(['symbol', 'ann_date'], keep='last')
    dates = price.index
    roe_out = pd.DataFrame(float('nan'), index=dates, columns=price.columns)
    debt_out = roe_out.copy()
    left = pd.DataFrame({'date': dates})
    for symbol, group in fin.groupby('symbol'):
        if symbol not in price.columns:
            continue
        g = group.dropna(subset=['ann_date']).sort_values('ann_date')
        merged = pd.merge_asof(left, g[['ann_date', 'roe', 'debt_to_assets']],
                               left_on='date', right_on='ann_date',
                               direction='backward')
        roe_out[symbol] = merged['roe'].to_numpy()
        debt_out[symbol] = merged['debt_to_assets'].to_numpy()
    return roe_out, debt_out


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    panel, base, evidence = load_ohlc_panel(PROTOCOL)
    price = base['price']
    eligible = (base['usable']
                & price.notna().rolling(PROTOCOL['minimum_history_sessions']).sum()
                .eq(PROTOCOL['minimum_history_sessions'])
                & base['amount'].rolling(PROTOCOL['liquidity_window'], min_periods=PROTOCOL['liquidity_window'])
                .median().ge(PROTOCOL['minimum_median_daily_amount_local_currency']))
    labels = forward_labels(price, PROTOCOL['horizon_sessions'])
    roe, debt = pit_quality(price)
    atomic_json(DEST / 'inputs.json', evidence)
    print(f"PIT 覆盖：ROE {roe.notna().mean().mean():.0%} | 负债率 {debt.notna().mean().mean():.0%}", flush=True)

    signals = {'Q1_roe': roe, 'Q2_low_debt': -debt}
    results, pvals = {}, {}
    for name, sig in signals.items():
        daily = daily_statistics(sig, labels, eligible, PROTOCOL['minimum_cross_section'])
        daily.to_csv(DEST / f'{name}_daily.csv', index_label='signal_date')
        parts = run_partitions(daily, PROTOCOL)
        results[name] = parts
        t = parts['validation']['hac_t_stat']
        pvals[name] = 1.0 if t is None else float(2 * sps.norm.sf(abs(t)))
        print(f"{name}: train IC {parts['train']['mean_ic'] if parts['train']['mean_ic'] is None else round(parts['train']['mean_ic'],4)} | "
              f"val IC {parts['validation']['mean_ic'] if parts['validation']['mean_ic'] is None else round(parts['validation']['mean_ic'],4)} | "
              f"t={t if t is None else round(t,2)}")

    from scipy.stats import false_discovery_control
    names_sorted = sorted(pvals, key=pvals.get)
    p = __import__('numpy').array([pvals[n] for n in names_sorted])
    rejected = false_discovery_control(p, method='bh') <= 0.10
    survivors = [n for n, r in zip(names_sorted, rejected) if r]
    print(f"\nFDR(q=0.10) 存活: {survivors or '无'}")
    atomic_json(DEST / 'summary.json', {'signals': results, 'p_values': pvals,
                                        'survivors': survivors})
    roe.to_parquet(DEST / 'pit_roe.parquet')
    (-debt).to_parquet(DEST / 'pit_low_debt.parquet')
    from research.factor_screen2 import register
    register(f'质量因子2检验: {survivors or "无存活"}', label='quality_screen')
    print(f"证据: {DEST}")


if __name__ == '__main__':
    main()
