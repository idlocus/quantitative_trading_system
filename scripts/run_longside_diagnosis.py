#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""多头端结构诊断（处方3，2026-11-09）——描述性分析，非假设检验，不计入登记册。

问题：D16 IC +0.076 但多头组合不赚——排序收益落在多头端还是空头端？
方法：逐日按信号十分位组合，统计各分位前瞻5日收益（train/val 分报），
并单看 Top5%/Top1% 极端多头。输出每分位日均超额（vs 当日全市场均值）。
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
from research.factor_screen2 import PROTOCOL, load_ohlc_panel
from research.signal_screen import forward_labels

DEST = Path('ai-created/research/longside_diagnosis_2026-11-09')

FACTORS = ['D16_rel_atr', 'D5_boll_bandwidth', 'A2_idio_reversal',
           'D4_boll_pctb', 'A6_vol_term']


def decile_profile(signal: pd.DataFrame, labels: pd.DataFrame,
                   eligible: pd.DataFrame, protocol: dict) -> dict:
    """逐日十分位（Q1=最看多）+ 极端多头的前瞻收益，train/val 分报"""
    paired = eligible & signal.notna() & labels.notna()
    ranks = signal.where(paired).rank(axis=1, pct=True)
    out = {}
    splits = {'train': (protocol['start'], protocol['train_end']),
              'validation': (pd.Timestamp(protocol['train_end']) + pd.Timedelta(days=1),
                             protocol['validation_end'])}
    for split, (lo, hi) in splits.items():
        rows = []
        for day in signal.index:
            if not (pd.Timestamp(lo) <= day <= pd.Timestamp(hi)):
                continue
            lab = labels.loc[day]
            rk = ranks.loc[day]
            valid = rk.dropna()
            if len(valid) < 500:
                continue
            market = lab.reindex(valid.index).mean()
            row = {'date': day, 'market': market}
            for q in range(10):
                # Q1 = 最低分位 = 信号最看多（信号已定向为值大看多 → rank 低=看多）
                lo_q, hi_q = q / 10, (q + 1) / 10
                members = valid[(valid > lo_q) & (valid <= hi_q)].index
                if len(members) >= 30:
                    row[f'Q{q + 1}'] = lab.reindex(members).mean() - market
            for tag, thr in (('T5', 0.05), ('T1', 0.01)):
                members = valid[valid <= thr].index
                if len(members) >= 20:
                    row[tag] = lab.reindex(members).mean() - market
            rows.append(row)
        df = pd.DataFrame(rows).set_index('date')
        out[split] = df
    return out


def summarize(df: pd.DataFrame) -> dict:
    res = {}
    for col in df.columns:
        if col == 'market':
            continue
        v = df[col].dropna()
        res[col] = {'mean_daily': float(v.mean()),
                    'monthly': float((1 + v.mean()) ** 21 - 1),
                    'days': int(len(v))}
    return res


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
    labels = forward_labels(price, protocol['horizon_sessions'])
    signals = ts.build_all_signals(panel)
    atomic_json(DEST / 'inputs.json', evidence)

    report = {}
    for name in FACTORS:
        parts = decile_profile(signals[name], labels, eligible, protocol)
        report[name] = {split: summarize(df) for split, df in parts.items()}
        parts['train'].to_csv(DEST / f'{name}_train_deciles.csv')
        parts['validation'].to_csv(DEST / f'{name}_validation_deciles.csv')
        del signals[name]

    # 渲染
    for name in FACTORS:
        print(f'\n== {name}（日均超额 vs 市场，Q1=最看多 … Q10=最看空）==')
        for split in ('train', 'validation'):
            r = report[name][split]
            line = ' '.join(f"Q{q}:{r[f'Q{q}']['mean_daily'] * 100:+.2f}bp"
                            for q in range(1, 11) if f'Q{q}' in r)
            t5 = r.get('T5', {}).get('mean_daily')
            t1 = r.get('T1', {}).get('mean_daily')
            print(f"  {split:<10} {line}")
            if t5 is not None:
                print(f"  {'':<10} Top5% {t5 * 100:+.2f}bp | Top1% "
                      f"{t1 * 100:+.2f}bp" if t1 is not None else
                      f"  {'':<10} Top5% {t5 * 100:+.2f}bp")
    atomic_json(DEST / 'decile_summary.json', report)
    print(f"\n证据: {DEST}")


if __name__ == '__main__':
    main()
