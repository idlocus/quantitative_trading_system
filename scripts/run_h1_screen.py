#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""L 系：1 日持有口径因子重筛（标签=T+1开盘→T+2开盘，预注册见 L 系节）。"""
import sys, os
if __name__ == '__main__':
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats as sps
from scipy.stats import false_discovery_control

from data.preparation import atomic_json
from research import technical_signals as ts
from research.factor_screen2 import PROTOCOL, load_ohlc_panel
from research.signal_screen import daily_statistics

DEST = Path('ai-created/research/l_h1_screen_2026-11-11')


def open_to_open_labels(open_px: pd.DataFrame) -> pd.DataFrame:
    """T 信号日 → T+1开盘 至 T+2开盘 收益（次日开盘卖出口径）"""
    return (open_px.shift(-2) / open_px.shift(-1) - 1).where(lambda x: np.isfinite(x))


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    panel, base, evidence = load_ohlc_panel(PROTOCOL)
    price = base['price']
    eligible = (base['usable']
                & price.notna().rolling(PROTOCOL['minimum_history_sessions']).sum()
                .eq(PROTOCOL['minimum_history_sessions'])
                & base['amount'].rolling(PROTOCOL['liquidity_window'], min_periods=PROTOCOL['liquidity_window'])
                .median().ge(PROTOCOL['minimum_median_daily_amount_local_currency']))
    labels = open_to_open_labels(panel['open'])
    signals = ts.build_all_signals(panel)
    atomic_json(DEST / 'inputs.json', evidence)

    results, pvals = {}, {}
    label_end = pd.Series(labels.index, index=labels.index).shift(-2)
    for name, sig in signals.items():
        daily = daily_statistics(sig, labels, eligible, 30)
        parts = {}
        for split, lo, hi in [('train', '2016-01-01', '2020-12-31'),
                              ('validation', '2021-01-01', '2023-12-31')]:
            mask = (daily.index >= lo) & (daily.index <= hi) & label_end.le(pd.Timestamp(hi))
            sel = daily.loc[mask, 'ic'].dropna()
            if len(sel) < 100:
                parts[split] = {'days': len(sel), 'mean_ic': None, 't': None}
                continue
            t = float(sel.mean() / (sel.std(ddof=1) / np.sqrt(len(sel))))
            parts[split] = {'days': len(sel), 'mean_ic': float(sel.mean()), 't': t,
                            'win_rate': float((sel > 0).mean())}
        results[name] = parts
        vt = parts['validation']['t']
        pvals[name] = 1.0 if vt is None else float(2 * sps.norm.sf(abs(vt)))

    names_sorted = sorted(pvals, key=pvals.get)
    p = np.array([pvals[n] for n in names_sorted])
    rejected = false_discovery_control(p, method='bh') <= 0.10
    fdr = dict(zip(names_sorted, map(bool, rejected)))

    print('== L 系：1 日持有口径（标签=次日开盘→次次日开盘）==')
    print(f"{'因子':<24}{'训练IC':>9}{'验证IC':>9}{'验证t':>8}{'胜率':>7}{'FDR':>5}")
    rows = []
    for n in sorted(results, key=lambda k: -(results[k]['validation']['mean_ic'] or -9)):
        v, t = results[n]['validation'], results[n]['train']
        if v['mean_ic'] is None:
            continue
        rows.append((n, t['mean_ic'], v['mean_ic'], v['t'], v.get('win_rate'), fdr.get(n)))
    for n, tic, vic, vt, wr, fd in rows:
        print(f"{n:<24}{tic:>+9.4f}{vic:>+9.4f}{vt:>+8.2f}{wr:>7.0%}{'✓' if fd else '✗':>5}")
    # 成本覆盖：验证期 top50 日均毛超额（用最强因子的日度 top/bottom decile 差近似）
    best = rows[0][0] if rows else None
    if best:
        daily = daily_statistics(signals[best], labels, eligible, 30)
        sel = daily.loc[(daily.index > '2021-01-01') & (daily.index <= '2023-12-31')]
        spread = (sel['top_decile_forward_mean'] - sel['bottom_decile_forward_mean']).dropna()
        top = sel['top_decile_forward_mean'].dropna()
        print(f"最强因子[{best}] 验证期：头组日均毛收益 {top.mean()*10000:+.1f}bp"
              f" | 头尾组日均价差 {spread.mean()*10000:+.1f}bp；成本覆盖需按实际换手验证")
    atomic_json(DEST / 'summary.json', {'signals': results, 'p_values': pvals, 'fdr': fdr})
    from research.factor_screen2 import register
    survivors = [n for n, ok in fdr.items() if ok]
    register(f'L系1日窗重筛32: FDR存活{len(survivors)}', label='h1_screen')
    print(f"\n证据: {DEST}")


if __name__ == '__main__':
    main()
