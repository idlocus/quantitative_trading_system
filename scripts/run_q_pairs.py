#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Q 系：全因子两两组合扫描（预注册见 Q 系节）。"""


def main():
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

    from pathlib import Path
    from itertools import combinations
    import numpy as np
    import pandas as pd
    from scipy import stats as sps
    from scipy.stats import false_discovery_control

    from data.preparation import atomic_json
    from research import technical_signals as ts
    from research.factor_screen2 import PROTOCOL, load_ohlc_panel
    from research.signal_screen import daily_statistics

    DEST = Path('ai-created/research/q_pairs_2026-11-11')
    DEST.mkdir(parents=True, exist_ok=True)

    ALIVE = ['D16_rel_atr', 'D5_boll_bandwidth', 'A2_idio_reversal', 'D4_boll_pctb',
             'A1_risk_adj_reversal', 'D13_wr14', 'A6_vol_term', 'D3_kdj_j',
             'D15_psy12', 'A3_quiet_reversal', 'J4_kdj_squeeze', 'A5_overnight_share']

    panel, base, evidence = load_ohlc_panel(PROTOCOL)
    price = base['price']
    eligible = (base['usable']
                & price.notna().rolling(PROTOCOL['minimum_history_sessions']).sum()
                .eq(PROTOCOL['minimum_history_sessions'])
                & base['amount'].rolling(PROTOCOL['liquidity_window'], min_periods=PROTOCOL['liquidity_window'])
                .median().ge(PROTOCOL['minimum_median_daily_amount_local_currency']))
    labels = (panel['open'].shift(-2) / panel['open'].shift(-1) - 1).where(lambda x: np.isfinite(x))
    signals = ts.build_all_signals(panel)
    atomic_json(DEST / 'inputs.json', evidence)
    print('信号构造完成，开始66对扫描...', flush=True)

    label_end = pd.Series(labels.index, index=labels.index).shift(-2)
    rows = []
    for a, b in combinations(ALIVE, 2):
        ra = signals[a].rank(axis=1, pct=True)
        rb = signals[b].rank(axis=1, pct=True)
        combo = (ra + rb) / 2.0
        daily = daily_statistics(combo, labels, eligible, 30)
        val = daily.loc[(daily.index > '2021-01-01') & (daily.index <= '2023-12-31')
                        & label_end.le(pd.Timestamp('2023-12-31')), 'ic'].dropna()
        if len(val) < 100:
            continue
        t = float(val.mean() / (val.std(ddof=1) / np.sqrt(len(val))))
        p = float(2 * sps.norm.sf(abs(t)))
        rows.append({'pair': f'{a.split("_")[0]}+{b.split("_")[0]}', 'full': f'{a}|{b}',
                     'val_ic': float(val.mean()), 't': t, 'p': p})
        print('.', end='', flush=True)
    print()
    df = pd.DataFrame(rows).sort_values('val_ic', ascending=False)
    # FDR
    rej = false_discovery_control(df['p'].to_numpy(), method='bh') <= 0.10
    df['fdr_pass'] = rej
    df.to_csv(DEST / 'pair_ic_scan.csv', index=False, encoding='utf-8-sig')
    atomic_json(DEST / 'summary.json', {'n_pairs': len(df), 'd16_baseline_ic': 0.0500,
                                        'top10': df.head(10)[['pair','val_ic','t','fdr_pass']].to_dict('records')})
    print("== 66 对信号层扫描（验证期 1 日 IC；D16 基线 +0.050）==")
    print(df.head(15).to_string(index=False))
    n_above = int((df.val_ic > 0.050).sum())
    print(f"\n超过 D16 基线的对数: {n_above} / {len(df)}")
    from research.factor_screen2 import register
    register(f'Q系66对IC扫描: 超 D16 基线 {n_above} 对', label='q_pairs')
    print(f"证据: {DEST}")


if __name__ == '__main__':
    main()
