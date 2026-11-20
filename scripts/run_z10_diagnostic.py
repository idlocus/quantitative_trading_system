#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Z10 design diagnostic: signal overlap structure of the Z8 six.

Descriptive ONLY (no strategy returns, no performance peeking):
1. Per-commodity signal occupancy (how often each fires)
2. Simultaneity distribution (when >=1 fires, how many fire together)
3. Depth of compression available for ranking (v20_pct spread when overlapped)
This informs whether 'concentrate on strongest signal' is a sizing change
or a genuine selection problem.
"""


def main():
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

    from pathlib import Path
    import pickle

    import numpy as np
    import pandas as pd

    DEST = Path('ai-created/research/z8_full_stop')
    SYMS = {'AU0': 'Gold', 'CU0': 'Copper', 'RB0': 'Rebar',
            'I0': 'Iron', 'P0': 'Palm', 'HC0': 'HRC'}
    VOL_WINDOW = 20
    VOL_LOOKBACK = 250
    VOL_THRESHOLD = 0.20

    data = pickle.load(open(DEST / 'data.pkl', 'rb'))

    signals = {}   # signed signal per commodity
    depths = {}    # compression depth (v20_pct) for ranking
    for sym in SYMS:
        close = data[sym]['收盘价'].astype(float)
        ret = close.pct_change()
        v20 = ret.rolling(VOL_WINDOW).std()
        v20_pct = v20.rolling(VOL_LOOKBACK).rank(pct=True)
        ma20 = close.rolling(VOL_WINDOW).mean()
        sig = (v20_pct < VOL_THRESHOLD) * np.sign(close - ma20)
        signals[sym] = sig
        depths[sym] = v20_pct

    sig_df = pd.DataFrame(signals).dropna(how='all')
    sig_df = sig_df.fillna(0)
    n_active = sig_df.abs().sum(axis=1)

    total_days = len(n_active)
    active_days = int((n_active >= 1).sum())
    print('=' * 70)
    print('Z8 六品种信号重叠结构（诊断，不含任何收益信息）')
    print('=' * 70)

    print('\n1. 单品种信号占用率（|signal|=1 的天数占比）:')
    for sym, name in SYMS.items():
        occ = (sig_df[sym].abs() > 0).mean()
        print(f'  {name:<8} {occ:>6.1%}')

    print(f'\n2. 同时性: {total_days}个交易日中有{active_days}天至少1个信号'
          f'（组合占用 {active_days/total_days:.1%}）')
    dist = n_active[n_active >= 1].value_counts().sort_index()
    for k, v in dist.items():
        print(f'  {int(k)}个品种同时信号: {v:>4}天 ({v/active_days:>5.1%} of active)')

    multi = int((n_active >= 2).sum())
    print(f'  → ≥2品种重叠天数: {multi}（占活跃日 {multi/active_days:.1%}，'
          f'占全部日 {multi/total_days:.1%}）')

    # When 2+ fire, how spread out is compression depth (ranking headroom)?
    dep_df = pd.DataFrame({s: depths[s] for s in SYMS}).reindex(sig_df.index)
    mask = n_active >= 2
    if mask.sum() > 0:
        rows = []
        for dt in sig_df.index[mask]:
            act = [s for s in SYMS if sig_df.loc[dt, s] != 0]
            vals = [dep_df.loc[dt, s] for s in act]
            rows.append(max(vals) - min(vals))
        spread = pd.Series(rows)
        print(f'\n3. 重叠日的压缩深度差（v20_pct 最大-最小, 排序空间）:')
        print(f'   mean={spread.mean():.3f}  median={spread.median():.3f}  '
              f'P90={spread.quantile(0.9):.3f}')
        print('   （0=完全并列无排序信息, 越大=强度区分越清晰）')

    # Average signal duration per commodity (event granularity)
    print('\n4. 单品种信号段平均长度（天）:')
    for sym, name in SYMS.items():
        s = sig_df[sym].abs()
        on = (s > 0) & (s.shift(1) == 0)
        off = (s == 0) & (s.shift(1) > 0)
        n_on = on.sum()
        if n_on > 0 and off.sum() > 0:
            avg_len = (s > 0).sum() / n_on
            print(f'  {name:<8} 触发{n_on:>3}段, 平均持续{avg_len:>5.1f}天')


if __name__ == '__main__':
    main()
