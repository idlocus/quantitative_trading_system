#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 2-4: Full backtest with exit optimization and correlation check."""
import sys, os
if __name__ == '__main__':
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from pathlib import Path
import pickle

import numpy as np
import pandas as pd

DEST = Path('ai-created/research/z9_minute_signal')
COST = 0.00025


def load_data():
    f = DEST / 'min5_2yr_6com.pkl'
    if not f.exists():
        print(f'Data not ready: {f}')
        sys.exit(1)
    from scripts.run_z9_batch_download import load_verified_product
    return load_verified_product(f)


def run_strategy(df, vol_win=96, vol_pct_win=None, vol_thr=0.20,
                 ma_win=None, max_hold=48, trail=None, close_overnight=True):
    """Run minute-level vol compression strategy.

    vol_win: bars for rolling vol (96=8h)
    vol_thr: percentile threshold (0.20)
    ma_win: MA window (default=same as vol_win)
    max_hold: max holding bars (48=4h)
    trail: trailing stop pct (None=no trail)
    close_overnight: close all positions at end of day
    """
    if vol_pct_win is None: vol_pct_win = vol_win * 8
    if ma_win is None: ma_win = vol_win

    close = df['close'].astype(float)
    ret = close.pct_change()
    vol = ret.rolling(vol_win).std()
    vol_pct = vol.rolling(vol_pct_win).rank(pct=True)
    ma = close.rolling(ma_win).mean()
    compressed = (vol_pct < vol_thr).fillna(False)
    direction = np.sign(close - ma)

    daily = {}
    pos = 0; entry_i = 0; entry_px = 0; best_px = 0

    for i in range(len(close)):
        dt = close.index[i]
        dk = dt.date()
        r = ret.iloc[i] if np.isfinite(ret.iloc[i]) else 0

        # Day boundary
        if i > 0 and close.index[i-1].date() != dk:
            if close_overnight and pos != 0:
                pos = 0; best_px = 0

        if pos == 0:
            if compressed.iloc[i] and direction.iloc[i] != 0:
                pos = int(direction.iloc[i])
                entry_i = i; entry_px = close.iloc[i]; best_px = close.iloc[i]
        else:
            # Track best
            if pos == 1: best_px = max(best_px, close.iloc[i])
            else: best_px = min(best_px, close.iloc[i])

            # Trail stop
            if trail is not None:
                if pos == 1 and close.iloc[i] <= best_px * (1 - trail):
                    pos = 0
                elif pos == -1 and close.iloc[i] >= best_px * (1 + trail):
                    pos = 0

            # Exit: vol exits compression
            if pos != 0 and not compressed.iloc[i]:
                pos = 0
            # Exit: max holding
            elif pos != 0 and (i - entry_i) >= max_hold:
                pos = 0

        if dk not in daily:
            daily[dk] = 0.0
        if pos != 0:
            daily[dk] += r * pos

    ds = pd.Series(daily)
    ds.index = pd.to_datetime(ds.index)

    # Apply cost (charge per day with entries)
    entries_per_day = ((compressed & (compressed.shift(1) == False)).fillna(False))
    entries_by_day = entries_per_day.groupby(close.index.date).sum()
    cost_by_day = {d: n * 2 * COST for d, n in entries_by_day.items()}
    for d in ds.index:
        dd = d.date()
        if dd in cost_by_day:
            ds[d] -= cost_by_day[dd]

    return ds


def metrics(daily, label):
    if len(daily) < 20:
        return
    yrs = len(daily) / 250
    ann = (1 + daily).prod() ** (1/yrs) - 1
    sh = daily.mean() / daily.std() * np.sqrt(250) if daily.std() > 0 else 0
    nav = (1 + daily).cumprod()
    mdd = float((nav / nav.cummax() - 1).min())
    wr = (daily > 0).mean()
    print(f'  {label:<40} ann={ann:>+7.1%} sh={sh:>5.2f} dd={mdd:>7.1%} wr={wr:.0%}')
    return ann, sh, mdd


def main():
    data = load_data()
    print(f'Loaded {len(data)} commodities')

    print('\n' + '=' * 75)
    print('Step 2: 全量回测（22个月数据）')
    print('=' * 75)

    # Test multiple exit rules
    EXIT_RULES = [
        ('Fixed 4h', dict(max_hold=48)),
        ('Fixed 2h', dict(max_hold=24)),
        ('Fixed 8h', dict(max_hold=96)),
        ('Trail 1%', dict(max_hold=48, trail=0.01)),
        ('Trail 2%', dict(max_hold=48, trail=0.02)),
        ('Signal exit only', dict(max_hold=999999)),
    ]

    all_daily = {}
    for sym, name in [('CU.SHF', 'Copper'), ('RB.SHF', 'Rebar')]:
        if sym not in data:
            continue
        df = data[sym]
        print(f'\n--- {name} ({len(df)} bars) ---')
        for label, params in EXIT_RULES:
            daily = run_strategy(df, **params)
            all_daily[f'{name}_{label}'] = daily
            metrics(daily, label)

    # Best combination portfolio
    print(f'\n--- 组合（等权） ---')
    for label, _ in EXIT_RULES:
        c_daily = all_daily.get(f'Copper_{label}')
        r_daily = all_daily.get(f'Rebar_{label}')
        if c_daily is not None and r_daily is not None:
            port = pd.DataFrame({'C': c_daily, 'R': r_daily}).mean(axis=1)
            metrics(port, f'Portfolio: {label}')

    # Step 4: Correlation with daily returns
    print(f'\n{"="*75}')
    print('Step 4: 与日频收益的独立性检验')
    print(f'{"="*75}')
    # Compare with daily Z8 net returns if available
    z8_file = Path('ai-created/research/z8_full_stop/data.pkl')
    if z8_file.exists():
        z8_data = pickle.load(open(z8_file, 'rb'))
        if 'AU0' in z8_data:
            z8_close = z8_data['AU0']['收盘价'].astype(float)
            z8_ret = z8_close.pct_change()
            z8_vol = z8_ret.rolling(20).std()
            z8_pct = z8_vol.rolling(250).rank(pct=True)
            z8_ma = z8_close.rolling(20).mean()
            z8_sig = (z8_pct < 0.20) * np.sign(z8_close - z8_ma)
            z8_daily = (z8_sig.shift(1) * z8_ret).dropna()
            print(f'  Daily Z8 available: {len(z8_daily)} days')

            # Check correlation with minute Copper
            if 'Copper_Fixed 4h' in all_daily:
                min_cu = all_daily['Copper_Fixed 4h']
                common = pd.DataFrame({'minute': min_cu, 'daily_z8': z8_daily}).dropna()
                if len(common) > 30:
                    corr = common['minute'].corr(common['daily_z8'])
                    print(f'  相关性(minute Copper vs daily Z8): {corr:.3f}')
    else:
        print('  Daily Z8 data not available for comparison')

    print(f'\nEvidence: {DEST}')


if __name__ == '__main__':
    main()
