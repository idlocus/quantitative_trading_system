#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Minute-level CTA prototype: vol compression breakout on best signal commodities."""


def main():
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

    from pathlib import Path
    import pickle
    import numpy as np
    import pandas as pd

    DEST = Path('ai-created/research/z9_minute_signal')

    # Load data
    data = pickle.load(open(DEST / 'min5_21commodities.pkl', 'rb'))

    COST = 0.00025  # one-way, commodity futures
    VOL_WINDOW = 96  # 96 bars = 8 hours
    VOL_PCTILE_WIN = 96 * 8  # percentile lookback
    VOL_THRESHOLD = 0.20
    MA_WINDOW = 96  # same as vol window
    MAX_HOLD_BARS = 48  # max 4 hours holding

    CORE = {'CU.SHF': 'Copper', 'RB.SHF': 'Rebar'}  # strongest signals


    def run_strategy(sym, df, use_signal=True):
        """Run minute-level vol compression strategy for one commodity.
    Returns daily returns series.
    """
        close = df['close'].astype(float)
        ret = close.pct_change()

        vol = ret.rolling(VOL_WINDOW).std()
        vol_pct = vol.rolling(VOL_PCTILE_WIN).rank(pct=True)
        ma = close.rolling(MA_WINDOW).mean()
        direction = np.sign(close - ma)
        compressed = (vol_pct < VOL_THRESHOLD).fillna(False)

        # Position tracking
        daily = {}
        pos = 0
        entry_idx = 0
        bars_held = 0

        for i in range(len(close)):
            dt = close.index[i]
            date_key = dt.date()
            r = ret.iloc[i] if np.isfinite(ret.iloc[i]) else 0

            # New day: close overnight positions
            if i > 0 and close.index[i-1].date() != date_key:
                if pos != 0:
                    pos = 0  # no overnight holding
                    bars_held = 0

            if pos == 0:
                if compressed.iloc[i] and direction.iloc[i] != 0:
                    pos = int(direction.iloc[i])
                    entry_idx = i
                    bars_held = 0
            else:
                bars_held = i - entry_idx
                # Exit conditions
                if not compressed.iloc[i]:  # vol exits compression
                    pos = 0
                elif bars_held >= MAX_HOLD_BARS:  # max holding
                    pos = 0

            # Record daily P&L
            if date_key not in daily:
                daily[date_key] = 0.0
            if pos != 0:
                daily[date_key] += r * pos

        # Convert to Series, apply cost per entry
        daily_s = pd.Series(daily)
        daily_s.index = pd.to_datetime(daily_s.index)

        # Estimate entries per day (rough: count signal transitions)
        entries = ((compressed & (compressed.shift(1) == False)).fillna(False)).groupby(close.index.date).sum()
        cost_per_day = {d: n * 2 * COST for d, n in entries.items()}
        cost_s = pd.Series(cost_per_day)
        cost_s.index = pd.to_datetime(cost_s.index)

        # Align and subtract cost
        for d in daily_s.index:
            if d in cost_s.index:
                daily_s[d] -= cost_s[d]

        return daily_s


    # Run for core commodities
    print('=' * 70)
    print('分钟级CTA原型：8h波动压缩→MA方向→最长4h持有→日内平仓')
    print(f'品种：{", ".join(CORE.values())}')
    print('=' * 70)

    all_daily = {}
    for sym, name in CORE.items():
        if sym not in data:
            continue
        daily = run_strategy(sym, data[sym])
        all_daily[name] = daily
        yrs = len(daily) / 250
        ann = (1 + daily).prod() ** (1 / yrs) - 1 if len(daily) > 10 else 0
        sh = daily.mean() / daily.std() * np.sqrt(250) if daily.std() > 0 else 0
        wr = (daily > 0).mean() if len(daily) > 0 else 0
        print(f'\n  {name}: {len(daily)} days')
        print(f'    ann={ann:>+7.1%} sh={sh:>5.2f} wr={wr:.0%}')
        print(f'    avg_daily={daily.mean()*10000:+.1f}bp')

    # Portfolio (equal weight)
    if all_daily:
        port = pd.DataFrame(all_daily).mean(axis=1)
        yrs = len(port) / 250
        ann = (1 + port).prod() ** (1 / yrs) - 1
        sh = port.mean() / port.std() * np.sqrt(250) if port.std() > 0 else 0
        nav = (1 + port).cumprod()
        mdd = float((nav / nav.cummax() - 1).min())
        wr = (port > 0).mean()

        print('\n' + '='*70)
        print(f'组合（等权 {len(all_daily)} 品种）')
        print(f'  ann={ann:>+7.1%} sh={sh:>5.2f} dd={mdd:>7.1%} wr={wr:.0%}')
        print(f'  天数={len(port)} avg_daily={port.mean()*10000:+.1f}bp')

        # By month
        print('\n  月度收益:')
        monthly = port.resample('ME').sum()
        for idx, v in monthly.items():
            print(f'    {idx.strftime("%Y-%m")}: {v:>+7.2%}')

    print(f'\nEvidence: {DEST}')


if __name__ == '__main__':
    main()
