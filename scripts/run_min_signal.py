#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Download minute data + exploratory signal analysis."""
import sys, os
if __name__ == '__main__':
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from pathlib import Path
import time
import pickle

import numpy as np
import pandas as pd

DEST = Path('ai-created/research/z9_minute_signal')

WIND_CODES = {
    'AU.SHF': 'Gold', 'CU.SHF': 'Copper', 'RB.SHF': 'Rebar',
    'I.DCE': 'Iron', 'P.DCE': 'Palm', 'HC.SHF': 'HRC',
}


def download_wind_minute(code, start, end):
    """Download 5-min bars from Wind."""
    from WindPy import w
    data = w.wsi(code, 'close,volume,open,high,low', f'{start} 09:00:00', f'{end} 15:00:00', 'barsize=5')
    if data.ErrorCode != 0 or len(data.Times) == 0:
        return None
    n = len(data.Times)
    cols = ['close', 'volume', 'open', 'high', 'low']
    values = {c: data.Data[i] if i < len(data.Data) else [np.nan]*n for i, c in enumerate(cols)}
    df = pd.DataFrame(values, index=pd.DatetimeIndex(data.Times))
    return df[~df.index.duplicated(keep='last')].sort_index()


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    # Download or load cached data
    cache = DEST / 'min5_data.pkl'
    if cache.exists():
        data = pickle.load(open(cache, 'rb'))
        print(f'Cached: {len(data)} commodities')
    else:
        from WindPy import w
        w.start()
        data = {}
        for code, name in WIND_CODES.items():
            print(f'  Downloading {name}...', flush=True)
            df = download_wind_minute(code, '2025-11-01', '2026-11-12')
            if df is not None and len(df) > 1000:
                data[code] = df
                print(f'    {len(df)} bars: {df.index[0]} ~ {df.index[-1]}', flush=True)
            time.sleep(0.5)
        w.stop()
        pickle.dump(data, open(cache, 'wb'))
        print(f'Downloaded: {len(data)} commodities')

    if not data:
        print('No data!')
        return

    # Exploratory analysis for each commodity
    print('\n' + '=' * 75)
    print('分钟级信号探索：波动率压缩在5分钟尺度是否存在并预测突破')
    print('=' * 75)

    for code, df in data.items():
        name = WIND_CODES[code]
        close = df['close'].astype(float)
        ret = close.pct_change()

        # Multiple volatility windows (in 5-min bars)
        # 20 bars = 100 min (~1.5 hrs)
        # 48 bars = 240 min (~4 hrs = 1 session)
        # 240 bars = ~5 sessions
        for lb in [20, 48, 96]:
            vol = ret.rolling(lb).std()
            # Percentile within trailing window
            pct_window = lb * 10  # lookback for percentile
            vol_pct = vol.rolling(pct_window).rank(pct=True)

            # Forward returns (next 12 bars = 1 hour, 48 bars = 4 hours)
            for fwd in [12, 48]:
                fwd_ret = close.shift(-fwd) / close - 1

                # Split into compressed vs normal
                compressed = vol_pct < 0.20
                if compressed.sum() < 100:
                    continue

                # Direction from short MA
                ma = close.rolling(lb).mean()
                direction = np.sign(close - ma)
                signed_ret = fwd_ret * direction

                comp_ret = signed_ret[compressed].dropna()
                norm_ret = signed_ret[~compressed].dropna()

                if len(comp_ret) < 50 or len(norm_ret) < 50:
                    continue

                t_stat = (comp_ret.mean() - norm_ret.mean()) / np.sqrt(
                    comp_ret.var()/len(comp_ret) + norm_ret.var()/len(norm_ret))

                print(f'\n  {name} vol{lb}bars({lb*5/60:.0f}h) fwd{fwd}bars({fwd*5/60:.0f}h):')
                print(f'    压缩组: n={len(comp_ret):>6} 均值 {comp_ret.mean()*10000:>+6.1f}bp')
                print(f'    正常组: n={len(norm_ret):>6} 均值 {norm_ret.mean()*10000:>+6.1f}bp')
                print(f'    差: {(comp_ret.mean()-norm_ret.mean())*10000:>+6.1f}bp (t={t_stat:+.2f})')

                # Breakout analysis: direction of next move after compression
                if fwd == 12:  # 1 hour ahead
                    # Check if direction from MA predicts forward return direction
                    correct = (np.sign(fwd_ret[compressed]) == direction[compressed])
                    hit_rate = correct.mean()
                    print(f'    方向命中率: {hit_rate:.1%}')

    print('\n' + '=' * 75)
    print(f'Evidence: {DEST}')


if __name__ == '__main__':
    main()
