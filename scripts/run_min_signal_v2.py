#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Minute signal exploration: batch download + analysis."""
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

# Download period: 3 months (fast enough for Wind)
START = '2026-08-01'
END = '2026-11-12'


def download_all():
    from WindPy import w
    w.start()
    data = {}
    for code, name in WIND_CODES.items():
        print(f'  {name}...', end='', flush=True)
        d = w.wsi(code, 'close,volume,open,high,low',
                  f'{START} 09:00:00', f'{END} 15:00:00', 'barsize=5')
        if d.ErrorCode == 0 and len(d.Times) > 100:
            n = len(d.Times)
            cols = ['close', 'volume', 'open', 'high', 'low']
            vals = {c: d.Data[i] if i < len(d.Data) else [np.nan]*n for i, c in enumerate(cols)}
            df = pd.DataFrame(vals, index=pd.DatetimeIndex(d.Times))
            df = df[~df.index.duplicated(keep='last')].sort_index()
            data[code] = df
            print(f' {len(df)} bars OK', flush=True)
        else:
            print(f' FAILED (err={d.ErrorCode})', flush=True)
        time.sleep(1)
    w.stop()
    pickle.dump(data, open(DEST / 'min5_3mo.pkl', 'wb'))
    print(f'\nSaved {len(data)} commodities to {DEST}/min5_3mo.pkl')
    return data


def analyze(data):
    print('\n' + '=' * 75)
    print('分钟级信号探索：波动率压缩→突破 在5分钟尺度是否存在')
    print('=' * 75)

    for code, df in data.items():
        name = WIND_CODES[code]
        close = df['close'].astype(float)
        ret = close.pct_change()
        if len(ret) < 2000:
            continue

        print(f'\n{"="*50}')
        print(f'{name} ({len(df)} bars, {df.index[0].date()} ~ {df.index[-1].date()})')
        print(f'{"="*50}')

        # Test multiple vol windows (in 5-min bars)
        # 20 bars=1.7h, 48 bars=4h(1 session), 96 bars=8h(2 sessions)
        for lb in [20, 48, 96]:
            vol = ret.rolling(lb).std()
            pct_win = lb * 8  # percentile lookback (shorter for 3mo data)
            vol_pct = vol.rolling(pct_win).rank(pct=True)
            ma = close.rolling(lb).mean()
            direction = np.sign(close - ma)

            # Forward returns: 12 bars=1h, 48 bars=4h
            for fwd in [12, 48]:
                fwd_ret = close.shift(-fwd) / close - 1
                compressed = vol_pct < 0.20
                if compressed.sum() < 50:
                    continue

                signed = fwd_ret * direction
                comp = signed[compressed].dropna()
                norm = signed[~compressed].dropna()
                if len(comp) < 30 or len(norm) < 30:
                    continue

                t = (comp.mean() - norm.mean()) / np.sqrt(
                    comp.var()/len(comp) + norm.var()/len(norm))

                # Direction hit rate
                correct = (np.sign(fwd_ret[compressed]) == direction[compressed])
                hr = correct.mean()

                tag = f'vol{lb*5/60:.0f}h fwd{fwd*5/60:.0f}h'
                sig = '***' if abs(t) > 3 else '**' if abs(t) > 2 else '*' if abs(t) > 1.5 else ''
                print(f'  {tag:<16} comp:{comp.mean()*10000:>+6.1f}bp  norm:{norm.mean()*10000:>+6.1f}bp  '
                      f'diff:{(comp.mean()-norm.mean())*10000:>+6.1f}bp  t={t:>+5.1f}{sig}  hit={hr:.0%}')


if __name__ == '__main__':
    DEST.mkdir(parents=True, exist_ok=True)
    cache = DEST / 'min5_3mo.pkl'
    if cache.exists():
        data = pickle.load(open(cache, 'rb'))
        print(f'Cached: {len(data)} commodities')
    else:
        data = download_all()

    if data:
        analyze(data)
    else:
        print('No data downloaded!')
