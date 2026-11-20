#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Minute signal exploration: expanded to 21 commodities."""
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

ALL_CODES = {
    # Original 6
    'AU.SHF': 'Gold', 'CU.SHF': 'Copper', 'RB.SHF': 'Rebar',
    'I.DCE': 'Iron', 'P.DCE': 'Palm', 'HC.SHF': 'HRC',
    # New: precious/base metals
    'AG.SHF': 'Silver', 'NI.SHF': 'Nickel', 'AL.SHF': 'Alum',
    'ZN.SHF': 'Zinc', 'SS.SHF': 'Stainless',
    # New: energy/chemical
    'SC.INE': 'Crude', 'TA.CZC': 'PTA', 'MA.CZC': 'Methanol',
    'FG.CZC': 'Glass', 'BU.SHF': 'Asphalt', 'L.DCE': 'LLDPE',
    'V.DCE': 'PVC', 'PP.DCE': 'PP',
    # New: black metal chain
    'J.DCE': 'Coke', 'JM.DCE': 'CokingCoal', 'ZC.CZC': 'ThermalCoal',
    # New: agricultural
    'Y.DCE': 'SoyOil', 'SR.CZC': 'Sugar', 'CF.CZC': 'Cotton',
    'M.DCE': 'SoyMeal', 'C.DCE': 'Corn', 'AP.CZC': 'Apple',
}

START = '2026-08-01'
END = '2026-11-12'


def download_all():
    from WindPy import w
    w.start()
    data = {}
    failed = []
    for code, name in ALL_CODES.items():
        print(f'  {name}({code})...', end='', flush=True)
        try:
            d = w.wsi(code, 'close,volume', f'{START} 09:00:00', f'{END} 15:00:00', 'barsize=5')
            if d.ErrorCode == 0 and len(d.Times) > 500:
                n = len(d.Times)
                vals = {'close': d.Data[0], 'volume': d.Data[1] if len(d.Data) > 1 else [0]*n}
                df = pd.DataFrame(vals, index=pd.DatetimeIndex(d.Times))
                df = df[~df.index.duplicated(keep='last')].sort_index()
                data[code] = df
                print(f' {len(df)} OK', flush=True)
            else:
                failed.append(code)
                print(f' FAIL({d.ErrorCode})', flush=True)
        except Exception:
            failed.append(code)
            print(' ERR', flush=True)
        time.sleep(0.5)
    w.stop()
    pickle.dump(data, open(DEST / 'min5_21commodities.pkl', 'wb'))
    print(f'\nDownloaded: {len(data)} OK, {len(failed)} failed')
    if failed:
        print(f'Failed: {failed}')
    return data


def analyze_one(code, df, name):
    close = df['close'].astype(float)
    ret = close.pct_change()
    if len(ret) < 1500:
        return None

    results = {}
    # Key test: 8h vol window, 4h forward (best from initial exploration)
    for lb in [48, 96]:  # 4h and 8h windows
        vol = ret.rolling(lb).std()
        pct_win = lb * 8
        vol_pct = vol.rolling(pct_win).rank(pct=True)
        ma = close.rolling(lb).mean()
        direction = np.sign(close - ma)

        for fwd in [12, 48]:  # 1h and 4h forward
            fwd_ret = close.shift(-fwd) / close - 1
            compressed = (vol_pct < 0.20).fillna(False)
            if compressed.sum() < 30:
                continue

            signed = (fwd_ret * direction)[compressed].dropna()
            normal = (fwd_ret * direction)[~compressed].dropna()
            if len(signed) < 20 or len(normal) < 20:
                continue

            diff_bp = (signed.mean() - normal.mean()) * 10000
            se = np.sqrt(signed.var()/len(signed) + normal.var()/len(normal))
            t = diff_bp / 10000 / se if se > 0 else 0
            correct = (np.sign(fwd_ret[compressed]) == direction[compressed])
            hr = correct.mean()

            key = f'v{lb*5//60}h_f{fwd*5//60}h'
            results[key] = {'diff_bp': diff_bp, 't': t, 'hit': hr, 'n_comp': len(signed)}

    return results


if __name__ == '__main__':
    DEST.mkdir(parents=True, exist_ok=True)
    cache = DEST / 'min5_21commodities.pkl'
    if cache.exists():
        data = pickle.load(open(cache, 'rb'))
        print(f'Cached: {len(data)} commodities')
    else:
        data = download_all()

    if not data:
        print('No data!')
        sys.exit(1)

    # Analyze all
    all_results = {}
    for code, df in data.items():
        name = ALL_CODES.get(code, code)
        r = analyze_one(code, df, name)
        if r:
            all_results[name] = r

    # Summary table
    print('\n' + '=' * 90)
    print('分钟级波动率压缩信号 × 全品种汇总（2026.08 ~ 2026.11，3个月）')
    print('=' * 90)
    print(f'{"品种":<12} {"4h窗/1h前":>20} {"4h窗/4h前":>20} {"8h窗/1h前":>20} {"8h窗/4h前":>20}')
    print(f'{"":12} {"diff(bp)/t/hit":>20} {"diff(bp)/t/hit":>20} {"diff(bp)/t/hit":>20} {"diff(bp)/t/hit":>20}')
    print('-' * 90)

    # Sort by 8h/4h signal strength
    sorted_names = sorted(all_results.keys(),
                         key=lambda n: all_results[n].get('v8h_f4h', {}).get('t', 0),
                         reverse=True)

    for name in sorted_names:
        r = all_results[name]
        cells = []
        for key in ['v4h_f1h', 'v4h_f4h', 'v8h_f1h', 'v8h_f4h']:
            d = r.get(key)
            if d:
                sig = '***' if abs(d['t']) > 3 else '**' if abs(d['t']) > 2 else ''
                cells.append(f'{d["diff_bp"]:+6.1f} t={d["t"]:+4.1f}{sig} {d["hit"]:.0%}')
            else:
                cells.append(f'{"—":>18}')
        print(f'{name:<12} {cells[0]:>20} {cells[1]:>20} {cells[2]:>20} {cells[3]:>20}')

    # Summary statistics
    v8h_t = [all_results[n].get('v8h_f4h', {}).get('t', 0) for n in all_results]
    pos = sum(1 for t in v8h_t if t > 2)
    neg = sum(1 for t in v8h_t if t < -2)
    neu = len(v8h_t) - pos - neg
    print(f'\n{"="*90}')
    print(f'8h波动窗/4h前瞻: 正信号 {pos} | 负信号 {neg} | 中性 {neu} (共{len(all_results)}品种)')
    print(f'Evidence: {DEST}')
