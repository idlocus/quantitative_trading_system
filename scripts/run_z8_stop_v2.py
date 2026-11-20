#!/usr/bin/env python3
# -*- coding: utf-8 -*-


def main():
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

    from pathlib import Path
    import numpy as np
    import pandas as pd
    import akshare as ak
    import time
    import pickle

    DEST = Path('ai-created/research/z8_full_stop')
    DEST.mkdir(parents=True, exist_ok=True)

    SYMS = ['AU0', 'CU0', 'RB0', 'I0', 'P0', 'HC0']
    COST = 0.00025
    TARGET_VOL = 0.15


    def load_data():
        cf = DEST / 'data.pkl'
        if cf.exists():
            return pickle.load(open(cf, 'rb'))
        data = {}
        for sym in SYMS:
            df = ak.futures_main_sina(symbol=sym, start_date='20150101', end_date='20261112')
            df['dt'] = pd.to_datetime(df['日期'])
            data[sym] = df.set_index('dt').sort_index()
            time.sleep(0.3)
        pickle.dump(data, open(cf, 'wb'))
        return data


    def make_signal(sym, data):
        close = data[sym]['收盘价'].astype(float)
        ret = close.pct_change()
        v20 = ret.rolling(20).std()
        v20_pct = v20.rolling(250).rank(pct=True)
        ma20 = close.rolling(20).mean()
        sig = (v20_pct < 0.20) * np.sign(close - ma20)
        return close, ret, sig


    def run_baseline(data):
        all_daily = {}
        for sym in SYMS:
            close, ret, sig = make_signal(sym, data)
            all_daily[sym] = (sig.shift(1) * ret).dropna()
        port = pd.DataFrame(all_daily).mean(axis=1)
        pv = port.rolling(60).std() * np.sqrt(252)
        lev = (TARGET_VOL / pv).clip(upper=2).shift(1)
        return (port * lev).dropna()


    def run_stop(data, trig, cd=0):
        all_daily = {}
        n_stops = 0
        for sym in SYMS:
            close, ret, sig = make_signal(sym, data)
            high = data[sym]['最高价'].astype(float).fillna(close)
            low = data[sym]['最低价'].astype(float).fillna(close)
            daily = pd.Series(0.0, index=close.index)
            pos = 0; entry = 0; best = 0; be = False; cool = 0
            ssig = sig.shift(1)
            for i in range(len(close)):
                s = ssig.iloc[i]; raw = sig.iloc[i]
                r = ret.iloc[i] if np.isfinite(ret.iloc[i]) else 0
                h = high.iloc[i] if np.isfinite(high.iloc[i]) else close.iloc[i]
                l = low.iloc[i] if np.isfinite(low.iloc[i]) else close.iloc[i]
                if cool > 0: cool -= 1
                if not np.isfinite(raw) or raw == 0:
                    be = False; cool = 0
                if pos == 0:
                    if np.isfinite(s) and s != 0 and cool == 0:
                        pos = int(s); entry = close.iloc[i]; best = close.iloc[i]; be = False
                        daily.iloc[i] = r * pos
                else:
                    if pos == 1: best = max(best, h)
                    else: best = min(best, l)
                    if trig is not None and not be:
                        if pos == 1 and best >= entry * (1 + trig): be = True
                        elif pos == -1 and best <= entry * (1 - trig): be = True
                    stopped = False
                    if trig is not None and be:
                        if pos == 1 and l <= entry: stopped = True
                        elif pos == -1 and h >= entry: stopped = True
                    if stopped:
                        daily.iloc[i] = 0 - COST
                        pos = 0; be = False; cool = cd; n_stops += 1
                    else:
                        daily.iloc[i] = r * pos
            all_daily[sym] = daily
        port = pd.DataFrame(all_daily).mean(axis=1)
        pv = port.rolling(60).std() * np.sqrt(252)
        lev = (TARGET_VOL / pv).clip(upper=2).shift(1)
        return (port * lev).dropna(), n_stops


    def rep(net, lab, ns=0):
        sp = '2020-12-31'
        sfx = f' stops:{ns}' if ns else ''
        print(f'\n  {lab}{sfx}')
        for nm, r in [('Full', net), ('15-20', net[net.index<=sp]), ('21-26', net[net.index>sp])]:
            if len(r) < 100: continue
            y = len(r)/252
            a = (1+r).prod()**(1/y)-1
            sh = r.mean()/r.std()*np.sqrt(252) if r.std()>0 else 0
            np_v = (1+r).cumprod()
            m = float((np_v/np_v.cummax()-1).min())
            print(f'    {nm:<8} ann{a:>+7.1%} sh{sh:>5.2f} dd{m:>7.1%}')


    data = load_data()
    print(f'Data: {len(data)} symbols')
    print('='*70)
    print('Z8 Baseline + Breakeven Stop')
    print('='*70)
    nb = run_baseline(data)
    rep(nb, 'BASELINE (expect ~+8.6%, sh~0.88)')
    for t in [0.01, 0.02, 0.03, 0.05]:
        for c in [0, 5, 10]:
            ns, st = run_stop(data, t, c)
            rep(ns, f'Stop {t:.0%} cd{c}d', st)


if __name__ == '__main__':
    main()
