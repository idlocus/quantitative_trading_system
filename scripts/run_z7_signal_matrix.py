#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Z7: 8 new signals x 20 commodities matrix."""


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

    DEST = Path('ai-created/research/z7_signal_matrix_2026-11-17')
    DEST.mkdir(parents=True, exist_ok=True)
    COST = 0.00025
    SPLIT = '2020-12-31'

    SYMS = ['AU0','AG0','CU0','CF0','RB0','I0','M0','TA0','FG0','SC0',
            'J0','JM0','P0','SR0','HC0','NI0','AL0','ZN0','V0','PP0']
    NAMES = {'AU0':'gold','AG0':'silver','CU0':'copper','CF0':'cotton','RB0':'rebar',
             'I0':'iron','M0':'soymeal','TA0':'PTA','FG0':'glass','SC0':'crude',
             'J0':'coke','JM0':'cokingcoal','P0':'palm','SR0':'sugar','HC0':'HRC',
             'NI0':'nickel','AL0':'alum','ZN0':'zinc','V0':'PVC','PP0':'PP'}
    SECTORS = {
        'precious': ['AU0','AG0'],
        'metals': ['CU0','NI0','AL0','ZN0'],
        'ferrous': ['RB0','I0','HC0','J0','JM0'],
        'energy_chem': ['SC0','TA0','V0','PP0','FG0'],
        'agri': ['M0','P0','SR0','CF0'],
    }

    cache_file = DEST / 'data.pkl'
    if cache_file.exists():
        data = pickle.load(open(cache_file, 'rb'))
        print(f'cache: {len(data)} symbols', flush=True)
    else:
        data = {}
        for sym in SYMS:
            try:
                df = ak.futures_main_sina(symbol=sym, start_date='20150101', end_date='20261112')
                df['日期'] = pd.to_datetime(df['日期'])
                df = df.set_index('日期').sort_index()
                data[sym] = {'close': df['收盘价'].astype(float),
                             'oi': df['持仓量'].astype(float),
                             'volume': df['成交量'].astype(float)}
                time.sleep(0.2)
            except Exception:
                pass
        pickle.dump(data, open(cache_file, 'wb'))
        print(f'fetched: {len(data)} symbols', flush=True)

    close_mx = pd.DataFrame({s: data[s]['close'] for s in data}).sort_index()
    oi_mx = pd.DataFrame({s: data[s]['oi'] for s in data}).sort_index()
    vol_mx = pd.DataFrame({s: data[s]['volume'] for s in data}).sort_index()
    ret_mx = close_mx.pct_change()


    def make_signals(sym):
        """Return dict of signal name -> Series for one symbol."""
        sigs = {}
        close = close_mx[sym]
        ret = ret_mx[sym]
        oi = oi_mx[sym]
        volume = vol_mx[sym]

        # O2: OI momentum
        oi_chg = oi.pct_change(5)
        px_dir = np.sign(ret.rolling(5).sum())
        sigs['O2_OImom'] = (oi_chg > 0.01) * px_dir

        # O3: OI divergence
        oi_chg10 = oi.pct_change(10)
        px_chg10 = ret.rolling(10).sum().abs()
        dir3 = np.sign(ret.rolling(3).sum())
        sigs['O3_OIdiv'] = ((oi_chg10 > 0.02) & (px_chg10 < 0.01)) * dir3

        # O4: vol/OI ratio spec heat
        ratio = volume / oi
        pct = ratio.rolling(250).rank(pct=True)
        sigs['O4_voloi'] = -(pct > 0.8) * np.sign(ret.rolling(5).sum())

        # O5: OI extreme
        oi_pct = oi.rolling(250).rank(pct=True)
        sigs['O5_OIextreme'] = -(oi_pct > 0.9) * np.sign(ret.rolling(20).sum())

        # V1: vol breakout
        v20 = ret.rolling(20).std()
        v60 = ret.rolling(60).std()
        brk = (v20 > v60) & (v20.shift(1) <= v60.shift(1))
        sigs['V1_volbrk'] = brk * np.sign(ret.rolling(5).sum())

        # V2: vol squeeze
        v20_pct = v20.rolling(250).rank(pct=True)
        ma20 = close.rolling(20).mean()
        sigs['V2_volsqz'] = (v20_pct < 0.2) * np.sign(close - ma20)

        # X1: steel chain (only for RB0, HC0)
        if sym in ('RB0', 'HC0') and 'JM0' in ret_mx and 'J0' in ret_mx:
            jm5 = ret_mx['JM0'].rolling(5).sum()
            j5 = ret_mx['J0'].rolling(5).sum()
            sig = pd.Series(0.0, index=close.index)
            sig[(jm5 > 0) & (j5 > 0)] = 1.0
            sig[(jm5 < 0) & (j5 < 0)] = -1.0
            sigs['X1_steel'] = sig
        else:
            sigs['X1_steel'] = pd.Series(0.0, index=close.index)

        # X2: sector consensus
        sector = None
        for sec, members in SECTORS.items():
            if sym in members:
                sector = members
                break
        if sector and len(sector) >= 3:
            rets5 = pd.DataFrame({m: ret_mx[m].rolling(5).sum() for m in sector if m in ret_mx})
            n_pos = (rets5 > 0).sum(axis=1)
            n_neg = (rets5 < 0).sum(axis=1)
            n_tot = rets5.notna().sum(axis=1).replace(0, 1)
            sig = pd.Series(0.0, index=close.index)
            sig[(n_pos / n_tot) >= 0.6] = 1.0
            sig[(n_neg / n_tot) >= 0.6] = -1.0
            sigs['X2_sectcons'] = sig
        else:
            sigs['X2_sectcons'] = pd.Series(0.0, index=close.index)

        return sigs


    print('\n===== Z7: 8 signals x 20 commodities =====')
    print(f"{'Signal':<14} {'Sym':<8} {'P1 ann':>9} {'P2 ann':>9} {'All ann':>9} {'Sharpe':>8} {'Eff':>4}")

    results = []
    for sym in SYMS:
        if sym not in data:
            continue
        sigs = make_signals(sym)
        ret = ret_mx[sym]
        for sig_name, sig in sigs.items():
            s = sig.shift(1)
            daily = (s * ret).dropna()
            if len(daily) < 500:
                continue
            p1 = daily[daily.index <= SPLIT]
            p2 = daily[daily.index > SPLIT]
            y1 = max(len(p1) / 252, 0.1)
            y2 = max(len(p2) / 252, 0.1)
            a1 = (1 + p1).prod() ** (1/y1) - 1
            a2 = (1 + p2).prod() ** (1/y2) - 1
            ya = len(daily) / 252
            aa = (1 + daily).prod() ** (1/ya) - 1
            sh = daily.mean() / daily.std() * np.sqrt(252) if daily.std() > 0 else 0
            eff = aa > 0 and a1 > 0 and a2 > 0
            results.append({'signal': sig_name, 'symbol': sym,
                            'name': NAMES.get(sym, sym),
                            'ann_p1': a1, 'ann_p2': a2, 'ann_all': aa,
                            'sharpe': sh, 'effective': eff})
            e = 'Y' if eff else 'N'
            print(f'{sig_name:<14} {NAMES.get(sym,sym):<8} {a1:>+9.1%} {a2:>+9.1%} {aa:>+9.1%} {sh:>8.2f} {e:>4}')

    df = pd.DataFrame(results)
    df.to_csv(DEST / 'signal_matrix.csv', index=False, encoding='utf-8-sig')

    print('\n===== Summary =====')
    for sn in sorted(df['signal'].unique()):
        sub = df[df['signal'] == sn]
        n_eff = int(sub['effective'].sum())
        n_tot = len(sub)
        if n_tot > 0:
            best_idx = sub['ann_all'].idxmax()
            best = sub.loc[best_idx]
            print(f'{sn:<14} eff {n_eff:>2}/{n_tot:<3} best: {best["name"]} {best["ann_all"]:+.1%} (sh={best["sharpe"]:.2f})')

    from research.factor_screen2 import register
    n_e = int(df['effective'].sum())
    register(f'Z7 signal matrix: {n_e}/{len(df)} effective', label='z7_signal_matrix')
    print(f'\nevidence: {DEST}')


if __name__ == '__main__':
    main()
