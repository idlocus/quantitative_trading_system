#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Z8 full backtest with integrated breakeven stop (properly parameterized).

⚠ 结果已作废（2026-11-18 勘误，见 docs/CTA_PREREG_2026-11-16.md 勘误节）：
  1. 止损后 no_reentry 永不解除（清除分支要求 pos!=0，止损时 pos 已=0）
     → 止损后该品种余下全样本不再交易（实证：stop_2pct 后半段夏普 NaN）
  2. 止损日只记手续费，遗漏昨收→止损成交价的当日损益；全程无进出场成本
  3. n_entries/n_stops 在品种循环内清零，返回值只是最后一个品种的计数

修复前不得引用本脚本任何输出。仅存档供复现，运行需显式设置
环境变量 QUANT_VOID_REPRO=1。
"""


def main():
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

    if os.environ.get('QUANT_VOID_REPRO') != '1':
        print('=' * 70)
        print('本脚本结果已作废（止损禁入永不解除/止损日记账错/计数器错）。')
        print('详情见 docs/CTA_PREREG_2026-11-16.md 勘误节。')
        print('如确需复现作废结果：set QUANT_VOID_REPRO=1 后重跑。')
        print('=' * 70)
        sys.exit(1)

    from pathlib import Path
    import numpy as np
    import pandas as pd
    import akshare as ak
    import time
    import pickle

    DEST = Path('ai-created/research/z8_full_stop')
    DEST.mkdir(parents=True, exist_ok=True)

    SYMS = {'AU0': 'Gold', 'CU0': 'Copper', 'RB0': 'Rebar',
            'I0': 'Iron', 'P0': 'Palm', 'HC0': 'HRC'}
    COST = 0.00025
    TARGET_VOL = 0.15
    VOL_WINDOW = 20
    VOL_LOOKBACK = 250
    VOL_THRESHOLD = 0.20


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


    def backtest(data, be_trigger=None):
        """Z8 backtest. be_trigger=None means no stop.
    be_trigger=0.02 means breakeven activated at 2% profit.
    Stop-out = no re-entry until signal resets to 0 then re-fires.
    """
        all_daily = {}

        for sym in SYMS:
            df = data[sym]
            close = df['收盘价'].astype(float)
            high = df['收盘价'].astype(float)  # fallback
            if '最高价' in df.columns:
                high = df['最高价'].astype(float).fillna(close)
            low = close.copy()
            if '最低价' in df.columns:
                low = df['最低价'].astype(float).fillna(close)
            ret = close.pct_change()
            v20 = ret.rolling(VOL_WINDOW).std()
            v20_pct = v20.rolling(VOL_LOOKBACK).rank(pct=True)
            ma20 = close.rolling(VOL_WINDOW).mean()
            signal = (v20_pct < VOL_THRESHOLD) * np.sign(close - ma20)

            daily_ret = pd.Series(0.0, index=close.index)
            pos = 0
            entry_px = 0.0
            best_px = 0.0
            be_on = False
            no_reentry = False  # after stop-out, until signal resets

            n_entries = 0
            n_stops = 0

            for i in range(len(close)):
                raw = signal.iloc[i]

                # Signal cycle detection: raw goes to 0 = cycle over
                if not np.isfinite(raw) or raw == 0:
                    if pos != 0:
                        # Signal exit
                        daily_ret.iloc[i] = ret.iloc[i] * pos if np.isfinite(ret.iloc[i]) else 0
                        pos = 0
                        be_on = False
                        no_reentry = False
                    continue

                # Signal is active
                if pos == 0:
                    if no_reentry:
                        continue
                    # Enter
                    pos = int(raw)
                    entry_px = close.iloc[i]
                    best_px = close.iloc[i]
                    be_on = False
                    n_entries += 1
                    daily_ret.iloc[i] = 0  # no return on entry day (simplification)
                else:
                    # Track best
                    h = high.iloc[i] if np.isfinite(high.iloc[i]) else close.iloc[i]
                    l = low.iloc[i] if np.isfinite(low.iloc[i]) else close.iloc[i]
                    if pos == 1:
                        best_px = max(best_px, h)
                    else:
                        best_px = min(best_px, l)

                    # Breakeven activation
                    if be_trigger is not None and not be_on:
                        if pos == 1 and best_px >= entry_px * (1 + be_trigger):
                            be_on = True
                        elif pos == -1 and best_px <= entry_px * (1 - be_trigger):
                            be_on = True

                    # Stop check
                    stopped = False
                    if be_trigger is not None and be_on:
                        if pos == 1 and l <= entry_px:
                            stopped = True
                        elif pos == -1 and h >= entry_px:
                            stopped = True

                    if stopped:
                        daily_ret.iloc[i] = 0 - 2 * COST  # breakeven minus cost
                        pos = 0
                        be_on = False
                        no_reentry = True
                        n_stops += 1
                        continue

                    # Normal P&L
                    daily_ret.iloc[i] = ret.iloc[i] * pos if np.isfinite(ret.iloc[i]) else 0

            all_daily[sym] = daily_ret

        port = pd.DataFrame(all_daily).mean(axis=1)
        port_vol = port.rolling(60).std() * np.sqrt(252)
        lev = (TARGET_VOL / port_vol).clip(upper=2).shift(1)
        net = (port * lev).dropna()
        return net, n_entries, n_stops


    def report(net, label, n_ent=0, n_stop=0):
        split = '2020-12-31'
        print(f'\n  {label}')
        if n_ent:
            print(f'    Entries: {n_ent} | Stop-outs: {n_stop}')
        for name, r in [('Full', net), ('2015-20', net[net.index <= split]),
                        ('2021-26', net[net.index > split])]:
            if len(r) < 100:
                continue
            yrs = len(r) / 252
            ann = (1 + r).prod() ** (1/yrs) - 1
            sh = r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else 0
            nav_p = (1 + r).cumprod()
            mdd = float((nav_p / nav_p.cummax() - 1).min())
            wr = (r > 0).mean()
            print(f'    {name:<10} ann{ann:>+7.1%} sh{sh:>5.2f} dd{mdd:>7.1%} wr{wr:.0%}')


    data = load_data()
    print(f'Data: {len(data)} symbols')
    print('=' * 70)
    print('Z8 Full Backtest: Breakeven Stop Comparison')
    print('=' * 70)

    # Test multiple trigger levels
    results = {}
    for trig in [None, 0.01, 0.02, 0.03, 0.05]:
        if trig is None:
            label = 'Original (no stop)'
        else:
            label = f'Breakeven stop (trigger {trig:.0%})'
        net, n_ent, n_stop = backtest(data, be_trigger=trig)
        report(net, label, n_ent, n_stop)
        results[str(trig)] = net

    # Save best result
    results['None'].to_csv(DEST / 'orig_daily.csv')
    results['0.02'].to_csv(DEST / 'stop_2pct_daily.csv')
    print(f'\nEvidence: {DEST}')


if __name__ == '__main__':
    main()
