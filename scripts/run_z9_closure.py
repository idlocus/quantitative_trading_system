#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Z9 minute-level CTA closure: all 6 commodities, two-period split,
and same-bar attribution bias quantification.

Questions to answer for the final verdict:
1. Does the minute vol-compression signal generalize across all 6 downloaded
   commodities on 21 months (not just CU/RB)?
2. Is it stable across the two halves of the sample?
3. How much of the (already weak) performance comes from the favorable
   same-bar attribution (decide on close[i], get credited ret[i])?
   lag=0 reproduces run_z9_full_backtest.py; lag=1 is the implementable
   version (decide on close[i], earn close[i+1]/close[i]-1).
"""
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

SYMS = {
    'CU.SHF': 'Copper', 'RB.SHF': 'Rebar', 'AP.CZC': 'Apple',
    'BU.SHF': 'Asphalt', 'TA.CZC': 'PTA', 'M.DCE': 'SoyMeal',
}

# Prototype parameters (same as run_min_prototype.py / Fixed 4h in full backtest)
VOL_WIN = 96          # 96 bars = 8h rolling vol
VOL_PCT_WIN = 96 * 8  # percentile lookback
VOL_THR = 0.20
MA_WIN = 96
MAX_HOLD = 48         # 4h


def decide_positions(df):
    """Run the signal loop, return per-bar DECIDED position series.

    decided[i] = position after processing bar i (uses close[i]).
    Identical logic to run_z9_full_backtest.run_strategy body.
    """
    close = df['close'].astype(float)
    ret = close.pct_change()
    vol = ret.rolling(VOL_WIN).std()
    vol_pct = vol.rolling(VOL_PCT_WIN).rank(pct=True)
    ma = close.rolling(MA_WIN).mean()
    compressed = (vol_pct < VOL_THR).fillna(False)
    direction = np.sign(close - ma)

    decided = np.zeros(len(close), dtype=int)
    pos = 0
    entry_i = 0
    idx = close.index

    for i in range(len(close)):
        if i > 0 and idx[i - 1].date() != idx[i].date():
            pos = 0  # no overnight holding
        if pos == 0:
            if compressed.iloc[i] and direction.iloc[i] != 0:
                pos = int(direction.iloc[i])
                entry_i = i
        else:
            if not compressed.iloc[i]:
                pos = 0
            elif (i - entry_i) >= MAX_HOLD:
                pos = 0
        decided[i] = pos

    return pd.Series(decided, index=idx)


def daily_returns(df, decided, lag, charge_cost=True):
    """Convert decided positions to daily returns.

    lag=0: credit ret[i] to decided[i]  (same-bar, favorable bias)
    lag=1: credit ret[i] to decided[i-1] (implementable: enter at signal close)
    """
    close = df['close'].astype(float)
    ret = close.pct_change()
    held = decided.shift(lag).fillna(0)
    pnl = held * ret

    daily = pnl.groupby(idx_date(close.index)).sum()
    daily.index = pd.to_datetime(daily.index)

    if charge_cost:
        entries = held.diff().abs().fillna(0) / 2  # 0->1 or 0->-1 = 1 entry
        entries = entries[entries > 0]
        cost_by_day = entries.groupby(idx_date(entries.index)).sum() * 2 * COST
        cost_by_day.index = pd.to_datetime(cost_by_day.index)
        daily = daily.sub(cost_by_day, fill_value=0.0)

    return daily


def idx_date(idx):
    return pd.Index([d.date() for d in idx])


def stats(daily):
    if len(daily) < 20 or daily.std() == 0:
        return dict(days=len(daily), ann=0.0, sh=0.0, mdd=0.0, wr=0.0)
    yrs = len(daily) / 250
    ann = (1 + daily).prod() ** (1 / yrs) - 1
    sh = daily.mean() / daily.std() * np.sqrt(250)
    nav = (1 + daily).cumprod()
    mdd = float((nav / nav.cummax() - 1).min())
    return dict(days=len(daily), ann=ann, sh=sh, mdd=mdd,
                wr=float((daily > 0).mean()))


def fmt(s):
    return (f'ann={s["ann"]:>+7.1%} sh={s["sh"]:>5.2f} '
            f'dd={s["mdd"]:>7.1%} wr={s["wr"]:.0%} days={s["days"]}')


def split_two_periods(daily):
    mid = daily.index[len(daily) // 2]
    return daily[daily.index < mid], daily[daily.index >= mid]


def main():
    from scripts.run_z9_batch_download import load_verified_product
    data = load_verified_product(DEST / 'min5_2yr_6com.pkl')
    print('=' * 78)
    print('Z9 分钟级CTA收尾检验：6品种 × 21个月（2025-01 ~ 2026-09）')
    print('参数=原型版：8h波动窗/20%分位压缩/8h MA方向/最长4h/日内平仓')
    print('=' * 78)

    per_sym = {}
    for sym, name in SYMS.items():
        if sym not in data:
            print(f'{name}: no data, skipped')
            continue
        decided = decide_positions(data[sym])
        d_lag0 = daily_returns(data[sym], decided, lag=0)
        d_lag1 = daily_returns(data[sym], decided, lag=1)
        d_gross = daily_returns(data[sym], decided, lag=1, charge_cost=False)
        per_sym[name] = d_lag1

        p1, p2 = split_two_periods(d_lag1)
        util = (decided != 0).mean()

        print(f'\n--- {name} ---  bar占用率={util:.0%}')
        print(f'  同bar归因(偏差版,lag=0)  {fmt(stats(d_lag0))}')
        print(f'  次bar归因(可实现,lag=1)  {fmt(stats(d_lag1))}')
        print(f'  次bar毛收益(未扣费)      {fmt(stats(d_gross))}')
        print(f'  两期: 前半 {stats(p1)["ann"]:>+6.1%} | '
              f'后半 {stats(p2)["ann"]:>+6.1%}')

    # Equal-weight 6-commodity portfolio (implementable version)
    print('\n' + '=' * 78)
    print('等权6品种组合（可实现版 lag=1）')
    print('=' * 78)
    port = pd.DataFrame(per_sym).mean(axis=1)
    print(f'  全样本          {fmt(stats(port))}')
    p1, p2 = split_two_periods(port)
    print(f'  两期: 前半 {stats(p1)["ann"]:>+6.1%} (sh={stats(p1)["sh"]:.2f}) | '
          f'后半 {stats(p2)["ann"]:>+6.1%} (sh={stats(p2)["sh"]:.2f})')
    monthly = port.resample('ME').sum()
    pos_months = (monthly > 0).sum()
    print(f'  月度: {len(monthly)}个月 正{pos_months} 负{len(monthly)-pos_months} '
          f'最差月 {monthly.min():+.2%}')

    # Correlation with daily Z8 net returns (independence check)
    z8_file = Path('ai-created/research/z8_full_stop/data.pkl')
    if z8_file.exists():
        z8_data = pickle.load(open(z8_file, 'rb'))
        corrs = []
        for code in z8_data:
            z8_close = z8_data[code]['收盘价'].astype(float)
            z8_ret = z8_close.pct_change()
            z8_vol = z8_ret.rolling(20).std()
            z8_pct = z8_vol.rolling(250).rank(pct=True)
            z8_ma = z8_close.rolling(20).mean()
            z8_sig = (z8_pct < 0.20) * np.sign(z8_close - z8_ma)
            z8_daily = (z8_sig.shift(1) * z8_ret).dropna()
            common = pd.DataFrame({'m': port, 'd': z8_daily}).dropna()
            if len(common) > 30:
                corrs.append((code, common['m'].corr(common['d'])))
        if corrs:
            print('\n与日频Z8各品种收益相关性(分钟组合):')
            for code, c in corrs:
                print(f'  {code:<8} corr={c:+.3f}')

    print(f'\nEvidence: {DEST}')


if __name__ == '__main__':
    main()
