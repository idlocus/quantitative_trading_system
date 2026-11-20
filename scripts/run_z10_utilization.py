#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Z10 capital utilization profile: how much of the sleeve is actually
deployed, when, and at what margin usage.

Descriptive statistics of the preregistered variants (no new strategy test):
- signal occupancy (days with a book)
- vol-target leverage actually applied (mean, % days at 2x cap)
- effective notional exposure = sum|w| * lev
- margin usage = exposure * 11% (exchange ~10-12%)
- idle/deployed streak structure (how long capital sits out)
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

DEST = Path('ai-created/research/z10_concentration')
SYMS = ['AU0', 'CU0', 'RB0', 'I0', 'P0', 'HC0']
MARGIN = 0.11  # average exchange margin across the six
COST = 0.00025
TARGET_VOL = 0.15
VOL_TARGET_WIN = 60
LEV_CAP = 2


def build_inputs(data):
    info = {}
    for sym in SYMS:
        close = data[sym]['收盘价'].astype(float)
        ret = close.pct_change()
        v20_pct = ret.rolling(20).std().rolling(250).rank(pct=True)
        ma20 = close.rolling(20).mean()
        info[sym] = pd.DataFrame({
            'ret': ret, 'depth': v20_pct,
            'comp': (v20_pct < 0.20).fillna(False),
            'dir': np.sign(close - ma20)})
    idx = sorted(set().union(*[data[s]['收盘价'].index for s in SYMS]))
    for sym in SYMS:
        info[sym] = info[sym].reindex(idx)
    return info, pd.DatetimeIndex(idx)


def run(info, idx, k):
    """Replicate Z10 book + track exposure. Returns per-day exposure and book."""
    held = {}
    gross = np.zeros(len(idx))
    expo = np.zeros(len(idx))          # notional exposure before overlay
    n_active = np.zeros(len(idx))
    for i, dt in enumerate(idx):
        for sym, w in held.items():
            r = info[sym]['ret'].iloc[i]
            if np.isfinite(r):
                gross[i] += w * r
        active = []
        for j, sym in enumerate(SYMS):
            f = info[sym]
            if bool(f['comp'].iloc[i]) and np.isfinite(f['depth'].iloc[i]):
                d = f['dir'].iloc[i]
                if np.isfinite(d) and d != 0:
                    active.append((f['depth'].iloc[i], j, sym, d))
        n_active[i] = len(active)
        desired = {}
        if active:
            if k == 0:
                for dep, j, sym, d in active:
                    desired[sym] = d / len(SYMS)
            else:
                active.sort()
                kk = min(k, len(active))
                for dep, j, sym, d in active[:kk]:
                    desired[sym] = d / kk
        for sym in list(held):
            if sym not in desired:
                del held[sym]
        held.update(desired)
        expo[i] = sum(abs(w) for w in held.values())

    g = pd.Series(gross, index=idx)
    vol = g.rolling(VOL_TARGET_WIN).std() * np.sqrt(252)
    lev = (TARGET_VOL / vol).clip(upper=LEV_CAP).shift(1).fillna(0)
    eff = pd.Series(expo, index=idx) * lev          # effective notional
    return eff, pd.Series(n_active, index=idx)


def streaks(deployed):
    """Lengths of consecutive True/False runs."""
    runs = []
    cur, n = None, 0
    for v in deployed:
        if v == cur:
            n += 1
        else:
            if cur is not None:
                runs.append((cur, n))
            cur, n = v, 1
    runs.append((cur, n))
    idle = [n for v, n in runs if not v]
    on = [n for v, n in runs if v]
    return idle, on


def main():
    data = pickle.load(open('ai-created/research/z8_full_stop/data.pkl', 'rb'))
    info, idx = build_inputs(data)

    print('=' * 72)
    print('Z10 资金利用率画像（描述统计，2025口径同回测期 2015-04~2026-11）')
    print(f'保证金假设：名义 × {MARGIN:.0%}（交易所水平10-12%取中值）')
    print('=' * 72)

    for label, k in [('BASE=Z8固定1/6槽', 0), ('TOP1=全给最强', 1),
                     ('TOP2=最强两个', 2)]:
        eff, n_active = run(info, idx, k)
        deployed = eff > 1e-9
        eff_on = eff[deployed]
        idle, on = streaks(deployed.values)
        # leverage-at-cap share: see precise TOP1 recompute at file end
        print(f'\n--- {label} ---')
        print(f'  有名义仓位天数: {deployed.mean():.1%}')
        print(f'  有效名义暴露(含波动率目标杠杆): 均值 {eff.mean():.2f} | '
              f'持仓日均值 {eff_on.mean():.2f} | 中位 {eff_on.median():.2f} | '
              f'P90 {eff_on.quantile(0.9):.2f}')
        print(f'  保证金占用(占 sleeve 资金): 均值 {eff.mean()*MARGIN:.1%} | '
              f'持仓日均值 {eff_on.mean()*MARGIN:.1%} | '
              f'峰值 P90 {eff_on.quantile(0.9)*MARGIN:.1%}')
        if idle:
            print(f'  空仓连续期: 平均 {np.mean(idle):.0f}天 | '
                  f'最长 {max(idle)}天 | 次数 {len(idle)}')
        if on:
            print(f'  持仓连续期: 平均 {np.mean(on):.0f}天 | 最长 {max(on)}天')

    # TOP1: how often the 2x cap binds (lev saturation)
    print('\n--- TOP1 杠杆饱和度 ---')
    # recompute lev directly for TOP1
    held = {}
    gross = np.zeros(len(idx))
    for i, dt in enumerate(idx):
        for sym, w in held.items():
            r = info[sym]['ret'].iloc[i]
            if np.isfinite(r):
                gross[i] += w * r
        active = []
        for j, sym in enumerate(SYMS):
            f = info[sym]
            if bool(f['comp'].iloc[i]) and np.isfinite(f['depth'].iloc[i]):
                d = f['dir'].iloc[i]
                if np.isfinite(d) and d != 0:
                    active.append((f['depth'].iloc[i], j, sym, d))
        desired = {}
        if active:
            active.sort()
            kk = min(1, len(active))
            for dep, j, sym, d in active[:kk]:
                desired[sym] = d / kk
        held = {s: w for s, w in desired.items()}
    g = pd.Series(gross, index=idx)
    vol = g.rolling(VOL_TARGET_WIN).std() * np.sqrt(252)
    lev = (TARGET_VOL / vol).clip(upper=LEV_CAP).shift(1).fillna(0)
    on_days = lev[lev > 0]
    print(f'  杠杆>0天数 {len(on_days)/len(idx):.1%} | '
          f'其中打满2倍上限天数 {(lev >= LEV_CAP - 1e-9).sum()} '
          f'({(lev >= LEV_CAP - 1e-9).sum()/max(1,(lev>0).sum()):.0%} of 持仓日) | '
          f'平均杠杆 {on_days.mean():.2f}')

    print(f'\nEvidence: {DEST}')


if __name__ == '__main__':
    main()
