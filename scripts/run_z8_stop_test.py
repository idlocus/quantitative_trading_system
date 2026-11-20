#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Test Plan A (breakeven stop) and Plan B (trailing stop) on Z8 trades."""


def main():
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

    from pathlib import Path
    import numpy as np
    import pandas as pd
    import pickle

    DEST = Path('ai-created/research/z8_trades_report')
    data = pickle.load(open(DEST / 'raw_data.pkl', 'rb'))

    SYMS = {'AU0': '黄金', 'CU0': '沪铜', 'RB0': '螺纹钢', 'I0': '铁矿石', 'P0': '棕榈油', 'HC0': '热卷'}
    COST = 0.00025  # one-way


    def simulate_stops(sym, trigger_pct=0.02, trail_pct=0.05):
        """Simulate trades with Plan A (breakeven) and Plan B (trailing).

    trigger_pct: profit needed to activate breakeven stop (Plan A)
    trail_pct: trailing distance from best price (Plan B)
    Returns DataFrame of trades with original/A/B returns.
    """
        df = data[sym]
        close = df['收盘价'].astype(float)
        high = df['最高价'].astype(float)
        low = df['最低价'].astype(float)
        ret = close.pct_change()
        v20 = ret.rolling(20).std()
        v20_pct = v20.rolling(250).rank(pct=True)
        ma20 = close.rolling(20).mean()
        signal = (v20_pct < 0.20) * np.sign(close - ma20)

        trades = []
        pos = 0
        entry_px = entry_date = None
        best_px = None
        be_active = False  # breakeven activated?

        for i in range(len(close)):
            sig = signal.iloc[i]
            d = close.index[i]
            c = close.iloc[i]
            h = high.iloc[i] if np.isfinite(high.iloc[i]) else c
            l = low.iloc[i] if np.isfinite(low.iloc[i]) else c

            if pos == 0 and np.isfinite(sig) and sig != 0:
                pos = int(sig)
                entry_px = c
                entry_date = d
                best_px = c
                be_active = False
                exit_a = None
                exit_a_px = None
                exit_b = None
                exit_b_px = None

            elif pos != 0:
                # Track best price
                if pos == 1:  # long
                    best_px = max(best_px, h)
                else:  # short
                    best_px = min(best_px, l)

                # Plan A: breakeven stop
                if not be_active:
                    if pos == 1 and l <= entry_px * (1 - trigger_pct):
                        be_active = True
                    elif pos == -1 and h >= entry_px * (1 + trigger_pct):
                        be_active = True

                if be_active:
                    if pos == 1 and l <= entry_px:
                        exit_a = d
                        exit_a_px = entry_px  # breakeven
                    elif pos == -1 and h >= entry_px:
                        exit_a = d
                        exit_a_px = entry_px

                # Plan B: trailing stop
                if pos == 1:
                    trail_stop = best_px * (1 - trail_pct)
                else:
                    trail_stop = best_px * (1 + trail_pct)

                if pos == 1 and l <= trail_stop:
                    exit_b = d
                    exit_b_px = trail_stop
                elif pos == -1 and h >= trail_stop:
                    exit_b = d
                    exit_b_px = trail_stop

                # Signal exit (original)
                if not np.isfinite(sig) or sig == 0:
                    # Signal gone, close everything
                    orig_ret = ((c / entry_px - 1) * pos) - 2 * COST

                    # Plan A result
                    if exit_a is not None:
                        a_ret = ((exit_a_px / entry_px - 1) * pos) - 2 * COST
                    else:
                        a_ret = orig_ret

                    # Plan B result
                    if exit_b is not None:
                        b_ret = ((exit_b_px / entry_px - 1) * pos) - 2 * COST
                    else:
                        b_ret = orig_ret

                    trades.append({
                        'symbol': sym, 'name': SYMS[sym],
                        'entry_date': str(entry_date.date()),
                        'direction': 'L' if pos == 1 else 'S',
                        'hold_days': int((d - entry_date).days),
                        'orig': round(float(orig_ret), 4),
                        'planA': round(float(a_ret), 4),
                        'planB': round(float(b_ret), 4),
                        'a_exit': str(exit_a.date()) if exit_a else 'signal',
                        'b_exit': str(exit_b.date()) if exit_b else 'signal',
                    })
                    pos = 0
                    continue

                # If A/B triggered but signal still active, we need to handle re-entry
                # Simplification: if A or B triggered, close that leg (track separately)
                # For simplicity, we assume no re-entry after stop-out

        return pd.DataFrame(trades)


    # Test different parameters
    print('=' * 80)
    print('Z8 止损方案测试（基于日线 OHLC 模拟盘中触发）')
    print('=' * 80)

    for trigger, trail in [(0.02, 0.05), (0.03, 0.07), (0.01, 0.03)]:
        all_results = []
        for sym in SYMS:
            if sym not in data:
                continue
            tdf = simulate_stops(sym, trigger, trail)
            if len(tdf) > 0:
                all_results.append(tdf)

        if not all_results:
            continue
        rdf = pd.concat(all_results, ignore_index=True)

        orig_total = rdf['orig'].sum()
        a_total = rdf['planA'].sum()
        b_total = rdf['planB'].sum()
        orig_wr = (rdf['orig'] > 0).mean()
        a_wr = (rdf['planA'] > 0).mean()
        b_wr = (rdf['planB'] > 0).mean()

        # Count how many trades A/B actually changed
        a_changed = (rdf['planA'] != rdf['orig']).sum()
        b_changed = (rdf['planB'] != rdf['orig']).sum()

        print(f'\n--- 触发{trigger:.0%} / 追踪{trail:.0%} ---')
        print(f'  交易数: {len(rdf)}')
        print(f'  原始: 总收益{orig_total:+.1%} | 胜率{orig_wr:.0%}')
        print(f'  方案A(保本): 总收益{a_total:+.1%} | 胜率{a_wr:.0%} | 改变{a_changed}笔')
        print(f'  方案B(追踪): 总收益{b_total:+.1%} | 胜率{b_wr:.0%} | 改变{b_changed}笔')
        print(f'  A vs 原始: {"改善" if a_total > orig_total else "变差"} {abs(a_total - orig_total):.1%}')
        print(f'  B vs 原始: {"改善" if b_total > orig_total else "变差"} {abs(b_total - orig_total):.1%}')

        # Show biggest improvements and regressions for the best parameter set
        if trigger == 0.02 and trail == 0.05:
            rdf['a_diff'] = rdf['planA'] - rdf['orig']
            rdf['b_diff'] = rdf['planB'] - rdf['orig']
            best_a = rdf.nlargest(5, 'a_diff')
            worst_a = rdf.nsmallest(5, 'a_diff')
            print('\n  方案A最大改善:')
            for _, t in best_a.iterrows():
                if t['a_diff'] > 0.001:
                    print(f'    {t["name"]} {t["entry_date"]} {t["direction"]}: {t["orig"]:+.1%} -> {t["planA"]:+.1%} (改善{t["a_diff"]:+.1%})')
            print('  方案A最大损害:')
            for _, t in worst_a.iterrows():
                if t['a_diff'] < -0.001:
                    print(f'    {t["name"]} {t["entry_date"]} {t["direction"]}: {t["orig"]:+.1%} -> {t["planA"]:+.1%} (损害{t["a_diff"]:+.1%})')

    print('\n' + '=' * 80)
    print('注: 假设止损单在触发价成交(实际可能有滑点)')
    print('    方案A = 浮盈>触发%后止损移至入场价(保本)')
    print('    方案B = 止损始终追踪最佳价的固定回撤%')


if __name__ == '__main__':
    main()
