#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Z8 CTA full trade report with K-line charts."""


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

    DEST = Path('ai-created/research/z8_trades_report')
    DEST.mkdir(parents=True, exist_ok=True)

    SYMS = {
        'AU0': 'Gold', 'CU0': 'Copper', 'RB0': 'Rebar',
        'I0': 'IronOre', 'P0': 'Palm', 'HC0': 'HRC',
    }
    COST = 0.00025

    cache = DEST / 'raw_data.pkl'
    if cache.exists():
        data = pickle.load(open(cache, 'rb'))
    else:
        data = {}
        for sym in SYMS:
            try:
                df = ak.futures_main_sina(symbol=sym, start_date='20150101', end_date='20261112')
                df['日期'] = pd.to_datetime(df['日期'])
                data[sym] = df.set_index('日期').sort_index()
                time.sleep(0.3)
            except Exception:
                pass
        pickle.dump(data, open(cache, 'wb'))
    print(f'data: {len(data)} symbols', flush=True)


    def get_signal(sym):
        close = data[sym]['收盘价'].astype(float)
        ret = close.pct_change()
        v20 = ret.rolling(20).std()
        v20_pct = v20.rolling(250).rank(pct=True)
        ma20 = close.rolling(20).mean()
        signal = (v20_pct < 0.20) * np.sign(close - ma20)
        return close, signal, v20_pct


    def extract_trades(sym):
        close, signal, v20_pct = get_signal(sym)
        trades = []
        pos = 0
        entry_px = entry_date = entry_sig = None
        for i in range(len(close)):
            sig = signal.iloc[i]
            d = close.index[i]
            px = close.iloc[i]
            if pos == 0 and sig != 0 and np.isfinite(sig):
                pos = int(sig)
                entry_px = px
                entry_date = d
                entry_sig = float(v20_pct.iloc[i])
            elif pos != 0 and (sig == 0 or not np.isfinite(sig)):
                ret_gross = (px / entry_px - 1) * pos
                ret_net = ret_gross - 2 * COST
                trades.append({
                    'symbol': sym, 'name': SYMS[sym],
                    'entry_date': str(entry_date.date()),
                    'entry_price': round(float(entry_px), 1),
                    'exit_date': str(d.date()),
                    'exit_price': round(float(px), 1),
                    'direction': 'L' if pos == 1 else 'S',
                    'hold_days': int((d - entry_date).days),
                    'net_ret': round(float(ret_net), 4),
                    'entry_vol_pct': round(entry_sig, 2),
                })
                pos = 0
        return pd.DataFrame(trades)


    all_trades = []
    for sym in SYMS:
        trades = extract_trades(sym)
        if len(trades) > 0:
            all_trades.append(trades)
            print(f'  {SYMS[sym]}: {len(trades)} trades', flush=True)

    all_df = pd.concat(all_trades, ignore_index=True).sort_values('entry_date')
    all_df.to_csv(DEST / 'all_trades.csv', index=False, encoding='utf-8-sig')

    # Generate text report
    lines = []
    lines.append('# Z8 CTA Complete Trade Record (2015-2026)')
    lines.append('')
    wr = (all_df['net_ret'] > 0).mean()
    lines.append(f'Total trades: {len(all_df)} | Win rate: {wr:.0%} | Avg per trade: {all_df["net_ret"].mean():+.2%}')
    lines.append('')

    lines.append('## Per-Commodity Summary')
    lines.append('')
    header = '| Commodity | Trades | Win Rate | Avg Return | Best | Worst | Cumulative |'
    lines.append(header)
    lines.append('|---|---:|---:|---:|---:|---:|---:|')
    for sym in SYMS:
        sub = all_df[all_df['symbol'] == sym]
        if len(sub) == 0:
            continue
        total = (1 + sub['net_ret']).prod() - 1
        lines.append('| {} | {} | {:.0%} | {:+.2%} | {:+.1%} | {:+.1%} | {:+.1%} |'.format(
            SYMS[sym], len(sub), (sub['net_ret'] > 0).mean(),
            sub['net_ret'].mean(), sub['net_ret'].max(), sub['net_ret'].min(), total))
    lines.append('')

    lines.append('## All Trades (sorted by entry date)')
    lines.append('')
    lines.append('| # | Commodity | Dir | Entry Date | Entry Px | Exit Date | Exit Px | Days | Net Ret | Vol Pct |')
    lines.append('|---|---|---|---|---:|---|---:|---:|---:|---:|')
    for i, (_, t) in enumerate(all_df.iterrows()):
        mark = '+' if t['net_ret'] > 0 else ''
        lines.append('| {} | {} | {} | {} | {} | {} | {} | {} | {}{:.1%} | {:.0%} |'.format(
            i + 1, t['name'], t['direction'], t['entry_date'], t['entry_price'],
            t['exit_date'], t['exit_price'], t['hold_days'], mark, t['net_ret'],
            t['entry_vol_pct']))
    lines.append('')

    # Yearly stats
    lines.append('## Yearly Performance')
    lines.append('')
    all_df['year'] = pd.to_datetime(all_df['entry_date']).dt.year
    lines.append('| Year | Trades | Win Rate | Avg | Total (sum) |')
    lines.append('|---|---:|---:|---:|---:|')
    for yr, sub in all_df.groupby('year'):
        lines.append('| {} | {} | {:.0%} | {:+.2%} | {:+.1%} |'.format(
            yr, len(sub), (sub['net_ret'] > 0).mean(), sub['net_ret'].mean(), sub['net_ret'].sum()))
    lines.append('')

    report = '\n'.join(lines)
    (DEST / 'trade_report.md').write_text(report, encoding='utf-8')
    print(f'\nreport saved: {DEST}/trade_report.md')

    # Generate charts
    print('\ngenerating charts...')
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates

        for sym in SYMS:
            close, signal, v20_pct = get_signal(sym)
            trades = extract_trades(sym)
            ma20 = close.rolling(20).mean()

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10),
                                           sharex=True, gridspec_kw={'hspace': 0.05})
            name = SYMS[sym]
            fig.suptitle('Z8 CTA Signals - {}'.format(name), fontsize=14, fontweight='bold')

            x = close.index
            ax1.plot(x, close.values, color='gray', lw=0.5, alpha=0.5)
            ax1.plot(x, ma20.values, color='blue', lw=1, alpha=0.7, label='MA20')

            for _, t in trades.iterrows():
                ed = pd.Timestamp(t['entry_date'])
                xd = pd.Timestamp(t['exit_date'])
                ep = t['entry_price']
                xp = t['exit_price']
                if t['direction'] == 'L':
                    ax1.scatter(ed, ep, marker='^', color='red', s=100, zorder=5)
                    ax1.scatter(xd, xp, marker='v', color='darkred', s=80, zorder=5)
                    ax1.plot([ed, xd], [ep, xp], 'r-', lw=1.5, alpha=0.5)
                else:
                    ax1.scatter(ed, ep, marker='v', color='green', s=100, zorder=5)
                    ax1.scatter(xd, xp, marker='^', color='darkgreen', s=80, zorder=5)
                    ax1.plot([ed, xd], [ep, xp], 'g-', lw=1.5, alpha=0.5)
                # Annotate return
                offset = 10 if t['net_ret'] > 0 else -15
                color = 'red' if t['net_ret'] > 0 else 'green'
                ax1.annotate('{:+.0%}'.format(t['net_ret']), (xd, xp),
                            textcoords='offset points', xytext=(5, offset), fontsize=7, color=color)

            ax1.set_ylabel('Price')
            ax1.legend(['Price', 'MA20'], loc='upper left', fontsize=8)
            ax1.grid(True, alpha=0.3)

            # Vol percentile subplot
            ax2.fill_between(x, v20_pct.values, 0, alpha=0.3, color='orange')
            ax2.axhline(y=0.20, color='red', linestyle='--', lw=1, label='Trigger 20%')
            ax2.set_ylabel('Vol Percentile')
            ax2.set_xlabel('Date')
            ax2.legend(loc='upper left', fontsize=8)
            ax2.grid(True, alpha=0.3)
            ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
            plt.xticks(rotation=45)
            plt.tight_layout()
            fig.savefig(str(DEST / '{}_chart.png'.format(sym)), dpi=150, bbox_inches='tight')
            plt.close()
            print('  {}: chart saved'.format(name))

    except ImportError as e:
        print('matplotlib unavailable: {}'.format(e))

    print('\nall files in: {}/'.format(DEST))
    print('  trade_report.md - full trade record')
    print('  all_trades.csv - trade data')
    print('  *_chart.png - K-line charts with signals')


if __name__ == '__main__':
    main()
