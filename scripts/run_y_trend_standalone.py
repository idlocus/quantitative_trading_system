#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Y 系：独立趋势策略——市场启动日事件驱动（预注册见 Y 系节）。"""


def main():
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

    from pathlib import Path
    import pandas as pd

    DEST = Path('ai-created/research/y_trend_standalone_2026-11-16')
    DEST.mkdir(parents=True, exist_ok=True)

    # 加载日线
    oracle = Path('data_cache/preparation/oracle')
    frames = []
    for year in range(2022, 2027):
        for month in range(1, 13):
            f = oracle / 'normalized_a' / f'{year}/{month:02d}.parquet'
            if not f.exists():
                continue
            frames.append(pd.read_parquet(f, columns=['symbol', 'date', 'adjclose', 'open', 'close', 'preclose', 'amount']))
    daily = pd.concat(frames, ignore_index=True)
    daily['date'] = pd.to_datetime(daily['date'])
    daily = daily.sort_values(['symbol', 'date']).reset_index(drop=True)
    daily['ret'] = daily['adjclose'] / daily.groupby('symbol')['adjclose'].shift(1) - 1
    print(f'日线: {len(daily)} 行', flush=True)

    # 市场启动日
    mkt = daily.groupby('date')['ret'].mean().reset_index()
    mkt.columns = ['date', 'mkt_ret']
    mkt['mkt_ret_prev'] = mkt['mkt_ret'].shift(1)
    mkt['is_start'] = (mkt['mkt_ret'] >= 0.02) & (mkt['mkt_ret_prev'] < 0.01)
    start_dates = mkt[mkt['is_start']]['date'].tolist()
    all_dates = sorted(daily['date'].unique())
    print(f'启动日: {len(start_dates)} | 期间: {all_dates[0]} ~ {all_dates[-1]}', flush=True)

    # 逐事件回测
    COST = 0.0025
    results = []
    nav_curve = []
    nav = 1_000_000
    prev_exit = None

    for start_d in start_dates:
        pos = all_dates.index(start_d) if start_d in all_dates else None
        if pos is None or pos + 1 >= len(all_dates):
            continue
        # T 日选股：涨幅≥3%、非涨停、流动性
        day_data = daily[daily['date'] == start_d].copy()
        day_data['ret'] = day_data['adjclose'] / day_data['preclose'] - 1 if 'preclose' in day_data.columns else day_data['ret']
        candidates = day_data[(day_data['ret'] >= 0.03) & (day_data['ret'] < 0.095)]
        if len(candidates) < 10:
            continue
        selected = candidates.nlargest(50, 'ret')
        syms = set(selected['symbol'])

        for hold_days in (3, 5):
            entry_pos = pos + 1
            exit_pos = min(entry_pos + hold_days, len(all_dates) - 1)
            if entry_pos >= len(all_dates):
                continue
            entry_d = all_dates[entry_pos]
            exit_d = all_dates[exit_pos]
            # 跳过与上次持仓重叠（简化：独立测每次事件）
            entry_px = daily[(daily['date'] == entry_d) & (daily['symbol'].isin(syms))][['symbol', 'open']]
            exit_px = daily[(daily['date'] == exit_d) & (daily['symbol'].isin(syms))][['symbol', 'open']]
            if entry_px.empty or exit_px.empty:
                continue
            merged = entry_px.merge(exit_px, on='symbol', suffixes=('_in', '_out'))
            merged['stock_ret'] = merged['open_out'] / merged['open_in'] - 1
            gross = merged['stock_ret'].mean()
            net = gross - 2 * COST
            results.append({'start_date': start_d, 'entry': entry_d, 'exit': exit_d,
                            'hold': hold_days, 'n_stocks': len(merged),
                            'gross': gross, 'net': net})

    df = pd.DataFrame(results)
    df.to_csv(DEST / 'events.csv', index=False, encoding='utf-8-sig')

    print('\n== Y 系：独立趋势策略（事件驱动，2022-2026）==')
    for h in (3, 5):
        sub = df[df['hold'] == h]
        if len(sub) < 5:
            print(f'持有{h}日: 事件不足({len(sub)})')
            continue
        # 年化：事件次数 × 平均净收益（按全资金计，闲置期=0）
        years = (all_dates[-1] - all_dates[0]).days / 365.25
        events_per_year = len(sub) / years
        ann_return = events_per_year * sub['net'].mean()
        # 逐事件累计净值
        cum_nav = (1 + sub['net']).cumprod()
        max_dd = float((cum_nav / cum_nav.cummax() - 1).min())
        win_rate = (sub['net'] > 0).mean()
        print(f'\n--- 持有{h}日 ---')
        print(f'  事件数: {len(sub)} ({events_per_year:.1f} 次/年)')
        print(f'  每次毛收益: {sub["gross"].mean():+.2%} | 净(扣50bp): {sub["net"].mean():+.2%}')
        print(f'  胜率: {win_rate:.0%}')
        print(f'  年化净收益(全资金): {ann_return:+.1%}')
        print(f'  累计净值: {cum_nav.iloc[-1]:.3f} | 最大回撤: {max_dd:.1%}')
        print('  对照 G1 同期年化: ~+7%')

    from research.factor_screen2 import register
    register('Y系独立趋势策略: 见y_trend_standalone证据', label='y_trend_standalone')
    print('\n证据:', DEST)


if __name__ == '__main__':
    main()
