#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Y 系验证：样本外 + 参数敏感性 + 保守成本 + 进出优化。"""


def main():
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

    from pathlib import Path
    import pandas as pd

    DEST = Path('ai-created/research/y_validation_2026-11-16')
    DEST.mkdir(parents=True, exist_ok=True)

    # 加载全部日线（2016-2026）
    oracle = Path('data_cache/preparation/oracle')
    frames = []
    for year in range(2016, 2027):
        for month in range(1, 13):
            f = oracle / 'normalized_a' / f'{year}/{month:02d}.parquet'
            if not f.exists():
                continue
            frames.append(pd.read_parquet(f, columns=['symbol', 'date', 'adjclose', 'open', 'close', 'preclose', 'amount']))
    daily = pd.concat(frames, ignore_index=True)
    daily['date'] = pd.to_datetime(daily['date'])
    daily = daily.sort_values(['symbol', 'date']).reset_index(drop=True)
    daily['ret'] = daily['adjclose'] / daily.groupby('symbol')['adjclose'].shift(1) - 1
    all_dates = sorted(daily['date'].unique())
    print(f'日线: {len(daily)} 行 | {all_dates[0]} ~ {all_dates[-1]}', flush=True)

    # 市场日收益
    mkt = daily.groupby('date')['ret'].mean().reset_index()
    mkt.columns = ['date', 'mkt_ret']
    mkt['mkt_ret_prev'] = mkt['mkt_ret'].shift(1)

    def run_events(start_year, end_year, mkt_thresh=0.02, hold=5, top_n=50, cost=0.0025, entry='open'):
        """返回事件列表"""
        sub_mkt = mkt[(mkt['date'] >= f'{start_year}-01-01') & (mkt['date'] <= f'{end_year}-12-31')]
        starts = sub_mkt[(sub_mkt['mkt_ret'] >= mkt_thresh) & (sub_mkt['mkt_ret_prev'] < 0.01)]['date'].tolist()
        events = []
        for sd in starts:
            pos = all_dates.index(sd) if sd in all_dates else None
            if pos is None or pos + 1 >= len(all_dates):
                continue
            exit_pos = min(pos + 1 + hold, len(all_dates) - 1)
            entry_d, exit_d = all_dates[pos + 1], all_dates[exit_pos]
            day_data = daily[daily['date'] == sd].copy()
            cands = day_data[(day_data['ret'] >= 0.03) & (day_data['ret'] < 0.095)]
            if len(cands) < 10:
                continue
            sel = cands.nlargest(top_n, 'ret')
            syms = set(sel['symbol'])
            ep_col = 'open' if entry == 'open' else 'close'
            entry_px = daily[(daily['date'] == entry_d) & (daily['symbol'].isin(syms))][['symbol', ep_col]]
            exit_px = daily[(daily['date'] == exit_d) & (daily['symbol'].isin(syms))][['symbol', 'open']]
            if entry_px.empty or exit_px.empty:
                continue
            entry_px.columns = ['symbol', 'px_in']
            exit_px.columns = ['symbol', 'px_out']
            merged = entry_px.merge(exit_px, on='symbol')
            merged['stock_ret'] = merged['px_out'] / merged['px_in'] - 1
            gross = merged['stock_ret'].mean()
            net = gross - 2 * cost
            events.append({'start': sd, 'n': len(merged), 'gross': gross, 'net': net})
        return pd.DataFrame(events)

    def report(df, label):
        if len(df) < 3:
            print(f'{label}: 事件不足({len(df)})')
            return
        years = (df.start.max() - df.start.min()).days / 365.25
        per_year = len(df) / max(years, 0.5)
        ann = per_year * df['net'].mean()
        cum = (1 + df['net']).prod()
        wr = (df['net'] > 0).mean()
        nav_s = (1 + df['net']).cumprod()
        mdd = float((nav_s / nav_s.cummax() - 1).min())
        print(f'{label}: n={len(df):>3} | 每次{df["net"].mean():+.2%} | 胜率{wr:.0%} | '
              f'年化{ann:+.1%} | 累计{cum:.2f} | maxDD{mdd:.1%}')

    # ── 验证1：样本外 2016-2021 ──
    print('\n===== 验证1：样本外 2016-2021 =====')
    v1 = run_events(2016, 2021)
    report(v1, 'Y2-5日持有 2016-2021')
    v1_3 = run_events(2016, 2021, hold=3)
    report(v1_3, 'Y1-3日持有 2016-2021')

    # ── 验证2：参数敏感性（2022-2026）──
    print('\n===== 验证2：参数敏感性（2022-2026）=====')
    print(f"{'阈值':>6}{'持有':>4}{'选股':>4} | {'次数':>4}{'每次':>8}{'胜率':>5}{'年化':>7}")
    for thr in (0.015, 0.02, 0.025):
        for h in (3, 5, 7):
            for n in (30, 50, 70):
                df = run_events(2022, 2026, mkt_thresh=thr, hold=h, top_n=n)
                if len(df) < 3:
                    continue
                years = (df.start.max() - df.start.min()).days / 365.25
                ann = len(df) / max(years, 0.5) * df['net'].mean()
                wr = (df['net'] > 0).mean()
                print(f"{thr:>6.1%}{h:>4}{n:>4} | {len(df):>4}{df['net'].mean():>+8.2%}{wr:>5.0%}{ann:>+7.1%}")

    # ── 验证3：保守成本 ──
    print('\n===== 验证3：保守成本（50bp单边=100bp双边）=====')
    v3 = run_events(2022, 2026, cost=0.005)
    report(v3, 'Y2 50bp单边')

    # ── 验证4：进入时点 ──
    print('\n===== 验证4：进入时点（T+1收盘 vs T+1开盘）=====')
    v4_open = run_events(2022, 2026, entry='open')
    report(v4_open, 'Y2 T+1开盘入')
    v4_close = run_events(2022, 2026, entry='close')
    report(v4_close, 'Y2 T+1收盘入')

    from research.factor_screen2 import register
    register('Y系验证4项: 见y_validation证据', label='y_validation')
    print(f'\n证据: {DEST}')


if __name__ == '__main__':
    main()
