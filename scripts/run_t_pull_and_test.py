#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""T 系：5 分钟级涨停前买入检验（预注册见 T 系节）。"""


def main():
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

    from pathlib import Path
    import pandas as pd

    DEST = Path('ai-created/research/t_pre_limitup_5m_2026-11-12')
    DEST.mkdir(parents=True, exist_ok=True)

    # ── 第1步：Oracle 日线锁定候选（high ≥ prev_close × 1.08）──
    oracle = Path('data_cache/preparation/oracle')
    candidates = []
    for year in range(2022, 2027):
        for month in range(1, 13):
            try:
                f = oracle / 'normalized_a' / f'{year}/{month:02d}.parquet'
                if not f.exists():
                    continue
                df = pd.read_parquet(f, columns=['symbol', 'date', 'preclose', 'high', 'close', 'open'])
                mask = df['high'] / df['preclose'] >= 1.08
                for _, r in df[mask].iterrows():
                    candidates.append({'symbol': r['symbol'], 'date': str(r['date'])[:10],
                                       'preclose': r['preclose'], 'high': r['high'],
                                       'close': r['close'], 'open': r['open']})
            except Exception:
                continue
    cand = pd.DataFrame(candidates)
    cand.to_csv(DEST / 'candidates.csv', index=False, encoding='utf-8-sig')
    print(f'候选（日线 high≥+8%）: {len(cand)} 条 | {cand.symbol.nunique()} 只 | '
          f'{cand.date.min()} ~ {cand.date.max()}', flush=True)

    # ── 第2步：baostock 拉候选的 5 分钟线 ──
    import baostock as bs
    lg = bs.login()

    # 按股票分组，一次拉该股票的全部日期
    results = []
    for i, (sym, grp) in enumerate(cand.groupby('symbol')):
        bs_code = sym.replace('.SH', '.sh').replace('.SZ', '.sz')
        dates = sorted(grp['date'].unique())
        date_set = set(dates)
        # 拉该股票候选日的 5 分钟线（逐月拉避免超时）
        for d in dates:
            rs = bs.query_history_k_data_plus(
                bs_code, 'date,time,open,high,low,close,volume',
                start_date=d, end_date=d, frequency='5', adjustflag='2')
            bars = []
            while rs.next():
                bars.append(rs.get_row_data())
            if not bars:
                continue
            bars = pd.DataFrame(bars, columns=['date', 'time', 'open', 'high', 'low', 'close', 'volume'])
            for col in ('open', 'high', 'low', 'close', 'volume'):
                bars[col] = pd.to_numeric(bars[col], errors='coerce')
            prev = float(grp[grp['date'] == d]['preclose'].iloc[0])
            bars['cum_ret'] = bars['close'] / prev - 1
            bars['high_ret'] = bars['high'] / prev - 1

            # 找首次触 +8% 的 bar
            hit = bars[bars['high_ret'] >= 0.08]
            if hit.empty:
                continue
            first = hit.iloc[0]
            entry_price = first['close']  # 以该 bar 收盘价近似入场
            final_close = bars['close'].iloc[-1]
            final_ret = final_close / entry_price - 1
            # 是否封板（收盘涨幅≥9.7%）
            daily_ret = final_close / prev - 1
            limit_up = daily_ret >= 0.093
            # 盘中最高涨幅（封后是否打开）
            max_ret = bars['high_ret'].max()

            results.append({'symbol': sym, 'date': d, 'entry_ret': first['cum_ret'],
                            'ret_to_close': final_ret, 'daily_ret': daily_ret,
                            'limit_up': limit_up, 'max_ret': max_ret,
                            'n_bars_after': len(bars) - hit.index[0]})
        if (i + 1) % 100 == 0:
            print(f'  进度: {i+1}/{cand.symbol.nunique()} 只', flush=True)

    bs.logout()
    res = pd.DataFrame(results)
    res.to_csv(DEST / 'pre_limitup_results.csv', index=False, encoding='utf-8-sig')

    print('== T 系：首次触+8%追入到收盘（5分钟级，验证期 2022-2026）==')
    print(f'事件数: {len(res)}')
    print(f'日均追入收益: {res.ret_to_close.mean()*10000:+.1f}bp | 中位 {res.ret_to_close.median()*10000:+.1f}bp')
    print(f'封板率: {res.limit_up.mean():.1%} | 封后打开率: {(res.max_ret >= 0.097).mean():.1%}')
    print(f'收益分布: P25 {res.ret_to_close.quantile(0.25)*100:+.1f}% | P75 {res.ret_to_close.quantile(0.75)*100:+.1f}%')

    from research.factor_screen2 import register
    register(f'T系5分钟涨停前: n={len(res)} 均{res.ret_to_close.mean()*10000:+.1f}bp 封板率{res.limit_up.mean():.0%}', label='t_pull_and_test')
    print('证据:', DEST)


if __name__ == '__main__':
    main()
