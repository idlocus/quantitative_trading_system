#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""V 系：板块启动第1天 → 板块内滞涨股补涨（预注册见 V 系节）。"""


def main():
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

    from pathlib import Path
    import pandas as pd
    from scipy import stats as sps

    DEST = Path('ai-created/research/v_sector_start_2026-11-13')
    DEST.mkdir(parents=True, exist_ok=True)

    # 加载日线
    oracle = Path('data_cache/preparation/oracle')
    rows = []
    for year in range(2022, 2027):
        for month in range(1, 13):
            f = oracle / 'normalized_a' / f'{year}/{month:02d}.parquet'
            if not f.exists():
                continue
            df = pd.read_parquet(f, columns=['symbol', 'date', 'preclose', 'close', 'adjclose'])
            rows.append(df)
    daily = pd.concat(rows, ignore_index=True)
    daily['date'] = pd.to_datetime(daily['date'])

    # 用复权价算收益
    daily = daily.sort_values(['symbol', 'date'])
    daily['adj_prev'] = daily.groupby('symbol')['adjclose'].shift(1)
    daily['ret'] = daily['adjclose'] / daily['adj_prev'] - 1
    # 未来收益
    for h in (1, 3, 5):
        daily[f'fwd_{h}d'] = daily.groupby('symbol')['adjclose'].shift(-h) / daily['adjclose'] - 1

    # 板块（简化用代码前缀）
    def sector(sym):
        code = sym.split('.')[0]
        if code.startswith('688'):
            return 'STAR'
        elif code.startswith('300'):
            return 'ChiNext'
        elif code.startswith('60'):
            return 'SH_main'
        elif code.startswith('00'):
            return 'SZ_main'
        return 'Other'

    daily['sector'] = daily['symbol'].map(sector)
    print(f'日线 {len(daily)} 行 | 板块: {daily.sector.unique().tolist()}', flush=True)

    # 板块日等权涨幅
    sec_daily = daily.groupby(['date', 'sector'])['ret'].mean().reset_index()
    sec_daily.columns = ['date', 'sector', 'sector_ret']
    sec_daily = sec_daily.sort_values(['sector', 'date'])
    sec_daily['sector_ret_prev'] = sec_daily.groupby('sector')['sector_ret'].shift(1)

    # 板块启动 = 今日≥2% 且 昨日<1%
    sec_daily['is_start'] = (sec_daily['sector_ret'] >= 0.02) & (sec_daily['sector_ret_prev'] < 0.01)
    start_days = sec_daily[sec_daily['is_start']]
    print(f'板块启动日: {len(start_days)} 天次', flush=True)

    # 匹配到个股
    daily = daily.merge(sec_daily[['date', 'sector', 'sector_ret', 'sector_ret_prev', 'is_start']],
                        on=['date', 'sector'], how='left')
    # 全市场等权日收益（作为基准）
    mkt_ret = daily.groupby('date')['ret'].mean().rename('mkt_ret')
    daily = daily.merge(mkt_ret, on='date', how='left')

    # ── V1: 启动日 → 板块内滞涨股（<1%）的未来收益 ──
    v1_lag = daily[(daily['is_start'] == True) & (daily['ret'] < 0.01)]
    v1_lead = daily[(daily['is_start'] == True) & (daily['ret'] >= 0.03)]
    v3_lag_normal = daily[(daily['is_start'] == False) & (daily['ret'] < 0.01)]

    print(f'\n== V 系：板块启动第1天（n={len(start_days)} 天次）==')
    print(f'V1 启动日滞涨股(<1%): n={len(v1_lag)}')
    print(f'V2 启动日已涨股(≥3%): n={len(v1_lead)}')
    print(f'V3 普通日滞涨股: n={len(v3_lag_normal)}')

    for h in (1, 3, 5):
        col = f'fwd_{h}d'
        v1_r = v1_lag[col].dropna()
        v2_r = v1_lead[col].dropna()
        v3_r = v3_lag_normal[col].dropna()
        if len(v1_r) < 100 or len(v3_r) < 100:
            continue
        # 减去同期全市场等权收益作超额（近似：用V3作为无条件基准）
        t13 = sps.ttest_ind(v1_r, v3_r, equal_var=False)
        t12 = sps.ttest_ind(v1_r, v2_r, equal_var=False)
        print(f'\n--- 未来{h}日收益 ---')
        print(f'  V1启动滞涨: 均 {v1_r.mean()*100:+.2f}% (n={len(v1_r)})')
        print(f'  V2启动已涨: 均 {v2_r.mean()*100:+.2f}% (n={len(v2_r)})')
        print(f'  V3普通滞涨: 均 {v3_r.mean()*100:+.2f}% (n={len(v3_r)})')
        print(f'  V1 vs V3: 差 {(v1_r.mean()-v3_r.mean())*100:+.3f}% (t={t13.statistic:+.2f}, p={t13.pvalue:.4f})')
        print(f'  V1 vs V2: 差 {(v1_r.mean()-v2_r.mean())*100:+.3f}% (t={t12.statistic:+.2f}, p={t12.pvalue:.4f})')

    from research.factor_screen2 import register
    register('V系板块启动传导3检验: 见v_sector_start证据', label='v_sector_start')
    print(f'\n证据: {DEST}')


if __name__ == '__main__':
    main()
