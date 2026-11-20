#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""V2 系：申万行业级传导 + 市场启动日指标（预注册见 V2 系节）。"""


def main():
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

    from pathlib import Path
    import json
    import pandas as pd
    from scipy import stats as sps

    DEST = Path('ai-created/research/v2_sw_industry_2026-11-13')
    DEST.mkdir(parents=True, exist_ok=True)

    # ── 1) 拉申万一级行业成分（PIT）──
    import oracledb
    from data.oracle_preparation import connection_settings, connect_readonly
    ind_info = json.load(open('data_cache/sw_industry/industries.json', encoding='utf-8'))
    sw_codes = {f'{c}.SI': c for c in ind_info['codes']}  # 801010 -> 801010.SI

    settings = connection_settings(Path('.env'), os.environ)
    with connect_readonly(settings, oracledb) as reader:
        rows = reader.rows("""SELECT S_INFO_WINDCODE, S_CON_WINDCODE,
                          S_CON_INDATE, S_CON_OUTDATE
                          FROM SWINDEXMEMBERS
                          WHERE S_INFO_WINDCODE IN (
                          '""" + "','".join(sw_codes.keys()) + """')""")
    members = pd.DataFrame(rows, columns=['ind_code', 'symbol', 'in_date', 'out_date'])
    print(f'行业成分: {len(members)} 条 | {members.ind_code.nunique()} 个行业', flush=True)
    members['in_date'] = pd.to_datetime(members['in_date'], format='%Y%m%d', errors='coerce')
    members['out_date'] = pd.to_datetime(members['out_date'], format='%Y%m%d', errors='coerce')

    # ── 2) 加载日线 ──
    oracle = Path('data_cache/preparation/oracle')
    frames = []
    for year in range(2022, 2027):
        for month in range(1, 13):
            f = oracle / 'normalized_a' / f'{year}/{month:02d}.parquet'
            if not f.exists():
                continue
            df = pd.read_parquet(f, columns=['symbol', 'date', 'adjclose'])
            frames.append(df)
    daily = pd.concat(frames, ignore_index=True)
    daily['date'] = pd.to_datetime(daily['date'])
    daily = daily.sort_values(['symbol', 'date'])
    daily['adj_prev'] = daily.groupby('symbol')['adjclose'].shift(1)
    daily['ret'] = daily['adjclose'] / daily['adj_prev'] - 1
    for h in (1, 3, 5):
        daily[f'fwd_{h}d'] = daily.groupby('symbol')['adjclose'].shift(-h) / daily['adjclose'] - 1
    print(f'日线: {len(daily)} 行', flush=True)

    # ── 3) PIT 行业映射：对每个 (symbol, date) 找当日的行业 ──
    # 高效方式：把 members 展开为 (symbol, start, end, industry)，与 daily merge
    members_clean = members.dropna(subset=['in_date']).copy()
    members_clean['end'] = members_clean['out_date'].fillna(pd.Timestamp('2026-12-31'))
    # 对每只股票构建行业区间列表
    daily = daily.merge(
        members_clean[['symbol', 'ind_code', 'in_date', 'end']].rename(
            columns={'in_date': 'ind_start'}),
        on='symbol', how='left')
    daily = daily[(daily['date'] >= daily['ind_start']) & (daily['date'] < daily['end'])]
    daily['industry'] = daily['ind_code'].map(sw_codes)
    daily['ind_name'] = daily['industry'].map(ind_info['names'])
    print(f'行业映射后: {len(daily)} 行 | {daily.industry.nunique()} 个行业', flush=True)

    # ── 4) 行业日等权涨幅 ──
    ind_daily = daily.groupby(['date', 'industry'])['ret'].mean().reset_index()
    ind_daily.columns = ['date', 'industry', 'ind_ret']
    ind_daily = ind_daily.sort_values(['industry', 'date'])
    ind_daily['ind_ret_prev'] = ind_daily.groupby('industry')['ind_ret'].shift(1)
    ind_daily['ind_start_day'] = (ind_daily['ind_ret'] >= 0.02) & (ind_daily['ind_ret_prev'] < 0.01)
    n_starts = ind_daily['ind_start_day'].sum()
    print(f'行业启动日: {n_starts} 天次', flush=True)

    # 匹配回个股
    daily = daily.merge(ind_daily[['date', 'industry', 'ind_ret', 'ind_start_day']],
                        on=['date', 'industry'], how='left')

    # 全市场启动日指标
    mkt = daily.groupby('date')['ret'].mean().rename('mkt_ret').reset_index()
    mkt['mkt_ret_prev'] = mkt['mkt_ret'].shift(1)
    mkt['market_start'] = (mkt['mkt_ret'] >= 0.02) & (mkt['mkt_ret_prev'] < 0.01)
    daily = daily.merge(mkt[['date', 'mkt_ret', 'market_start']], on='date', how='left')

    # 保存市场启动日指标
    mkt_out = mkt[['date', 'mkt_ret', 'market_start']].copy()
    mkt_out['date'] = mkt_out['date'].dt.strftime('%Y-%m-%d')
    mkt_out.to_csv('runtime/market_start_indicator.csv', index=False, encoding='utf-8-sig')
    print(f'市场启动日指标已保存 runtime/market_start_indicator.csv ({int(mkt_out.market_start.sum())} 个启动日)', flush=True)

    # ── 5) 检验 ──
    v1_ind = daily[(daily['ind_start_day'] == True) & (daily['ret'] < 0.01)]
    v2_ind = daily[(daily['ind_start_day'] == True) & (daily['ret'] >= 0.03)]
    v3_norm = daily[(daily['ind_start_day'] == False) & (daily['ret'] < 0.01)]
    # 市场启动版（对照）
    v1_mkt = daily[(daily['market_start'] == True) & (daily['ret'] < 0.01)]
    v2_mkt = daily[(daily['market_start'] == True) & (daily['ret'] >= 0.03)]

    print(f'\n== V2 系：申万行业级传导（{n_starts} 个行业启动日）==')
    for h in (1, 3, 5):
        col = f'fwd_{h}d'
        r1 = v1_ind[col].dropna()
        r2 = v2_ind[col].dropna()
        r3 = v3_norm[col].dropna()
        r1m = v1_mkt[col].dropna()
        r2m = v2_mkt[col].dropna()
        if min(len(r1), len(r3)) < 100:
            continue
        t13 = sps.ttest_ind(r1, r3, equal_var=False)
        t12 = sps.ttest_ind(r1, r2, equal_var=False)
        print(f'\n--- 未来{h}日 ---')
        print(f'  行业启动滞涨: 均 {r1.mean()*100:+.2f}% (n={len(r1)})')
        print(f'  行业启动已涨: 均 {r2.mean()*100:+.2f}% (n={len(r2)})')
        print(f'  普通日滞涨:  均 {r3.mean()*100:+.2f}% (n={len(r3)})')
        print(f'  传导(vs普通): {(r1.mean()-r3.mean())*100:+.3f}% (t={t13.statistic:+.2f})')
        print(f'  滞涨vs龙头:  {(r1.mean()-r2.mean())*100:+.3f}% (t={t12.statistic:+.2f})')
        if len(r1m) > 100:
            print(f'  [对照]市场启动滞涨: {r1m.mean()*100:+.2f}% | 市场启动已涨: {r2m.mean()*100:+.2f}%')

    from research.factor_screen2 import register
    register('V2申万行业级传导3检验: 见v2_sw_industry证据', label='v2_sw_industry')
    print(f'\n证据: {DEST}')


if __name__ == '__main__':
    main()
