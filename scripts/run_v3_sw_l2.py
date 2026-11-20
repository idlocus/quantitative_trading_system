#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""V3 系：申万二级行业级传导（预注册见 V3 系节）。"""


def main():
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

    from pathlib import Path
    import re
    import pandas as pd
    from scipy import stats as sps

    DEST = Path('ai-created/research/v3_sw_l2_2026-11-13')
    DEST.mkdir(parents=True, exist_ok=True)

    import oracledb
    from data.oracle_preparation import connection_settings, connect_readonly

    settings = connection_settings(Path('.env'), os.environ)
    with connect_readonly(settings, oracledb) as reader:
        # 1) 识别二级行业（801 开头，非一级，名称不含指数/风格等）
        desc = reader.rows("""SELECT S_INFO_WINDCODE, S_INFO_COMPNAME
                          FROM AINDEXDESCRIPTION
                          WHERE S_INFO_WINDCODE LIKE '801%'""")
        names = {r[0]: r[1] for r in desc}
        level1 = {'801010.SI','801030.SI','801040.SI','801050.SI','801080.SI','801880.SI',
                  '801110.SI','801120.SI','801130.SI','801140.SI','801150.SI','801160.SI',
                  '801170.SI','801180.SI','801200.SI','801210.SI','801230.SI','801710.SI',
                  '801720.SI','801730.SI','801740.SI','801750.SI','801760.SI','801770.SI',
                  '801780.SI','801790.SI','801890.SI','801950.SI','801960.SI','801970.SI','801980.SI'}
        exclude = re.compile(r'指数|风格|主题|策略|规模|价值|成长|红利|等权')
        l2_codes = sorted([c for c, n in names.items()
                           if c not in level1 and not exclude.search(n)])
        print(f'二级行业: {len(l2_codes)} 个', flush=True)

        # 2) 拉成分
        ph = ','.join(f"'{c}'" for c in l2_codes)
        rows = reader.rows(f"""SELECT S_INFO_WINDCODE, S_CON_WINDCODE,
                               S_CON_INDATE, S_CON_OUTDATE
                               FROM SWINDEXMEMBERS WHERE S_INFO_WINDCODE IN ({ph})""")
        members = pd.DataFrame(rows, columns=['ind_code', 'symbol', 'in_date', 'out_date'])
        members['in_date'] = pd.to_datetime(members['in_date'], format='%Y%m%d', errors='coerce')
        members['out_date'] = pd.to_datetime(members['out_date'], format='%Y%m%d', errors='coerce')
        members['end'] = members['out_date'].fillna(pd.Timestamp('2026-12-31'))
        members['ind_name'] = members['ind_code'].map(names)
        print(f'成分: {len(members)} 条 | {members.ind_code.nunique()} 个行业', flush=True)

    # 3) 加载日线
    oracle = Path('data_cache/preparation/oracle')
    frames = []
    for year in range(2022, 2027):
        for month in range(1, 13):
            f = oracle / 'normalized_a' / f'{year}/{month:02d}.parquet'
            if not f.exists():
                continue
            frames.append(pd.read_parquet(f, columns=['symbol', 'date', 'adjclose']))
    daily = pd.concat(frames, ignore_index=True)
    daily['date'] = pd.to_datetime(daily['date'])
    daily = daily.sort_values(['symbol', 'date'])
    daily['ret'] = daily.groupby('symbol')['adjclose'].pct_change()
    for h in (1, 3, 5):
        daily[f'fwd_{h}d'] = daily.groupby('symbol')['adjclose'].shift(-h) / daily['adjclose'] - 1

    # 4) PIT 行业映射
    daily = daily.merge(members[['symbol', 'ind_code', 'ind_name', 'in_date', 'end']],
                        on='symbol', how='inner')
    daily = daily[(daily['date'] >= daily['in_date']) & (daily['date'] < daily['end'])]
    print(f'映射后: {len(daily)} 行', flush=True)

    # 5) 二级行业日收益
    ind2 = daily.groupby(['date', 'ind_code'])['ret'].mean().reset_index()
    ind2.columns = ['date', 'ind_code', 'ind2_ret']
    ind2 = ind2.sort_values(['ind_code', 'date'])
    ind2['ind2_ret_prev'] = ind2.groupby('ind_code')['ind2_ret'].shift(1)
    ind2['l2_start'] = (ind2['ind2_ret'] >= 0.02) & (ind2['ind2_ret_prev'] < 0.01)
    print(f'二级启动日: {int(ind2["l2_start"].sum())} 天次', flush=True)

    daily = daily.merge(ind2[['date', 'ind_code', 'l2_start']], on=['date', 'ind_code'], how='left')

    # 市场启动日（复用V2指标）
    mkt = pd.read_csv('runtime/market_start_indicator.csv')
    mkt['date'] = pd.to_datetime(mkt['date'])
    daily = daily.merge(mkt[['date', 'market_start']], on='date', how='left')

    # 6) 检验
    v1 = daily[(daily['l2_start'] == True) & (daily['ret'] < 0.01)]
    v2 = daily[(daily['l2_start'] == True) & (daily['ret'] >= 0.03)]
    v3 = daily[(daily['l2_start'] == False) & (daily['ret'] < 0.01)]
    v1m = daily[(daily['market_start'] == True) & (daily['ret'] < 0.01)]
    v2m = daily[(daily['market_start'] == True) & (daily['ret'] >= 0.03)]

    print('\n== V3 系：申万二级行业级传导 ==')
    for h in (1, 3, 5):
        col = f'fwd_{h}d'
        r1, r2, r3 = v1[col].dropna(), v2[col].dropna(), v3[col].dropna()
        r1m, r2m = v1m[col].dropna(), v2m[col].dropna()
        if min(len(r1), len(r3)) < 100:
            continue
        t13 = sps.ttest_ind(r1, r3, equal_var=False)
        print(f'\n--- 未来{h}日 ---')
        print(f'  L2启动滞涨: 均 {r1.mean()*100:+.2f}% (n={len(r1)})')
        print(f'  L2启动已涨: 均 {r2.mean()*100:+.2f}% (n={len(r2)})')
        print(f'  普通日滞涨: 均 {r3.mean()*100:+.2f}% (n={len(r3)})')
        print(f'  传导: {(r1.mean()-r3.mean())*100:+.3f}% (t={t13.statistic:+.2f})')
        if len(r1m) > 100:
            print(f'  [对照]市场启动滞涨: {r1m.mean()*100:+.2f}% | 已涨: {r2m.mean()*100:+.2f}%')

    from research.factor_screen2 import register
    register('V3申万二级传导: 见v3_sw_l2证据', label='v3_sw_l2')
    print(f'\n证据: {DEST}')


if __name__ == '__main__':
    main()
