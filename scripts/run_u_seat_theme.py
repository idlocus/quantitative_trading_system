#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""U 系：席位资金 × 题材热度 对涨停前事件的条件分组（预注册见 U 系节）。"""


def main():
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

    from pathlib import Path
    import pandas as pd
    from scipy import stats as sps

    DEST = Path('ai-created/research/u_seat_theme_2026-11-13')
    DEST.mkdir(parents=True, exist_ok=True)

    # ── 加载 T 系事件 ──
    events = pd.read_csv('ai-created/research/t_pre_limitup_5m_2026-11-12/pre_limitup_results.csv')
    events['date'] = pd.to_datetime(events['date']).dt.strftime('%Y-%m-%d')
    print(f'T系事件: {len(events)} 条', flush=True)

    # ── 加载 LHB 明细 + 机构 ──
    lhb = pd.read_parquet('data_cache/lhb/detail.parquet')
    lhb['上榜日'] = pd.to_datetime(lhb['上榜日']).dt.strftime('%Y-%m-%d')
    inst = pd.read_parquet('data_cache/lhb/inst.parquet')
    inst['上榜日期'] = pd.to_datetime(inst['上榜日期']).dt.strftime('%Y-%m-%d')

    # ── 构造行业日线（用 Oracle 数据算行业等权涨幅 + 涨停数）──
    oracle = Path('data_cache/preparation/oracle')
    daily_rows = []
    for year in range(2022, 2027):
        for month in range(1, 13):
            f = oracle / 'normalized_a' / f'{year}/{month:02d}.parquet'
            if not f.exists():
                continue
            df = pd.read_parquet(f, columns=['symbol', 'date', 'preclose', 'close'])
            df['ret'] = df['close'] / df['preclose'] - 1
            daily_rows.append(df[['symbol', 'date', 'ret']])
    daily = pd.concat(daily_rows, ignore_index=True)
    daily['date'] = pd.to_datetime(daily['date']).dt.strftime('%Y-%m-%d')
    print(f'日线: {len(daily)} 行', flush=True)

    # 行业映射（简化：用申万一级行业近似——此处用代码前缀做粗分）
    # 完整版应从 Wind 拉取行业分类
    def simple_industry(sym):
        code = sym.split('.')[0]
        if code.startswith('6'):
            return 'SH_main'
        elif code.startswith('0'):
            return 'SZ_main'
        elif code.startswith('3'):
            return 'ChiNext'
        return 'Other'

    daily['industry'] = daily['symbol'].map(simple_industry)
    events['industry'] = events['symbol'].map(simple_industry)

    # 行业日涨幅和涨停数
    ind_stats = daily.groupby(['date', 'industry']).agg(
        industry_ret=('ret', 'mean'),
        industry_limitup=('ret', lambda x: (x >= 0.093).sum())).reset_index()
    events = events.merge(ind_stats, on=['date', 'industry'], how='left')

    # ── LHB 信号 ──
    # U1a: 席位集中度（买入前5总额/总成交额）
    # detail 表有 龙虎榜买入额/龙虎榜成交额/市场总成交额
    lhb['buy_ratio'] = lhb['龙虎榜买入额'] / lhb['市场总成交额']
    lhb_signal = lhb[['代码', '上榜日', 'buy_ratio', '龙虎榜净买额']].rename(
        columns={'代码': 'code', '上榜日': 'date'})
    lhb_signal['symbol'] = lhb_signal['code'].astype(str).str.zfill(6)
    # 映射到 .SH/.SZ 后缀
    def to_wind_code(code):
        code = str(code).zfill(6)
        if code.startswith('6'):
            return code + '.SH'
        return code + '.SZ'
    lhb_signal['symbol'] = lhb_signal['symbol'].map(to_wind_code)
    events = events.merge(lhb_signal[['symbol', 'date', 'buy_ratio', '龙虎榜净买额']],
                          on=['symbol', 'date'], how='left')

    # U1b: 机构参与
    inst['has_inst'] = (inst['机构买入净额'] > 0).astype(int)
    inst_signal = inst[['代码', '上榜日期', 'has_inst']].rename(
        columns={'代码': 'code', '上榜日期': 'date'})
    inst_signal['symbol'] = inst_signal['code'].astype(str).str.zfill(6).map(to_wind_code)
    events = events.merge(inst_signal[['symbol', 'date', 'has_inst']],
                          on=['symbol', 'date'], how='left')
    events['has_inst'] = events['has_inst'].fillna(0).astype(int)

    events.to_csv(DEST / 'events_with_signals.csv', index=False, encoding='utf-8-sig')

    # ── 分组分析 ──
    print('\n== U 系：条件分组（追入收益 bp；基线=全体 -89bp）==')

    def group_report(ev, mask_high, mask_low, label):
        hi = ev[mask_high]['ret_to_close'].dropna()
        lo = ev[mask_low]['ret_to_close'].dropna()
        if len(hi) < 30 or len(lo) < 30:
            print(f'  {label}: 样本不足 (hi={len(hi)}, lo={len(lo)})')
            return
        t = sps.ttest_ind(hi, lo, equal_var=False)
        print(f'  {label}:')
        print(f'    高信号组: n={len(hi):>6} | 均 {hi.mean()*10000:+.1f}bp | 封板率 {ev[mask_high]["limit_up"].mean():.0%}')
        print(f'    低信号组: n={len(lo):>6} | 均 {lo.mean()*10000:+.1f}bp | 封板率 {ev[mask_low]["limit_up"].mean():.0%}')
        print(f'    差: {(hi.mean()-lo.mean())*10000:+.1f}bp (t={t.statistic:+.2f}, p={t.pvalue:.4f})')

    on_lhb = events['buy_ratio'].notna()
    print(f'\n事件中有龙虎榜数据: {int(on_lhb.sum())} / {len(events)} ({on_lhb.mean():.0%})')

    # U1a: 席位集中度
    print('\n--- U1a 席位集中度（买入前5/总成交）---')
    group_report(events,
                 on_lhb & (events['buy_ratio'] >= 0.5),
                 on_lhb & (events['buy_ratio'] < 0.3),
                 '集中≥50% vs 分散<30%')

    # U1b: 机构参与
    print('\n--- U1b 机构净买入 ---')
    group_report(events,
                 events['has_inst'] == 1,
                 events['has_inst'] == 0,
                 '有机构 vs 无机构')

    # U1c: 净买额不对称
    print('\n--- U1c 龙虎榜净买额 ---')
    group_report(events,
                 on_lhb & (events['龙虎榜净买额'] > 0),
                 on_lhb & (events['龙虎榜净买额'] < 0),
                 '净买入 vs 净卖出')

    # U2a: 行业涨停数
    print('\n--- U2a 行业共振（板块内涨停数）---')
    group_report(events,
                 events['industry_limitup'] >= 3,
                 events['industry_limitup'] <= 1,
                 '板块涨停≥3 vs ≤1')

    # U2b: 行业涨幅
    print('\n--- U2b 行业整体涨幅 ---')
    group_report(events,
                 events['industry_ret'] >= 0.02,
                 events['industry_ret'] < 0,
                 '板块涨≥2% vs 板块跌')

    from research.factor_screen2 import register
    register('U系席位×题材5检验: 见u_seat_theme证据', label='u_seat_theme')
    print(f'\n证据: {DEST}')


if __name__ == '__main__':
    main()
