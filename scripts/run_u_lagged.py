#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""U 系修正：T-1 滞后信号消除前视偏差。"""


def main():
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

    from pathlib import Path
    import pandas as pd
    from scipy import stats as sps

    DEST = Path('ai-created/research/u_seat_theme_2026-11-13')

    ev = pd.read_csv(DEST / 'events_with_signals.csv')
    ev['date'] = pd.to_datetime(ev['date'])

    # 构造 T-1 信号：把每个 symbol 的龙虎榜信号 shift(1) 按 symbol 时间序对齐
    ev = ev.sort_values(['symbol', 'date']).reset_index(drop=True)
    for col, lag_name in [('buy_ratio', 'buy_ratio_lag1'),
                           ('龙虎榜净买额', 'lhb_net_lag1'),
                           ('has_inst', 'has_inst_lag1')]:
        ev[lag_name] = ev.groupby('symbol')[col].shift(1)
    # 行业信号也滞后
    for col in ['industry_ret', 'industry_limitup']:
        ev[f'{col}_lag1'] = ev.groupby(['industry', 'date'])[col].transform('first')
        # 行业信号按(行业,日期)分组后对每个symbol shift 不对——应该按日期 shift
    # 更正确的方式：按日期对齐行业指标的前一交易日值
    ind_daily = ev.groupby(['date', 'industry'])[['industry_ret', 'industry_limitup']].first().reset_index()
    ind_daily = ind_daily.sort_values(['industry', 'date'])
    for col in ['industry_ret', 'industry_limitup']:
        ind_daily[f'{col}_lag1'] = ind_daily.groupby('industry')[col].shift(1)
    ev = ev.drop(columns=[c for c in ev.columns if c.endswith('_lag1') and c.startswith('industry_')], errors='ignore')
    ev = ev.merge(ind_daily[['date', 'industry', 'industry_ret_lag1', 'industry_limitup_lag1']],
                  on=['date', 'industry'], how='left')

    ev.to_csv(DEST / 'events_with_lagged_signals.csv', index=False, encoding='utf-8-sig')

    print('== U 系修正：T-1 滞后信号（可执行时序）==')
    print(f'基线（全体追入）: {ev.ret_to_close.mean()*10000:+.1f}bp\n')

    def group_report(mask_hi, mask_lo, label):
        hi = ev.loc[mask_hi, 'ret_to_close'].dropna()
        lo = ev.loc[mask_lo, 'ret_to_close'].dropna()
        if len(hi) < 30 or len(lo) < 30:
            print(f'{label}: 样本不足 (hi={len(hi)}, lo={len(lo)})')
            return
        t = sps.ttest_ind(hi, lo, equal_var=False)
        hi_lu = ev.loc[mask_hi, 'limit_up'].mean()
        lo_lu = ev.loc[mask_lo, 'limit_up'].mean()
        sig = '***' if t.pvalue < 0.001 else '**' if t.pvalue < 0.01 else '*' if t.pvalue < 0.05 else ''
        print(f'{label}:')
        print(f'  高组: n={len(hi):>6} 均 {hi.mean()*10000:+.1f}bp 封板率 {hi_lu:.0%}')
        print(f'  低组: n={len(lo):>6} 均 {lo.mean()*10000:+.1f}bp 封板率 {lo_lu:.0%}')
        print(f'  差: {(hi.mean()-lo.mean())*10000:+.1f}bp (t={t.statistic:+.2f}, p={t.pvalue:.4f}){sig}\n')

    on_lhb = ev['lhb_net_lag1'].notna()

    print('--- U1c-lag1: 昨日龙虎榜净买入 ---')
    group_report(on_lhb & (ev['lhb_net_lag1'] > 0),
                 on_lhb & (ev['lhb_net_lag1'] < 0),
                 '昨净买入 vs 昨净卖出')

    print('--- U1b-lag1: 昨日有机构参与 ---')
    group_report(ev['has_inst_lag1'] == 1,
                 ev['has_inst_lag1'] == 0,
                 '昨有机构 vs 昨无机构')

    print('--- U1a-lag1: 昨日席位集中度 ---')
    group_report(on_lhb & (ev['buy_ratio_lag1'] >= 0.5),
                 on_lhb & (ev['buy_ratio_lag1'] < 0.3),
                 '昨集中≥50% vs 昨分散<30%')

    print('--- U2b-lag1: 昨日板块涨幅 ---')
    group_report(ev['industry_ret_lag1'] >= 0.02,
                 ev['industry_ret_lag1'] < 0,
                 '昨板块涨≥2% vs 昨板块跌')

    # 组合：昨日净买入 + 今日触+8%
    print('--- 组合最强信号：昨日龙虎榜净买入 → 今日追+8% ---')
    best = ev[on_lhb & (ev['lhb_net_lag1'] > 0)]
    if len(best) >= 30:
        r = best['ret_to_close'].dropna()
        print(f'  n={len(best)} | 均 {r.mean()*10000:+.1f}bp | 中位 {r.median()*10000:+.1f}bp')
        print(f'  胜率 {(r>0).mean():.1%} | 封板率 {best["limit_up"].mean():.0%}')
        print(f'  P25 {r.quantile(0.25)*100:+.2f}% | P75 {r.quantile(0.75)*100:+.2f}%')

    from research.factor_screen2 import register
    register('U系修正T-1滞后: 见u_seat_theme证据', label='u_lagged')
    print(f'\n证据: {DEST}')


if __name__ == '__main__':
    main()
