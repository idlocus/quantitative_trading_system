#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""G1（每日决策版）终极区间验证（预注册见 G1 终极区间验证节）。"""
import sys, os
if __name__ == '__main__':
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from pathlib import Path
import pandas as pd

from data.preparation import atomic_json
from research.portfolio_study import publish_study, read_published_verdict
from research import technical_signals as ts
from research.composite_study import PORTFOLIO_PROTOCOL, portfolio_layer
from research.factor_screen2 import PROTOCOL, load_ohlc_panel, valuation_input
from research.portfolio_metrics import performance

DEST = Path('ai-created/research/g1_holdout_pool_v3')  # 审计2026-11-19重跑(仿真器语义修复+指标口径更正)
HOLDOUT_START = pd.Timestamp('2024-01-01')


def _compute(dest):
    dest.mkdir(parents=True, exist_ok=True)
    protocol = dict(PROTOCOL, start='2023-06-01', validation_end='2026-10-30',
                    train_end='2023-12-31')
    panel, base, evidence = load_ohlc_panel(protocol)
    price = base['price']
    eligible = (base['usable']
                & price.notna().rolling(protocol['minimum_history_sessions']).sum()
                .eq(protocol['minimum_history_sessions'])
                & base['amount'].rolling(protocol['liquidity_window'], min_periods=protocol['liquidity_window'])
                .median().ge(protocol['minimum_median_daily_amount_local_currency']))
    d16 = ts.d16_rel_atr(panel['high'], panel['low'], panel['close'])
    fund = Path('data_cache/fundamental')
    pb = valuation_input(fund / 'pb_lf.parquet', evidence)
    pe = valuation_input(fund / 'pe_ttm.parquet', evidence)
    bp_rank = (1.0 / pb.where(pb > 0)).reindex(
        index=price.index, columns=price.columns).where(eligible).rank(axis=1, pct=True)
    loss = pe.reindex(index=price.index, columns=price.columns).isna()
    d7_rank = ts.d7_expma_gap(panel['close']).where(eligible).rank(axis=1, pct=True)
    g1 = (((d16.where(eligible).rank(axis=1, pct=True) + bp_rank) / 2.0)
          .where(~loss).where(d7_rank < 0.90))
    sim_protocol = dict(PORTFOLIO_PROTOCOL, study='g1_holdout',
                        start='2023-06-01', validation_end='2026-10-30',
                        train_end='2023-12-31', risk_variants=['budget_guard'],
                        rebalance_sessions=1,
                        hold_buffer_ranks={'G1_daily': 100})
    availability = {'G1_daily': (d16.notna() & bp_rank.notna() & d7_rank.notna()).any(axis=1)}
    portfolio_layer({'G1_daily': g1}, eligible, d16, base, evidence,
                     sim_protocol, dest=dest, availability=availability)

    name = 'G1_daily_budget_guard_25bps'
    rows = pd.read_csv(dest / f'{name}_nav.csv')
    rows['date'] = pd.to_datetime(rows['date'])
    # 审计 2026-11-19 一-1：月度指标统一走 performance()——首月基准=
    # 上年（2023）最后交易日净值、月基准逐月 shift、峰值含 initial 跨期
    # 延续、月数=自然月数。旧口径"本月首日收盘为月初基准 + 交易日数/21
    # 估算月数"漏掉每月首日盈亏且月数失真，已废弃。
    pre = rows[rows['date'] < HOLDOUT_START]
    seg = rows[rows['date'] >= HOLDOUT_START]
    seg_nav = seg.set_index('date')['nav']
    initial = float(pre['nav'].iloc[-1])
    perf = performance(seg_nav, initial=initial)
    gmr = perf['geometric_monthly_return']
    mdd = perf['max_drawdown']

    issues = pd.read_csv(dest / f'{name}_issues.csv')
    issues = issues[pd.to_datetime(issues['date']) >= HOLDOUT_START]
    crit = issues[~issues['issue'].isin(
        {'unknown_cash_payment_locked', 'unknown_bonus_listing_locked',
         'documented_suspension_valuation_carry',
         'adjustment_change_no_distribution_recorded'})]
    # 审计 2026-11-19 一-2：换手改金额口径（先裁样本外日期再汇总），
    # 逐成交日买卖金额/当日收盘净值按自然月平均；成交笔数单独报告。
    # 旧口径 len(tr)/月数/50 是笔数冒充换手，与容量/手续费判断不可比。
    tr = pd.read_csv(dest / f'{name}_trades.csv')
    tr['date'] = pd.to_datetime(tr['date'])
    tr_seg = tr[tr['date'] >= HOLDOUT_START].copy()
    tr_seg['amount'] = tr_seg['price'] * tr_seg['quantity']
    nav_by_date = rows.set_index('date')['nav']
    amt_by_day = tr_seg.groupby('date')['amount'].sum()
    turnover_double = float((amt_by_day / nav_by_date.reindex(amt_by_day.index)).sum()) \
        / perf['months']
    n_buy = int((tr_seg['side'] == 'BUY').sum())
    n_sell = int((tr_seg['side'] == 'SELL').sum())
    buy_amt = float(tr_seg.loc[tr_seg['side'] == 'BUY', 'amount'].sum())
    sell_amt = float(tr_seg.loc[tr_seg['side'] == 'SELL', 'amount'].sum())
    exposure = (seg['market_value'] / seg['nav']).mean()

    print('== G1（每日决策版）终极区间 2024-01~2026-10（2026-11-19 更正口径）==')
    print(f"月收益 {gmr:+.2%}（年化 {(1+gmr)**12-1:+.1%}） | maxDD {abs(mdd):.1%} | "
          f"最差月 {perf['worst_calendar_month']:+.1%} | "
          f"月内最差 {perf['worst_daily_month_to_date']:+.1%}")
    print(f"换手 {turnover_double:.2f}x/月（双边金额口径）| 成交 "
          f"{len(tr_seg)} 笔（买{n_buy}/卖{n_sell}，样本外前 "
          f"{int((tr['date'] < HOLDOUT_START).sum())} 笔不入统计）")
    print(f"买入金额 {buy_amt:,.0f} | 卖出金额 {sell_amt:,.0f} | "
          f"平均仓位 {exposure:.0%} | critical {len(crit)}条")

    verdict = {'monthly': gmr, 'max_dd': mdd,
               'worst_month': perf['worst_calendar_month'],
               'worst_mtd': perf['worst_daily_month_to_date'],
               'months': perf['months'], 'holdout_initial_nav': initial,
               'turnover_double': turnover_double,
               'trades_before_holdout': int((tr['date'] < HOLDOUT_START).sum()),
               'trades_in_holdout': len(tr_seg),
               'trades_buy': n_buy, 'trades_sell': n_sell,
               'buy_amount': buy_amt, 'sell_amount': sell_amt,
               'exposure': exposure, 'critical': len(crit),
               'PASSED': bool(gmr > 0 and not len(crit)
                              and not seg[['opening_breach', 'closing_breach']].any().any()
                              and seg['drawdown'].min() >= -.15 and perf['worst_daily_month_to_date'] >= -.15),
               'lifetime_drawdown_in_holdout': float(seg['drawdown'].min()),
               'correction_note': ('2026-11-19 审计更正：月度基准=上年最后净值、'
                                   '月数=自然月、换手=金额口径双边（原 4.39x 为'
                                   '笔数口径，不可比）。更正算法见 '
                                   'ai-created/2026-11-19-review-fixes/'
                                   'recompute_g1_metrics.py（已对保存净值复算验收）。')}
    atomic_json(dest / 'verdict.json', verdict)
    return verdict


def main(register_trial=True, *, published_only=False):
    manifest_name = 'holdout_manifest.json'
    if not published_only:
        with publish_study(DEST, manifest_name) as stage:
            verdict = _compute(stage)
        if not register_trial:
            return verdict
    verdict, release_sha256 = read_published_verdict(DEST, manifest_name)
    if register_trial:
        from research.overfitting import TrialRegistry
        note = (f"G1 每日决策版样本外（合格池内排名口径），{verdict['months']} 个自然月，"
                f"几何月收益 {verdict['monthly']:+.2%}，最大回撤 {verdict['max_dd']:.2%}，"
                f"月均双边换手 {verdict['turnover_double']:.3f} 倍")
        metrics = {k: verdict[k] for k in ('monthly', 'max_dd', 'turnover_double', 'months',
                                           'trades_in_holdout', 'PASSED')}
        count = TrialRegistry().record(
            'G1 样本外验证（合格池内排名口径）', release_sha256=release_sha256,
            extra={'verdict': note, 'metrics': metrics, 'release_manifest': (DEST / manifest_name).as_posix()})
        print(f'{note}\n登记册总数 {count}，证据 {DEST}')
    return verdict


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--register-published', action='store_true',
                        help='核验并登记已发布的 G1 结果')
    main(published_only=parser.parse_args().register_published)
