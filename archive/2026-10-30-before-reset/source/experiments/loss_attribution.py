#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
亏损归因实验 —— 拆解当前策略的亏损来源

四组对照（同一数据、同一费用、同一风控基础）：
  A: 当前生产策略（每日信号+每日调仓+缓冲带20%+止损5%）
  B: 信号不变，改为月度调仓（检验：每日换仓是否消耗收益）
  C: B + 排名缓冲（入Top100/出Top150才卖）（检验：排名边缘反复进出是否值得）
  D: B + 止损后不回买（检验：止损次日回买是否造成无效交易）

登记假设（experiments registry 格式）：
  - 假设：当前策略亏损主要来自交易成本（高频换仓+止损回买），而非选股无效
  - 淘汰条件：如果B的毛收益（不扣费）与A无差异 → 选股本身无效，降低频率只能减亏不能盈利
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import date, timedelta

import pandas as pd

from data.layered import LayeredDataSource
from data.base.data_loader import DataLoader
from portfolio import composite_from_matrices
from portfolio.daily_replay import DailyReplayBacktester, ReplayConfig
from trading import config as chain

D5 = chain.PRODUCTION_FACTORS
ERAS = [('fullmarket_2022_2026', date(2021, 11, 1), date(2026, 10, 9), date(2022, 1, 1))]
BENCHES = ['000300.SH']


def build_scores(ds, pool, d0, d1):
    from data.fundamentals import WindFundamentalProvider, build_fundamental_matrix
    from data.fundamental_pit import PITFinancialProvider
    prov = WindFundamentalProvider(ds)
    pit = PITFinancialProvider(ds)
    mats = {}
    for f in ['ep', 'bp', 'sp', 'size']:
        fm = build_fundamental_matrix(prov, f, pool, d0, d1)
        if not fm.empty:
            mats[f] = fm.loc[pd.Timestamp(d0):pd.Timestamp(d1)]
    lev = pit.get_factor_matrix('leverage', pool, d0, d1)
    if not lev.empty:
        mats['leverage'] = lev.loc[pd.Timestamp(d0):pd.Timestamp(d1)]
    return composite_from_matrices(mats, directions=D5, experimental=True)


def monthly_signal(score_mx, rebalance_day=1):
    """只在每月第N个交易日更新信号（其余日沿用最近排名）"""
    dates = score_mx.index
    monthly_dates = set()
    last_month = None
    for d in dates:
        m = (d.year, d.month)
        if m != last_month:
            monthly_dates.add(d)
            last_month = m
    return score_mx.loc[sorted(monthly_dates)]


def run_experiment(name, cache, score_mx, ts, d1, desc,
                   band=chain.BUFFER_BAND, use_gate=True,
                   stop_loss=None, signal_override=None):
    """运行单组实验"""
    cfg = ReplayConfig(
        top_n=chain.TOP_N,
        band=band,
        use_risk_gate=use_gate,
        min_listing_days=chain.MIN_LISTING_DAYS,
        min_daily_turnover=chain.MIN_DAILY_TURNOVER,
    )
    if stop_loss is not None and use_gate:
        from live.risk_gate import GateConfig
        cfg.gate_config = GateConfig(stop_loss_pct=stop_loss)

    effective_score = signal_override if signal_override is not None else score_mx
    rp = DailyReplayBacktester(cache, cfg)
    result = rp.run(effective_score, start_date=str(ts), end_date=str(d1))

    # 成本拆分
    total_cost_pct = result.total_cost / result.initial_capital
    n_trades = len(result.trades)
    n_days = result.n_dates
    turnover_per_day = result.avg_turnover

    print(f"\n{'='*60}")
    print(f"实验 {name}: {desc}")
    print(f"{'='*60}")
    print(f"  总收益:     {result.total_return:+.2%}")
    print(f"  年化:       {result.annual_return:+.2%}")
    print(f"  夏普:       {result.sharpe:.2f}")
    print(f"  最大回撤:   {result.max_drawdown:.2%}")
    print(f"  累计成本:   {total_cost_pct:.2%} ({result.total_cost:,.0f}元)")
    print(f"  成交笔数:   {n_trades}")
    print(f"  平均换手/日: {turnover_per_day:.2%}")
    print(f"  止损卖出:   {result.n_stop_sells}笔")
    if result.terminal_date:
        print(f"  ⚠ 终态:     {result.terminal_date}起停机 "
              f"(运行期{result.active_period_return:+.1%})")

    # 毛收益估算（总收益+成本≈毛收益，不含冲击）
    gross_est = result.total_return + total_cost_pct
    print(f"  毛收益估算: {gross_est:+.2%}（净收益+成本）")
    return result


def main():
    ds = LayeredDataSource()
    loader = DataLoader.__new__(DataLoader)
    loader.data_source_type = 'layered'
    loader.data_source = ds
    loader._cache = None

    era, d0, d1, ts = ERAS[0]
    print(f"===== {era} =====")
    pool = sorted(ds.list_symbols(as_of=ts))
    print(f"池 {len(pool)} 只", flush=True)
    cache = loader.preload_klines(pool + BENCHES, d0, d1)._cache
    score = build_scores(ds, pool, d0, d1)
    score = score[score.columns.intersection(cache.keys())]
    print(f"因子矩阵 {score.shape[0]}日 × {score.shape[1]}只", flush=True)

    # 月度信号（只在月初第一个交易日更新排名）
    score_monthly = monthly_signal(score)
    print(f"月度信号日: {len(score_monthly.index)}天（全量{len(score.index)}天）")

    results = {}

    # A: 当前生产策略（每日信号+每日调仓+止损5%）
    results['A'] = run_experiment(
        'A', cache, score, ts, d1,
        '当前生产策略（每日信号+止损5%）')

    # B: 月度调仓（信号不变，只在月初更新）
    results['B'] = run_experiment(
        'B', cache, score_monthly, ts, d1,
        '月度调仓（信号不变，月初更新排名）')

    # C: B + 排名缓冲（缓冲带加宽到50%，近似入100/出150的效果）
    results['C'] = run_experiment(
        'C', cache, score_monthly, ts, d1,
        '月度调仓+宽缓冲带50%（近似入100出150）',
        band=0.50)

    # D: B + 无止损（检验止损是否在消耗收益）
    results['D'] = run_experiment(
        'D', cache, score_monthly, ts, d1,
        '月度调仓+无止损（stop_loss=1.0=不触发）',
        stop_loss=1.0)

    # 归因对比表
    print(f"\n{'='*60}")
    print("亏损归因对比")
    print(f"{'='*60}")
    print(f"{'实验':>4s} {'净收益':>8s} {'成本':>6s} {'毛收益':>8s} "
          f"{'换手/日':>7s} {'止损':>4s} {'结论':s}")
    for k in 'ABCD':
        r = results[k]
        cost = r.total_cost / r.initial_capital
        gross = r.total_return + cost
        conclusion = ''
        if k == 'B':
            diff_cost = (results['A'].total_cost - r.total_cost) / r.initial_capital
            conclusion = f"成本节省{diff_cost:+.1%}"
        elif k == 'C':
            conclusion = f"vs B换手差{(results['B'].avg_turnover - r.avg_turnover):+.1%}"
        elif k == 'D':
            conclusion = f"止损{(results['B'].n_stop_sells - r.n_stop_sells):+d}笔"
        print(f"  {k}   {r.total_return:>7.1%} {cost:>5.1%} {gross:>7.1%} "
              f"{r.avg_turnover:>6.1%} {r.n_stop_sells:>4d}  {conclusion}")

    # 关键判断
    a_gross = results['A'].total_return + results['A'].total_cost / results['A'].initial_capital
    b_gross = results['B'].total_return + results['B'].total_cost / results['B'].initial_capital
    print(f"\n关键结论：")
    print(f"  A毛收益 {a_gross:+.1%} vs B毛收益 {b_gross:+.1%}")
    if a_gross < 0 and b_gross < 0:
        print(f"  → 未扣费的选股收益也无优势——问题在选股，降低频率只能减亏")
    elif a_gross < 0 and b_gross > 0:
        print(f"  → 月度调仓后毛收益转正——问题主要在交易成本")
    elif a_gross > 0 and b_gross > 0:
        print(f"  → 毛收益有优势但净收益可能被成本消耗——优化换手和执行")
    else:
        print(f"  → 需进一步分析")


if __name__ == '__main__':
    main()
