#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
组合策略回测 —— 质量价值选股 + ETF趋势控制总仓位

核心思想：
  质量价值策略在股票池内选股（+30.5%），但无法决定"该不该持有股票"。
  ETF趋势策略判断资产方向（+17.5%），但不做个股选择。
  两者组合：趋势决定股票仓位0-100%，质量价值决定持有哪些股票。

规则：
  1. ETF趋势层：沪深300过去6个月趋势>0 → 股票仓位=100%；否则 → 股票仓位=0%（全现金/债券）
  2. 股票选择层：质量过滤+价值合成+趋势叠加(0.2)，Top100等权，月度调仓
  3. 两层独立运行，趋势层只影响总仓位倍数

对照：
  A. 纯质量价值（无趋势层，始终100%股票仓位）
  B. 纯ETF趋势（无选股，等权持有趋势为正的资产）
  C. 60/40固定比例+质量价值选股（60%仓位用质量价值，40%债券）

淘汰条件：
  1. 组合净收益 > 纯质量价值
  2. 组合最大回撤 < 纯质量价值的回撤
  3. 组合夏普 > 纯质量价值
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import date, timedelta

import numpy as np
import pandas as pd

from data.layered import LayeredDataSource
from data.base.data_loader import DataLoader
from portfolio.daily_replay import DailyReplayBacktester, ReplayConfig
from portfolio.tradable_benchmark import tradable_equal_weight
from trading import config as chain

ERAS = [('fullmarket_2022_2026', date(2021, 11, 1), date(2026, 10, 9), date(2022, 1, 1))]


def build_stock_signal(ds, pool, d0, d1):
    """质量过滤+价值+趋势(0.2)的月度信号"""
    from experiments.quality_value_candidate import (
        build_value_score, build_quality_filter, apply_quality_filter, monthly_signal)
    from experiments.momentum_overlay import build_momentum_matrix, combine_value_momentum

    value_score = build_value_score(ds, pool, d0, d1)
    value_score = value_score[value_score.columns.intersection(
        set(ds.preload_batch(pool, d0, d1).keys()))] if False else value_score
    quality = build_quality_filter(ds, pool, d0, d1)
    score_qv = apply_quality_filter(value_score, quality)

    # 趋势叠加
    loader = DataLoader.__new__(DataLoader)
    loader.data_source_type = 'layered'
    loader.data_source = ds
    loader._cache = None
    cache = loader.preload_klines(pool + ['000300.SH'], d0, d1)._cache
    momentum = build_momentum_matrix(cache, d0, d1)
    score_full = combine_value_momentum(score_qv, momentum, chain.MOMENTUM_WEIGHT)

    return monthly_signal(score_full), cache


def build_equity_trend(ds, d0, d1, lookback=126):
    """沪深300趋势信号（用于控制总仓位）"""
    cache = ds.preload_batch(['000300.SH'], d0, d1)
    bench = cache.get('000300.SH')
    if bench is None or bench.empty:
        return None
    idx = pd.DatetimeIndex(pd.to_datetime(bench['date'])).as_unit('ns')
    adj_col = 'adjclose' if 'adjclose' in bench.columns else 'close'
    px = pd.Series(pd.to_numeric(bench[adj_col], errors='coerce').values, index=idx)
    trend = (px / px.shift(lookback) - 1) > 0
    return trend


def run_combined_strategy(cache, score_monthly, equity_trend, ts, d1,
                          initial_capital=1_000_000, cost_rate=0.001):
    """
    组合策略：趋势层控制总仓位 × 质量价值层选股

    趋势信号>0 → 股票仓位100%（用质量价值的日度重放）
    趋势信号≤0 → 股票仓位0%（全现金；重新入场需趋势恢复）
    """
    dates = sorted(set(score_monthly.index) |
                   set(equity_trend.index[equity_trend.index >= pd.Timestamp(ts)]))
    dates = [d for d in dates if pd.Timestamp(ts) <= d <= pd.Timestamp(d1)]

    nav = initial_capital
    navs = {}
    holdings = {}       # {sym: weight}（股票层内部权重）
    stock_on = True     # 当前是否持有股票
    total_cost = 0.0

    # 获取价格矩阵
    close_dfs = {}
    for sym, df in cache.items():
        if df is None or df.empty or sym == '000300.SH':
            continue
        idx = pd.DatetimeIndex(pd.to_datetime(df['date'])).as_unit('ns')
        close_dfs[sym] = pd.Series(
            pd.to_numeric(df['close'], errors='coerce').values, index=idx)
    close_mx = pd.DataFrame(close_dfs).sort_index()
    prev_mx = close_mx.shift(1)

    for t in dates:
        if t in close_mx.index:
            # 检查趋势信号（月度）
            et = equity_trend.reindex([t], method='ffill').iloc[0] if t in equity_trend.index or \
                (equity_trend.index <= t).any() else None
            if et is not None and not np.isnan(et):
                should_hold_stock = bool(et)
            else:
                should_hold_stock = stock_on  # 无信号时维持

            if should_hold_stock and not stock_on:
                # 趋势转正 → 开始持有股票
                stock_on = True
                # 重建持仓（简化：用当前月度信号）
                if t in score_monthly.index:
                    row = score_monthly.loc[t].dropna()
                    targets = row.sort_values(ascending=False).head(chain.TOP_N).index.tolist()
                    w = 1.0 / max(len(targets), 1)
                    holdings = {s: w for s in targets if s in close_mx.columns}
                total_cost += 0.001  # 建仓成本
                nav *= 0.999

            elif not should_hold_stock and stock_on:
                # 趋势转负 → 清仓股票
                stock_on = False
                holdings = {}
                total_cost += 0.001  # 清仓成本
                nav *= 0.999

            # 月度调仓（股票层内部）
            if stock_on and t in score_monthly.index:
                row = score_monthly.loc[t].dropna()
                targets = row.sort_values(ascending=False).head(chain.TOP_N).index.tolist()
                w = 1.0 / max(len(targets), 1)
                new_holdings = {s: w for s in targets if s in close_mx.columns}
                # 换手成本
                all_s = set(holdings) | set(new_holdings)
                turnover = sum(abs(new_holdings.get(s, 0) - holdings.get(s, 0))
                               for s in all_s) / 2
                cost = turnover * cost_rate
                total_cost += cost
                nav *= (1 - cost)
                holdings = new_holdings

            # 逐日估值
            if stock_on and holdings:
                port_ret = 0.0
                px_row = close_mx.loc[t]
                prev_row = prev_mx.loc[t]
                for s, w in holdings.items():
                    px, prev_px = px_row.get(s, np.nan), prev_row.get(s, np.nan)
                    if not np.isnan(px) and not np.isnan(prev_px) and prev_px > 0:
                        port_ret += w * (px / prev_px - 1)
                nav *= (1 + port_ret)

        navs[t] = nav

    eq = pd.Series(navs).sort_index()
    if len(eq) < 2:
        return None
    rets = eq.pct_change().dropna()
    sharpe = float(rets.mean() * 252 / (rets.std() * np.sqrt(252))) if rets.std() > 0 else 0
    mdd = float((eq / eq.cummax() - 1).min())
    total_return = float(eq.iloc[-1] / eq.iloc[0] - 1)
    n_years = len(eq) / 252

    return {
        'equity': eq, 'total_return': total_return, 'sharpe': sharpe,
        'max_drawdown': mdd, 'total_cost': total_cost,
        'annual_return': (1 + total_return) ** (1 / max(n_years, 0.5)) - 1,
    }


def main():
    ds = LayeredDataSource()
    era, d0, d1, ts = ERAS[0]
    print(f"===== 组合策略回测 @ {era} =====\n")

    pool = sorted(ds.list_symbols(as_of=ts))
    print(f"构建信号...", flush=True)

    # 质量价值+趋势信号
    score_monthly, cache = build_stock_signal(ds, pool, d0, d1)

    # 沪深300趋势
    equity_trend = build_equity_trend(ds, d0, d1, lookback=126)
    trend_positive_pct = equity_trend.mean()
    print(f"信号就绪：月度信号{len(score_monthly.index)}天 | "
          f"趋势为正比例{trend_positive_pct:.0%}")

    # ---- 运行组合策略 ----
    print(f"\n运行组合策略（趋势控制仓位 × 质量价值选股）...", flush=True)
    combined = run_combined_strategy(cache, score_monthly, equity_trend, ts, d1)

    # ---- 对照A：纯质量价值（始终100%仓位） ----
    print(f"运行纯质量价值对照...", flush=True)
    cfg_stock = ReplayConfig(
        top_n=chain.TOP_N, band=chain.BUFFER_BAND,
        use_risk_gate=False,  # 不用风控闸门（纯粹对比选股能力）
        min_listing_days=chain.MIN_LISTING_DAYS,
        min_daily_turnover=chain.MIN_DAILY_TURNOVER,
    )
    rp = DailyReplayBacktester(cache, cfg_stock)
    stock_only = rp.run(score_monthly, start_date=str(ts), end_date=str(d1))

    # ---- 汇总 ----
    print(f"\n{'='*60}")
    print("组合策略 vs 纯选股 vs 纯趋势")
    print(f"{'='*60}")

    stock_cost = stock_only.total_cost / stock_only.initial_capital
    print(f"{'策略':>20s} {'净收益':>7s} {'年化':>7s} {'夏普':>5s} {'回撤':>6s} {'成本':>5s}")
    if combined:
        print(f"  {'组合(趋势×选股)':>18s} {combined['total_return']:>6.1%} "
              f"{combined['annual_return']:>6.1%} {combined['sharpe']:>5.2f} "
              f"{combined['max_drawdown']:>5.1%} {combined['total_cost']:>4.1%}")
    print(f"  {'纯质量价值':>18s} {stock_only.total_return:>6.1%} "
          f"{stock_only.annual_return:>6.1%} {stock_only.sharpe:>5.2f} "
          f"{stock_only.max_drawdown:>5.1%} {stock_cost:>4.1%}")

    # 淘汰判定
    if combined:
        print("\n淘汰判定：")
        c1 = '✅' if combined['total_return'] > stock_only.total_return else '❌'
        print(f"  净收益>纯选股: {c1} ({combined['total_return']:+.1%} vs {stock_only.total_return:+.1%})")
        c2 = '✅' if combined['max_drawdown'] > stock_only.max_drawdown else '❌'
        print(f"  回撤<纯选股: {c2} ({combined['max_drawdown']:.1%} vs {stock_only.max_drawdown:.1%})")
        c3 = '✅' if combined['sharpe'] > stock_only.sharpe else '❌'
        print(f"  夏普>纯选股: {c3} ({combined['sharpe']:.2f} vs {stock_only.sharpe:.2f})")


if __name__ == '__main__':
    main()
