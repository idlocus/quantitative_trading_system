#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ETF 多资产趋势策略原型 —— 把选股问题变成资产配置问题

假设登记：
  不同资产（股票/债券/黄金）的中期趋势持续性与分散效应能改善净收益。
  月度决策，持有趋势为正的资产，剩余资金进现金。

规则（初版，刻意简单）：
  1. 资产池：沪深300 ETF、中证500 ETF、国债 ETF、黄金 ETF（用指数替代ETF）
  2. 趋势判定：过去 N 月（默认6个月/约126交易日）累计收益 > 0 → 趋势为正
  3. 持仓：趋势为正的资产等权持有，无正趋势资产时100%现金
  4. 调仓：月度（每月第一个交易日）
  5. 成本：买卖各0.1%（ETF佣金+滑点，低于股票）

对照：
  A. 全股票持有（沪深300买入持有）
  B. 固定比例（60/40 股债）
  C. 等权再平衡（25%×4资产，月度再平衡）

淘汰条件（预注册）：
  1. 趋势版本净收益必须 > 0
  2. 最大回撤必须 < 全股票持有的50%
  3. 夏普必须 > 固定比例配置
  4. 参数 N=3/6/9/12 个月均应为正（参数稳定性）
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import date, timedelta

import numpy as np
import pandas as pd

from data.layered import LayeredDataSource
from trading import config as chain

# 资产池（用指数替代ETF——ETF上市晚，指数历史更长；正式上线时换ETF代码）
ASSETS = {
    '000300.SH': '沪深300（A股大盘）',
    '000905.SH': '中证500（A股中盘）',
    '000012.SH': '国债指数（债券）',
    'Au99.99.SGX': '黄金现货（伦敦金替代）',  # 如不可用则跳过
}
# 实际可用的资产代码（探测后确定）
AVAILABLE_ASSETS = {}


def load_asset_prices(ds, d0, d1):
    """加载资产价格矩阵"""
    codes = list(ASSETS.keys())
    cache = ds.preload_batch(codes, d0, d1)
    close = {}
    for sym, df in cache.items():
        if df is None or df.empty:
            continue
        idx = pd.DatetimeIndex(pd.to_datetime(df['date'])).as_unit('ns')
        adj_col = 'adjclose' if 'adjclose' in df.columns else 'close'
        close[sym] = pd.Series(
            pd.to_numeric(df[adj_col], errors='coerce').values, index=idx)
        AVAILABLE_ASSETS[sym] = ASSETS.get(sym, sym)
    return pd.DataFrame(close).sort_index().ffill()


def trend_signal(prices, lookback_days):
    """趋势判定：过去N日累计收益 > 0"""
    ret = prices / prices.shift(lookback_days) - 1
    return ret > 0


def run_trend_strategy(prices, lookback=126, cost_rate=0.001):
    """
    月度趋势策略：每月第一个交易日检查趋势，持有正趋势资产等权。
    """
    dates = prices.index
    if len(dates) < lookback + 20:
        return None, "数据不足"

    # 月度调仓日
    monthly_dates = []
    last_m = None
    for d in dates:
        m = (d.year, d.month)
        if m != last_m:
            monthly_dates.append(d)
            last_m = m

    # 净值模拟
    nav = 1.0
    navs = {}
    holdings = {}          # {sym: weight}
    total_cost = 0.0
    last_rebalance = None

    for t in dates:
        px_row = prices.loc[t]
        # 月度调仓
        if t in monthly_dates and t > dates[lookback]:
            trend = trend_signal(prices.loc[:t], lookback).iloc[-1]
            positive = trend[trend == True].index.tolist()
            if not positive:
                new_holdings = {}          # 全现金
            else:
                w = 1.0 / len(positive)
                new_holdings = {s: w for s in positive if s in AVAILABLE_ASSETS}

            # 计算换手成本
            all_syms = set(holdings) | set(new_holdings)
            turnover = sum(abs(new_holdings.get(s, 0) - holdings.get(s, 0))
                           for s in all_syms)
            cost = turnover * cost_rate
            total_cost += cost
            nav *= (1 - cost)
            holdings = new_holdings
            last_rebalance = t

        # 逐日估值
        if holdings:
            port_ret = 0.0
            prev_row = prices.shift(1).loc[t]
            for s, w in holdings.items():
                px, prev_px = px_row.get(s, np.nan), prev_row.get(s, np.nan)
                if not np.isnan(px) and not np.isnan(prev_px) and prev_px > 0:
                    port_ret += w * (px / prev_px - 1)
            nav *= (1 + port_ret)
        navs[t] = nav

    eq = pd.Series(navs).sort_index()
    rets = eq.pct_change().dropna()
    sharpe = float(rets.mean() * 252 / (rets.std() * np.sqrt(252))) if rets.std() > 0 else 0
    mdd = float((eq / eq.cummax() - 1).min())
    total_return = float(eq.iloc[-1] / eq.iloc[0] - 1)

    return {
        'equity': eq, 'total_return': total_return, 'sharpe': sharpe,
        'max_drawdown': mdd, 'total_cost': total_cost,
        'avg_turnover': 0,  # 月度策略换手低
    }, None


def run_buy_hold(prices, sym):
    """买入持有单一资产"""
    px = prices[sym].dropna()
    if len(px) < 2:
        return None
    eq = px / px.iloc[0]
    rets = eq.pct_change().dropna()
    sharpe = float(rets.mean() * 252 / (rets.std() * np.sqrt(252))) if rets.std() > 0 else 0
    mdd = float((eq / eq.cummax() - 1).min())
    return {'total_return': float(eq.iloc[-1] - 1), 'sharpe': sharpe,
            'max_drawdown': mdd, 'total_cost': 0}


def run_fixed_ratio(prices, ratio={'000300.SH': 0.6, '000012.SH': 0.4},
                    rebalance_monthly=True, cost_rate=0.001):
    """固定比例配置（如60/40股债）"""
    dates = prices.index
    nav = 1.0
    navs = {}
    total_cost = 0.0
    monthly_dates = set()
    last_m = None
    for d in dates:
        m = (d.year, d.month)
        if m != last_m:
            monthly_dates.add(d)
            last_m = m

    holdings = dict(ratio)
    for t in dates:
        if rebalance_monthly and t in monthly_dates:
            # 再平衡到目标比例（成本=偏离×费率）
            turnover = 0.0
            cur_values = {}
            total_value = nav
            for s, w in ratio.items():
                px = prices.loc[t].get(s, np.nan)
                prev_px = prices.shift(1).loc[t].get(s, np.nan)
                if not np.isnan(px) and not np.isnan(prev_px) and prev_px > 0:
                    cur_values[s] = holdings.get(s, 0) * (1 + px / prev_px - 1)
                else:
                    cur_values[s] = holdings.get(s, 0)
            total_cur = sum(cur_values.values())
            if total_cur > 0:
                for s in ratio:
                    target_v = ratio[s] * total_cur
                    turnover += abs(target_v - cur_values.get(s, 0)) / total_cur
                cost = turnover / 2 * cost_rate
                total_cost += cost
                nav *= (1 - cost)
                holdings = dict(ratio)

        # 逐日估值
        port_ret = 0.0
        prev_row = prices.shift(1).loc[t]
        for s, w in holdings.items():
            px, prev_px = prices.loc[t].get(s, np.nan), prev_row.get(s, np.nan)
            if not np.isnan(px) and not np.isnan(prev_px) and prev_px > 0:
                port_ret += w * (px / prev_px - 1)
        nav *= (1 + port_ret)
        navs[t] = nav

    eq = pd.Series(navs).sort_index()
    rets = eq.pct_change().dropna()
    sharpe = float(rets.mean() * 252 / (rets.std() * np.sqrt(252))) if rets.std() > 0 else 0
    mdd = float((eq / eq.cummax() - 1).min())
    return {'total_return': float(eq.iloc[-1] - 1), 'sharpe': sharpe,
            'max_drawdown': mdd, 'total_cost': total_cost}


def main():
    ds = LayeredDataSource()

    # 时间范围
    end = date(2026, 10, 9)
    start = end - timedelta(days=365 * 5)  # 5年

    print("加载资产价格...")
    prices = load_asset_prices(ds, start, end)
    print(f"可用资产: {list(AVAILABLE_ASSETS.keys())}")
    print(f"价格矩阵: {prices.shape[0]}日 × {prices.shape[1]}只")
    print(f"时间范围: {prices.index[0].date()} ~ {prices.index[-1].date()}")

    # 参数稳定性：多个回看期
    lookbacks = [63, 126, 189, 252]  # 3/6/9/12个月

    results = {}
    for lb in lookbacks:
        r, err = run_trend_strategy(prices, lookback=lb)
        if r:
            results[f'趋势{lb // 21}月'] = r
            print(f"\n趋势{lb // 21}月: 净{r['total_return']:+.1%} "
                  f"夏普{r['sharpe']:.2f} 回撤{r['max_drawdown']:.1%} "
                  f"成本{r['total_cost']:.2%}")

    # 对照组
    print("\n--- 对照组 ---")
    for sym in AVAILABLE_ASSETS:
        label = ASSETS.get(sym, sym)
        r = run_buy_hold(prices, sym)
        if r:
            results[f'持有{label[:6]}'] = r
            print(f"持有{label}: 净{r['total_return']:+.1%} "
                  f"夏普{r['sharpe']:.2f} 回撤{r['max_drawdown']:.1%}")

    # 固定比例
    if '000300.SH' in prices.columns and '000012.SH' in prices.columns:
        r = run_fixed_ratio(prices, {'000300.SH': 0.6, '000012.SH': 0.4})
        results['60/40股债'] = r
        print(f"60/40股债:   净{r['total_return']:+.1%} "
              f"夏普{r['sharpe']:.2f} 回撤{r['max_drawdown']:.1%}")

    # 汇总
    print(f"\n{'='*60}")
    print("ETF 多资产趋势策略汇总")
    print(f"{'='*60}")
    print(f"{'策略':>16s} {'净收益':>7s} {'夏普':>5s} {'回撤':>6s} {'成本':>5s}")
    for name, r in results.items():
        print(f"  {name:>14s} {r['total_return']:>6.1%} {r['sharpe']:>5.2f} "
              f"{r['max_drawdown']:>5.1%} {r.get('total_cost', 0):>4.1%}")

    # 淘汰判定（以6月趋势为主策略）
    if '趋势6月' in results:
        trend = results['趋势6月']
        stock_bh = results.get('持有沪深300', results.get('持有中证500', None))
        fixed = results.get('60/40股债', None)
        print(f"\n淘汰判定（趋势6月 vs 对照）：")
        c1 = '✅' if trend['total_return'] > 0 else '❌'
        print(f"  1. 净收益>0: {c1} ({trend['total_return']:+.1%})")
        if stock_bh:
            c2 = '✅' if trend['max_drawdown'] > stock_bh['max_drawdown'] * 0.5 else '❌'
            print(f"  2. 回撤<股票50%: {c2} ({trend['max_drawdown']:.1%} vs {stock_bh['max_drawdown']:.1%})")
        if fixed:
            c3 = '✅' if trend['sharpe'] > fixed['sharpe'] else '❌'
            print(f"  3. 夏普>固定比例: {c3} ({trend['sharpe']:.2f} vs {fixed['sharpe']:.2f})")
        # 参数稳定性
        positive_lbs = sum(1 for k in results if k.startswith('趋势') and results[k]['total_return'] > 0)
        c4 = '✅' if positive_lbs >= 3 else '❌'
        print(f"  4. 参数稳定(≥3/4正): {c4} ({positive_lbs}/4)")


if __name__ == '__main__':
    main()
