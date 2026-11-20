#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ETF 趋势引擎唯一实现 —— 生产（paper/实盘）与验证实验共用。

禁止任何路径自行维护资产池/回看期/成本（参数单一来源=trading/config）。
规则（与原型预注册一致，2026-10-15 起在真实 ETF 代码上生效）：
  月度调仓日：过去 ETF_TREND_LOOKBACK 交易日累计收益（复权）> 0 → 趋势为正；
  正趋势资产等权持有，无正趋势资产时 100% 现金。
"""
import logging
from datetime import date

import pandas as pd

from trading import config as chain

logger = logging.getLogger(__name__)


def build_etf_signal(ds, as_of: date, lookback: int = None):
    """T-1 收盘数据下生成目标权重（与股票策略同 T-1 语义，无前视）。

    Returns:
        (targets: {code: 权重}, cache: dict, trend: {code: 区间收益})
        targets 为空 dict = 100% 现金。
    """
    lookback = lookback or chain.ETF_TREND_LOOKBACK
    codes = list(chain.ETF_ASSETS)
    # 预热窗口：交易日→日历日换算（A股年交易日≈243）+60日缓冲；
    # 只取1自然年对12个月回看不够（243<253），曾致长回看参数永远空仓
    days_back = int(lookback * 365 / 243) + 60
    try:
        d0 = date(as_of.year, as_of.month, as_of.day)
    except ValueError:
        d0 = date(as_of.year, as_of.month, 28)
    from datetime import timedelta
    d0 = d0 - timedelta(days=days_back)
    cache = ds.preload_batch(codes, d0, as_of)
    px = {}
    for sym, df in cache.items():
        if df is None or df.empty:
            continue
        idx = pd.DatetimeIndex(pd.to_datetime(df['date'])).as_unit('ns')
        col = 'adjclose' if 'adjclose' in df.columns else 'close'  # 收益口径=复权
        px[sym] = pd.Series(pd.to_numeric(df[col], errors='coerce').values,
                            index=idx)
    prices = pd.DataFrame(px).sort_index().dropna(how='all')
    if prices.empty:
        logger.error("[etf] 无可用价格数据")
        return {}, cache, {}

    hist = prices.loc[:pd.Timestamp(as_of)].dropna(how='all')
    if len(hist) < lookback + 1:
        logger.error(f"[etf] 历史不足（{len(hist)} < {lookback}+1）")
        return {}, cache, {}

    trend = {}
    for sym in codes:
        if sym not in hist.columns:
            continue
        s = hist[sym].dropna()
        if len(s) < lookback + 1:
            continue
        trend[sym] = float(s.iloc[-1] / s.iloc[-lookback - 1] - 1)

    positive = [c for c, r in trend.items() if r > 0]
    targets = {c: 1.0 / len(positive) for c in positive} if positive else {}
    return targets, cache, trend


def is_rebalance_day(trading_dates, as_of) -> bool:
    """月度调仓：as_of 是其所在月在 trading_dates 中的首个交易日。"""
    ts = pd.DatetimeIndex(trading_dates)
    prior = ts[ts < pd.Timestamp(as_of)]
    return len(prior) == 0 or prior[-1].month != pd.Timestamp(as_of).month


def build_etf_orders(targets: dict, current_weights: dict, nav: float,
                     band: float = 0.10):
    """目标权重 vs 当前权重 → 调仓订单（金额制；带宽外才调，降低换手）。

    权重差超过 band×目标权重的资产才交易；移出目标的全部卖出。
    ETF sleeve 唯一订单生成实现（与股票 build_orders 分离——资产数、
    权重语义、费率不同，共用反而引入适配 bug）。"""
    from live.broker import Order
    orders = []
    tset = set(targets)
    for s, w in current_weights.items():
        if s not in tset and w > 0.005:
            orders.append(Order(s, 'SELL', 0.0, amount=w * nav, note='移出(负趋势)'))
    for s, w in targets.items():
        cw = current_weights.get(s, 0.0)
        if abs(w - cw) > max(band * w, 0.01):
            if w > cw:
                orders.append(Order(s, 'BUY', w, amount=(w - cw) * nav,
                                    note='趋势建仓/加仓'))
            else:
                orders.append(Order(s, 'SELL', w, amount=(cw - w) * nav,
                                    note='趋势减仓'))
    return orders
