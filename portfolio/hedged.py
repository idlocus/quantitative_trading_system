#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
市场中性模拟 —— β 对冲 + 贴水成本

对冲收益（日频）：r_hedged = r_long − β_t·r_bench − cost_daily
- β_t：组合收益对基准的滚动回归（trailing 窗口，时点正确）
- cost_daily：股指期货贴水对冲成本（年化，默认6%——IF/IC历史中枢）

简化声明（V1）：保证金占用与资金利用率忽略；期货以指数现货近似
（V2 接 CFutureEODPrices 主力连续合约与移仓）。
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class HedgedResult:
    nav: pd.Series               # 对冲后净值
    beta_series: pd.Series       # 滚动 β
    total_return: float
    annual_return: float
    sharpe: float
    max_drawdown: float
    market_correlation: float    # 对冲后与基准的相关（应≈0）
    hedged_cost_total: float     # 对冲成本累计

    def summary(self) -> str:
        return (f"中性组合: 总收益{self.total_return:+.1%} 年化{self.annual_return:+.1%} "
                f"夏普{self.sharpe:.2f} 回撤{self.max_drawdown:.1%} | "
                f"市场相关{self.market_correlation:+.2f} 对冲成本{self.hedged_cost_total:.1%}")


def beta_hedge(equity: pd.Series, bench_close: pd.Series,
               beta_window: int = 60,
               cost_annual: float = 0.06) -> HedgedResult:
    """
    构造 β 对冲后的中性净值

    Args:
        equity: 多头组合净值序列
        bench_close: 基准收盘序列（对齐日期）
        beta_window: β 估计滚动窗口（trailing，时点正确）
        cost_annual: 对冲成本（年化贴水+交易，默认6%）
    """
    bench = bench_close.reindex(equity.index).ffill()
    r_p = equity.pct_change().fillna(0.0)
    r_b = bench.pct_change().fillna(0.0)
    common = r_p.index
    r_p, r_b = r_p.loc[common], r_b.loc[common]

    # 滚动 β（trailing，前 beta_window 日用 expanding）
    cov = r_p.rolling(beta_window, min_periods=20).cov(r_b)
    var = r_b.rolling(beta_window, min_periods=20).var()
    beta = (cov / var).fillna(0.0).clip(0.0, 2.0)

    cost_daily = cost_annual / 252
    r_hedged = r_p - beta * r_b - cost_daily
    nav = (1 + r_hedged).cumprod()

    n = len(nav)
    total = float(nav.iloc[-1] - 1)
    annual = float(nav.iloc[-1] ** (252 / max(n, 1)) - 1)
    rets = r_hedged.iloc[1:]
    sharpe = float(rets.mean() / rets.std() * np.sqrt(252)) if rets.std() > 0 else 0.0
    dd = float((nav / nav.cummax() - 1).min())
    # 市场相关性（对冲后 vs 基准）
    valid = (r_hedged != 0) & r_b.notna()
    corr = float(np.corrcoef(r_hedged[valid].iloc[1:], r_b[valid].iloc[1:])[0, 1]) \
        if valid.sum() > 30 else float('nan')

    return HedgedResult(nav=nav, beta_series=beta, total_return=total,
                        annual_return=annual, sharpe=sharpe,
                        max_drawdown=dd, market_correlation=corr,
                        hedged_cost_total=float(cost_daily * n))
