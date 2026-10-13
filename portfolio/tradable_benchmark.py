#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
可交易等权对照 —— 与策略完全同约束的真实回测（非统计参照）

与 close_mx.pct_change().mean(axis=1) 的统计口径（无过滤/无成本/无再平衡）不同，
本对照走完整 PortfolioBacktester：上市天数/流动性/ST 过滤、板块涨跌停、申报单位、
费用+市场冲击、缓冲带、现金底线、定期再平衡——全部与策略一致，仅"选股能力"为零
（常数分数 + top=全池）。

用法（实验工件）：experiments/mainchain_v5.py 每年代生成 equaltradable_v5_*；
单次交互：python run_portfolio_backtest.py --all-equal。
"""

from typing import Dict

import pandas as pd

from .backtester import PortfolioBacktester, PortfolioConfig

BENCH_INDEXES = ('000300.SH', '000905.SH', '000852.SH')


def all_equal_scores(close_index: pd.DatetimeIndex, symbols) -> pd.DataFrame:
    """常数分数矩阵：全标的同分 → top=全池时不产生任何排序效应"""
    cols = [s for s in symbols if s not in BENCH_INDEXES]
    return pd.DataFrame(1.0, index=pd.DatetimeIndex(close_index), columns=cols)


def tradable_equal_weight(cache: Dict[str, pd.DataFrame],
                          config: PortfolioConfig = None,
                          start_date: str = None,
                          end_date: str = None):
    """
    运行可交易等权对照，返回 (backtester, result, capital_scale)。

    资金等比放大：全池等权时单票分配 = 资金/N，若低于最低申报（主板100股≈千元级）
    会导致对照无法建仓（曾出现成本0%、持仓3034只的退化结果）。按 ceil(N/100) 放大
    初始资金使单票分配≥可交易规模——收益率/夏普/回撤规模无关，与策略仍可比。

    Args:
        cache:   K线缓存（与策略回测同一份，保证同数据）
        config:  策略配置模板（top_n 将被覆盖为全池；其余约束原样继承）
    """
    cfg = config or PortfolioConfig()
    probe = PortfolioBacktester(cache, cfg)
    score = all_equal_scores(probe.close_mx.index, probe.close_mx.columns)
    capital_scale = max(1, -(-score.shape[1] // 100))     # ceil(N/100)
    ctrl_cfg = PortfolioConfig(
        top_n=max(score.shape[1], cfg.top_n),          # 全池：不做任何截断
        rebalance_days=cfg.rebalance_days,
        initial_capital=cfg.initial_capital * capital_scale,
        slippage=cfg.slippage,
        enforce_price_limits=cfg.enforce_price_limits,
        max_weight=cfg.max_weight,
        weight_method=cfg.weight_method,
        benchmark_symbol=cfg.benchmark_symbol,
        limit_tolerance=cfg.limit_tolerance,
        buffer_band=cfg.buffer_band,
        max_participation=cfg.max_participation,
        min_daily_turnover=cfg.min_daily_turnover,      # 流动性过滤同策略口径
        impact_coeff=cfg.impact_coeff,
        min_listing_days=cfg.min_listing_days,
        risk_limits=cfg.risk_limits,
        industry_map=cfg.industry_map,
        st_matrix=cfg.st_matrix,                        # ST 过滤同策略口径
    )
    bt = PortfolioBacktester(cache, ctrl_cfg)
    if end_date:
        score = score.loc[:pd.Timestamp(end_date)]   # run() 仅支持 start_date，尾部截断在此处理
    result = bt.run(score, start_date=start_date)
    return bt, result, capital_scale
