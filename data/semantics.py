#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
口径收敛层 —— 所有"怎么算"的唯一实现

此前项目中的同一概念有多处不同实现（如成交额在 signals.py 和 backtester.py
各算一遍），每次修一个漏一个。本模块收敛为唯一函数，所有调用方经由这里。

新增"怎么算"的逻辑时：
1. 先检查是否已有现成函数
2. 如果没有，加到本模块并在调用方 import
3. 禁止在业务代码中内联计算（哪怕只有一行）
"""

from typing import Optional

import numpy as np
import pandas as pd


# ================= 价格基准 =================

def get_price_column(df: pd.DataFrame, basis: str = 'raw') -> str:
    """
    根据声明的基准返回正确的列名。

    Args:
        basis: 'raw'=不复权（实盘成交/成本/止损/涨跌停）
               'adjusted'=前复权（回测净值/收益/因子前瞻收益）
    """
    if basis == 'raw':
        return 'close'         # 不复权列名 = close
    elif basis == 'adjusted':
        if 'adjclose' in df.columns:
            return 'adjclose'
        return 'close'          # 退化：无复权列时用不复权（客户端源）
    else:
        raise ValueError(f"未知价格基准 '{basis}'（可选: raw / adjusted）")


def extract_prices(cache: dict, symbols: list, as_of,
                   basis: str = 'raw') -> dict:
    """
    从K线缓存提取价格字典。

    Args:
        basis: 'raw'=实盘撮合 | 'adjusted'=回测净值
    """
    col = None
    prices = {}
    for sym in symbols:
        df = cache.get(sym)
        if df is None or df.empty:
            continue
        if col is None:
            col = get_price_column(df, basis)
        d = df[df['date'] <= pd.Timestamp(as_of)] if 'date' in df.columns else df
        if d.empty:
            continue
        px = float(d[col].iloc[-1]) if col in d.columns else 0.0
        if px > 0:
            prices[sym] = px
    return prices


# ================= 成交额 =================

def daily_turnover(df: pd.DataFrame) -> Optional[pd.Series]:
    """
    日成交额（元）——amount列优先，退化 close×volume。

    唯一实现：所有需要成交额的地方（流动性过滤/参与率/冲击成本）
    必须经由本函数，禁止自行 close×volume。
    """
    if df is None or df.empty:
        return None
    if 'amount' in df.columns:
        amt = pd.to_numeric(df['amount'], errors='coerce')
        if amt.notna().any():
            filled = amt.where(amt > 0)
            if filled.notna().all():
                return filled
            fallback = (pd.to_numeric(df['close'], errors='coerce') *
                        pd.to_numeric(df['volume'], errors='coerce'))
            return filled.fillna(fallback)
    return (pd.to_numeric(df['close'], errors='coerce') *
            pd.to_numeric(df['volume'], errors='coerce'))


def avg_daily_turnover(df: pd.DataFrame, window: int = 20) -> Optional[float]:
    """近 window 日均成交额（元），数据不足返回 None。"""
    amt = daily_turnover(df)
    if amt is None:
        return None
    tail = amt.dropna().tail(window)
    if len(tail) < window:
        return None
    return float(tail.mean())


def turnover_matrix(close_mx: pd.DataFrame, volume_mx: pd.DataFrame,
                    amount_mx: pd.DataFrame = None) -> pd.DataFrame:
    """
    成交额矩阵（元）：amount优先，退化 close×volume（矩阵级操作）。

    唯一实现：backtester 和 daily_replay 的 20日均成交额计算共用。
    """
    fallback = close_mx * volume_mx
    if amount_mx is not None and not amount_mx.empty:
        return amount_mx.where(amount_mx > 0).fillna(fallback)
    return fallback


# ================= 信号质量 =================

def factor_coverage(mx: pd.DataFrame, date=None) -> float:
    """
    因子矩阵单日覆盖率（0-1）。

    唯一实现：信号质量闸门、日报、实验脚本共用。
    """
    if mx is None or mx.empty:
        return 0.0
    row = mx.loc[date] if date is not None else mx.iloc[-1]
    return float(row.notna().sum()) / max(mx.shape[1], 1)


def should_halt_signal(coverage: float, n_targets: int,
                       min_coverage: float = 0.30,
                       min_targets: int = 20) -> tuple:
    """
    信号质量判定：是否应进入数据停机。

    Returns:
        (should_halt: bool, reason: str)
    """
    if coverage < min_coverage:
        return True, f"因子覆盖率{coverage:.0%}<{min_coverage:.0%}（数据管道故障）"
    if n_targets < min_targets:
        return True, f"目标持仓{n_targets}只<{min_targets}只（数据链路异常）"
    return False, ""


# ================= 净值与回撤 =================

def drawdown_from_series(nav: pd.Series) -> float:
    """
    从净值序列计算最大回撤。

    唯一实现：RiskGate、回测器、报告共用。
    """
    if nav is None or len(nav) < 2:
        return 0.0
    return float((nav / nav.cummax() - 1).min())


def current_drawdown(nav_history: list) -> tuple:
    """
    当前回撤（基于账本净值序列，排除存取款影响）。

    Returns:
        (drawdown: float, peak_nav: float)
    """
    if not nav_history:
        return 0.0, 0.0
    navs = [h['nav'] for h in nav_history]
    peak, last = max(navs), navs[-1]
    if peak <= 0:
        return 0.0, peak
    return 1 - last / peak, peak


# ================= 仓位与权重 =================

def planned_exposure(n_targets: int, top_n: int) -> tuple:
    """
    计划仓位与现金比例。

    Returns:
        (exposure: float, cash_ratio: float)
    """
    target_w = 1.0 / max(top_n, 1)
    exposure = min(1.0, n_targets * target_w)
    return exposure, 1.0 - exposure


def actual_cash_ratio(cash: float, positions: dict, prices: dict) -> float:
    """
    当前账户现金比例（用最后有效价兜底）。
    """
    mv = sum(q * prices.get(s, 0) for s, q in positions.items())
    total = cash + mv
    if total <= 0:
        return 1.0
    return cash / total
