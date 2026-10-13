#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
日成交额计算（回测与生产共用，单一口径）

优先使用 amount 列（数据层归一化为元：DB 千元×1000 / 客户端 wsd amt=元）；
amount 缺失时退化为 close × volume（volume 归一化为股）。
禁止在消费端再自行 close×volume（曾因 DB 层 volume 单位为手而低估 100 倍）。
"""

from typing import Optional

import pandas as pd


def daily_turnover(df: pd.DataFrame) -> Optional[pd.Series]:
    """逐日成交额（元）。df 需含 close/volume，可选 amount。"""
    if df is None or df.empty:
        return None
    if 'amount' in df.columns:
        amt = pd.to_numeric(df['amount'], errors='coerce')
        if amt.notna().any():
            filled = amt.where(amt > 0)
            if filled.notna().all():
                return filled
            # 部分缺失：缺失日用 close×volume 补
            fallback = pd.to_numeric(df['close'], errors='coerce') * \
                pd.to_numeric(df['volume'], errors='coerce')
            return filled.fillna(fallback)
    return pd.to_numeric(df['close'], errors='coerce') * \
        pd.to_numeric(df['volume'], errors='coerce')


def avg_daily_turnover(df: pd.DataFrame, window: int = 20) -> Optional[float]:
    """近 window 日均成交额（元），数据不足返回 None。"""
    amt = daily_turnover(df)
    if amt is None:
        return None
    tail = amt.dropna().tail(window)
    if len(tail) < window:
        return None
    return float(tail.mean())
