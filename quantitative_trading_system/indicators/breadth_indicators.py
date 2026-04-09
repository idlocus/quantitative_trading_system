#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
市场广度指标
包含: Advance-Decline Line, McClellan Oscillator
"""

import pandas as pd
import numpy as np


def AdvanceDeclineLine(advances, declines):
    """
    A/D Line (腾落线)

    公式:
    A/D = 累计(上涨股数 - 下跌股数)

    信号:
    - A/D创新高: 市场广度改善，上涨趋势健康
    - A/D创新低: 市场广度恶化，下跌趋势蔓延
    - 价格创新高但A/D未创新高: 潜在顶部
    - 价格创新低但A/D未创新低: 潜在底部
    - A/D与价格背离

    Args:
        advances: 上涨股票数量序列
        declines: 下跌股票数量序列
    """
    net_advances = pd.Series(advances - declines)
    ad_line = net_advances.cumsum()

    return pd.Series(ad_line, name='ad_line')


def McClellanOscillator(advances, declines, fast_period=19, slow_period=39):
    """
    McClellan Oscillator (麦克莱恩摆动指标)

    公式:
    OSC = 19日EMA(净上涨) - 39日EMA(净上涨)
    净上涨 = 上涨股数 - 下跌股数

    信号:
    - OSC > 0: 市场偏多
    - OSC < 0: 市场偏空
    - OSC穿越0: 趋势变化
    - OSC极度正/负值: 可能的反转信号
    """
    net_advances = pd.Series(advances - declines)

    # 计算快速和慢速EMA
    fast_ema = net_advances.ewm(span=fast_period, adjust=False).mean()
    slow_ema = net_advances.ewm(span=slow_period, adjust=False).mean()

    oscillator = fast_ema - slow_ema

    return pd.Series(oscillator, name='mcclellan_oscillator')


def McClellanSummationIndex(advances, declines):
    """
    McClellan Summation Index

    公式:
    Sum Index = 累计McClellan Oscillator

    用途:
    - 长期趋势确认
    - 与价格形成更大级别的背离
    """
    net_advances = pd.Series(advances - declines)
    fast_ema = net_advances.ewm(span=19, adjust=False).mean()
    slow_ema = net_advances.ewm(span=39, adjust=False).mean()
    oscillator = fast_ema - slow_ema

    sum_index = oscillator.cumsum()

    return pd.Series(sum_index, name='mcclellan_sum_index')


def ArmsIndex(advances, declines, volume_up, volume_down):
    """
    Arms Index (TRIN) - 阿姆氏指标

    公式:
    Arms Index = (上涨股数/下跌股数) / (上涨成交量/下跌成交量)

    信号:
    - Arms < 1: 偏多 (资金流入强势股)
    - Arms > 1: 偏空 (资金流入弱势股)
    - 极端值: 可能反转信号
    """
    ratio = (pd.Series(advances) / pd.Series(declines)) / \
            (pd.Series(volume_up) / pd.Series(volume_down))

    return pd.Series(ratio, name='arms_index')


def HighLowIndex(highs, lows, period=10):
    """
    High-Low Index - 新高新低指数

    公式:
    HLI = 100 * (N日内创新高的股票数) / (N日内创新高 + 创新低的股票数)

    信号:
    - HLI > 50: 上涨趋势中
    - HLI < 50: 下跌趋势中
    - HLI创历史新低: 熊市
    - HLI创历史新高: 牛市
    """
    # 简化版本：使用涨跌家数比
    highs = pd.Series(highs)
    lows = pd.Series(lows)

    hli = 100 * highs / (highs + lows)

    return pd.Series(hli, name='high_low_index')
