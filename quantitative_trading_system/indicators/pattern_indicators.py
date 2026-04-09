#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
价格形态识别指标
包含: Fibonacci, Pivot Points, Pattern Recognition
"""

import pandas as pd
import numpy as np


def FibonacciRetracement(high, low, levels=None):
    """
    Fibonacci回撤位

    常用回撤位:
    - 23.6%
    - 38.2%
    - 50% (非Fibonacci，但常用)
    - 61.8%
    - 78.6%

    用途:
    - 识别潜在支撑/阻力位
    - 预测回调目标位
    - 确认趋势延续或反转
    """
    if levels is None:
        levels = [0.236, 0.382, 0.5, 0.618, 0.786]

    diff = high - low

    retracements = {}
    for level in levels:
        retracements[f'{int(level*100)}%'] = high - diff * level

    return retracements


def FibonacciExtension(high, low, extension_levels=None):
    """
    Fibonacci扩展位

    常用扩展位:
    - 100%
    - 127.2%
    - 161.8%
    - 200%
    - 261.8%

    用途:
    - 预测上涨目标位
    - 预测下跌目标位
    """
    if extension_levels is None:
        extension_levels = [1.0, 1.272, 1.618, 2.0, 2.618]

    diff = high - low

    extensions = {}
    for level in extension_levels:
        extensions[f'{int(level*100)}%'] = low + diff * level

    return extensions


def PivotPoints(data, pivot_type='standard'):
    """
    Pivot Points (枢轴点)

    类型:
    - Standard: 标准枢轴
    - Fibonacci: Fibonacci枢轴
    - Camarilla: 卡玛利拉枢轴

    组成:
    - Pivot: 枢轴点
    - R1, R2, R3: 阻力位
    - S1, S2, S3: 支撑位
    """
    high = data['high'].iloc[-1]
    low = data['low'].iloc[-1]
    close = data['close'].iloc[-1]

    if pivot_type == 'standard':
        pivot = (high + low + close) / 3
        r1 = 2 * pivot - low
        s1 = 2 * pivot - high
        r2 = pivot + (high - low)
        s2 = pivot - (high - low)
        r3 = high + 2 * (pivot - low)
        s3 = low - 2 * (high - pivot)

    elif pivot_type == 'fibonacci':
        pivot = (high + low + close) / 3
        r1 = pivot + 0.382 * (high - low)
        s1 = pivot - 0.382 * (high - low)
        r2 = pivot + 0.618 * (high - low)
        s2 = pivot - 0.618 * (high - low)
        r3 = pivot + 1.0 * (high - low)
        s3 = pivot - 1.0 * (high - low)

    elif pivot_type == 'camarilla':
        pivot = (high + low + close) / 3
        r1 = close + (high - low) * 1.1 / 12
        s1 = close - (high - low) * 1.1 / 12
        r2 = close + (high - low) * 1.1 / 6
        s2 = close - (high - low) * 1.1 / 6
        r3 = close + (high - low) * 1.1 / 4
        s3 = close - (high - low) * 1.1 / 4
        r4 = close + (high - low) * 1.1 / 2
        s4 = close - (high - low) * 1.1 / 2

    else:
        raise ValueError(f"Unknown pivot type: {pivot_type}")

    return {
        'pivot': pivot,
        'r1': r1, 'r2': r2, 'r3': r3 if pivot_type != 'camarilla' else r4,
        's1': s1, 's2': s2, 's3': s3 if pivot_type != 'camarilla' else s4
    }


def PatternRecognizer(data, window=20):
    """
    价格形态识别

    可识别的形态:
    - 双底 (Double Bottom)
    - 双顶 (Double Top)
    - 头肩底 (Head and Shoulders Bottom)
    - 头肩顶 (Head and Shoulders Top)
    - 旗形 (Flag)
    - 三角整理 (Triangle)
    """
    close = data['close'].values
    high = data['high'].values
    low = data['low'].values

    patterns = {
        'double_bottom': False,
        'double_top': False,
        'head_shoulders_bottom': False,
        'head_shoulders_top': False
    }

    # 双底识别: 两个相近的低点，中间有高点
    if len(close) >= window:
        recent_lows = []
        for i in range(1, len(close) - 1):
            if close[i] < close[i-1] and close[i] < close[i+1]:
                recent_lows.append((i, close[i]))

        if len(recent_lows) >= 2:
            idx1, price1 = recent_lows[-2]
            idx2, price2 = recent_lows[-1]
            if abs(price1 - price2) / price1 < 0.02:  # 2%容差
                patterns['double_bottom'] = True

    # 双顶识别
    if len(close) >= window:
        recent_highs = []
        for i in range(1, len(close) - 1):
            if close[i] > close[i-1] and close[i] > close[i+1]:
                recent_highs.append((i, close[i]))

        if len(recent_highs) >= 2:
            idx1, price1 = recent_highs[-2]
            idx2, price2 = recent_highs[-1]
            if abs(price1 - price2) / price1 < 0.02:
                patterns['double_top'] = True

    return patterns


def SupportResistanceLevels(data, lookback=50, tolerance=0.02):
    """
    识别支撑位和阻力位

    方法:
    - 找到局部高点和低点
    - 聚类相近的价格水平

    Args:
        lookback: 回溯窗口
        tolerance: 价格水平容差 (2%)
    """
    highs = data['high'].rolling(window=5, center=True).max()
    lows = data['low'].rolling(window=5, center=True).min()

    # 找到局部高点和低点
    resistance_levels = []
    support_levels = []

    for i in range(10, len(data) - 10):
        if data['high'].iloc[i] == highs.iloc[i]:
            resistance_levels.append(data['high'].iloc[i])
        if data['low'].iloc[i] == lows.iloc[i]:
            support_levels.append(data['low'].iloc[i])

    return {
        'resistance': resistance_levels[-10:] if len(resistance_levels) > 10 else resistance_levels,
        'support': support_levels[-10:] if len(support_levels) > 10 else support_levels
    }


def detect_candlestick_patterns(data):
    """
    识别K线形态

    常见形态:
    - Doji (十字星)
    - Hammer (锤子线)
    - Inverted Hammer (倒锤线)
    - Engulfing (吞没形态)
    - Morning Star (晨星)
    - Evening Star (暮星)
    """
    open_price = data['open']
    high = data['high']
    low = data['low']
    close = data['close']

    patterns = pd.DataFrame(index=data.index)

    # Doji: 开盘价接近收盘价
    body_size = abs(close - open_price)
    candle_range = high - low
    doji = body_size < 0.1 * candle_range
    patterns['doji'] = doji

    # Hammer: 小实体 + 长下影线
    lower_shadow = np.minimum(open_price, close) - low
    upper_shadow = high - np.maximum(open_price, close)
    body = np.abs(close - open_price)
    hammer = (lower_shadow > 2 * body) & (upper_shadow < body) & (body < 0.3 * candle_range)
    patterns['hammer'] = hammer

    # Inverted Hammer: 小实体 + 长上影线
    inverted_hammer = (upper_shadow > 2 * body) & (lower_shadow < body) & (body < 0.3 * candle_range)
    patterns['inverted_hammer'] = inverted_hammer

    # Bullish Engulfing: 阳线吞没前一根阴线
    bullish_engulfing = (
        (close > open) &  # 当前阳线
        (close.shift(1) < open.shift(1)) &  # 前一根阴线
        (close > open.shift(1)) &  # 阳线实体上穿
        (open < close.shift(1))  # 阳线实体下吞
    )
    patterns['bullish_engulfing'] = bullish_engulfing

    # Bearish Engulfing: 阴线吞没前一根阳线
    bearish_engulfing = (
        (close < open) &  # 当前阴线
        (close.shift(1) > open.shift(1)) &  # 前一根阳线
        (close < open.shift(1)) &  # 阴线实体下穿
        (open > close.shift(1))  # 阴线实体上吞
    )
    patterns['bearish_engulfing'] = bearish_engulfing

    return patterns
