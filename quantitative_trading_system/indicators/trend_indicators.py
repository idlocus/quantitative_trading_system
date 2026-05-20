#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
趋势指标
包含: SMA, EMA, MACD, DMI+ADX, IchimokuCloud, GoldenCross, TMAC
"""

import pandas as pd
import numpy as np

from .registry import IndicatorRegistry


@IndicatorRegistry.register('sma', category='trend')
def SMA(data, period=20, column='close'):
    """
    简单移动平均线 (Simple Moving Average)

    公式: SMA = (C1 + C2 + ... + Cn) / n

    用途:
    - 识别趋势方向
    - 动态支撑/阻力位
    - 金叉/死叉信号
    """
    return data[column].rolling(window=period).mean()


@IndicatorRegistry.register('ema', category='trend')
def EMA(data, period=20, column='close'):
    """
    指数移动平均线 (Exponential Moving Average)

    公式: EMA = (Close - EMA_prev) * k + EMA_prev
          其中 k = 2 / (period + 1)

    用途:
    - 比SMA对价格变化更敏感
    - 用于MACD计算
    - 短期趋势识别
    """
    return data[column].ewm(span=period, adjust=False).mean()


@IndicatorRegistry.register('macd', category='trend')
def MACD(data, fast_period=12, slow_period=26, signal_period=9, column='close'):
    """
    MACD (Moving Average Convergence Divergence)

    组成:
    - MACD线 = Fast EMA - Slow EMA
    - Signal线 = MACD的EMA
    - Histogram = MACD线 - Signal线

    信号:
    - 金叉 (MACD > Signal): 买入
    - 死叉 (MACD < Signal): 卖出
    - 底背离: 买入
    - 顶背离: 卖出
    """
    fast_ema = EMA(data, fast_period, column)
    slow_ema = EMA(data, slow_period, column)

    macd_line = fast_ema - slow_ema
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    histogram = macd_line - signal_line

    return pd.DataFrame({
        'macd': macd_line,
        'signal': signal_line,
        'histogram': histogram
    })


@IndicatorRegistry.register('dmi', category='trend')
def DMI_ADX(data, period=14):
    """
    DMI (Directional Movement Index) + ADX (Average Directional Index)

    组成:
    - +DI: 正向指标
    - -DI: 负向指标
    - ADX: 趋势强度指标

    信号:
    - +DI上穿-DI: 买入
    - -DI上穿+DI: 卖出
    - ADX > 25: 趋势市场
    - ADX < 20: 震荡市场
    """
    high = data['high']
    low = data['low']
    close = data['close']

    # 计算True Range
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # 计算Directional Movement
    up_move = high - high.shift()
    down_move = low.shift() - low

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)

    # 计算平滑值
    atr = tr.rolling(window=period).mean()
    plus_dm_smooth = pd.Series(plus_dm).rolling(window=period).mean()
    minus_dm_smooth = pd.Series(minus_dm).rolling(window=period).mean()

    # 计算DI
    plus_di = 100 * plus_dm_smooth / atr
    minus_di = 100 * minus_dm_smooth / atr

    # 计算DX
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)

    # 计算ADX
    adx = dx.rolling(window=period).mean()

    return pd.DataFrame({
        'plus_di': plus_di,
        'minus_di': minus_di,
        'adx': adx
    })


def IchimokuCloud(data, tenkan_period=9, kijun_period=26, senkou_b_period=52, displacement=26):
    """
    Ichimoku Cloud (一目均衡表)

    组成:
    - Tenkan-sen (转换线): (9日高 + 9日低) / 2
    - Kijun-sen (基准线): (26日高 + 26日低) / 2
    - Senkou Span A (先行线A): (转换线 + 基准线) / 2
    - Senkou Span B (先行线B): (52日高 + 52日低) / 2
    - Chikou Span (延迟线): 当前收盘价向后移26天

    云层:
    - 价格上涨时云层为绿色
    - 价格下跌时云层为红色

    信号:
    - 转换线 > 基准线: 买入
    - 转换线 < 基准线: 卖出
    - 价格在云层上方: 强势
    - 价格在云层下方: 弱势
    """
    high = data['high']
    low = data['low']

    # Tenkan-sen (Conversion Line)
    tenkan_sen = (high.rolling(window=tenkan_period).max() +
                  low.rolling(window=tenkan_period).min()) / 2

    # Kijun-sen (Base Line)
    kijun_sen = (high.rolling(window=kijun_period).max() +
                 low.rolling(window=kijun_period).min()) / 2

    # Senkou Span A (Leading Span A)
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(displacement)

    # Senkou Span B (Leading Span B)
    senkou_span_b = ((high.rolling(window=senkou_b_period).max() +
                      low.rolling(window=senkou_b_period).min()) / 2).shift(displacement)

    # Chikou Span (Lagging Span)
    chikou_span = data['close'].shift(-displacement)

    return pd.DataFrame({
        'tenkan_sen': tenkan_sen,
        'kijun_sen': kijun_sen,
        'senkou_span_a': senkou_span_a,
        'senkou_span_b': senkou_span_b,
        'chikou_span': chikou_span
    })


def GoldenCrossDeathCross(data, short_period=50, long_period=200, column='close'):
    """
    Golden Cross & Death Cross

    定义:
    - Golden Cross: 短期MA上穿长期MA (买入信号)
    - Death Cross: 短期MA下穿长期MA (卖出信号)

    用途:
    - 确认长期趋势变化
    - 通常用于50/200 SMA组合
    """
    short_ma = SMA(data, short_period, column)
    long_ma = SMA(data, long_period, column)

    # 生成信号
    signal = pd.Series(0, index=data.index)
    signal[short_ma > long_ma] = 1   # Golden Cross
    signal[short_ma < long_ma] = -1  # Death Cross

    # 检测交叉点
    crossover = signal.diff()
    buy_signal = crossover == 2   # 从-1变到1
    sell_signal = crossover == -2  # 从1变到-1

    return pd.DataFrame({
        'short_ma': short_ma,
        'long_ma': long_ma,
        'signal': signal,
        'crossover': crossover,
        'buy_signal': buy_signal,
        'sell_signal': sell_signal
    })


def TripleMovingAverageCross(data, short_period=5, medium_period=10, long_period=20, column='close'):
    """
    Triple Moving Average Cross (TMAC)

    策略:
    - 当短期MA > 中期MA > 长期MA时买入
    - 当短期MA < 中期MA < 长期MA时卖出

    用途:
    - 过滤假信号
    - 确认趋势一致性
    """
    short_ma = SMA(data, short_period, column)
    medium_ma = SMA(data, medium_period, column)
    long_ma = SMA(data, long_period, column)

    # 生成信号
    signal = pd.Series(0, index=data.index)
    signal[(short_ma > medium_ma) & (medium_ma > long_ma)] = 1   # 三线多头
    signal[(short_ma < medium_ma) & (medium_ma < long_ma)] = -1  # 三线空头

    # 检测交叉点
    crossover = signal.diff()
    buy_signal = crossover == 2
    sell_signal = crossover == -2

    return pd.DataFrame({
        'short_ma': short_ma,
        'medium_ma': medium_ma,
        'long_ma': long_ma,
        'signal': signal,
        'crossover': crossover,
        'buy_signal': buy_signal,
        'sell_signal': sell_signal
    })
