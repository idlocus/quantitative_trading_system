#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
波动率指标
包含: Bollinger Bands, ATR, Standard Deviation
"""

import pandas as pd
import numpy as np

from .registry import IndicatorRegistry


@IndicatorRegistry.register('bb', category='volatility')
def BollingerBands(data, period=20, std_dev=2, column='close'):
    """
    Bollinger Bands (布林带)

    组成:
    - 中轨: N日简单移动平均线
    - 上轨: 中轨 + 2倍标准差
    - 下轨: 中轨 - 2倍标准差

    信号:
    - 价格触及上轨: 超买，可能回调
    - 价格触及下轨: 超卖，可能反弹
    - 布林带收缩: 波动率降低，预示突破
    - 布林带扩张: 波动率增加，趋势可能加速
    """
    middle = data[column].rolling(window=period).mean()
    std = data[column].rolling(window=period).std()

    upper = middle + (std_dev * std)
    lower = middle - (std_dev * std)

    bandwidth = (upper - lower) / middle * 100
    percent_b = (data[column] - lower) / (upper - lower)

    return pd.DataFrame({
        'upper': upper,
        'middle': middle,
        'lower': lower,
        'bandwidth': bandwidth,
        'percent_b': percent_b
    })


@IndicatorRegistry.register('atr', category='volatility')
def ATR(data, period=14):
    """
    ATR (Average True Range) - 平均真实波幅

    True Range计算:
    TR = max(High - Low, |High - PrevClose|, |Low - PrevClose|)

    用途:
    - 衡量市场波动性
    - 设置止损位
    - 趋势确认 (趋势强劲时ATR上升)
    """
    high = data['high']
    low = data['low']
    close = data['close']

    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()

    return pd.Series(atr, name='atr')


def StandardDeviation(data, period=20, column='close'):
    """
    Standard Deviation (标准差)

    用途:
    - 衡量价格波动程度
    - Bollinger Bands的基础
    - 确认趋势强度
    """
    std = data[column].rolling(window=period).std()

    return pd.Series(std, name='std')


def KeltnerChannels(data, ema_period=20, atr_period=10, multiplier=2):
    """
    Keltner Channels (肯特纳通道)

    组成:
    - 中轨: EMA
    - 上轨: EMA + 2 * ATR
    - 下轨: EMA - 2 * ATR

    信号:
    - 价格上穿上轨: 趋势走强
    - 价格下穿下轨: 趋势走弱
    """
    ema = data['close'].ewm(span=ema_period, adjust=False).mean()
    atr = ATR(data, atr_period)

    upper = ema + multiplier * atr
    lower = ema - multiplier * atr

    return pd.DataFrame({
        'upper': upper,
        'middle': ema,
        'lower': lower
    })


def DonchianChannels(data, period=20):
    """
    Donchian Channels (唐奇安通道)

    组成:
    - 上轨: N日最高价
    - 下轨: N日最低价
    - 中轨: (上轨 + 下轨) / 2
    """
    upper = data['high'].rolling(window=period).max()
    lower = data['low'].rolling(window=period).min()
    middle = (upper + lower) / 2

    return pd.DataFrame({
        'upper': upper,
        'middle': middle,
        'lower': lower
    })
