#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
成交量指标
包含: OBV, Accumulation/Distribution, CMF, MFI
"""

import pandas as pd
import numpy as np


def OBV(data, column='close'):
    """
    OBV (On-Balance Volume) - 能量潮

    公式:
    - 上涨日: OBV = OBV_prev + Volume
    - 下跌日: OBV = OBV_prev - Volume
    - 持平日: OBV = OBV_prev

    信号:
    - OBV创新高: 上涨趋势确认
    - OBV创新低: 下跌趋势确认
    - 价格涨但OBV跌: 潜在顶部
    - 价格跌但OBV涨: 潜在底部
    - OBV与价格背离
    """
    close = data[column]
    volume = data['volume']

    obv = pd.Series(index=data.index, dtype=float)
    obv.iloc[0] = volume.iloc[0]

    for i in range(1, len(data)):
        if close.iloc[i] > close.iloc[i-1]:
            obv.iloc[i] = obv.iloc[i-1] + volume.iloc[i]
        elif close.iloc[i] < close.iloc[i-1]:
            obv.iloc[i] = obv.iloc[i-1] - volume.iloc[i]
        else:
            obv.iloc[i] = obv.iloc[i-1]

    return pd.Series(obv, name='obv')


def AccumulationDistribution(data):
    """
    A/D (Accumulation/Distribution Line) - 累积/派发线

    公式:
    MFV = ((Close - Low) - (High - Close)) / (High - Low) * Volume
    A/D = 累计MFV

    信号:
    - A/D上升: 主力累积
    - A/D下降: 主力派发
    - 价格与A/D背离
    """
    high = data['high']
    low = data['low']
    close = data['close']
    volume = data['volume']

    mf_multiplier = ((close - low) - (high - close)) / (high - low)
    mf_multiplier = mf_multiplier.fillna(0)
    mf_volume = mf_multiplier * volume

    ad_line = mf_volume.cumsum()

    return pd.Series(ad_line, name='ad_line')


def CMF(data, period=20):
    """
    CMF (Chaikin Money Flow) - 蔡金资金流

    公式:
    CMF = N日累计MFV / N日累计成交量

    信号:
    - CMF > 0: 资金流入
    - CMF < 0: 资金流出
    - CMF穿越0: 趋势变化信号
    """
    high = data['high']
    low = data['low']
    close = data['close']
    volume = data['volume']

    mf_multiplier = ((close - low) - (high - close)) / (high - low)
    mf_multiplier = mf_multiplier.fillna(0)
    mf_volume = mf_multiplier * volume

    cmf = mf_volume.rolling(window=period).sum() / volume.rolling(window=period).sum()

    return pd.Series(cmf, name='cmf')


def MFI(data, period=14):
    """
    MFI (Money Flow Index) - 资金流量指标

    公式:
    TP = (High + Low + Close) / 3
    Raw Money Flow = TP * Volume
    Money Flow Ratio = Positive MF / Negative MF
    MFI = 100 - (100 / (1 + Money Flow Ratio))

    信号:
    - MFI > 80: 超买
    - MFI < 20: 超卖
    - MFI与价格背离
    """
    high = data['high']
    low = data['low']
    close = data['close']
    volume = data['volume']

    tp = (high + low + close) / 3
    raw_mf = tp * volume

    # 计算正向和负向资金流
    positive_mf = pd.Series(index=data.index, dtype=float)
    negative_mf = pd.Series(index=data.index, dtype=float)

    for i in range(1, len(data)):
        if tp.iloc[i] > tp.iloc[i-1]:
            positive_mf.iloc[i] = raw_mf.iloc[i]
            negative_mf.iloc[i] = 0
        elif tp.iloc[i] < tp.iloc[i-1]:
            positive_mf.iloc[i] = 0
            negative_mf.iloc[i] = raw_mf.iloc[i]
        else:
            positive_mf.iloc[i] = 0
            negative_mf.iloc[i] = 0

    # 计算资金流比率和MFI
    mf_ratio = positive_mf.rolling(window=period).sum() / negative_mf.rolling(window=period).sum()
    mfi = 100 - (100 / (1 + mf_ratio))

    return pd.Series(mfi, name='mfi')


def VWAP(data):
    """
    VWAP (Volume Weighted Average Price) - 成交量加权平均价

    公式:
    VWAP = 累计(Price * Volume) / 累计(Volume)

    用途:
    - 机构投资者参考基准
    - 价格在VWAP上方: 偏多
    - 价格在VWAP下方: 偏空
    """
    tp = (data['high'] + data['low'] + data['close']) / 3
    cum_vol = (tp * data['volume']).cumsum() / data['volume'].cumsum()

    return pd.Series(cum_vol, name='vwap')
