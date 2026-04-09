#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
动量指标模块
包含: RSI, Stochastic, ROC, CCI, RPS

所有指标都通过 @IndicatorRegistry.register 装饰器注册到指标注册表
"""

import pandas as pd
import numpy as np

from .registry import IndicatorRegistry


@IndicatorRegistry.register('rsi', category='momentum')
def RSI(data, period=14, column='close'):
    """
    RSI (Relative Strength Index) - 相对强弱指数

    公式:
    RSI = 100 - (100 / (1 + RS))
    其中 RS = 平均涨幅 / 平均跌幅

    信号:
    - RSI > 70: 超买
    - RSI < 30: 超卖
    - 底背离 (价格创新低, RSI更高): 买入
    - 顶背离 (价格创新高, RSI更低): 卖出
    """
    delta = data[column].diff()

    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()

    # 使用指数移动平均进行平滑 (Wilder平滑)
    for i in range(period, len(data)):
        avg_gain.iloc[i] = (avg_gain.iloc[i-1] * (period - 1) + gain.iloc[i]) / period
        avg_loss.iloc[i] = (avg_loss.iloc[i-1] * (period - 1) + loss.iloc[i]) / period

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    return pd.Series(rsi, name='rsi')


@IndicatorRegistry.register('stochastic', category='momentum')
def Stochastic(data, k_period=14, d_period=3, smooth_k=3):
    """
    Stochastic Oscillator (随机指标)

    公式:
    %K = 100 * (Close - LowestLow) / (HighestHigh - LowestLow)
    %D = %K的移动平均

    信号:
    - %K > 80: 超买
    - %K < 20: 超卖
    - %K上穿%D: 买入
    - %K下穿%D: 卖出
    """
    low_min = data['low'].rolling(window=k_period).min()
    high_max = data['high'].rolling(window=k_period).max()

    k = 100 * (data['close'] - low_min) / (high_max - low_min)
    k_smooth = k.rolling(window=smooth_k).mean()
    d = k_smooth.rolling(window=d_period).mean()

    return pd.DataFrame({
        'k': k,
        'k_smooth': k_smooth,
        'd': d
    })


@IndicatorRegistry.register('roc', category='momentum')
def ROC(data, period=12, column='close'):
    """
    ROC (Rate of Change) - 变化率指标

    公式:
    ROC = 100 * (Close - Close n periods ago) / Close n periods ago

    信号:
    - ROC > 0: 上涨动能
    - ROC < 0: 下跌动能
    - ROC从负转正: 买入
    - ROC从正转负: 卖出
    """
    roc = 100 * (data[column] - data[column].shift(period)) / data[column].shift(period)

    return pd.Series(roc, name='roc')


@IndicatorRegistry.register('cci', category='momentum')
def CCI(data, period=20):
    """
    CCI (Commodity Channel Index) - 商品通道指数

    公式:
    CCI = (Typical Price - SMA of TP) / (0.015 * Mean Deviation)
    其中 Typical Price = (High + Low + Close) / 3

    信号:
    - CCI > 100: 超买
    - CCI < -100: 超卖
    - CCI上穿100: 买入
    - CCI下穿-100: 卖出
    """
    tp = (data['high'] + data['low'] + data['close']) / 3
    sma_tp = tp.rolling(window=period).mean()

    # 计算平均偏差
    mean_deviation = tp.rolling(window=period).apply(
        lambda x: np.abs(x - x.mean()).mean(), raw=True
    )

    cci = (tp - sma_tp) / (0.015 * mean_deviation)

    return pd.Series(cci, name='cci')


def detect_divergence(price, indicator, lookback=20):
    """
    检测价格与指标之间的背离

    Args:
        price: 价格序列
        indicator: 指标序列
        lookback: 回溯窗口

    Returns:
        bullish_divergence: 底背离信号
        bearish_divergence: 顶背离信号
    """
    signals = pd.DataFrame(index=price.index)

    # 底背离: 价格创新低但指标没有创新低
    # 顶背离: 价格创新高但指标没有创新高

    price_low = price.rolling(window=lookback).min()
    price_high = price.rolling(window=lookback).max()
    ind_low = indicator.rolling(window=lookback).min()
    ind_high = indicator.rolling(window=lookback).max()

    # 检测局部低点和高点
    bullish_divergence = (
        (price == price_low) &
        (indicator > ind_low)
    )

    bearish_divergence = (
        (price == price_high) &
        (indicator < ind_high)
    )

    signals['bullish_divergence'] = bullish_divergence
    signals['bearish_divergence'] = bearish_divergence

    return signals


@IndicatorRegistry.register('rps', category='momentum')
def RPS(data, period=20, column='close'):
    """
    RPS (Relative Price Strength) - 相对价格强度

    原理: 衡量股票在一定时期内的相对表现。
    RPS值越高，表示该股票在同期内表现越强。

    基于欧奈尔CAN SLIM系统的RPS概念，
    计算N日内价格变化率在历史分布中的百分位排名

    信号:
    - RPS > 80: 强势股，可能继续跑赢
    - RPS < 20: 弱势股，可能继续跑输

    Returns:
        pd.Series: RPS值 (0-100)
    """
    # 计算N日价格变化率
    roc = 100 * (data[column] - data[column].shift(period)) / data[column].shift(period)

    # 计算滚动百分位排名
    # 使用expand窗口，从period开始逐渐增加
    rps = pd.Series(index=data.index, dtype=float)

    for i in range(period, len(data)):
        # 历史ROC序列
        hist_roc = roc.iloc[period:i+1].dropna()
        current_roc = roc.iloc[i]

        if len(hist_roc) > 0 and not np.isnan(current_roc):
            # 计算当前ROC超过多少比例的历史数据 (0-100)
            rank = (current_roc >= hist_roc).sum() / len(hist_roc) * 100
            rps.iloc[i] = rank
        else:
            rps.iloc[i] = 50  # 默认中性值

    return pd.Series(rps, name='rps')


def RPS_with_benchmark(data, benchmark, period=20, column='close'):
    """
    RPS with benchmark - 市场调整后的相对价格强度

    Args:
        data: DataFrame，包含个股价格
        benchmark: 基准指数数据 (Series)
        period: 计算周期
        column: 价格列名

    Returns:
        pd.Series: 市场调整后的RPS值 (0-100)
    """
    # 计算个股ROC
    stock_roc = 100 * (data[column] - data[column].shift(period)) / data[column].shift(period)

    # 计算基准ROC
    benchmark_roc = 100 * (benchmark - benchmark.shift(period)) / benchmark.shift(period)

    # 市场调整后的ROC
    adjusted_roc = stock_roc - benchmark_roc

    # 计算百分位排名
    rps = pd.Series(index=data.index, dtype=float)

    for i in range(period, len(data)):
        hist_roc = adjusted_roc.iloc[period:i+1].dropna()
        current_roc = adjusted_roc.iloc[i]

        if len(hist_roc) > 0 and not np.isnan(current_roc):
            rank = (current_roc >= hist_roc).sum() / len(hist_roc) * 100
            rps.iloc[i] = rank
        else:
            rps.iloc[i] = 50

    return pd.Series(rps, name='rps_market_adjusted')


# 保持向后兼容的别名
RPS_simple = RPS
RS_ratio = None  # 已废弃，使用 RPS_with_benchmark


def RS_ratio(data, period=20, benchmark=None, column='close'):
    """
    RS Ratio (Relative Strength Ratio) - 相对强度比率

    原理: 个股价格 / 基准指数价格，反映个股相对于市场的强弱

    注意: 此函数已废弃，请使用 RPS 或 RPS_with_benchmark

    Returns:
        DataFrame with rs_ratio, rs_ma, rs_trend, rs_roc
    """
    import warnings
    warnings.warn(
        "RS_ratio is deprecated. Use RPS (for self-relative) or "
        "RPS_with_benchmark (for market-adjusted) instead.",
        DeprecationWarning
    )

    if benchmark is None:
        benchmark = data[column]

    rs_ratio = data[column] / benchmark * 100
    rs_ma = rs_ratio.rolling(window=period).mean()

    rs_trend = pd.Series(index=data.index, dtype=str)
    rs_trend[rs_ratio > rs_ma] = 'above_avg'
    rs_trend[rs_ratio <= rs_ma] = 'below_avg'

    rs_roc = rs_ratio.pct_change(period) * 100

    return pd.DataFrame({
        'rs_ratio': rs_ratio,
        'rs_ma': rs_ma,
        'rs_trend': rs_trend,
        'rs_roc': rs_roc
    })
