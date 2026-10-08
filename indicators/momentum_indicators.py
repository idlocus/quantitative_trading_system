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

    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    return pd.Series(rsi, name='rsi')


@IndicatorRegistry.register('stochastic', category='momentum', primary_column='d')
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

    # 计算平均偏差（避免 rolling.apply 逐行运算，使用预计算差值后求均值）
    deviations = tp - sma_tp
    mean_deviation = deviations.abs().rolling(window=period).mean()

    cci = (tp - sma_tp) / (0.015 * mean_deviation)

    return pd.Series(cci, name='cci')


@IndicatorRegistry.register('divergence', category='momentum')
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
    [已弃用] 时序自排名 RPS —— 当前 ROC 在本股票自身历史中的百分位。

    注意：这与欧奈尔 RPS（全市场横截面排名）语义不同。
    横截面 RPS 请使用选股配置中的 rps_rank_gt 操作符
    （由 BacktestEngine 每日从全市场缓存计算并注入 ScanConstraint.rps_rank）。

    信号:
    - 值 > 80: 近期动量处于自身历史高位
    - 值 < 20: 近期动量处于自身历史低位

    Returns:
        pd.Series: RPS值 (0-100)
    """
    from scipy.stats import rankdata

    # 计算N日价格变化率
    roc = 100 * (data[column] - data[column].shift(period)) / data[column].shift(period)

    # expanding 百分位排名：每个位置 i 的 RPS = 当前 ROC 在历史 ROC 中的百分位
    rps = pd.Series(index=data.index, dtype=float)
    for i in range(period, len(data)):
        hist_roc = roc.iloc[period:i+1].dropna()
        current_roc = roc.iloc[i]
        if len(hist_roc) > 0 and not np.isnan(current_roc):
            rps.iloc[i] = rankdata([current_roc] + list(hist_roc))[0] / (len(hist_roc) + 1) * 100
        else:
            rps.iloc[i] = 50.0

    return pd.Series(rps, name='rps')


@IndicatorRegistry.register('rps_bench', category='momentum')
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
    from scipy.stats import rankdata

    # 计算个股ROC
    stock_roc = 100 * (data[column] - data[column].shift(period)) / data[column].shift(period)

    # 计算基准ROC
    benchmark_roc = 100 * (benchmark - benchmark.shift(period)) / benchmark.shift(period)

    # 市场调整后的ROC
    adjusted_roc = stock_roc - benchmark_roc

    # expanding 百分位排名
    rps = pd.Series(index=data.index, dtype=float)
    for i in range(period, len(data)):
        hist_roc = adjusted_roc.iloc[period:i+1].dropna()
        current_roc = adjusted_roc.iloc[i]
        if len(hist_roc) > 0 and not np.isnan(current_roc):
            rps.iloc[i] = rankdata([current_roc] + list(hist_roc))[0] / (len(hist_roc) + 1) * 100
        else:
            rps.iloc[i] = 50.0

    return pd.Series(rps, name='rps_market_adjusted')


# 保持向后兼容的别名
RPS_simple = RPS


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
