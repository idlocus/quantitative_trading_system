#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
技术因子
基于技术指标的高级因子封装
"""

from typing import Dict
import pandas as pd
import numpy as np

from indicators import SMA, EMA, MACD, DMI_ADX
from indicators import RSI, Stochastic, RPS
from indicators import BollingerBands, ATR
from .base import Factor, FactorDirection, FactorResult, CompositeFactor, FactorRegistry


@FactorRegistry.register('trend')
class TrendFactor(Factor):
    """
    趋势因子

    综合评估价格趋势方向和强度
    包含: 均线排列、MACD、DMI
    """

    def __init__(self, ma_short=10, ma_medium=50, ma_long=200):
        super().__init__(
            name='trend',
            weight=0.40,
            direction=FactorDirection.POSITIVE
        )
        self.ma_short = ma_short
        self.ma_medium = ma_medium
        self.ma_long = ma_long

    def compute_raw(self, data: pd.DataFrame) -> pd.Series:
        """计算趋势评分 0-100"""
        current_price = data['close']

        # 均线值
        ma_s = SMA(data, self.ma_short)
        ma_m = SMA(data, self.ma_medium)
        ma_l = SMA(data, self.ma_long)

        # 趋势评分
        trend_score = pd.Series(0.0, index=data.index)

        # 1. 均线多头排列加分
        # 价格 > 短期 > 中期 > 长期 = 完全多头
        price_above_short = (current_price > ma_s).astype(float) * 15
        short_above_medium = (ma_s > ma_m).astype(float) * 10
        medium_above_long = (ma_m > ma_l).astype(float) * 10

        # 2. MACD 评分
        macd = MACD(data)
        macd_value = macd['macd']
        macd_signal = macd['signal']
        macd_bullish = (macd_value > macd_signal).astype(float) * 15

        # MACD 柱状图方向
        macd_hist = macd['histogram']
        macd_histogram_positive = (macd_hist > 0).astype(float) * 5

        # 3. 价格相对均线位置
        price_position = (current_price - ma_l) / (ma_s - ma_l + 0.001) * 15
        price_position = price_position.clip(0, 15)

        # 4. 均线收敛程度（布林带宽度作为补充）
        bb = BollingerBands(data)
        bb_width = bb['bandwidth']
        volatility_normalized = (bb_width / 50).clip(0, 1) * 5  # 波动率低加分

        # 综合评分
        trend_score = (
            price_above_short +
            short_above_medium +
            medium_above_long +
            macd_bullish +
            macd_histogram_positive +
            price_position +
            volatility_normalized
        )

        # 限制在 0-100
        return trend_score.clip(0, 100)

    def analyze(self, data: pd.DataFrame) -> FactorResult:
        """分析趋势"""
        raw_values = self.compute_raw(data)
        normalized = raw_values  # 已经标准化到 0-100

        latest = normalized.iloc[-1] if len(normalized) > 0 else 0

        # 判断趋势方向
        ma_s = SMA(data, self.ma_short)
        ma_m = SMA(data, self.ma_medium)
        current_price = data['close'].iloc[-1]

        if ma_s.iloc[-1] > ma_m.iloc[-1] and current_price > ma_s.iloc[-1]:
            direction = 'strong_up'
        elif current_price > ma_s.iloc[-1]:
            direction = 'up'
        elif current_price < ma_s.iloc[-1]:
            direction = 'down'
        else:
            direction = 'neutral'

        return FactorResult(
            name=self.name,
            value=latest,
            normalized_value=latest,
            percentile=latest,
            direction=FactorDirection.POSITIVE if 'up' in direction else FactorDirection.NEGATIVE if 'down' in direction else FactorDirection.NEUTRAL,
            confidence=1.0,
            metadata={
                'direction': direction,
                'ma_short': ma_s.iloc[-1],
                'ma_medium': ma_m.iloc[-1],
                'current_price': current_price
            }
        )


@FactorRegistry.register('momentum')
class MomentumFactor(Factor):
    """
    动量因子

    综合评估价格动量
    包含: RSI, Stochastic, RPS
    """

    def __init__(self, rsi_period=14, stoch_period=14, rps_period=20):
        super().__init__(
            name='momentum',
            weight=0.35,
            direction=FactorDirection.POSITIVE
        )
        self.rsi_period = rsi_period
        self.stoch_period = stoch_period
        self.rps_period = rps_period

    def compute_raw(self, data: pd.DataFrame) -> pd.Series:
        """计算动量评分 0-100"""
        # RSI 评分 (RSI 在 40-60 为中性，高于60偏多，低于40偏空但可能反弹)
        rsi = RSI(data, self.rsi_period)

        # RSI 转换到 0-100
        # RSI=50 为中性，偏离越大分数越低
        rsi_score = 100 - abs(rsi - 50) * 2
        rsi_score = rsi_score.clip(0, 100)

        # Stochastic 评分
        stoch = Stochastic(data, k_period=self.stoch_period)
        stoch_k = stoch['k']
        stoch_d = stoch['d']

        # %K 在 20-80 之间为正常
        stoch_score = 100 - abs(stoch_k - 50) * 1.5
        stoch_score = stoch_score.clip(0, 100)

        # Stochastic 金叉加分
        stoch_cross = (stoch_k > stoch_d).astype(float) * 10

        # RPS 评分
        rps = RPS(data, period=self.rps_period)
        rps_score = rps  # 已经是 0-100

        # 综合动量评分
        momentum = (
            rsi_score * 0.35 +
            stoch_score * 0.30 +
            stoch_cross * 0.10 +
            rps_score * 0.25
        )

        return momentum.clip(0, 100)

    def analyze(self, data: pd.DataFrame) -> FactorResult:
        """分析动量"""
        raw_values = self.compute_raw(data)
        normalized = raw_values

        latest = normalized.iloc[-1] if len(normalized) > 0 else 0
        rsi_val = RSI(data, self.rsi_period).iloc[-1]

        # 信号判断
        if latest >= 70:
            signal = 'strong_buy'
        elif latest >= 55:
            signal = 'buy'
        elif latest >= 45:
            signal = 'neutral'
        elif latest >= 30:
            signal = 'sell'
        else:
            signal = 'strong_sell'

        return FactorResult(
            name=self.name,
            value=latest,
            normalized_value=latest,
            percentile=latest,
            direction=FactorDirection.POSITIVE if 'buy' in signal else FactorDirection.NEGATIVE if 'sell' in signal else FactorDirection.NEUTRAL,
            confidence=1.0,
            metadata={
                'signal': signal,
                'rsi': rsi_val
            }
        )


@FactorRegistry.register('volatility')
class VolatilityFactor(Factor):
    """
    波动率因子

    评估市场波动状态
    包含: ATR, Bollinger Bands
    """

    def __init__(self, atr_period=14, bb_period=20):
        super().__init__(
            name='volatility',
            weight=0.25,
            direction=FactorDirection.NEGATIVE  # 低波动率 = 高分数
        )
        self.atr_period = atr_period
        self.bb_period = bb_period

    def compute_raw(self, data: pd.DataFrame) -> pd.Series:
        """计算波动率评分 0-100 (低波动 = 高分)"""
        current_price = data['close']

        # ATR 百分比
        atr = ATR(data, self.atr_period)
        atr_pct = (atr / current_price) * 100

        # ATR 波动率评分 (ATR% 越低分数越高)
        # 正常 ATR% 在 1-5% 之间
        atr_score = 100 - atr_pct * 10
        atr_score = atr_score.clip(0, 100)

        # 布林带评分
        bb = BollingerBands(data, self.bb_period)
        bb_percent = bb['percent_b']
        bb_bandwidth = bb['bandwidth']

        # %B 在 0.3-0.7 之间为正常
        bb_position_score = 100 - abs(bb_percent - 0.5) * 100
        bb_position_score = bb_position_score.clip(0, 100)

        # 布林带收缩 = 低波动加分
        bb_squeeze = (bb_bandwidth < 3).astype(float) * 15

        # 综合波动率评分
        volatility = (
            atr_score * 0.40 +
            bb_position_score * 0.40 +
            bb_squeeze * 0.20
        )

        return volatility.clip(0, 100)

    def analyze(self, data: pd.DataFrame) -> FactorResult:
        """分析波动率"""
        raw_values = self.compute_raw(data)
        normalized = raw_values

        latest = normalized.iloc[-1] if len(normalized) > 0 else 50

        atr = ATR(data, self.atr_period)
        atr_pct = (atr.iloc[-1] / data['close'].iloc[-1]) * 100

        if atr_pct > 5:
            status = 'high'
        elif atr_pct > 2:
            status = 'medium'
        else:
            status = 'low'

        return FactorResult(
            name=self.name,
            value=latest,
            normalized_value=latest,
            percentile=latest,
            direction=FactorDirection.NEUTRAL,  # 波动率方向不重要
            confidence=1.0,
            metadata={
                'status': status,
                'atr_pct': atr_pct
            }
        )


# 保持向后兼容的别名
CompositeFactor = CompositeFactor
