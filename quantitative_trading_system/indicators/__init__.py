#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
技术指标模块
包含BofA Technical Primer中提到的所有技术指标实现

使用指标注册表统一管理，支持通过 IndicatorRegistry.list_all() 查看所有指标
"""

# 导入注册表和基类
from .registry import IndicatorRegistry, registry
from .base import Indicator, CompositeIndicator, IndicatorResult

# 导入所有指标（触发注册）
from .trend_indicators import (
    SMA, EMA, MACD, DMI_ADX, IchimokuCloud,
    GoldenCrossDeathCross, TripleMovingAverageCross
)

from .momentum_indicators import (
    RSI, Stochastic, ROC, CCI, RPS, detect_divergence
)

from .volatility_indicators import (
    BollingerBands, ATR, StandardDeviation,
    KeltnerChannels, DonchianChannels
)

from .volume_indicators import (
    OBV, AccumulationDistribution, CMF, MFI, VWAP
)

from .breadth_indicators import (
    AdvanceDeclineLine, McClellanOscillator,
    McClellanSummationIndex, ArmsIndex, HighLowIndex
)

from .pattern_indicators import (
    FibonacciRetracement, FibonacciExtension,
    PivotPoints, PatternRecognizer, SupportResistanceLevels,
    detect_candlestick_patterns
)

# 市场状态分析
from .market_regime import (
    MarketRegime, MarketRegimeAnalyzer, MarketRegimeResult,
    MarketBreadthAnalyzer, MarketBreadthResult,
    MarketSentimentAnalyzer, TrendDirection,
    get_strategy_by_regime, generate_market_report
)

# 导出列表
__all__ = [
    # Registry
    'IndicatorRegistry',
    'registry',
    'Indicator',
    'CompositeIndicator',
    'IndicatorResult',
    # Trend
    'SMA', 'EMA', 'MACD', 'DMI_ADX', 'IchimokuCloud',
    'GoldenCrossDeathCross', 'TripleMovingAverageCross',
    # Momentum
    'RSI', 'Stochastic', 'ROC', 'CCI', 'RPS', 'detect_divergence',
    # Volatility
    'BollingerBands', 'ATR', 'StandardDeviation',
    'KeltnerChannels', 'DonchianChannels',
    # Volume
    'OBV', 'AccumulationDistribution', 'CMF', 'MFI', 'VWAP',
    # Breadth
    'AdvanceDeclineLine', 'McClellanOscillator',
    'McClellanSummationIndex', 'ArmsIndex', 'HighLowIndex',
    # Patterns
    'FibonacciRetracement', 'FibonacciExtension',
    'PivotPoints', 'PatternRecognizer', 'SupportResistanceLevels',
    'detect_candlestick_patterns',
    # Market Regime
    'MarketRegime',
    'MarketRegimeAnalyzer',
    'MarketRegimeResult',
    'MarketBreadthAnalyzer',
    'MarketBreadthResult',
    'MarketSentimentAnalyzer',
    'TrendDirection',
    'get_strategy_by_regime',
    'generate_market_report'
]


def list_indicators(category: str = None) -> list:
    """
    列出所有指标

    Args:
        category: 可选，按类别过滤

    Returns:
        指标名称列表
    """
    if category:
        return IndicatorRegistry.list_by_category(category)
    return IndicatorRegistry.list_all()


def get_indicator(name: str):
    """获取指标函数"""
    return IndicatorRegistry.get(name)


def compute_indicator(name: str, data, **kwargs):
    """计算指标"""
    return IndicatorRegistry.compute(name, data, **kwargs)
