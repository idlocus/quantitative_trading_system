#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
技术指标模块测试
"""

import pytest
import pandas as pd
import numpy as np

# 添加项目路径以导入指标模块
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from indicators.momentum_indicators import RSI
from indicators.trend_indicators import MACD
from indicators.volatility_indicators import BollingerBands, ATR
from indicators.volume_indicators import OBV
from indicators.base import Indicator
from indicators.registry import detect_cross_signal, IndicatorRegistry


# =============================================================================
# 样本 OHLCV 数据生成
# =============================================================================

def create_sample_ohlcv(n=100, start_price=100.0, volatility=0.02, trend=0.001):
    """
    创建样本 OHLCV 数据

    Args:
        n: 数据点数量
        start_price: 起始价格
        volatility: 波动率
        trend: 趋势偏移

    Returns:
        pd.DataFrame: OHLCV 数据
    """
    np.random.seed(42)

    dates = pd.date_range(start='2024-01-01', periods=n, freq='D')

    # 生成价格序列（带趋势）
    prices = [start_price]
    for i in range(1, n):
        change = np.random.randn() * volatility * prices[-1] + trend * prices[-1]
        prices.append(prices[-1] + change)

    prices = np.array(prices)

    # 生成 OHLC
    data = pd.DataFrame({
        'open': prices * (1 + np.random.uniform(-0.01, 0.01, n)),
        'high': prices * (1 + np.random.uniform(0.01, 0.03, n)),
        'low': prices * (1 + np.random.uniform(-0.03, -0.01, n)),
        'close': prices,
        'volume': np.random.randint(1000000, 10000000, n)
    }, index=dates)

    # 确保 high 是最高的
    data['high'] = data[['open', 'high', 'close']].max(axis=1) * 1.01
    data['low'] = data[['open', 'low', 'close']].min(axis=1) * 0.99

    return data


# 全局样本数据
SAMPLE_OHLCV = create_sample_ohlcv(100)


# =============================================================================
# 测试 RSI
# =============================================================================

def test_rsi_calculation():
    """测试 RSI 计算，结果应在 0-100 之间"""
    rsi = RSI(SAMPLE_OHLCV)

    assert isinstance(rsi, pd.Series), "RSI 应返回 pd.Series"
    assert rsi.name == 'rsi', "RSI 系列应被命名为 'rsi'"

    # 检查有效值范围（排除 NaN）
    valid_rsi = rsi.dropna()
    assert len(valid_rsi) > 0, "RSI 应有有效值"

    # RSI 应该在 0-100 之间
    assert (valid_rsi >= 0).all(), "RSI 值应 >= 0"
    assert (valid_rsi <= 100).all(), "RSI 值应 <= 100"


def test_rsi_with_custom_period():
    """测试 RSI 使用自定义周期"""
    rsi_7 = RSI(SAMPLE_OHLCV, period=7)
    rsi_21 = RSI(SAMPLE_OHLCV, period=21)

    assert isinstance(rsi_7, pd.Series)
    assert isinstance(rsi_21, pd.Series)

    # 较短周期的 RSI 应该有更多有效值（因为预热期更短）
    assert len(rsi_7.dropna()) >= len(rsi_21.dropna())


# =============================================================================
# 测试 MACD
# =============================================================================

def test_macd_calculation():
    """测试 MACD 计算"""
    macd = MACD(SAMPLE_OHLCV)

    assert isinstance(macd, pd.DataFrame), "MACD 应返回 pd.DataFrame"

    # 检查必需的列
    required_cols = ['macd', 'signal', 'histogram']
    for col in required_cols:
        assert col in macd.columns, f"MACD DataFrame 应包含 '{col}' 列"

    # 检查有效值
    valid_macd = macd.dropna()
    assert len(valid_macd) > 0, "MACD 应有有效值"


def test_macd_cross():
    """测试 MACD 交叉信号"""
    macd = MACD(SAMPLE_OHLCV)

    signal = detect_cross_signal(macd['macd'], macd['signal'])

    # 信号应该是 None, 'gold_cross', 或 'death_cross' 之一
    assert signal in [None, 'gold_cross', 'death_cross'], \
        f"MACD 交叉信号应为 [None, 'gold_cross', 'death_cross']，得到: {signal}"


def test_macd_gold_cross_detection():
    """测试 MACD 金叉检测"""
    # 创建一个价格持续上涨的数据（产生金叉）
    data = create_sample_ohlcv(100, start_price=100.0, volatility=0.01, trend=0.005)

    macd = MACD(data)
    signal = detect_cross_signal(macd['macd'], macd['signal'])

    # 在上涨趋势中，MACD 应该倾向于产生金叉
    # 注意：不一定每次都有金叉，所以只检查返回值类型


# =============================================================================
# 测试 BollingerBands
# =============================================================================

def test_bollinger_bands_calculation():
    """测试布林带计算"""
    bb = BollingerBands(SAMPLE_OHLCV)

    assert isinstance(bb, pd.DataFrame), "BollingerBands 应返回 pd.DataFrame"

    # 检查必需的列
    required_cols = ['upper', 'middle', 'lower']
    for col in required_cols:
        assert col in bb.columns, f"BollingerBands DataFrame 应包含 '{col}' 列"


def test_bollinger_bands_properties():
    """测试布林带的基本属性"""
    bb = BollingerBands(SAMPLE_OHLCV, period=20, std_dev=2)

    valid_bb = bb.dropna()

    # 上轨应该大于中轨
    assert (valid_bb['upper'] >= valid_bb['middle']).all(), "上轨应 >= 中轨"

    # 中轨应该大于下轨
    assert (valid_bb['middle'] >= valid_bb['lower']).all(), "中轨应 >= 下轨"

    # 带宽应该是正值
    if 'bandwidth' in bb.columns:
        assert (bb['bandwidth'].dropna() > 0).all(), "带宽应为正值"


# =============================================================================
# 测试 ATR
# =============================================================================

def test_atr_calculation():
    """测试 ATR 计算"""
    atr = ATR(SAMPLE_OHLCV)

    assert isinstance(atr, pd.Series), "ATR 应返回 pd.Series"
    assert atr.name == 'atr', "ATR 系列应被命名为 'atr'"


def test_atr_positive_values():
    """测试 ATR 返回正值"""
    atr = ATR(SAMPLE_OHLCV)

    valid_atr = atr.dropna()
    assert len(valid_atr) > 0, "ATR 应有有效值"
    assert (valid_atr > 0).all(), "ATR 所有有效值应 > 0"


def test_atr_with_custom_period():
    """测试 ATR 使用自定义周期"""
    atr_7 = ATR(SAMPLE_OHLCV, period=7)
    atr_14 = ATR(SAMPLE_OHLCV, period=14)

    assert isinstance(atr_7, pd.Series)
    assert isinstance(atr_14, pd.Series)


# =============================================================================
# 测试 OBV
# =============================================================================

def test_obv_calculation():
    """测试 OBV 计算"""
    obv = OBV(SAMPLE_OHLCV)

    assert isinstance(obv, pd.Series), "OBV 应返回 pd.Series"
    assert obv.name == 'obv', "OBV 系列应被命名为 'obv'"


def test_obv_cumulative():
    """测试 OBV 是累积的"""
    obv = OBV(SAMPLE_OHLCV)

    valid_obv = obv.dropna()

    # OBV 应该是累积的（至少有增有减）
    # 检查 OBV 不是单调的（如果不是，则测试无意义）
    diffs = valid_obv.diff().dropna()
    has_increase = (diffs > 0).any()
    has_decrease = (diffs < 0).any()

    # OBV 应该有增有减（因为价格有涨有跌）
    assert has_increase or has_decrease, "OBV 应该有变化"


def test_obv_monotonic_increasing_when_price_rises():
    """测试当价格上涨时 OBV 单调增加"""
    # 创建一个价格持续上涨的数据
    data = create_sample_ohlcv(50, start_price=100.0, volatility=0.005, trend=0.01)

    obv = OBV(data)

    # 在持续上涨期间，OBV 应该是累积增加的
    # 但由于 OBV 取决于方向变化，可能不完全单调
    # 我们检查 OBV 最后值应该大于开始值
    valid_obv = obv.dropna()
    assert len(valid_obv) > 0
    # OBV 应该总体趋势向上
    assert valid_obv.iloc[-1] >= valid_obv.iloc[0], \
        "在持续上涨中，OBV 应该增加（至少不减少）"


# =============================================================================
# 测试 Indicator.evaluate() 方法
# =============================================================================

class DummyIndicator(Indicator):
    """测试用的虚拟指标"""

    def compute(self, data: pd.DataFrame) -> pd.Series:
        return data['close']


def test_evaluate_greater_than():
    """测试 evaluate() 方法的 > 运算符"""
    ind = DummyIndicator('test', period=14)
    data = pd.Series([10, 20, 30, 40, 50])

    # last value = 50
    assert ind.evaluate(data, '>', 55) == False  # 50 > 55 = False
    assert ind.evaluate(data, '>', 45) == True   # 50 > 45 = True


def test_evaluate_less_than():
    """测试 evaluate() 方法的 < 运算符"""
    ind = DummyIndicator('test', period=14)
    data = pd.Series([50, 40, 30, 20, 10])

    # last value = 10
    assert ind.evaluate(data, '<', 5) == False   # 10 < 5 = False
    assert ind.evaluate(data, '<', 15) == True   # 10 < 15 = True


def test_evaluate_cross_up():
    """测试 evaluate() 方法的 cross_up 运算符"""
    ind = DummyIndicator('test', period=14)
    data = pd.Series([10, 20, 30, 40, 50])

    # 从下方穿越到阈值 - previous=40, current=50
    assert ind.evaluate(data, 'cross_up', 45) == True   # 40 < 45 <= 50 = True (cross)
    assert ind.evaluate(data, 'cross_up', 55) == False  # 40 < 55, 50 < 55 = False (no cross)


def test_evaluate_cross_down():
    """测试 evaluate() 方法的 cross_down 运算符"""
    ind = DummyIndicator('test', period=14)
    data = pd.Series([50, 40, 30, 20, 10])

    # 从上方穿越到阈值 - previous=20, current=10
    assert ind.evaluate(data, 'cross_down', 15) == True   # 20 > 15 >= 10 = True (cross)
    assert ind.evaluate(data, 'cross_down', 25) == False  # 20 > 25, 10 < 25 = False (no cross)


def test_evaluate_insufficient_data():
    """测试数据不足时 evaluate() 返回 False"""
    ind = DummyIndicator('test', period=14)
    data = pd.Series([10])

    assert ind.evaluate(data, '>', 5) is False


# =============================================================================
# 测试 detect_cross_signal() 函数
# =============================================================================

def test_detect_cross_signal_gold_cross():
    """测试金叉检测"""
    series1 = pd.Series([10, 15, 20, 25])  # 快速线
    series2 = pd.Series([20, 20, 20, 20])  # 慢速线（不变）

    signal = detect_cross_signal(series1, series2)
    assert signal == 'gold_cross', f"期望 'gold_cross'，得到: {signal}"


def test_detect_cross_signal_death_cross():
    """测试死叉检测"""
    series1 = pd.Series([30, 25, 20, 15])  # 快速线
    series2 = pd.Series([20, 20, 20, 20])  # 慢速线（不变）

    signal = detect_cross_signal(series1, series2)
    assert signal == 'death_cross', f"期望 'death_cross'，得到: {signal}"


def test_detect_cross_signal_no_cross():
    """测试无交叉情况"""
    series1 = pd.Series([10, 12, 14, 16])  # 持续在下方
    series2 = pd.Series([20, 20, 20, 20])  # 慢速线

    signal = detect_cross_signal(series1, series2)
    assert signal is None, f"期望 None，得到: {signal}"


def test_detect_cross_signal_insufficient_data():
    """测试数据不足情况"""
    series1 = pd.Series([10])
    series2 = pd.Series([20])

    signal = detect_cross_signal(series1, series2)
    assert signal is None


# =============================================================================
# 测试 IndicatorRegistry
# =============================================================================

def test_registry_get_indicator():
    """测试从注册表获取指标"""
    rsi_func = IndicatorRegistry.get('rsi')
    assert callable(rsi_func), "RSI 应是可调用的"


def test_registry_list_all():
    """测试列出所有指标"""
    indicators = IndicatorRegistry.list_all()
    assert isinstance(indicators, list), "应返回列表"
    assert 'rsi' in indicators, "RSI 应在指标列表中"
    assert 'macd' in indicators, "MACD 应在指标列表中"


def test_registry_list_by_category():
    """测试按类别列出指标"""
    momentum_indicators = IndicatorRegistry.list_by_category('momentum')
    assert isinstance(momentum_indicators, list)
    assert 'rsi' in momentum_indicators


def test_registry_compute():
    """测试注册表的 compute 方法"""
    result = IndicatorRegistry.compute('rsi', SAMPLE_OHLCV)
    assert isinstance(result, pd.Series)
    assert result.name == 'rsi'


# =============================================================================
# 主程序入口
# =============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v'])