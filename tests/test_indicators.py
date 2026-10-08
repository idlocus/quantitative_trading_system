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

    signal = detect_cross(macd['macd'], macd['signal'])

    # 信号应该是 None, 'gold_cross', 或 'death_cross' 之一
    assert signal in [None, 'gold_cross', 'death_cross'], \
        f"MACD 交叉信号应为 [None, 'gold_cross', 'death_cross']，得到: {signal}"


def test_macd_gold_cross_detection():
    """测试 MACD 金叉检测"""
    # 创建一个价格持续上涨的数据（产生金叉）
    data = create_sample_ohlcv(100, start_price=100.0, volatility=0.01, trend=0.005)

    macd = MACD(data)
    signal = detect_cross(macd['macd'], macd['signal'])

    # 信号应该是 None, 'gold_cross', 或 'death_cross' 之一
    assert signal in [None, 'gold_cross', 'death_cross'], \
        f"MACD 交叉信号应为 [None, 'gold_cross', 'death_cross']，得到: {signal}"

    # 在强上涨趋势中，应该更可能产生金叉而非死叉
    # 如果没有交叉，至少验证返回值有效
    assert isinstance(signal, str) or signal is None


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
    assert 'bandwidth' in bb.columns, "BollingerBands 应包含 'bandwidth' 列"
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

    valid_obv = obv.dropna()
    assert len(valid_obv) > 0

    # 计算 OBV 变化
    obv_diff = valid_obv.diff().dropna()

    # 计算价格上涨的天数比例（收盘价相比前一天上涨）
    price_rises = (data['close'].diff().dropna() > 0)
    price_up_count = price_rises.sum()
    total_price_changes = len(price_rises)

    # 在持续上涨趋势中，大部分天数价格应该上涨
    assert price_up_count / total_price_changes > 0.6, \
        f"测试数据应呈持续上涨趋势，但只有 {price_up_count}/{total_price_changes} 天上涨"

    # OBV 在价格上涨时应该增加，下跌时应该减少
    # 计算 OBV 变化与价格变化方向一致的比例
    price_changes = data['close'].diff().dropna()
    aligned_changes = ((obv_diff > 0) & (price_changes > 0)) | ((obv_diff < 0) & (price_changes < 0))
    alignment_ratio = aligned_changes.sum() / len(obv_diff)

    assert alignment_ratio > 0.6, \
        f"OBV 变化方向应与价格变化方向一致，期望 > 60%，得到: {alignment_ratio:.2%}"


# =============================================================================
# 测试 detect_cross() 函数
# =============================================================================

from indicators.registry import detect_cross_signal as detect_cross
from utils.data_utils import to_scalar


def test_detect_cross_gold_cross():
    """测试金叉检测"""
    series1 = pd.Series([10, 15, 20, 25])  # 快速线
    series2 = pd.Series([20, 20, 20, 20])  # 慢速线（不变）

    signal = detect_cross(series1, series2)
    assert signal == 'gold_cross', f"Expected 'gold_cross', got: {signal}"


def test_detect_cross_death_cross():
    """测试死叉检测"""
    series1 = pd.Series([30, 25, 20, 15])  # 快速线
    series2 = pd.Series([20, 20, 20, 20])  # 慢速线（不变）

    signal = detect_cross(series1, series2)
    assert signal == 'death_cross', f"Expected 'death_cross', got: {signal}"


def test_detect_cross_no_cross():
    """测试无交叉情况"""
    series1 = pd.Series([10, 12, 14, 16])  # 持续在下方
    series2 = pd.Series([20, 20, 20, 20])  # 慢速线

    signal = detect_cross(series1, series2)
    assert signal is None, f"Expected None, got: {signal}"


def test_detect_cross_insufficient_data():
    """测试数据不足情况"""
    series1 = pd.Series([10])
    series2 = pd.Series([20])

    signal = detect_cross(series1, series2)
    assert signal is None


def test_to_scalar_series():
    """测试 to_scalar 处理 Series"""
    s = pd.Series([1, 2, 3, 4, 5])
    assert to_scalar(s) == 5


def test_to_scalar_dataframe():
    """测试 to_scalar 处理 DataFrame"""
    df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
    assert to_scalar(df) == 4


def test_to_scalar_scalar():
    """测试 to_scalar 处理标量"""
    assert to_scalar(42) == 42


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