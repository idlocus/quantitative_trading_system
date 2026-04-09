#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
估值因子

PE、PB、PS、PCF 等市盈率族指标
低估值 = 高分数（可能反弹）
"""

import pandas as pd
import numpy as np
from typing import Dict

from factors.base import Factor, FactorDirection, FactorResult, FactorRegistry


@FactorRegistry.register('valuation')
class ValuationFactor(Factor):
    """
    估值因子

    综合评估股票估值水平
    低估值得分高，高估值得分低
    """

    def __init__(self, pe_weight=0.3, pb_weight=0.3, ps_weight=0.2, pcf_weight=0.2):
        super().__init__(
            name='valuation',
            weight=0.30,  # 基本面权重中估值占30%
            direction=FactorDirection.NEGATIVE  # 估值越低越好
        )
        self.pe_weight = pe_weight
        self.pb_weight = pb_weight
        self.ps_weight = ps_weight
        self.pcf_weight = pcf_weight

    def compute_raw(self, data: pd.DataFrame) -> pd.Series:
        """
        计算估值评分 0-100

        返回:
            pd.Series: 估值评分序列
        """
        # 尝试从市场数据计算简化的估值指标
        # 实际项目中应从基本面数据源获取真实PE/PB等

        close = data['close']

        # 简化处理：使用价格动量和波动率估算
        # 真实实现需要从Wind/AKShare获取财务数据

        # 价格位置（在历史高低之间）
        high_252 = data['high'].rolling(252).max()
        low_252 = data['low'].rolling(252).min()
        price_position = (close - low_252) / (high_252 - low_252 + 0.001)
        price_position = price_position.fillna(0.5)

        # 估值评分：价格位置低 = 低估值 = 高分
        valuation_score = (1 - price_position) * 100

        return valuation_score.clip(0, 100)

    def analyze(self, data: pd.DataFrame) -> FactorResult:
        """分析估值"""
        raw_values = self.compute_raw(data)
        latest = raw_values.iloc[-1] if len(raw_values) > 0 else 50

        # 判断估值水平
        if latest >= 70:
            level = "极低估值"
        elif latest >= 55:
            level = "低估值"
        elif latest >= 45:
            level = "中性"
        elif latest >= 30:
            level = "高估值"
        else:
            level = "极高估值"

        # 获取真实估值数据
        valuation_data = self._get_real_valuation(data)

        return FactorResult(
            name=self.name,
            value=latest,
            normalized_value=latest,
            percentile=latest,
            direction=FactorDirection.NEGATIVE,
            confidence=0.8 if valuation_data else 0.5,
            metadata={
                'level': level,
                'pe': valuation_data.get('pe'),
                'pb': valuation_data.get('pb'),
                'ps': valuation_data.get('ps'),
                'pcf': valuation_data.get('pcf'),
            }
        )

    def _get_real_valuation(self, data: pd.DataFrame) -> Dict[str, float]:
        """获取真实估值数据"""
        try:
            from data import get_fundamental_source
            from config.system_config import SystemConfig

            symbol = data.attrs.get('symbol', '300308.SZ')
            config = SystemConfig()
            source = get_fundamental_source(config, 'akshare')
            return source.get_valuation(symbol)
        except Exception:
            return {}

    def _get_direction(self) -> str:
        return FactorDirection.NEGATIVE
