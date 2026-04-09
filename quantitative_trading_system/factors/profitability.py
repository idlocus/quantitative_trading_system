#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
盈利能力因子

ROE、ROA、毛利率、资产负债率等
高盈利能力 = 高分数
"""

import pandas as pd
import numpy as np
from typing import Dict

from factors.base import Factor, FactorDirection, FactorResult, FactorRegistry


@FactorRegistry.register('profitability')
class ProfitabilityFactor(Factor):
    """
    盈利能力因子

    综合评估公司盈利能力
    高盈利能力得分高，低盈利能力得分低
    """

    def __init__(self, roe_weight=0.35, roa_weight=0.25, gross_margin_weight=0.25, debt_weight=0.15):
        super().__init__(
            name='profitability',
            weight=0.35,  # 基本面权重中盈利能力占35%
            direction=FactorDirection.POSITIVE  # 盈利能力越高越好
        )
        self.roe_weight = roe_weight
        self.roa_weight = roa_weight
        self.gross_margin_weight = gross_margin_weight
        self.debt_weight = debt_weight

    def compute_raw(self, data: pd.DataFrame) -> pd.Series:
        """
        计算盈利能力评分 0-100

        返回:
            pd.Series: 盈利能力评分序列
        """
        # 简化处理：使用毛利率代理指标
        # 真实实现需要从Wind/AKShare获取财务数据

        # 计算简化毛利率（使用OHLC数据估算）
        # 实际应该是 (收盘价 - 最低价) / 收盘价 的极端情况估算
        high = data['high']
        low = data['low']
        close = data['close']

        # 毛利率代理：价格在日内区间的位置
        gross_proxy = (close - low) / (high - low + 0.001)
        gross_proxy = gross_proxy.fillna(0.5)

        # ROE代理：使用价格动量
        returns = data['close'].pct_change(20)
        roe_proxy = (returns * 10 + 50).clip(0, 100)

        # 资产负债率代理：低波动率 -> 低风险 -> 较低负债
        volatility = close.rolling(20).std() / close.rolling(20).mean()
        debt_proxy = 50 - volatility * 200
        debt_proxy = debt_proxy.clip(0, 100)

        # 综合评分
        profitability_score = (
            gross_proxy * 30 +
            roe_proxy * 40 +
            debt_proxy * 30
        )

        return profitability_score.clip(0, 100)

    def analyze(self, data: pd.DataFrame) -> FactorResult:
        """分析盈利能力"""
        raw_values = self.compute_raw(data)
        latest = raw_values.iloc[-1] if len(raw_values) > 0 else 50

        # 判断盈利水平
        if latest >= 80:
            level = "极强盈利"
        elif latest >= 65:
            level = "强盈利"
        elif latest >= 50:
            level = "中等盈利"
        elif latest >= 35:
            level = "弱盈利"
        else:
            level = "极弱盈利"

        # 获取真实盈利数据
        profit_data = self._get_real_profitability(data)

        return FactorResult(
            name=self.name,
            value=latest,
            normalized_value=latest,
            percentile=latest,
            direction=FactorDirection.POSITIVE,
            confidence=0.7 if profit_data else 0.4,
            metadata={
                'level': level,
                'roe': profit_data.get('roe'),
                'roa': profit_data.get('roa'),
                'gross_margin': profit_data.get('gross_margin'),
                'debt_ratio': profit_data.get('debt_ratio'),
            }
        )

    def _get_real_profitability(self, data: pd.DataFrame) -> Dict[str, float]:
        """获取真实盈利数据"""
        try:
            from data import get_fundamental_source
            from config.system_config import SystemConfig

            symbol = data.attrs.get('symbol', '300308.SZ')
            config = SystemConfig()
            source = get_fundamental_source(config, 'akshare')
            return source.get_profitability_metrics(symbol)
        except Exception:
            return {}

    def _get_direction(self) -> str:
        return FactorDirection.POSITIVE
