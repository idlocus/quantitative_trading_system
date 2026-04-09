#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
成长因子

营收增速、净利润增速、EPS增速等
高成长 = 高分数
"""

import pandas as pd
import numpy as np
from typing import Dict

from factors.base import Factor, FactorDirection, FactorResult, FactorRegistry


@FactorRegistry.register('growth')
class GrowthFactor(Factor):
    """
    成长因子

    综合评估公司成长能力
    高成长得分高，低成长得分低
    """

    def __init__(self, revenue_weight=0.4, profit_weight=0.4, eps_weight=0.2):
        super().__init__(
            name='growth',
            weight=0.35,  # 基本面权重中成长占35%
            direction=FactorDirection.POSITIVE  # 成长越高越好
        )
        self.revenue_weight = revenue_weight
        self.profit_weight = profit_weight
        self.eps_weight = eps_weight

    def compute_raw(self, data: pd.DataFrame) -> pd.Series:
        """
        计算成长评分 0-100

        返回:
            pd.Series: 成长评分序列
        """
        # 简化处理：使用价格动量作为成长代理指标
        # 真实实现需要从Wind/AKShare获取财务数据

        # 价格变化率
        returns = data['close'].pct_change()

        # 20日动量作为成长代理
        momentum_20 = returns.rolling(20).sum() * 100
        momentum_60 = returns.rolling(60).sum() * 100
        momentum_120 = returns.rolling(120).sum() * 100

        # 加权动量
        growth_score = (
            momentum_20 * 0.5 +
            momentum_60 * 0.3 +
            momentum_120 * 0.2
        )

        # 转换为 0-100 评分
        # 动量 > 0 得高分，动量 < 0 得低分
        growth_score = 50 + growth_score * 2
        return growth_score.clip(0, 100)

    def analyze(self, data: pd.DataFrame) -> FactorResult:
        """分析成长性"""
        raw_values = self.compute_raw(data)
        latest = raw_values.iloc[-1] if len(raw_values) > 0 else 50

        # 判断成长水平
        if latest >= 75:
            level = "强劲增长"
        elif latest >= 60:
            level = "稳健增长"
        elif latest >= 45:
            level = "低速增长"
        elif latest >= 30:
            level = "零增长"
        else:
            level = "负增长"

        # 获取真实成长数据
        growth_data = self._get_real_growth(data)

        return FactorResult(
            name=self.name,
            value=latest,
            normalized_value=latest,
            percentile=latest,
            direction=FactorDirection.POSITIVE,
            confidence=0.7 if growth_data else 0.4,
            metadata={
                'level': level,
                'revenue_growth': growth_data.get('revenue_growth'),
                'profit_growth': growth_data.get('profit_growth'),
                'eps_growth': growth_data.get('eps_growth'),
            }
        )

    def _get_real_growth(self, data: pd.DataFrame) -> Dict[str, float]:
        """获取真实成长数据"""
        try:
            from data import get_fundamental_source
            from config.system_config import SystemConfig

            symbol = data.attrs.get('symbol', '300308.SZ')
            config = SystemConfig()
            source = get_fundamental_source(config, 'akshare')
            return source.get_growth_metrics(symbol)
        except Exception:
            return {}

    def _get_direction(self) -> str:
        return FactorDirection.POSITIVE
