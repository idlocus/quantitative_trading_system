#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
因子基类
所有技术因子都应该继承自 Factor 基类
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from enum import Enum
import pandas as pd
import numpy as np


class FactorDirection(Enum):
    """因子方向"""
    POSITIVE = "positive"       # 值越大越好 (如动量)
    NEGATIVE = "negative"       # 值越小越好 (如波动率)
    NEUTRAL = "neutral"          # 方向不重要


@dataclass
class FactorResult:
    """
    因子计算结果

    包含因子的所有相关信息
    """
    name: str                          # 因子名称
    value: float                       # 原始因子值
    normalized_value: float             # 标准化后的值 (0-100)
    percentile: float                   # 百分位排名 (0-100)
    direction: FactorDirection          # 因子方向
    confidence: float = 1.0           # 置信度 0-1
    metadata: Dict[str, Any] = field(default_factory=dict)  # 附加数据

    def to_dict(self) -> Dict:
        return {
            'name': self.name,
            'value': self.value,
            'normalized': self.normalized_value,
            'percentile': self.percentile,
            'direction': self.direction.value,
            'confidence': self.confidence
        }


class Factor(ABC):
    """
    因子抽象基类

    因子是对一个或多个技术指标的高级封装，
    提供标准化、归一化和综合评分功能
    """

    def __init__(self, name: str, weight: float = 1.0, direction: FactorDirection = FactorDirection.POSITIVE):
        """
        初始化因子

        Args:
            name: 因子名称
            weight: 在综合评分中的权重
            direction: 因子方向
        """
        self.name = name
        self.weight = weight
        self.direction = direction

    @abstractmethod
    def compute_raw(self, data: pd.DataFrame) -> pd.Series:
        """
        计算因子原始值

        Args:
            data: OHLCV 数据

        Returns:
            pd.Series: 原始因子值
        """
        pass

    def normalize(self, values: pd.Series) -> pd.Series:
        """
        将因子值标准化到 0-100

        默认使用百分位排名，可被子类重写

        Args:
            values: 原始因子值

        Returns:
            pd.Series: 标准化后的值
        """
        # 使用百分位排名
        result = pd.Series(index=values.index, dtype=float)

        for i in range(len(values)):
            if i < 20:  # 数据太少时使用原始值
                result.iloc[i] = 50
            else:
                window = values.iloc[max(0, i-252):i+1].dropna()  # 最多取1年
                if len(window) > 0 and not np.isnan(values.iloc[i]):
                    current = values.iloc[i]
                    rank = (current >= window).sum() / len(window) * 100
                    result.iloc[i] = rank
                else:
                    result.iloc[i] = 50

        return result

    def compute(self, data: pd.DataFrame) -> pd.Series:
        """
        计算因子值（raw + normalize）

        Args:
            data: OHLCV 数据

        Returns:
            pd.Series: 标准化后的因子值
        """
        raw = self.compute_raw(data)
        return self.normalize(raw)

    def analyze(self, data: pd.DataFrame) -> FactorResult:
        """
        完整分析：计算因子并返回结构化结果

        Args:
            data: OHLCV 数据

        Returns:
            FactorResult: 因子分析结果
        """
        raw_values = self.compute_raw(data)
        normalized = self.normalize(raw_values)

        latest_raw = raw_values.iloc[-1] if len(raw_values) > 0 else 0
        latest_normalized = normalized.iloc[-1] if len(normalized) > 0 else 50

        # 计算百分位
        if len(normalized) >= 20:
            hist_for_percentile = normalized.iloc[-252:] if len(normalized) > 252 else normalized
            percentile = (normalized.iloc[-1] >= hist_for_percentile).sum() / len(hist_for_percentile) * 100
        else:
            percentile = 50

        return FactorResult(
            name=self.name,
            value=latest_raw,
            normalized_value=latest_normalized,
            percentile=percentile,
            direction=self.direction,
            confidence=1.0,
            metadata={}
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', weight={self.weight})"


class CompositeFactor(Factor):
    """
    组合因子

    由多个子因子组合而成的因子
    """

    def __init__(self, name: str, factors: List[Factor], weights: List[float] = None):
        """
        Args:
            name: 因子名称
            factors: 子因子列表
            weights: 子因子权重列表，默认等权重
        """
        super().__init__(name, weight=1.0)
        self.factors = factors
        self.weights = weights or [1.0] * len(factors)

        # 归一化权重
        total = sum(self.weights)
        self.weights = [w / total for w in self.weights]

    def compute_raw(self, data: pd.DataFrame) -> pd.Series:
        """组合因子 raw 值 = 加权平均"""
        results = []
        for factor, weight in zip(self.factors, self.weights):
            normalized = factor.normalize(factor.compute_raw(data))
            results.append(normalized * weight)

        composite = sum(results)
        return composite

    def analyze(self, data: pd.DataFrame) -> FactorResult:
        """分析组合因子"""
        raw_values = self.compute_raw(data)
        normalized = raw_values  # 已经在 compute_raw 中归一化

        latest_normalized = normalized.iloc[-1] if len(normalized) > 0 else 50

        return FactorResult(
            name=self.name,
            value=latest_normalized,
            normalized_value=latest_normalized,
            percentile=latest_normalized,
            direction=FactorDirection.POSITIVE,
            confidence=1.0,
            metadata={'factors': [f.name for f in self.factors]}
        )


class FactorRegistry:
    """
    因子注册表

    统一管理所有因子
    """

    _factors: Dict[str, Factor] = {}

    @classmethod
    def register(cls, name: str = None):
        """装饰器：注册因子"""
        def decorator(factor_cls):
            factor_name = name if name else factor_cls.__name__
            cls._factors[factor_name] = factor_cls
            return factor_cls
        return decorator

    @classmethod
    def get(cls, name: str) -> Factor:
        if name not in cls._factors:
            raise KeyError(f"Factor '{name}' not found")
        return cls._factors[name]()

    @classmethod
    def list_all(cls) -> List[str]:
        return list(cls._factors.keys())

    @classmethod
    def create(cls, name: str, **kwargs) -> Factor:
        """创建因子实例"""
        factor_cls = cls.get(name)
        return factor_cls(**kwargs)
