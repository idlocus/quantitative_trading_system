#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
指标基类
所有技术指标都应该继承自 Indicator 基类
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import pandas as pd


class Indicator(ABC):
    """
    技术指标抽象基类

    所有指标都应该继承此类并实现 compute 方法
    """

    def __init__(self, name: str, period: int = 14, **kwargs):
        """
        初始化指标

        Args:
            name: 指标名称
            period: 计算周期
            **kwargs: 其他参数
        """
        self.name = name
        self.period = period
        self.params = kwargs

    @abstractmethod
    def compute(self, data: pd.DataFrame) -> pd.Series:
        """
        计算指标值

        Args:
            data: 包含 OHLCV 数据的 DataFrame

        Returns:
            pd.Series: 指标值序列
        """
        pass

    def validate_data(self, data: pd.DataFrame) -> bool:
        """
        验证数据格式

        Args:
            data: 待验证的数据

        Returns:
            bool: 数据是否有效
        """
        required_columns = ['open', 'high', 'low', 'close']
        return all(col in data.columns for col in required_columns)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', period={self.period})"


class CompositeIndicator(Indicator):
    """
    组合指标基类

    由多个基础指标组合而成的指标
    """

    def __init__(self, name: str, indicators: list, **kwargs):
        """
        Args:
            name: 指标名称
            indicators: 子指标列表
            **kwargs: 其他参数
        """
        super().__init__(name, **kwargs)
        self.indicators = indicators

    def compute(self, data: pd.DataFrame) -> pd.Series:
        """组合指标由子类实现"""
        raise NotImplementedError


class IndicatorResult:
    """
    指标计算结果包装器

    方便同时返回多个值
    """

    def __init__(self, name: str, data: pd.Series, **metadata):
        self.name = name
        self.data = data
        self.metadata = metadata

    def __getitem__(self, key):
        return self.data.iloc[key]

    def __len__(self):
        return len(self.data)

    @property
    def last(self):
        """返回最后一个值"""
        return self.data.iloc[-1] if len(self.data) > 0 else None

    def to_series(self) -> pd.Series:
        """转换为 Series"""
        return self.data.copy()

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'name': self.name,
            'last': self.last,
            'metadata': self.metadata
        }
