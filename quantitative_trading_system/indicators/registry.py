#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
指标注册表
提供统一的指标注册、发现和计算接口
"""

from typing import Callable, Dict, List, Optional, Any
import pandas as pd


class IndicatorRegistry:
    """
    指标注册表

    使用注册表模式统一管理所有技术指标，
    支持通过名称动态发现和计算指标
    """

    _indicators: Dict[str, Callable] = {}
    _categories: Dict[str, List[str]] = {
        'trend': [],
        'momentum': [],
        'volatility': [],
        'volume': [],
        'breadth': [],
        'pattern': []
    }

    @classmethod
    def register(cls, name: str = None, category: str = 'trend'):
        """
        装饰器：注册指标函数

        Args:
            name: 指标名称，默认使用函数名
            category: 指标类别

        Usage:
            @IndicatorRegistry.register('rsi', category='momentum')
            def RSI(data, period=14):
                ...

            # 或不带参数
            @IndicatorRegistry.register()
            def SMA(data, period):
                ...
        """
        def decorator(func: Callable) -> Callable:
            indicator_name = name if name else func.__name__
            func._indicator_name = indicator_name
            func._category = category

            if indicator_name in cls._indicators:
                raise ValueError(f"Indicator '{indicator_name}' already registered")

            cls._indicators[indicator_name] = func

            if category not in cls._categories:
                cls._categories[category] = []
            cls._categories[category].append(indicator_name)

            return func

        return decorator

    @classmethod
    def get(cls, name: str) -> Callable:
        """
        获取指标函数

        Args:
            name: 指标名称

        Returns:
            Callable: 指标函数

        Raises:
            KeyError: 指标不存在
        """
        if name not in cls._indicators:
            raise KeyError(f"Indicator '{name}' not found. Available: {cls.list_all()}")
        return cls._indicators[name]

    @classmethod
    def list_all(cls) -> List[str]:
        """列出所有已注册的指标"""
        return list(cls._indicators.keys())

    @classmethod
    def list_by_category(cls, category: str) -> List[str]:
        """按类别列出指标"""
        return cls._categories.get(category, [])

    @classmethod
    def list_categories(cls) -> List[str]:
        """列出所有类别"""
        return list(cls._categories.keys())

    @classmethod
    def compute(cls, name: str, data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        计算指标的统一入口

        Args:
            name: 指标名称
            data: OHLCV 数据
            **kwargs: 指标参数

        Returns:
            pd.Series: 指标值序列
        """
        func = cls.get(name)

        # 检查函数是否有 period 参数
        import inspect
        sig = inspect.signature(func)
        params = sig.parameters

        # 如果指标函数接受 period 参数但未提供，从 kwargs 获取
        if 'period' in params and 'period' not in kwargs:
            kwargs['period'] = kwargs.get('period', 14)

        result = func(data, **kwargs)

        # 统一转换为 pd.Series
        if isinstance(result, pd.DataFrame):
            # 取最后一列
            return result.iloc[:, -1]
        return result

    @classmethod
    def batch_compute(cls, indicators: List[str], data: pd.DataFrame, **kwargs) -> Dict[str, pd.Series]:
        """
        批量计算多个指标

        Args:
            indicators: 指标名称列表
            data: OHLCV 数据
            **kwargs: 共享的指标参数

        Returns:
            Dict[str, pd.Series]: 指标名称到指标值的映射
        """
        results = {}
        for name in indicators:
            try:
                results[name] = cls.compute(name, data, **kwargs)
            except Exception as e:
                print(f"Warning: Failed to compute indicator '{name}': {e}")
                results[name] = pd.Series(dtype=float)
        return results

    @classmethod
    def clear(cls):
        """清空注册表（主要用于测试）"""
        cls._indicators.clear()
        for category in cls._categories:
            cls._categories[category].clear()


# 全局实例，方便导入
registry = IndicatorRegistry
