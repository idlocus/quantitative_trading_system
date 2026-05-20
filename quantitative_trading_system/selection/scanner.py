#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
选股扫描器
基于技术指标规则扫描股票
"""

from typing import List, Dict, Any, Optional
import pandas as pd

from .config import SelectionConfig, ScoredStock, IndicatorRule
from indicators.registry import IndicatorRegistry


class StockScanner:
    """
    股票扫描器

    基于预配置的技术指标规则对股票进行扫描和评分
    """

    def __init__(self, data_source):
        """
        初始化扫描器

        Args:
            data_source: 数据源，需提供 get_kline(symbol) 方法返回 OHLCV DataFrame
        """
        self.data_source = data_source
        self.registry = IndicatorRegistry

    def scan(self, symbols: List[str], config: SelectionConfig) -> List[ScoredStock]:
        """
        扫描股票列表

        Args:
            symbols: 股票代码列表
            config: 选股配置

        Returns:
            按评分降序排列的股票列表
        """
        results = []
        for symbol in symbols:
            score, conditions, indicators = self._evaluate(symbol, config)
            if score >= config.min_score:
                results.append(ScoredStock(
                    symbol=symbol,
                    score=score,
                    conditions=conditions,
                    indicators=indicators
                ))
        return sorted(results, key=lambda x: x.score, reverse=True)

    def _evaluate(self, symbol: str, config: SelectionConfig):
        """
        评估单个股票

        Args:
            symbol: 股票代码
            config: 选股配置

        Returns:
            (总分, 满足条件列表, 指标值字典)
        """
        data = self.data_source.get_kline(symbol)
        if data is None or len(data) < 60:
            return 0, [], {}

        scores = []
        conditions = []
        indicators = {}

        for rule in config.indicators:
            indicator_func = self.registry.get(rule.name)
            values = indicator_func(data)

            # 获取当前值和前一个值（用于cross检测）
            if isinstance(values, pd.DataFrame):
                # 取最后一列作为主值
                current = values.iloc[-1, -1]
                prev = values.iloc[-2, -2] if len(values) > 1 else current
            else:
                current = values.iloc[-1]
                prev = values.iloc[-2] if len(values) > 1 else current

            indicators[rule.name] = current

            # 评估条件
            met = self._check_condition(current, prev, rule.operator, rule.threshold)
            if met:
                scores.append(rule.weight * 100)
                conditions.append({
                    'indicator': rule.name,
                    'operator': rule.operator,
                    'threshold': rule.threshold,
                    'value': current
                })

        # 计算总分
        if config.logic == "AND" and len(scores) != len(config.indicators):
            return 0, [], {}
        elif config.logic == "OR" and len(scores) == 0:
            return 0, [], {}

        total_score = min(100, int(sum(scores) / len(config.indicators)))
        return total_score, conditions, indicators

    def _check_condition(self, current: float, prev: float, operator: str, threshold) -> bool:
        """
        检查条件是否满足

        Args:
            current: 当前值
            prev: 前一个值
            operator: 操作符
            threshold: 阈值

        Returns:
            条件是否满足
        """
        if operator == ">":
            return current > threshold
        elif operator == "<":
            return current < threshold
        elif operator == ">=":
            return current >= threshold
        elif operator == "<=":
            return current <= threshold
        elif operator == "cross_up":
            return prev < threshold <= current
        elif operator == "cross_down":
            return prev > threshold >= current
        elif operator == "break_upper":
            # 价格突破布林带上轨
            return current > threshold
        elif operator == "break_lower":
            # 价格突破布林带下轨
            return current < threshold
        return False


def load_strategy_from_yaml(strategy_name: str, yaml_path: str) -> SelectionConfig:
    """
    从YAML文件加载选股策略配置

    Args:
        strategy_name: 策略名称
        yaml_path: YAML文件路径

    Returns:
        SelectionConfig对象

    Raises:
        FileNotFoundError: YAML文件不存在
        ValueError: 策略不存在或配置格式错误
    """
    import yaml

    with open(yaml_path, 'r', encoding='utf-8') as f:
        strategies = yaml.safe_load(f)

    if strategy_name not in strategies:
        available = list(strategies.keys())
        raise ValueError(f"Strategy '{strategy_name}' not found. Available: {available}")

    config_dict = strategies[strategy_name]

    # 解析indicators
    indicator_rules = []
    for rule_dict in config_dict.get('indicators', []):
        rule = IndicatorRule(
            name=rule_dict['name'],
            operator=rule_dict['operator'],
            threshold=rule_dict['threshold'],
            weight=rule_dict.get('weight', 1.0)
        )
        indicator_rules.append(rule)

    return SelectionConfig(
        name=config_dict['name'],
        indicators=indicator_rules,
        logic=config_dict.get('logic', 'AND'),
        min_score=config_dict.get('min_score', 60),
        max_positions=config_dict.get('max_positions', 5)
    )