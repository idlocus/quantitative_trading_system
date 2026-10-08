#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多策略组合

同时运行多个子策略，按权重合并信号，统一仓位分配。
降低单一策略风险，提高组合稳定性。

使用方式（strategies.yml）:
  multi_strategy:
    name: 多策略组合
    strategies:
      - name: trend_following_with_dmi
        weight: 0.3
      - name: momentum_with_sector_filter
        weight: 0.3
      - name: breakout
        weight: 0.2
      - name: short_term_momentum
        weight: 0.2
    logic: weighted_sum  # 信号合并方式
    min_score: 60
    max_positions: 5
"""

from typing import List, Dict
import os
from copy import copy

from selection.config import SelectionConfig, ScoredStock, ScanConstraint
from selection.scanner import StockScanner, load_strategy_from_yaml
from . import Strategy


class SubStrategy:
    """子策略（持有策略配置 + 权重）"""

    def __init__(self, name: str, weight: float, config: SelectionConfig, data_source):
        self.name = name
        self.weight = weight          # 0.0-1.0，信号权重
        self.config = config
        self.data_source = data_source
        self.scanner = StockScanner(data_source)
        self._last_scores: List[ScoredStock] = []

    def select(self, date, allowed_symbols=None, constraint=None) -> List[ScoredStock]:
        """执行子策略选股"""
        from data.base import DataLoader
        if hasattr(self.data_source, 'get_tradable_symbols'):
            symbols = self.data_source.get_tradable_symbols(date)
        else:
            symbols = []

        results = self.scanner.scan(
            symbols, self.config, ref_date=date,
            allowed_symbols=allowed_symbols,
            constraint=constraint
        )
        self._last_scores = results
        return results


class MultiStrategy(Strategy):
    """
    多策略组合

    并行运行多个子策略，按权重合并信号。
    子策略权重决定其在最终信号中的占比。
    """

    def __init__(self, config_name: str, data_source, strategy_yaml: str = None):
        """
        初始化多策略组合

        Args:
            config_name: strategies.yml 中的多策略配置名
            data_source: 数据源
            strategy_yaml: strategies.yml 路径
        """
        if strategy_yaml is None:
            strategy_yaml = os.path.join(
                os.path.dirname(__file__),
                '..',
                'selection',
                'strategies.yml'
            )
        self.strategy_yaml = strategy_yaml
        self.data_source = data_source
        self._load_config(config_name)

    def _load_config(self, config_name: str):
        """加载多策略配置"""
        import yaml
        with open(self.strategy_yaml, 'r', encoding='utf-8') as f:
            all_strategies = yaml.safe_load(f)

        if config_name not in all_strategies:
            raise ValueError(f"Multi-strategy '{config_name}' not found in strategies.yml")

        cfg = all_strategies[config_name]

        # 加载子策略
        self.sub_strategies: List[SubStrategy] = []
        for entry in cfg.get('strategies', []):
            sub_name = entry['name']
            weight = entry.get('weight', 0.25)

            # 加载子策略的 YAML 配置
            try:
                sub_config = load_strategy_from_yaml(sub_name, self.strategy_yaml)
            except Exception:
                continue  # 跳过不存在的策略

            self.sub_strategies.append(SubStrategy(
                name=sub_name,
                weight=weight,
                config=sub_config,
                data_source=self.data_source
            ))

        # 合并配置参数
        self.config = SelectionConfig(
            name=cfg.get('name', config_name),
            indicators=[],
            logic=cfg.get('logic', 'AND'),
            min_score=cfg.get('min_score', 60),
            max_positions=cfg.get('max_positions', 5),
        )

    def select(self, date, data, allowed_symbols: set = None,
               constraint: ScanConstraint = None) -> List[ScoredStock]:
        """
        执行多策略选股

        1. 各子策略独立选股
        2. 按权重合并评分
        3. 按综合评分排序，取前 max_positions 名
        """
        if constraint is None:
            constraint = ScanConstraint()

        # 1. 各子策略独立选股
        all_scores: Dict[str, ScoredStock] = {}
        strategy_results = {}

        for sub in self.sub_strategies:
            scores = sub.select(date, allowed_symbols, constraint)
            strategy_results[sub.name] = scores

            for sc in scores:
                if sc.symbol not in all_scores:
                    all_scores[sc.symbol] = ScoredStock(
                        symbol=sc.symbol,
                        name=sc.name,
                        score=0,
                        conditions=[],
                        indicators={}
                    )
                # 累加加权分
                all_scores[sc.symbol].score += sc.score * sub.weight
                all_scores[sc.symbol].conditions.extend(sc.conditions)
                all_scores[sc.symbol].indicators.update(sc.indicators)

        # 2. 封顶到 100（保持绝对信号强度，不做相对归一化——
        #    否则最弱的信号只要唯一也会被放大到 100 分拿满仓）
        for sc in all_scores.values():
            sc.score = min(100, int(sc.score))

        # 3. 过滤 min_score
        results = [s for s in all_scores.values() if s.score >= self.config.min_score]

        # 4. 排序
        results.sort(key=lambda x: x.score, reverse=True)

        # 5. 取前 max_positions
        results = results[:self.config.max_positions]

        # 记录当日结果（供分析用）
        self._last_results = results
        self._last_strategy_results = strategy_results

        return results

    @property
    def sub_strategy_names(self) -> List[str]:
        return [s.name for s in self.sub_strategies]

    @property
    def sub_strategy_weights(self) -> List[float]:
        return [s.weight for s in self.sub_strategies]
