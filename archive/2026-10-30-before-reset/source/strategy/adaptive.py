#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
自适应策略 - 根据市场环境自动选择选股策略

策略选择规则：
- trending (宽度扩张) + expanding (领先向上) → 趋势跟随 (trend_following_with_dmi)
- trending + cheap → 趋势 + 抄底 (atr_trend_strategy)
- ranging (震荡) + cheap → 动量反转 (short_term_momentum)
- ranging + expensive → 空仓或轻仓 (min_score 提高)
- position_factor < 0.3 → 极轻仓，只选最强信号
"""

from typing import List
import os

from selection.config import SelectionConfig, ScoredStock, ScanConstraint
from selection.scanner import StockScanner, load_strategy_from_yaml
from . import Strategy


class AdaptiveStrategy(Strategy):
    """
    自适应策略

    每日根据 ScanConstraint 中的市场环境信号，
    从预定义策略池中选择最合适的策略执行选股。

    使用方式：runner.run(strategy_name="adaptive")
    """

    # 策略池（名称 → YAML 中的策略名）
    STRATEGY_POOL = {
        'trend_following': 'trend_following_with_dmi',
        'atr_trend': 'atr_trend_strategy',
        'momentum': 'short_term_momentum',
        'breakout': 'breakout',
        'momentum_sector': 'momentum_with_sector_filter',
        'pure_trend': 'pure_trend_follow',
    }

    def __init__(self, config_name: str, data_source, strategy_yaml: str = None):
        """
        初始化自适应策略

        Args:
            config_name: 固定填 "adaptive"（在 strategies.yml 中定义）
            data_source: 数据源
            strategy_yaml: strategies.yml 路径（默认从项目目录加载）
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
        self.scanner = StockScanner(data_source)

        # 缓存已加载的策略配置
        self._config_cache = {}
        for yaml_name, strategy_name in self.STRATEGY_POOL.items():
            try:
                self._config_cache[yaml_name] = load_strategy_from_yaml(
                    strategy_name, strategy_yaml
                )
            except Exception:
                pass  # 策略不存在则跳过

    def select(self, date, data, allowed_symbols: set = None,
               constraint: ScanConstraint = None) -> List[ScoredStock]:
        """
        执行自适应选股

        根据 constraint 中的市场环境信号选择最优策略。
        """
        # 无市场信号时使用默认策略
        if constraint is None:
            constraint = ScanConstraint()

        strategy_name = self._select_strategy(constraint)
        config = self._config_cache.get(strategy_name)

        if config is None:
            # 兜底：使用动量策略
            config = self._config_cache.get('momentum')

        if config is None:
            return []

        # 动态调整 min_score（市场冷时提高门槛，热时降低）
        effective_min_score = self._adjust_min_score(config, constraint)

        # 克隆配置（不改原始对象）
        from copy import copy
        adjusted_config = copy(config)
        adjusted_config.min_score = effective_min_score

        symbols = data.get_tradable_symbols(date)
        results = self.scanner.scan(
            symbols, adjusted_config, ref_date=date,
            allowed_symbols=allowed_symbols,
            constraint=constraint
        )

        # 记录当日使用策略（供分析用）
        self._last_strategy = config.name
        self._last_constraint = constraint

        return results

    def _select_strategy(self, constraint: ScanConstraint) -> str:
        """
        根据市场环境选择策略（以 ADX 信号为核心判断依据）

        市场状态优先级:
        1. ADX > 30 → 趋势市场 → 纯趋势跟随
        2. ADX < 20 → 震荡市场 → 根据估值选择动量或突破
        3. 20 ≤ ADX ≤ 30 → 过渡状态 → 保守策略
        4. position_factor < 0.3 → 极低仓位 → 保守 ATR 趋势
        """
        pf = constraint.position_factor

        # 极低仓位：保守趋势跟随
        if pf < 0.3:
            return 'atr_trend'

        # 以 ADX 信号为主要判断依据
        if constraint.is_adx_trending():
            return 'pure_trend'      # 纯趋势跟随（新策略池key）

        if constraint.is_adx_ranging():
            if constraint.is_cheap():
                return 'momentum'    # 便宜 + 震荡 = RSI超卖反弹
            return 'breakout'        # 中性/贵 + 震荡 = 突破策略

        # ADX neutral（20-30区间）：混合状态
        if constraint.is_cheap() and constraint.is_seasonal_favorable():
            return 'momentum'
        return 'momentum_sector'

    def _adjust_min_score(self, config: SelectionConfig, constraint: ScanConstraint) -> int:
        """
        根据市场环境动态调整 min_score

        - 市场冷 (position_factor 低) → 提高门槛，少选
        - 市场热 (position_factor 高) → 降低门槛，多选
        """
        base = config.min_score
        pf = constraint.position_factor

        if pf >= 0.8:
            return max(50, base - 10)  # 市场很强，降低门槛
        elif pf >= 0.6:
            return base  # 正常
        elif pf >= 0.4:
            return min(70, base + 10)  # 市场一般，提高门槛
        else:
            return min(80, base + 20)  # 市场弱，极高门槛

    @property
    def last_strategy_used(self) -> str:
        """返回最近一次选股使用的策略名称"""
        return getattr(self, '_last_strategy', 'unknown')
