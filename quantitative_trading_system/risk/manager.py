#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
风控模块 - 仓位管理和风险控制
"""

from dataclasses import dataclass
from datetime import date
from typing import List, Dict, Optional

from selection.config import ScoredStock


@dataclass
class Position:
    """持仓信息"""
    symbol: str
    entry_price: float
    quantity: int
    entry_date: date
    signal_score: int
    conditions: Dict  # 买入时的指标状态


class RiskManager:
    """
    风险管理器，负责仓位分配和止损止盈判断
    """

    def __init__(self, stop_loss: float = 0.05, take_profit: float = 0.10,
                 max_position_size: float = 0.2, data_source=None):
        """
        初始化风险管理器

        Args:
            stop_loss: 止损比例 (默认 5%)
            take_profit: 止盈比例 (默认 10%)
            max_position_size: 最大仓位比例 (默认 20%)
            data_source: 数据源对象，需提供 get_realtime(symbol) 方法
        """
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.max_position_size = max_position_size
        self.data_source = data_source

    def allocate(self, signals: List[ScoredStock], capital: float) -> List[Position]:
        """
        根据选股信号分配仓位

        Args:
            signals: 排序后的选股信号列表
            capital: 总资金

        Returns:
            Position 列表
        """
        positions = []
        for signal in signals[:5]:  # 最多5个持仓
            weight = self._score_to_weight(signal.score)
            amount = capital * weight
            price = self._get_price(signal.symbol)
            quantity = int(amount / price)
            if quantity > 0:
                positions.append(Position(
                    symbol=signal.symbol,
                    entry_price=price,
                    quantity=quantity,
                    entry_date=date.today(),
                    signal_score=signal.score,
                    conditions=dict(signal.conditions) if signal.conditions else {}
                ))
        return positions

    def _score_to_weight(self, score: int) -> float:
        """
        将信号评分转换为仓位权重

        Args:
            score: 信号评分 (0-100)

        Returns:
            仓位权重比例
        """
        if score >= 80:
            return 0.20
        elif score >= 70:
            return 0.15
        elif score >= 60:
            return 0.10
        else:
            return 0.05

    def check_exit(self, position: Position, current_price: float) -> Optional[str]:
        """
        检查是否需要退出仓位

        Args:
            position: 持仓信息
            current_price: 当前价格

        Returns:
            "stop_loss" (止损), "take_profit" (止盈), 或 None (继续持有)
        """
        ret = (current_price - position.entry_price) / position.entry_price
        if ret <= -self.stop_loss:
            return "stop_loss"
        elif ret >= self.take_profit:
            return "take_profit"
        return None

    def _get_price(self, symbol: str) -> float:
        """
        获取股票实时价格

        Args:
            symbol: 股票代码

        Returns:
            当前价格
        """
        if self.data_source is not None and hasattr(self.data_source, 'get_realtime'):
            return self.data_source.get_realtime(symbol)
        # 默认返回价格，避免分配失败
        return 10.0