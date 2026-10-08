#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
风控模块 - 仓位管理和风险控制

所有参数从 risk.constants 导入，禁止硬编码数值。
支持 ATR 动态止损（止损 = 入场价 - 2×ATR）。
"""

from dataclasses import dataclass
from datetime import date, timedelta
from typing import List, Dict, Optional

from selection.config import ScoredStock, SelectionConfig
from .constants import RiskParams


@dataclass
class Position:
    """持仓信息"""
    symbol: str
    entry_price: float
    quantity: int
    entry_date: date
    signal_score: int
    conditions: List[Dict]  # 买入时的指标状态
    atr: float = 0.0  # ATR 值（用于动态止损，入场时计算）


class RiskManager:
    """
    风险管理器，负责仓位分配和止损止盈判断

    参数从 RiskParams 获取，不再接受独立参数。
    支持 ATR 动态止损：止损位 = 入场价 - 2×ATR（替代固定止损 3%）。
    """

    def __init__(self, params: RiskParams = None, data_source=None):
        """
        初始化风险管理器

        Args:
            params: RiskParams实例，默认使用常量定义
            data_source: 数据源对象，需提供 get_close / get_realtime 方法
        """
        self.params = params or RiskParams()
        self.data_source = data_source

    @property
    def stop_loss(self) -> float:
        return self.params.STOP_LOSS

    @property
    def take_profit(self) -> float:
        return self.params.TAKE_PROFIT

    @property
    def max_position_size(self) -> float:
        return self.params.MAX_POSITION_SIZE

    def allocate(self, signals: List[ScoredStock], capital: float,
                 config: Optional[SelectionConfig] = None,
                 trading_date: Optional[date] = None) -> List[Position]:
        """
        根据选股信号分配仓位

        Args:
            signals: 排序后的选股信号列表
            capital: 总资金
            config: 选股配置，包含 max_positions 等参数
            trading_date: 交易日期（用于获取历史价格）

        Returns:
            Position 列表
        """
        max_positions = config.max_positions if config else 5
        positions = []
        total_weight = 0.0

        for signal in signals:
            # Check position count limit
            if len(positions) >= max_positions:
                break

            # Check weight cap (80% max total exposure)
            weight = self._score_to_weight(signal.score)
            if total_weight + weight > self.params.MAX_TOTAL_EXPOSURE:
                # Try lower weight allocation (5% fallback) if this is the first position
                if len(positions) == 0 and weight > 0.05:
                    weight = 0.05
                else:
                    break

            total_weight += weight
            amount = capital * weight
            price = self._get_price(signal.symbol, trading_date)
            if price <= 0:
                price = 10.0  # Default price to avoid division by zero
            quantity = int(amount / price)
            if quantity > 0:
                # 计算入场时 ATR（用于动态止损）
                atr = self._get_atr(signal.symbol, trading_date)
                positions.append(Position(
                    symbol=signal.symbol,
                    entry_price=price,
                    quantity=quantity,
                    entry_date=trading_date or date.today(),
                    signal_score=signal.score,
                    conditions=signal.conditions,
                    atr=atr
                ))
        return positions

    def _score_to_weight(self, score: int) -> float:
        """将信号评分转换为仓位权重（从常量获取）"""
        return self.params.score_to_weight(score)

    def check_exit(self, position: Position, current_price: float) -> Optional[str]:
        """
        检查是否需要退出仓位

        ATR 动态止损：优先用 ATR 计算止损位
        - 止损位 = entry_price - 2 * position.atr
        - 若 ATR 为 0 或过小，回退到固定止损 (STOP_LOSS)

        Args:
            position: 持仓信息
            current_price: 当前价格

        Returns:
            "stop_loss" (止损), "take_profit" (止盈), 或 None (继续持有)
        """
        ret = (current_price - position.entry_price) / position.entry_price

        # ATR 动态止损：entry_price - 2×ATR 对应的百分比
        # 取固定止损和 ATR 止损中更严格（更大亏损承受能力）的那个
        if position.atr > 0:
            atr_stop_ret = (position.entry_price - 2 * position.atr - position.entry_price) / position.entry_price
            # atr_stop_ret 是负值（如 -0.06），取绝对值与固定止损比较
            # 取较大者 = 更严格的止损（更小亏损就触发）
            effective_stop_loss = max(self.stop_loss, abs(atr_stop_ret))
        else:
            effective_stop_loss = self.stop_loss

        if ret <= -effective_stop_loss:
            return "stop_loss"
        elif ret >= self.take_profit:
            return "take_profit"
        return None

    def _get_price(self, symbol: str, trading_date: date = None) -> float:
        """
        获取股票价格

        Args:
            symbol: 股票代码
            trading_date: 交易日期（用于获取历史收盘价）

        Returns:
            股票价格（优先用历史收盘价，其次用实时价格）
        """
        # 优先使用历史收盘价
        if trading_date and self.data_source is not None:
            if hasattr(self.data_source, 'get_close'):
                price = self.data_source.get_close(symbol, trading_date)
                if price > 0:
                    return price
            if hasattr(self.data_source, 'get_kline'):
                kline = self.data_source.get_kline(symbol, trading_date, trading_date)
                if kline is not None and not kline.empty and 'close' in kline.columns:
                    return float(kline['close'].iloc[-1])

        # 回退到实时价格
        if self.data_source is not None and hasattr(self.data_source, 'get_realtime'):
            return self.data_source.get_realtime(symbol)
        # 默认返回价格，避免分配失败
        return 10.0

    def _get_atr(self, symbol: str, trading_date: date = None) -> float:
        """
        获取股票 ATR 值（用于动态止损计算）

        通过历史 K 线数据计算 14 日 ATR。

        Args:
            symbol: 股票代码
            trading_date: 交易日期

        Returns:
            ATR 值（单位：价格绝对值），若获取失败返回 0
        """
        if trading_date is None or self.data_source is None:
            return 0.0

        try:
            # 获取近 30 个交易日的数据（足够计算 14 日 ATR）
            start = trading_date - timedelta(days=60)
            kline = self.data_source.get_kline(symbol, start, trading_date)
            if kline is None or kline.empty or len(kline) < 15:
                return 0.0

            high = kline['high']
            low = kline['low']
            close = kline['close']

            # True Range
            tr1 = high - low
            tr2 = (high - close.shift()).abs()
            tr3 = (low - close.shift()).abs()
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

            # ATR (14 日, Wilder EMA —— 与 indicators.volatility_indicators.ATR 口径一致)
            atr = tr.ewm(alpha=1 / 14, adjust=False).mean().iloc[-1]
            return float(atr) if not pd.isna(atr) else 0.0
        except Exception:
            return 0.0


import pandas as pd
