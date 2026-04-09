#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
交易信号和订单数据模型
使用 dataclass 提供类型安全的消息传递
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional, List, Dict, Any
import pandas as pd


class SignalType(Enum):
    """信号类型"""
    STRONG_BUY = "strong_buy"
    BUY = "buy"
    HOLD = "hold"
    SELL = "sell"
    STRONG_SELL = "strong_sell"

    @classmethod
    def from_score(cls, score: float) -> 'SignalType':
        """根据评分转换为信号类型"""
        if score >= 75:
            return cls.STRONG_BUY
        elif score >= 60:
            return cls.BUY
        elif score >= 45:
            return cls.HOLD
        elif score >= 30:
            return cls.SELL
        else:
            return cls.STRONG_SELL


class OrderStatus(Enum):
    """订单状态"""
    PENDING = "pending"
    SUBMITTED = "submitted"
    FILLED = "filled"
    PARTIAL_FILLED = "partial_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


class Direction(Enum):
    """交易方向"""
    LONG = "long"
    SHORT = "short"
    FLAT = "flat"


@dataclass
class Signal:
    """
    交易信号

    包含触发信号的所有相关信息
    """
    symbol: str                          # 股票代码
    signal_type: SignalType               # 信号类型
    strength: float                       # 信号强度 0-1
    timestamp: datetime                   # 信号时间
    price: float                         # 当前价格
    recommendation: str                  # 操作建议 (buy/sell/hold)
    composite_score: float = 0.0         # 综合评分 0-100
    trend_score: float = 0.0             # 趋势评分
    momentum_score: float = 0.0          # 动量评分
    volatility_score: float = 0.0        # 波动率评分
    indicators: Dict[str, Any] = field(default_factory=dict)  # 技术指标值
    risk_reward_ratio: float = 0.0       # 风险收益比
    stop_loss: Optional[float] = None    # 止损价
    take_profit: Optional[float] = None  # 止盈价
    position_size: float = 0.0           # 推荐仓位 0-1
    metadata: Dict[str, Any] = field(default_factory=dict)    # 附加数据

    @classmethod
    def from_framework_analysis(cls, symbol: str, analysis: Dict, timestamp: datetime = None) -> 'Signal':
        """从技术分析框架结果创建信号"""
        if timestamp is None:
            timestamp = datetime.now()

        signals = analysis.get('signals', {})
        recommendation_map = {
            'strong_buy': ('STRONG_BUY', '强烈买入'),
            'buy': ('BUY', '买入'),
            'hold': ('HOLD', '持有观望'),
            'sell': ('SELL', '卖出'),
            'strong_sell': ('STRONG_SELL', '强烈卖出')
        }
        rec = signals.get('recommendation', 'hold')
        sig_type_str, rec_cn = recommendation_map.get(rec, ('HOLD', '持有观望'))

        risk = analysis.get('risk', {})

        return cls(
            symbol=symbol,
            signal_type=SignalType[sig_type_str],
            strength=signals.get('composite', 50) / 100,
            timestamp=timestamp,
            price=analysis.get('trend', {}).get('current_price', 0),
            recommendation=rec_cn,
            composite_score=signals.get('composite', 0),
            trend_score=analysis.get('trend', {}).get('strength', 0),
            momentum_score=analysis.get('momentum', {}).get('value', 0),
            volatility_score=analysis.get('volatility', {}).get('value', 0),
            indicators={
                'rsi': analysis.get('momentum', {}).get('rsi'),
                'stoch_k': analysis.get('momentum', {}).get('stoch_k'),
                'stoch_d': analysis.get('momentum', {}).get('stoch_d'),
                'rps': analysis.get('momentum', {}).get('rps'),
                'adx': analysis.get('trend', {}).get('adx'),
                'atr': analysis.get('volatility', {}).get('atr'),
                'bb_percent': analysis.get('volatility', {}).get('bb_percent'),
            },
            risk_reward_ratio=risk.get('risk_reward_ratio', 0),
            stop_loss=risk.get('stop_loss'),
            take_profit=risk.get('take_profit'),
            position_size=risk.get('position_size_recommendation', 0)
        )


@dataclass
class Order:
    """
    订单

    包含订单的完整信息
    """
    order_id: str                        # 订单ID
    symbol: str                          # 股票代码
    direction: Direction                  # 交易方向
    quantity: float                       # 数量
    order_type: str                       # 订单类型: market, limit, stop
    price: Optional[float] = None        # 限价
    stop_price: Optional[float] = None   # 止损价
    status: OrderStatus = OrderStatus.PENDING
    timestamp: datetime = field(default_factory=datetime.now)
    fill_price: Optional[float] = None
    fill_quantity: Optional[float] = None
    fill_time: Optional[datetime] = None
    commission: float = 0.0               # 手续费
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_filled(self) -> bool:
        return self.status == OrderStatus.FILLED

    def is_closed(self) -> bool:
        return self.status in (OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED)


@dataclass
class Position:
    """
    持仓

    包含持仓的当前状态
    """
    symbol: str                          # 股票代码
    quantity: float                      # 持仓数量
    direction: Direction                 # 持仓方向
    entry_price: float                  # 入场价
    current_price: float = 0.0          # 当前价
    unrealized_pnl: float = 0.0         # 浮动盈亏
    unrealized_pnl_pct: float = 0.0     # 浮动盈亏百分比
    realized_pnl: float = 0.0           # 已实现盈亏
    entry_time: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def update_price(self, new_price: float):
        """更新当前价格和浮动盈亏"""
        self.current_price = new_price
        if self.direction == Direction.LONG:
            self.unrealized_pnl = (new_price - self.entry_price) * self.quantity
        else:
            self.unrealized_pnl = (self.entry_price - new_price) * self.quantity
        if self.entry_price > 0:
            self.unrealized_pnl_pct = (new_price - self.entry_price) / self.entry_price * 100


@dataclass
class Portfolio:
    """
    投资组合

    包含现金和所有持仓
    """
    initial_cash: float                 # 初始资金
    cash: float                         # 当前现金
    positions: List[Position] = field(default_factory=list)
    total_value: float = 0.0           # 总价值
    equity: float = 0.0                 # 权益 (现金 + 浮动盈亏)
    total_pnl: float = 0.0              # 总盈亏
    total_pnl_pct: float = 0.0         # 总盈亏百分比
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def update(self):
        """更新组合价值"""
        positions_value = sum(
            p.current_price * p.quantity if p.direction == Direction.LONG
            else 0 for p in self.positions
        )
        self.equity = self.cash + positions_value + sum(p.realized_pnl for p in self.positions)
        self.total_pnl = self.equity - self.initial_cash
        if self.initial_cash > 0:
            self.total_pnl_pct = self.total_pnl / self.initial_cash * 100

    def get_position(self, symbol: str) -> Optional[Position]:
        """获取指定股票的持仓"""
        for p in self.positions:
            if p.symbol == symbol:
                return p
        return None

    def add_position(self, position: Position):
        """添加持仓"""
        existing = self.get_position(position.symbol)
        if existing:
            # 合并持仓
            if existing.direction == position.direction:
                total_qty = existing.quantity + position.quantity
                existing.entry_price = (
                    (existing.entry_price * existing.quantity +
                     position.entry_price * position.quantity) / total_qty
                )
                existing.quantity = total_qty
            else:
                # 反向操作，平仓
                if existing.quantity > position.quantity:
                    existing.quantity -= position.quantity
                elif existing.quantity < position.quantity:
                    existing.direction = position.direction
                    existing.quantity = position.quantity - existing.quantity
                else:
                    self.positions.remove(existing)
        else:
            self.positions.append(position)


@dataclass
class BacktestTrade:
    """回测交易记录"""
    entry_time: datetime
    exit_time: datetime
    symbol: str
    direction: Direction
    entry_price: float
    exit_price: float
    quantity: float
    pnl: float
    pnl_pct: float
    commission: float = 0.0


@dataclass
class BacktestResult:
    """回测结果"""
    initial_cash: float
    final_value: float
    total_return: float
    total_return_pct: float
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    avg_holding_days: float = 0.0
    equity_curve: pd.Series = None
    trades: List[BacktestTrade] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        """回测结果摘要"""
        return f"""
========== Backtest Results ==========
Initial Cash:     ¥{self.initial_cash:,.2f}
Final Value:      ¥{self.final_value:,.2f}
Total Return:     ¥{self.total_return:+,.2f} ({self.total_return_pct:+.2f}%)

Sharpe Ratio:     {self.sharpe_ratio:.2f}
Sortino Ratio:    {self.sortino_ratio:.2f}
Max Drawdown:     {self.max_drawdown_pct:.2f}%

Win Rate:         {self.win_rate:.1f}%
Profit Factor:    {self.profit_factor:.2f}
Total Trades:     {self.total_trades}
Winning:          {self.winning_trades}
Losing:           {self.losing_trades}

Avg Win:          ¥{self.avg_win:,.2f}
Avg Loss:         ¥{self.avg_loss:,.2f}
======================================
"""
