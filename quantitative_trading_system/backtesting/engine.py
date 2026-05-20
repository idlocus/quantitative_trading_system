#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
回测引擎 - 核心回测逻辑
"""

from dataclasses import dataclass, field
from datetime import date
from typing import List, Dict, Optional

import pandas as pd
import numpy as np

from risk.manager import Position, RiskManager
from strategy import Strategy
from selection.config import ScoredStock


@dataclass
class BacktestConfig:
    """回测配置"""
    initial_capital: float = 1000000.0  # 初始资金
    commission: float = 0.0003          # 手续费率 (万3)
    stop_loss: float = 0.05             # 止损比例 (5%)
    take_profit: float = 0.10           # 止盈比例 (10%)
    max_positions: int = 5              # 最大持仓数


@dataclass
class Trade:
    """交易记录"""
    id: int
    symbol: str
    action: str              # "BUY" / "SELL"
    date: date
    price: float
    quantity: int
    commission: float
    reason: str              # 触发原因
    conditions: List[Dict]   # 当时的指标状态
    signal_score: int        # 信号评分


@dataclass
class BacktestResult:
    """回测结果"""
    initial_capital: float
    final_capital: float
    total_return: float
    annual_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    trade_count: int
    avg_holding_days: float
    equity_curve: pd.Series
    trades: List[Trade]
    positions: Dict[str, Position]  # 当前持仓


class MarketData:
    """
    市场数据接口 (用于回测)

    回测时需要提供以下方法:
    - trading_dates: List[date] - 交易日期列表
    - get_close(symbol: str, date: date) -> float - 获取收盘价
    """

    def __init__(self, data_source=None):
        """
        初始化市场数据

        Args:
            data_source: 数据源对象，需提供 get_close(symbol, date) 方法
        """
        self.data_source = data_source
        self._trading_dates = []
        self._price_cache = {}

    def add_trading_date(self, trading_date: date):
        """添加交易日期"""
        self._trading_dates.append(trading_date)

    @property
    def trading_dates(self) -> List[date]:
        """获取交易日期列表"""
        return sorted(self._trading_dates)

    def get_close(self, symbol: str, trading_date: date) -> float:
        """
        获取收盘价

        Args:
            symbol: 股票代码
            trading_date: 交易日期

        Returns:
            收盘价
        """
        cache_key = (symbol, trading_date)
        if cache_key in self._price_cache:
            return self._price_cache[cache_key]

        if self.data_source is not None and hasattr(self.data_source, 'get_close'):
            price = self.data_source.get_close(symbol, trading_date)
        else:
            price = 0.0

        self._price_cache[cache_key] = price
        return price


class BacktestEngine:
    """回测引擎"""

    def __init__(self, config: BacktestConfig):
        """
        初始化回测引擎

        Args:
            config: 回测配置
        """
        self.config = config
        self.risk_manager = RiskManager(
            stop_loss=config.stop_loss,
            take_profit=config.take_profit
        )
        self.trades = []
        self.positions = {}
        self.cash = config.initial_capital
        self.equity_curve = []
        self._trade_id = 0
        self._position_entry_dates = {}  # 记录持仓入场日期

    def run(self, strategy: Strategy, data: MarketData) -> BacktestResult:
        """
        运行回测

        Args:
            strategy: 策略对象
            data: 市场数据

        Returns:
            BacktestResult 回测结果
        """
        self._data = data
        for trading_date in data.trading_dates:
            # 1. 选股
            signals = strategy.select(trading_date, data)

            # 2. 仓位分配
            new_positions = self.risk_manager.allocate(
                signals, self.cash, config=None
            )

            # 3. 执行买入
            for pos in new_positions:
                if pos.symbol not in self.positions and len(self.positions) >= self.config.max_positions:
                    continue
                if pos.symbol not in self.positions:
                    self._execute_buy(pos, trading_date)

            # 4. 检查止损止盈
            self._check_exit_conditions(trading_date, data)

            # 5. 更新权益
            self._update_equity(trading_date, data)

        return self._generate_result()

    def _execute_buy(self, position: Position, trading_date: date):
        """执行买入"""
        cost = position.entry_price * position.quantity
        commission = cost * self.config.commission
        total_cost = cost + commission

        if total_cost > self.cash:
            # 资金不足，调整数量
            position.quantity = int(
                self.cash / (position.entry_price * (1 + self.config.commission))
            )
            cost = position.entry_price * position.quantity
            commission = cost * self.config.commission
            total_cost = cost + commission

        if position.quantity <= 0:
            return

        self.cash -= total_cost
        self.positions[position.symbol] = position
        self._position_entry_dates[position.symbol] = trading_date

        self.trades.append(Trade(
            id=self._trade_id,
            symbol=position.symbol,
            action="BUY",
            date=trading_date,
            price=position.entry_price,
            quantity=position.quantity,
            commission=commission,
            reason="买入信号",
            conditions=position.conditions,
            signal_score=position.signal_score
        ))
        self._trade_id += 1

    def _execute_sell(self, symbol: str, trading_date: date, price: float, reason: str):
        """执行卖出"""
        position = self.positions[symbol]
        sell_value = price * position.quantity
        commission = sell_value * self.config.commission
        net_value = sell_value - commission

        self.trades.append(Trade(
            id=self._trade_id,
            symbol=symbol,
            action="SELL",
            date=trading_date,
            price=price,
            quantity=position.quantity,
            commission=commission,
            reason=reason,
            conditions=position.conditions,
            signal_score=position.signal_score
        ))
        self._trade_id += 1

        self.cash += net_value
        del self.positions[symbol]
        del self._position_entry_dates[symbol]

    def _check_exit_conditions(self, trading_date: date, data: MarketData):
        """检查止损止盈条件"""
        for symbol in list(self.positions.keys()):
            position = self.positions[symbol]
            current_price = data.get_close(symbol, trading_date)
            exit_reason = self.risk_manager.check_exit(position, current_price)
            if exit_reason:
                self._execute_sell(symbol, trading_date, current_price, exit_reason)

    def _update_equity(self, trading_date: date, data: MarketData):
        """更新权益曲线"""
        position_value = 0.0
        for symbol, position in self.positions.items():
            current_price = data.get_close(symbol, trading_date)
            position_value += current_price * position.quantity

        total_equity = self.cash + position_value
        self.equity_curve.append(total_equity)

    def _generate_result(self) -> BacktestResult:
        """生成回测结果"""
        initial_capital = self.config.initial_capital
        final_capital = self.cash + sum(
            self._data.get_close(symbol, date) * pos.quantity
            for symbol, pos in self.positions.items()
        )

        # 计算总收益率
        total_return = (final_capital - initial_capital) / initial_capital

        # 计算年化收益率
        trading_days = len(self.equity_curve)
        if trading_days > 0:
            annual_return = (final_capital / initial_capital) ** (252 / trading_days) - 1
        else:
            annual_return = 0.0

        # 计算夏普比率
        sharpe_ratio = self._calculate_sharpe_ratio()

        # 计算最大回撤
        max_drawdown = self._calculate_max_drawdown()

        # 计算胜率
        win_rate = self._calculate_win_rate()

        # 计算平均持仓天数
        avg_holding_days = self._calculate_avg_holding_days()

        # 创建权益曲线 Series
        equity_series = pd.Series(self.equity_curve)

        return BacktestResult(
            initial_capital=initial_capital,
            final_capital=final_capital,
            total_return=total_return,
            annual_return=annual_return,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            trade_count=len(self.trades),
            avg_holding_days=avg_holding_days,
            equity_curve=equity_series,
            trades=self.trades,
            positions=self.positions.copy()
        )

    def _calculate_sharpe_ratio(self, risk_free_rate: float = 0.0) -> float:
        """计算夏普比率"""
        if len(self.equity_curve) < 2:
            return 0.0

        equity_series = pd.Series(self.equity_curve)
        returns = equity_series.pct_change().dropna()

        if len(returns) == 0 or returns.std() == 0:
            return 0.0

        annual_return = returns.mean() * 252
        annual_volatility = returns.std() * np.sqrt(252)

        if annual_volatility == 0:
            return 0.0

        sharpe_ratio = (annual_return - risk_free_rate) / annual_volatility
        return sharpe_ratio

    def _calculate_max_drawdown(self) -> float:
        """计算最大回撤"""
        if len(self.equity_curve) == 0:
            return 0.0

        equity_series = pd.Series(self.equity_curve)
        cumulative = equity_series
        peak = cumulative.expanding(min_periods=1).max()
        drawdown = (cumulative - peak) / peak

        max_drawdown = drawdown.min()
        return max_drawdown if not pd.isna(max_drawdown) else 0.0

    def _calculate_win_rate(self) -> float:
        """计算胜率 (盈利交易 / 总交易)"""
        sell_trades = [t for t in self.trades if t.action == "SELL"]
        if not sell_trades:
            return 0.0

        winning_trades = 0
        for i in range(0, len(sell_trades)):
            sell_trade = sell_trades[i]
            # 找到对应的买入交易
            symbol_trades = [t for t in self.trades if t.symbol == sell_trade.symbol]
            buy_trades = [t for t in symbol_trades if t.action == "BUY"]
            if not buy_trades:
                continue

            last_buy = None
            for t in buy_trades:
                if t.date <= sell_trade.date:
                    last_buy = t

            if last_buy is None:
                continue

            profit = (sell_trade.price - last_buy.price) * sell_trade.quantity - \
                     last_buy.commission - sell_trade.commission
            if profit > 0:
                winning_trades += 1

        return winning_trades / len(sell_trades) if sell_trades else 0.0

    def _calculate_avg_holding_days(self) -> float:
        """计算平均持仓天数"""
        sell_trades = [t for t in self.trades if t.action == "SELL"]
        if not sell_trades:
            return 0.0

        holding_days_list = []
        for i in range(0, len(sell_trades)):
            sell_trade = sell_trades[i]
            symbol_trades = [t for t in self.trades if t.symbol == sell_trade.symbol]
            buy_trades = [t for t in symbol_trades if t.action == "BUY"]
            if not buy_trades:
                continue

            last_buy = None
            for t in buy_trades:
                if t.date <= sell_trade.date:
                    last_buy = t

            if last_buy is None:
                continue

            holding_days = (sell_trade.date - last_buy.date).days
            holding_days_list.append(holding_days)

        return np.mean(holding_days_list) if holding_days_list else 0.0

    