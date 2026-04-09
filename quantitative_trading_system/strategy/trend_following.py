#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
趋势跟踪策略
"""

import logging
import pandas as pd
import numpy as np
from .base_strategy import BaseStrategy

class TrendFollowingStrategy(BaseStrategy):
    """
    趋势跟踪策略，基于移动平均线交叉
    """

    def __init__(self, config):
        """
        初始化趋势跟踪策略
        """
        super().__init__(config)
        self.logger = logging.getLogger(__name__)

        # 策略参数
        self.fast_period = self.params.get('fast_period', 10)
        self.slow_period = self.params.get('slow_period', 20)
        self.signal_period = self.params.get('signal_period', 9)

        # 风控参数
        self.stop_loss_pct = self.params.get('stop_loss_pct', 0.05)  # 止损5%
        self.take_profit_pct = self.params.get('take_profit_pct', 0.15)  # 止盈15%
        self.trailing_stop_pct = self.params.get('trailing_stop_pct', 0.08)  # 移动止损8%
        self.position_size = self.params.get('position_size', 1.0)  # 仓位比例1.0=100%

        # 持仓成本记录（用于止损止盈计算）
        self.entry_price = 0
        self.highest_price_since_entry = 0

        # 预处理的指标数据
        self.indicators = None
        self.current_index = 0
    
    def initialize(self):
        """
        初始化策略
        """
        self.logger.info(f"初始化趋势跟踪策略，参数: fast_period={self.fast_period}, slow_period={self.slow_period}, signal_period={self.signal_period}")
        
        # 重置状态
        self.position = 0
        self.signals = []
        self.orders = []
        self.current_index = 0
    
    def preprocess_data(self, data):
        """
        预处理数据，计算所有指标
        """
        self.logger.debug("预处理数据，计算技术指标")
        
        # 计算移动平均线
        fast_ma = data['close'].rolling(window=self.fast_period).mean()
        slow_ma = data['close'].rolling(window=self.slow_period).mean()
        
        # 计算MACD
        macd_line = fast_ma - slow_ma
        signal_line = macd_line.rolling(window=self.signal_period).mean()
        
        # 存储指标数据
        self.indicators = {
            'fast_ma': fast_ma,
            'slow_ma': slow_ma,
            'macd_line': macd_line,
            'signal_line': signal_line
        }
        
        self.logger.debug("指标计算完成")
    
    def on_data(self, row):
        """
        处理数据，生成交易信号
        """
        try:
            # 获取数据
            close = row.close
            timestamp = row.Index
            symbol = getattr(self, 'symbol', '300308.SZ')  # 使用策略属性中的股票代码
            
            # 生成信号
            signal = self._generate_signal(close, timestamp, symbol)
            
            # 更新索引
            self.current_index += 1
            
            return signal
            
        except Exception as e:
            self.logger.debug(f"处理数据失败: {str(e)}")
            symbol = getattr(self, 'symbol', '300308.SZ')
            return self.generate_signal(symbol, 'hold', 0, timestamp)
    
    def on_order_update(self, order):
        """
        处理订单更新
        """
        self.logger.debug(f"订单更新: {order}")
        self.orders.append(order)
    
    def on_position_update(self, position):
        """
        处理持仓更新
        """
        self.logger.debug(f"持仓更新: {position}")
        self.position = position
    
    def _generate_signal(self, price, timestamp, symbol):
        """
        生成交易信号
        """
        # 确保有指标数据
        if self.indicators is None:
            signal = self.generate_signal(symbol, 'hold', price, timestamp)
            signal['reason'] = '指标数据未初始化'
            self.signals.append(signal)
            return signal

        # 确保有足够的指标数据
        if self.current_index < max(self.slow_period + self.signal_period, 2):
            signal = self.generate_signal(symbol, 'hold', price, timestamp)
            signal['reason'] = '指标数据不足'
            self.signals.append(signal)
            return signal

        # 获取最新指标值
        current_macd = self.indicators['macd_line'].iloc[self.current_index]
        current_signal = self.indicators['signal_line'].iloc[self.current_index]
        previous_macd = self.indicators['macd_line'].iloc[self.current_index - 1]
        previous_signal = self.indicators['signal_line'].iloc[self.current_index - 1]

        # 生成信号
        if self.position == 0:
            # 无持仓，检查买入信号
            if current_macd > current_signal and previous_macd <= previous_signal:
                # MACD金叉，买入
                signal = self.generate_signal(symbol, 'buy', price, timestamp)
                signal['reason'] = 'MACD金叉'
                signal['indicators'] = {
                    'current_macd': current_macd,
                    'current_signal': current_signal,
                }
                self.update_position(1)
                # 记录买入价格
                self.entry_price = price
                self.highest_price_since_entry = price
                self.signals.append(signal)
                return signal
            else:
                # 无信号，持有
                signal = self.generate_signal(symbol, 'hold', price, timestamp)
                signal['reason'] = '未满足买入条件'
                signal['indicators'] = {
                    'current_macd': current_macd,
                    'current_signal': current_signal,
                }
                self.signals.append(signal)
                return signal

        elif self.position == 1:
            # 持有多头，先检查风控条件

            # 更新最高价
            if price > self.highest_price_since_entry:
                self.highest_price_since_entry = price

            # 计算当前盈亏
            profit_pct = (price - self.entry_price) / self.entry_price

            # 1. 止损检查
            if price < self.entry_price * (1 - self.stop_loss_pct):
                signal = self.generate_signal(symbol, 'sell', price, timestamp)
                signal['reason'] = f'止损（价格{price:.2f} < 触发价{self.entry_price * (1 - self.stop_loss_pct):.2f}）'
                signal['indicators'] = {'profit_pct': profit_pct}
                self.update_position(0)
                self._reset_entry()
                self.signals.append(signal)
                return signal

            # 2. 止盈检查（固定止盈）
            if price > self.entry_price * (1 + self.take_profit_pct):
                signal = self.generate_signal(symbol, 'sell', price, timestamp)
                signal['reason'] = f'止盈（价格{price:.2f} > 触发价{self.entry_price * (1 + self.take_profit_pct):.2f}）'
                signal['indicators'] = {'profit_pct': profit_pct}
                self.update_position(0)
                self._reset_entry()
                self.signals.append(signal)
                return signal

            # 3. 移动止损检查（保住部分利润）
            if self.highest_price_since_entry > self.entry_price * (1 + self.trailing_stop_pct):
                # 价格从最高点回撤超过移动止损幅度
                drawdown_from_high = (self.highest_price_since_entry - price) / self.highest_price_since_entry
                if drawdown_from_high > self.trailing_stop_pct:
                    signal = self.generate_signal(symbol, 'sell', price, timestamp)
                    signal['reason'] = f'移动止损（回撤{drawdown_from_high*100:.1f}%）'
                    signal['indicators'] = {'profit_pct': profit_pct}
                    self.update_position(0)
                    self._reset_entry()
                    self.signals.append(signal)
                    return signal

            # 4. MACD死叉检查
            if current_macd < current_signal and previous_macd >= previous_signal:
                signal = self.generate_signal(symbol, 'sell', price, timestamp)
                signal['reason'] = 'MACD死叉'
                signal['indicators'] = {
                    'current_macd': current_macd,
                    'current_signal': current_signal,
                    'profit_pct': profit_pct
                }
                self.update_position(0)
                self._reset_entry()
                self.signals.append(signal)
                return signal

            # 无信号，持有
            signal = self.generate_signal(symbol, 'hold', price, timestamp)
            signal['reason'] = f'持有中（浮盈{profit_pct*100:.1f}%）'
            signal['indicators'] = {
                'current_macd': current_macd,
                'current_signal': current_signal,
                'profit_pct': profit_pct
            }
            self.signals.append(signal)
            return signal

        else:
            # 其他情况，持有
            signal = self.generate_signal(symbol, 'hold', price, timestamp)
            signal['reason'] = '未知持仓状态'
            self.signals.append(signal)
            return signal

    def _reset_entry(self):
        """
        重置入场记录
        """
        self.entry_price = 0
        self.highest_price_since_entry = 0
