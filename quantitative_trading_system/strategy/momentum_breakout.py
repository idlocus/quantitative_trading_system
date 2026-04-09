#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
动量突破策略
适应最近市场环境的策略，结合动量和突破信号
"""

import logging
import pandas as pd
import numpy as np
from .base_strategy import BaseStrategy

class MomentumBreakoutStrategy(BaseStrategy):
    """
    动量突破策略
    结合动量指标和突破信号，适应震荡上行和波动较大的市场环境
    """
    
    def __init__(self, config):
        """
        初始化动量突破策略
        """
        super().__init__(config)
        self.logger = logging.getLogger(__name__)
        
        # 策略参数
        self.momentum_period = self.params.get('momentum_period', 20)  # 动量周期
        self.breakout_period = self.params.get('breakout_period', 50)  # 突破周期
        self.rsi_period = self.params.get('rsi_period', 14)  # RSI周期
        self.rsi_overbought = self.params.get('rsi_overbought', 70)  # RSI超买阈值
        self.rsi_oversold = self.params.get('rsi_oversold', 30)  # RSI超卖阈值
        self.stop_loss_pct = self.params.get('stop_loss_pct', 0.03)  # 止损百分比
        self.take_profit_pct = self.params.get('take_profit_pct', 0.06)  # 止盈百分比
        
        # 预处理的指标数据
        self.indicators = None
        self.current_index = 0
        self.last_entry_price = 0
    
    def initialize(self):
        """
        初始化策略
        """
        self.logger.info(f"初始化动量突破策略，参数: momentum_period={self.momentum_period}, breakout_period={self.breakout_period}, rsi_period={self.rsi_period}")
        
        # 重置状态
        self.position = 0
        self.signals = []
        self.orders = []
        self.current_index = 0
        self.last_entry_price = 0
    
    def preprocess_data(self, data):
        """
        预处理数据，计算所有指标
        """
        self.logger.debug("预处理数据，计算技术指标")
        
        # 计算动量指标
        momentum = data['close'].pct_change(periods=self.momentum_period)
        
        # 计算突破指标（最高价和最低价）
        highest_high = data['high'].rolling(window=self.breakout_period).max()
        lowest_low = data['low'].rolling(window=self.breakout_period).min()
        
        # 计算RSI
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        # 计算移动平均线
        ma20 = data['close'].rolling(window=20).mean()
        ma50 = data['close'].rolling(window=50).mean()
        
        # 存储指标数据
        self.indicators = {
            'momentum': momentum,
            'highest_high': highest_high,
            'lowest_low': lowest_low,
            'rsi': rsi,
            'ma20': ma20,
            'ma50': ma50
        }
        
        self.logger.debug("指标计算完成")
    
    def on_data(self, row):
        """
        处理数据，生成交易信号
        """
        try:
            # 获取数据
            close = row.close
            high = row.high
            low = row.low
            timestamp = row.Index
            symbol = getattr(self, 'symbol', '300308.SZ')  # 使用策略属性中的股票代码
            
            # 生成信号
            signal = self._generate_signal(close, high, low, timestamp, symbol)
            
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
    
    def _generate_signal(self, price, high, low, timestamp, symbol):
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
        if self.current_index < max(self.momentum_period, self.breakout_period, self.rsi_period):
            signal = self.generate_signal(symbol, 'hold', price, timestamp)
            signal['reason'] = '指标数据不足'
            self.signals.append(signal)
            return signal
        
        # 获取最新指标值
        current_momentum = self.indicators['momentum'].iloc[self.current_index]
        current_highest = self.indicators['highest_high'].iloc[self.current_index]
        current_lowest = self.indicators['lowest_low'].iloc[self.current_index]
        current_rsi = self.indicators['rsi'].iloc[self.current_index]
        current_ma20 = self.indicators['ma20'].iloc[self.current_index]
        current_ma50 = self.indicators['ma50'].iloc[self.current_index]
        
        # 检查止损和止盈
        if self.position == 1:
            # 多头持仓，检查止损和止盈
            if price <= self.last_entry_price * (1 - self.stop_loss_pct):
                # 触发止损
                signal = self.generate_signal(symbol, 'sell', price, timestamp)
                signal['reason'] = f'触发止损，当前价格: {price:.2f}，止损价格: {self.last_entry_price * (1 - self.stop_loss_pct):.2f}'
                signal['indicators'] = {
                    'current_rsi': current_rsi,
                    'current_momentum': current_momentum,
                    'current_ma20': current_ma20,
                    'current_ma50': current_ma50
                }
                self.update_position(0)
                self.signals.append(signal)
                return signal
            elif price >= self.last_entry_price * (1 + self.take_profit_pct):
                # 触发止盈
                signal = self.generate_signal(symbol, 'sell', price, timestamp)
                signal['reason'] = f'触发止盈，当前价格: {price:.2f}，止盈价格: {self.last_entry_price * (1 + self.take_profit_pct):.2f}'
                signal['indicators'] = {
                    'current_rsi': current_rsi,
                    'current_momentum': current_momentum,
                    'current_ma20': current_ma20,
                    'current_ma50': current_ma50
                }
                self.update_position(0)
                self.signals.append(signal)
                return signal
        
        # 生成信号
        if self.position == 0:
            # 无持仓，检查买入信号
            buy_conditions = [
                # 动量为正，价格有向上 momentum
                current_momentum > 0,
                # 价格突破近期高点
                high > current_highest,
                # RSI未超买
                current_rsi < self.rsi_overbought,
                # 短期均线上穿长期均线（金叉）
                current_ma20 > current_ma50
            ]
            
            if all(buy_conditions):
                # 满足买入条件
                signal = self.generate_signal(symbol, 'buy', price, timestamp)
                signal['reason'] = '满足买入条件: 正动量 + 突破新高 + 均线金叉'
                signal['indicators'] = {
                    'current_rsi': current_rsi,
                    'current_momentum': current_momentum,
                    'current_ma20': current_ma20,
                    'current_ma50': current_ma50,
                    'highest_high': current_highest
                }
                self.update_position(1)
                self.last_entry_price = price
                self.signals.append(signal)
                return signal
            else:
                # 无信号，持有
                signal = self.generate_signal(symbol, 'hold', price, timestamp)
                signal['reason'] = '未满足买入条件'
                signal['indicators'] = {
                    'current_rsi': current_rsi,
                    'current_momentum': current_momentum,
                    'current_ma20': current_ma20,
                    'current_ma50': current_ma50
                }
                self.signals.append(signal)
                return signal
        elif self.position == 1:
            # 持有多头，检查卖出信号
            sell_conditions = [
                # 动量为负，价格有向下 momentum
                current_momentum < 0,
                # RSI超买
                current_rsi > self.rsi_overbought,
                # 短期均线下穿长期均线（死叉）
                current_ma20 < current_ma50
            ]
            
            if any(sell_conditions):
                # 满足卖出条件
                signal = self.generate_signal(symbol, 'sell', price, timestamp)
                signal['reason'] = '满足卖出条件: 负动量 + RSI超买 + 均线死叉'
                signal['indicators'] = {
                    'current_rsi': current_rsi,
                    'current_momentum': current_momentum,
                    'current_ma20': current_ma20,
                    'current_ma50': current_ma50
                }
                self.update_position(0)
                self.signals.append(signal)
                return signal
            else:
                # 无信号，持有
                signal = self.generate_signal(symbol, 'hold', price, timestamp)
                signal['reason'] = '未满足卖出条件'
                signal['indicators'] = {
                    'current_rsi': current_rsi,
                    'current_momentum': current_momentum,
                    'current_ma20': current_ma20,
                    'current_ma50': current_ma50
                }
                self.signals.append(signal)
                return signal
        else:
            # 其他情况，持有
            signal = self.generate_signal(symbol, 'hold', price, timestamp)
            signal['reason'] = '未知持仓状态'
            signal['indicators'] = {
                'current_rsi': current_rsi,
                'current_momentum': current_momentum,
                'current_ma20': current_ma20,
                'current_ma50': current_ma50
            }
            self.signals.append(signal)
            return signal
