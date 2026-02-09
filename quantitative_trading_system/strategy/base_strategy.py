#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
策略基础类
"""

import abc
import logging
import pandas as pd

class BaseStrategy(metaclass=abc.ABCMeta):
    """
    策略基础类，定义策略的基本接口
    """
    
    def __init__(self, config):
        """
        初始化策略
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.strategy_config = config.strategy_config
        self.params = self.strategy_config.get('params', {})
        
        # 策略状态
        self.running = False
        self.position = 0  # 持仓状态: -1=空仓, 0=无持仓, 1=多仓
        self.signals = []  # 信号历史
        self.orders = []  # 订单历史
    
    @abc.abstractmethod
    def initialize(self):
        """
        初始化策略
        """
        pass
    
    @abc.abstractmethod
    def on_data(self, data):
        """
        处理数据，生成交易信号
        """
        pass
    
    @abc.abstractmethod
    def on_order_update(self, order):
        """
        处理订单更新
        """
        pass
    
    @abc.abstractmethod
    def on_position_update(self, position):
        """
        处理持仓更新
        """
        pass
    
    def start(self):
        """
        启动策略
        """
        try:
            self.initialize()
            self.running = True
            self.logger.info(f"策略 {self.__class__.__name__} 启动")
        except Exception as e:
            self.logger.error(f"策略启动失败: {str(e)}")
            raise
    
    def stop(self):
        """
        停止策略
        """
        self.running = False
        self.logger.info(f"策略 {self.__class__.__name__} 停止")
    
    def generate_signal(self, symbol, signal_type, price, timestamp):
        """
        生成交易信号
        """
        signal = {
            'symbol': symbol,
            'type': signal_type,  # 'buy', 'sell', 'hold'
            'price': price,
            'timestamp': timestamp,
            'strategy': self.__class__.__name__
        }
        
        self.signals.append(signal)
        self.logger.info(f"生成信号: {signal}")
        
        return signal
    
    def update_position(self, position):
        """
        更新持仓状态
        """
        old_position = self.position
        self.position = position
        self.on_position_update(position)
        self.logger.info(f"持仓更新: {old_position} -> {position}")
    
    def get_signals(self, start_time=None, end_time=None):
        """
        获取信号历史
        """
        if start_time is None and end_time is None:
            return self.signals
        
        filtered_signals = []
        for signal in self.signals:
            timestamp = signal.get('timestamp')
            if timestamp:
                if start_time and timestamp < start_time:
                    continue
                if end_time and timestamp > end_time:
                    continue
                filtered_signals.append(signal)
        
        return filtered_signals
    
    def get_orders(self, start_time=None, end_time=None):
        """
        获取订单历史
        """
        if start_time is None and end_time is None:
            return self.orders
        
        filtered_orders = []
        for order in self.orders:
            timestamp = order.get('timestamp')
            if timestamp:
                if start_time and timestamp < start_time:
                    continue
                if end_time and timestamp > end_time:
                    continue
                filtered_orders.append(order)
        
        return filtered_orders
    
    def is_running(self):
        """
        检查策略是否运行中
        """
        return self.running
    
    def get_position(self):
        """
        获取当前持仓状态
        """
        return self.position
    
    def set_params(self, params):
        """
        设置策略参数
        """
        self.params.update(params)
        self.logger.info(f"策略参数更新: {params}")
    
    def get_params(self):
        """
        获取策略参数
        """
        return self.params
