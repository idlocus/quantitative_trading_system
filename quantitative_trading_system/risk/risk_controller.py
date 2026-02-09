#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
风险管理模块
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class RiskController:
    """
    风险控制器类，用于监控和控制交易风险
    """
    
    def __init__(self, config):
        """
        初始化风险控制器
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.risk_config = config.risk_config
        
        # 风险参数
        self.max_position_size = self.risk_config.get('max_position_size', 0.2)
        self.max_drawdown = self.risk_config.get('max_drawdown', 0.1)
        self.stop_loss_pct = self.risk_config.get('stop_loss_pct', 0.02)
        self.take_profit_pct = self.risk_config.get('take_profit_pct', 0.04)
        
        # 风险状态
        self.risk_metrics = {}
        self.positions = {}
        self.portfolio_value = 0
        self.peak_value = 0
        self.drawdown = 0
        
        # 风险预警
        self.alerts = []
        self.risk_limits_exceeded = False
    
    def start(self):
        """
        启动风险控制器
        """
        try:
            self.initialize()
            self.logger.info("风险控制器启动")
        except Exception as e:
            self.logger.error(f"风险控制器启动失败: {str(e)}")
            raise
    
    def stop(self):
        """
        停止风险控制器
        """
        self.logger.info("风险控制器停止")
    
    def initialize(self):
        """
        初始化风险控制器
        """
        # 初始化风险指标
        self.risk_metrics = {
            'position_size': {},
            'drawdown': 0,
            'volatility': 0,
            'sharpe_ratio': 0,
            'max_leverage': 0,
            'trade_frequency': 0
        }
        
        # 初始化预警
        self.alerts = []
        self.risk_limits_exceeded = False
    
    def evaluate_risk(self, portfolio_value, positions, trades=None):
        """
        评估风险
        """
        try:
            # 更新组合价值
            self.portfolio_value = portfolio_value
            self.positions = positions
            
            # 计算最大回撤
            if portfolio_value > self.peak_value:
                self.peak_value = portfolio_value
            
            self.drawdown = (self.peak_value - portfolio_value) / self.peak_value if self.peak_value > 0 else 0
            self.risk_metrics['drawdown'] = self.drawdown
            
            # 计算持仓风险
            for symbol, position in positions.items():
                position_size = abs(position['value']) / portfolio_value if portfolio_value > 0 else 0
                self.risk_metrics['position_size'][symbol] = position_size
                
                # 检查单个持仓限制
                if position_size > self.max_position_size:
                    self._add_alert(
                        'position_size',
                        f"持仓过大: {symbol} 占比 {position_size:.2%}，超过限制 {self.max_position_size:.2%}"
                    )
            
            # 检查回撤限制
            if self.drawdown > self.max_drawdown:
                self._add_alert(
                    'drawdown',
                    f"回撤过大: {self.drawdown:.2%}，超过限制 {self.max_drawdown:.2%}"
                )
            
            # 检查交易频率
            if trades:
                self._check_trade_frequency(trades)
            
            # 计算整体风险指标
            self._calculate_portfolio_risk()
            
            # 检查是否超出风险限制
            self.risk_limits_exceeded = len([a for a in self.alerts if a['level'] == 'critical']) > 0
            
            return self.risk_metrics, self.alerts
            
        except Exception as e:
            self.logger.error(f"风险评估失败: {str(e)}")
            raise
    
    def should_execute_trade(self, trade):
        """
        判断是否应该执行交易
        """
        try:
            # 检查风险限制是否已超出
            if self.risk_limits_exceeded:
                self.logger.warning("风险限制已超出，拒绝执行交易")
                return False
            
            # 模拟执行交易后的风险
            symbol = trade['symbol']
            side = trade['side']
            quantity = trade['quantity']
            price = trade.get('price', 0)
            
            # 计算交易价值
            trade_value = quantity * price
            
            # 计算交易后持仓
            new_positions = self.positions.copy()
            if symbol in new_positions:
                current_position = new_positions[symbol]['value']
                if side == 'buy':
                    new_position_value = current_position + trade_value
                else:
                    new_position_value = current_position - trade_value
                new_positions[symbol]['value'] = new_position_value
            else:
                new_position_value = trade_value if side == 'buy' else -trade_value
                new_positions[symbol] = {
                    'value': new_position_value,
                    'quantity': quantity if side == 'buy' else -quantity,
                    'avg_price': price
                }
            
            # 计算交易后组合价值
            new_portfolio_value = self.portfolio_value
            if side == 'buy':
                new_portfolio_value -= trade_value
            else:
                new_portfolio_value += trade_value
            
            # 评估交易后风险
            risk_metrics, alerts = self.evaluate_risk(new_portfolio_value, new_positions)
            
            # 检查是否会超出风险限制
            for alert in alerts:
                if alert['level'] == 'critical':
                    self.logger.warning(f"交易将导致风险限制超出: {alert['message']}")
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"交易风险评估失败: {str(e)}")
            return False
    
    def get_stop_loss_price(self, symbol, entry_price, side):
        """
        获取止损价格
        """
        if side == 'buy':
            return entry_price * (1 - self.stop_loss_pct)
        else:
            return entry_price * (1 + self.stop_loss_pct)
    
    def get_take_profit_price(self, symbol, entry_price, side):
        """
        获取止盈价格
        """
        if side == 'buy':
            return entry_price * (1 + self.take_profit_pct)
        else:
            return entry_price * (1 - self.take_profit_pct)
    
    def _add_alert(self, alert_type, message, level='warning'):
        """
        添加风险预警
        """
        alert = {
            'type': alert_type,
            'message': message,
            'level': level,  # info, warning, critical
            'timestamp': datetime.now()
        }
        
        self.alerts.append(alert)
        self.logger.warning(f"风险预警: {message}")
    
    def _check_trade_frequency(self, trades):
        """
        检查交易频率
        """
        # 计算最近24小时的交易次数
        now = datetime.now()
        recent_trades = [t for t in trades if t['timestamp'] > now - timedelta(days=1)]
        trade_frequency = len(recent_trades)
        
        self.risk_metrics['trade_frequency'] = trade_frequency
        
        # 检查交易频率是否过高
        if trade_frequency > 100:  # 示例阈值
            self._add_alert(
                'trade_frequency',
                f"交易频率过高: 最近24小时 {trade_frequency} 笔交易"
            )
    
    def _calculate_portfolio_risk(self):
        """
        计算组合风险
        """
        # 计算总持仓价值
        total_position_value = sum(abs(p['value']) for p in self.positions.values())
        
        # 计算最大杠杆
        max_leverage = total_position_value / self.portfolio_value if self.portfolio_value > 0 else 0
        self.risk_metrics['max_leverage'] = max_leverage
        
        # 检查杠杆限制
        if max_leverage > 5:  # 示例阈值
            self._add_alert(
                'leverage',
                f"杠杆过高: {max_leverage:.2f}x"
            )
    
    def get_risk_metrics(self):
        """
        获取风险指标
        """
        return self.risk_metrics
    
    def get_alerts(self, since=None):
        """
        获取风险预警
        """
        if since:
            return [a for a in self.alerts if a['timestamp'] >= since]
        return self.alerts
    
    def reset_alerts(self):
        """
        重置风险预警
        """
        self.alerts = []
        self.risk_limits_exceeded = False
