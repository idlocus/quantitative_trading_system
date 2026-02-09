#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
交易所执行模块
"""

import logging
import ccxt
from datetime import datetime

class ExchangeExecution:
    """
    交易所执行类，用于与交易所API交互
    """
    
    def __init__(self, config):
        """
        初始化交易所执行
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.execution_config = config.execution_config
        
        # 交易所配置
        self.exchange_name = self.execution_config.get('exchange', 'binance')
        self.order_type = self.execution_config.get('order_type', 'market')
        self.testnet = self.execution_config.get('testnet', False)
        
        # 交易所连接
        self.exchange = None
        self.connected = False
    
    def connect(self):
        """
        连接交易所
        """
        try:
            # 获取API密钥
            data_config = self.config.data_config
            sources = data_config.get('sources', {})
            exchange_config = sources.get(self.exchange_name, {})
            
            api_key = exchange_config.get('api_key', '')
            api_secret = exchange_config.get('api_secret', '')
            
            # 初始化交易所
            if hasattr(ccxt, self.exchange_name):
                exchange_class = getattr(ccxt, self.exchange_name)
                self.exchange = exchange_class({
                    'apiKey': api_key,
                    'secret': api_secret,
                    'enableRateLimit': True,
                    'options': {
                        'defaultType': 'spot'
                    }
                })
                
                # 启用测试网
                if self.testnet and hasattr(self.exchange, 'set_sandbox_mode'):
                    self.exchange.set_sandbox_mode(True)
                
                # 测试连接
                markets = self.exchange.load_markets()
                self.logger.info(f"成功连接到 {self.exchange_name}，加载了 {len(markets)} 个交易对")
                
                self.connected = True
            else:
                raise Exception(f"不支持的交易所: {self.exchange_name}")
            
        except Exception as e:
            self.logger.error(f"连接交易所失败: {str(e)}")
            raise
    
    def disconnect(self):
        """
        断开交易所连接
        """
        try:
            # CCXT 没有明确的断开连接方法
            self.exchange = None
            self.connected = False
            self.logger.info(f"断开与 {self.exchange_name} 的连接")
            
        except Exception as e:
            self.logger.error(f"断开交易所连接失败: {str(e)}")
            raise
    
    def submit_order(self, order):
        """
        提交订单
        """
        try:
            if not self.connected:
                raise Exception("未连接到交易所")
            
            symbol = order['symbol']
            side = order['side']
            quantity = order['quantity']
            order_type = order['type']
            price = order.get('price')
            
            # 提交订单
            if order_type == 'market':
                response = self.exchange.create_order(
                    symbol=symbol,
                    type=order_type,
                    side=side,
                    amount=quantity
                )
            elif order_type == 'limit':
                if not price:
                    raise Exception("限价单必须指定价格")
                
                response = self.exchange.create_order(
                    symbol=symbol,
                    type=order_type,
                    side=side,
                    amount=quantity,
                    price=price
                )
            else:
                raise Exception(f"不支持的订单类型: {order_type}")
            
            # 处理响应
            submitted_order = {
                'order_id': response.get('id', order['order_id']),
                'status': self._map_status(response.get('status', 'open')),
                'filled_quantity': float(response.get('filled', 0)),
                'avg_price': float(response.get('average', 0)) if 'average' in response else price,
                'exchange_response': response
            }
            
            self.logger.info(f"订单提交成功: {submitted_order}")
            
            return submitted_order
            
        except Exception as e:
            self.logger.error(f"提交订单失败: {str(e)}")
            return {
                'order_id': order['order_id'],
                'status': 'rejected',
                'reason': str(e)
            }
    
    def cancel_order(self, order_id):
        """
        取消订单
        """
        try:
            if not self.connected:
                raise Exception("未连接到交易所")
            
            # 取消订单
            response = self.exchange.cancel_order(order_id)
            
            # 处理响应
            cancel_result = {
                'order_id': order_id,
                'status': self._map_status(response.get('status', 'canceled')),
                'exchange_response': response
            }
            
            self.logger.info(f"订单取消成功: {cancel_result}")
            
            return cancel_result
            
        except Exception as e:
            self.logger.error(f"取消订单失败: {str(e)}")
            return {
                'order_id': order_id,
                'status': 'rejected',
                'reason': str(e)
            }
    
    def get_order_status(self, order_id, symbol):
        """
        获取订单状态
        """
        try:
            if not self.connected:
                raise Exception("未连接到交易所")
            
            # 获取订单
            response = self.exchange.fetch_order(order_id, symbol)
            
            # 处理响应
            order_status = {
                'order_id': order_id,
                'status': self._map_status(response.get('status', 'open')),
                'filled_quantity': float(response.get('filled', 0)),
                'avg_price': float(response.get('average', 0)) if 'average' in response else 0,
                'exchange_response': response
            }
            
            return order_status
            
        except Exception as e:
            self.logger.error(f"获取订单状态失败: {str(e)}")
            raise
    
    def _map_status(self, exchange_status):
        """
        映射交易所状态到系统状态
        """
        status_map = {
            'open': 'submitted',
            'closed': 'filled',
            'canceled': 'cancelled',
            'rejected': 'rejected',
            'expired': 'cancelled',
            'pending': 'submitted'
        }
        
        return status_map.get(exchange_status, 'submitted')
    
    def get_balance(self):
        """
        获取账户余额
        """
        try:
            if not self.connected:
                raise Exception("未连接到交易所")
            
            balance = self.exchange.fetch_balance()
            return balance
            
        except Exception as e:
            self.logger.error(f"获取余额失败: {str(e)}")
            raise
    
    def get_market_price(self, symbol):
        """
        获取市场价格
        """
        try:
            if not self.connected:
                raise Exception("未连接到交易所")
            
            ticker = self.exchange.fetch_ticker(symbol)
            return {
                'symbol': symbol,
                'bid': ticker.get('bid', 0),
                'ask': ticker.get('ask', 0),
                'last': ticker.get('last', 0),
                'timestamp': ticker.get('timestamp')
            }
            
        except Exception as e:
            self.logger.error(f"获取市场价格失败: {str(e)}")
            raise
    
    def is_connected(self):
        """
        检查是否连接
        """
        return self.connected
