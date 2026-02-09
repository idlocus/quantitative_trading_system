#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
订单管理模块
"""

import logging
import uuid
from datetime import datetime

class OrderManager:
    """
    订单管理类，用于管理交易订单
    """
    
    def __init__(self, config):
        """
        初始化订单管理器
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.execution_config = config.execution_config
        
        # 订单状态
        self.orders = {}
        self.order_history = []
        
        # 执行接口
        self.execution = None
        
        # 连接状态
        self.connected = False
    
    def connect(self):
        """
        连接交易接口
        """
        try:
            # 初始化执行接口
            from .exchange_execution import ExchangeExecution
            self.execution = ExchangeExecution(self.config)
            self.execution.connect()
            
            self.connected = True
            self.logger.info("订单管理器连接成功")
            
        except Exception as e:
            self.logger.error(f"订单管理器连接失败: {str(e)}")
            raise
    
    def disconnect(self):
        """
        断开交易接口连接
        """
        try:
            if self.execution:
                self.execution.disconnect()
            
            self.connected = False
            self.logger.info("订单管理器断开连接")
            
        except Exception as e:
            self.logger.error(f"订单管理器断开连接失败: {str(e)}")
            raise
    
    def create_order(self, symbol, order_type, side, quantity, price=None, **kwargs):
        """
        创建订单
        """
        try:
            # 生成订单ID
            order_id = str(uuid.uuid4())
            
            # 创建订单对象
            order = {
                'order_id': order_id,
                'symbol': symbol,
                'type': order_type,  # market, limit
                'side': side,  # buy, sell
                'quantity': quantity,
                'price': price,
                'status': 'pending',  # pending, submitted, filled, cancelled, rejected
                'filled_quantity': 0,
                'avg_price': 0,
                'timestamp': datetime.now(),
                'submitted_timestamp': None,
                'filled_timestamp': None,
                'reason': None,
                **kwargs
            }
            
            # 保存订单
            self.orders[order_id] = order
            
            self.logger.info(f"创建订单: {order}")
            
            # 提交订单
            if self.connected:
                submitted_order = self.execution.submit_order(order)
                order['status'] = submitted_order.get('status', 'submitted')
                order['submitted_timestamp'] = datetime.now()
                
                # 如果订单已成交
                if order['status'] == 'filled':
                    order['filled_quantity'] = submitted_order.get('filled_quantity', quantity)
                    order['avg_price'] = submitted_order.get('avg_price', price)
                    order['filled_timestamp'] = datetime.now()
                    self.order_history.append(order)
                    del self.orders[order_id]
            
            return order
            
        except Exception as e:
            self.logger.error(f"创建订单失败: {str(e)}")
            raise
    
    def cancel_order(self, order_id):
        """
        取消订单
        """
        try:
            if order_id not in self.orders:
                self.logger.warning(f"订单不存在: {order_id}")
                return None
            
            order = self.orders[order_id]
            
            # 提交取消请求
            if self.connected:
                cancel_result = self.execution.cancel_order(order_id)
                order['status'] = cancel_result.get('status', 'cancelled')
                order['reason'] = cancel_result.get('reason', 'User cancelled')
            else:
                order['status'] = 'cancelled'
                order['reason'] = 'User cancelled (disconnected)'
            
            self.logger.info(f"取消订单: {order_id}, 状态: {order['status']}")
            
            # 移至历史订单
            self.order_history.append(order)
            del self.orders[order_id]
            
            return order
            
        except Exception as e:
            self.logger.error(f"取消订单失败: {str(e)}")
            raise
    
    def get_order(self, order_id):
        """
        获取订单信息
        """
        if order_id in self.orders:
            return self.orders[order_id]
        
        # 查找历史订单
        for order in self.order_history:
            if order['order_id'] == order_id:
                return order
        
        return None
    
    def get_orders(self, symbol=None, status=None):
        """
        获取订单列表
        """
        orders = list(self.orders.values())
        
        # 过滤订单
        if symbol:
            orders = [order for order in orders if order['symbol'] == symbol]
        
        if status:
            orders = [order for order in orders if order['status'] == status]
        
        return orders
    
    def get_order_history(self, symbol=None, start_date=None, end_date=None):
        """
        获取历史订单
        """
        orders = self.order_history.copy()
        
        # 过滤订单
        if symbol:
            orders = [order for order in orders if order['symbol'] == symbol]
        
        if start_date:
            orders = [order for order in orders if order['timestamp'] >= start_date]
        
        if end_date:
            orders = [order for order in orders if order['timestamp'] <= end_date]
        
        return orders
    
    def update_order_status(self, order_id, status, **kwargs):
        """
        更新订单状态
        """
        if order_id in self.orders:
            order = self.orders[order_id]
            order['status'] = status
            
            # 更新其他信息
            if 'filled_quantity' in kwargs:
                order['filled_quantity'] = kwargs['filled_quantity']
            
            if 'avg_price' in kwargs:
                order['avg_price'] = kwargs['avg_price']
            
            if 'reason' in kwargs:
                order['reason'] = kwargs['reason']
            
            # 如果订单已成交或取消，移至历史订单
            if status in ['filled', 'cancelled', 'rejected']:
                if status == 'filled':
                    order['filled_timestamp'] = datetime.now()
                
                self.order_history.append(order)
                del self.orders[order_id]
            
            self.logger.info(f"更新订单状态: {order_id}, 状态: {status}")
            
            return order
        
        return None
    
    def is_connected(self):
        """
        检查是否连接
        """
        return self.connected
