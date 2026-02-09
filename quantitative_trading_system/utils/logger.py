#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
日志工具模块
"""

import logging
import os
from datetime import datetime

class Logger:
    """
    日志类，用于配置和管理系统日志
    """
    
    def __init__(self, name, log_dir='logs', level=logging.INFO):
        """
        初始化日志
        """
        self.name = name
        self.log_dir = log_dir
        self.level = level
        self.logger = None
        
        # 确保日志目录存在
        os.makedirs(self.log_dir, exist_ok=True)
        
        # 初始化日志
        self.initialize()
    
    def initialize(self):
        """
        初始化日志配置
        """
        # 创建日志器
        self.logger = logging.getLogger(self.name)
        self.logger.setLevel(self.level)
        
        # 避免重复添加处理器
        if self.logger.handlers:
            self.logger.handlers.clear()
        
        # 创建格式化器
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # 创建文件处理器
        timestamp = datetime.now().strftime('%Y%m%d')
        log_file = os.path.join(self.log_dir, f'{self.name}_{timestamp}.log')
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(self.level)
        file_handler.setFormatter(formatter)
        
        # 创建控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(self.level)
        console_handler.setFormatter(formatter)
        
        # 添加处理器
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
    
    def debug(self, message):
        """
        记录调试信息
        """
        if self.logger:
            self.logger.debug(message)
    
    def info(self, message):
        """
        记录信息
        """
        if self.logger:
            self.logger.info(message)
    
    def warning(self, message):
        """
        记录警告信息
        """
        if self.logger:
            self.logger.warning(message)
    
    def error(self, message):
        """
        记录错误信息
        """
        if self.logger:
            self.logger.error(message)
    
    def critical(self, message):
        """
        记录严重错误信息
        """
        if self.logger:
            self.logger.critical(message)
    
    def exception(self, message):
        """
        记录异常信息
        """
        if self.logger:
            self.logger.exception(message)

# 创建全局日志实例
_system_logger = None

def get_logger(name='quantitative_trading_system', log_dir='logs', level=logging.INFO):
    """
    获取日志实例
    """
    global _system_logger
    if _system_logger is None:
        _system_logger = Logger(name, log_dir, level)
    return _system_logger.logger
