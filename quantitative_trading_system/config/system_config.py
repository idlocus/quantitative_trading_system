#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
系统配置管理
"""

import os
import json
from dotenv import load_dotenv

class SystemConfig:
    """
    系统配置类
    """
    
    def __init__(self):
        """
        初始化配置
        """
        # 加载环境变量
        load_dotenv()
        
        # 系统配置
        self.run_mode = os.getenv('RUN_MODE', 'backtest')  # 运行模式: backtest, live, paper
        self.log_level = os.getenv('LOG_LEVEL', 'INFO')  # 日志级别
        self.data_dir = os.getenv('DATA_DIR', 'data')  # 数据目录
        self.log_dir = os.getenv('LOG_DIR', 'logs')  # 日志目录
        
        # 确保目录存在
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        
        # 数据配置
        self.data_config = {
            'source_type': os.getenv('DATA_SOURCE_TYPE', 'mock'),
            'sources': {
                'binance': {
                    'api_key': os.getenv('BINANCE_API_KEY', ''),
                    'api_secret': os.getenv('BINANCE_API_SECRET', ''),
                    'base_url': 'https://api.binance.com'
                }
            },
            'symbols': os.getenv('SYMBOLS', 'BTC/USDT,ETH/USDT').split(','),
            'timeframes': os.getenv('TIMEFRAMES', '1m,5m,15m,1h,4h,1d').split(','),
            'historical_days': int(os.getenv('HISTORICAL_DAYS', '30'))
        }
        
        # 策略配置
        self.strategy_config = {
            'name': os.getenv('STRATEGY_NAME', 'TrendFollowing'),
            'params': {
                # 趋势跟踪策略参数
                'fast_period': int(os.getenv('FAST_PERIOD', '10')),
                'slow_period': int(os.getenv('SLOW_PERIOD', '20')),
                'signal_period': int(os.getenv('SIGNAL_PERIOD', '9')),
                # 动量突破策略参数
                'momentum_period': int(os.getenv('MOMENTUM_PERIOD', '20')),
                'breakout_period': int(os.getenv('BREAKOUT_PERIOD', '50')),
                'rsi_period': int(os.getenv('RSI_PERIOD', '14')),
                'rsi_overbought': int(os.getenv('RSI_OVERBOUGHT', '70')),
                'rsi_oversold': int(os.getenv('RSI_OVERSOLD', '30')),
                'stop_loss_pct': float(os.getenv('STOP_LOSS_PCT', '0.03')),
                'take_profit_pct': float(os.getenv('TAKE_PROFIT_PCT', '0.06'))
            }
        }
        
        # 回测配置
        self.backtest_config = {
            'initial_capital': float(os.getenv('INITIAL_CAPITAL', '10000')),
            'commission': float(os.getenv('COMMISSION', '0.001')),
            'slippage': float(os.getenv('SLIPPAGE', '0.0005')),
            'start_date': os.getenv('START_DATE', '2023-01-01'),
            'end_date': os.getenv('END_DATE', '2023-12-31')
        }
        
        # 执行配置
        self.execution_config = {
            'exchange': os.getenv('EXCHANGE', 'binance'),
            'order_type': os.getenv('ORDER_TYPE', 'market'),
            'testnet': os.getenv('TESTNET', 'False').lower() == 'true'
        }
        
        # 风险配置
        self.risk_config = {
            'max_position_size': float(os.getenv('MAX_POSITION_SIZE', '0.2')),  # 最大持仓比例
            'max_drawdown': float(os.getenv('MAX_DRAWDOWN', '0.1')),  # 最大回撤
            'stop_loss_pct': float(os.getenv('STOP_LOSS_PCT', '0.02')),  # 止损百分比
            'take_profit_pct': float(os.getenv('TAKE_PROFIT_PCT', '0.04'))  # 止盈百分比
        }
        
        # 加载配置文件（如果存在）
        self._load_config_file()
    
    def _load_config_file(self):
        """
        从配置文件加载配置
        """
        config_file = 'config.json'
        if os.path.exists(config_file):
            with open(config_file, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
                
                # 更新配置
                if 'data_config' in config_data:
                    self.data_config.update(config_data['data_config'])
                if 'strategy_config' in config_data:
                    self.strategy_config.update(config_data['strategy_config'])
                if 'backtest_config' in config_data:
                    self.backtest_config.update(config_data['backtest_config'])
                if 'execution_config' in config_data:
                    self.execution_config.update(config_data['execution_config'])
                if 'risk_config' in config_data:
                    self.risk_config.update(config_data['risk_config'])
    
    def save_config(self, config_file='config.json'):
        """
        保存配置到文件
        """
        config_data = {
            'data_config': self.data_config,
            'strategy_config': self.strategy_config,
            'backtest_config': self.backtest_config,
            'execution_config': self.execution_config,
            'risk_config': self.risk_config
        }
        
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, indent=2, ensure_ascii=False)
    
    def get(self, key, default=None):
        """
        获取配置项
        """
        # 简单的配置获取逻辑
        if hasattr(self, key):
            return getattr(self, key)
        return default
