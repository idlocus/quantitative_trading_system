#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
系统配置管理
只保留实际使用的配置项
"""

import os
from dotenv import load_dotenv

from risk.constants import RiskParams, BacktestParams


class SystemConfig:
    """系统配置"""

    def __init__(self):
        load_dotenv()

        self.run_mode = os.getenv('RUN_MODE', 'backtest')
        self.data_source_type = os.getenv('DATA_SOURCE_TYPE', 'wind')

        # 回测配置
        self.initial_capital = float(os.getenv('INITIAL_CAPITAL', '100000'))
        self.commission = float(os.getenv('COMMISSION', str(BacktestParams.COMMISSION)))
        self.slippage = float(os.getenv('SLIPPAGE', str(BacktestParams.SLIPPAGE)))
        self.start_date = os.getenv('START_DATE', '2023-01-01')
        self.end_date = os.getenv('END_DATE', '2026-02-27')

        # 风控参数（退出系统）
        self.stop_loss = float(os.getenv('STOP_LOSS', str(RiskParams.STOP_LOSS)))
        self.take_profit = float(os.getenv('TAKE_PROFIT', str(RiskParams.TAKE_PROFIT)))
        self.trailing_stop = float(os.getenv('TRAILING_STOP', str(BacktestParams.TRAILING_STOP)))
        self.max_holding_days = int(os.getenv('MAX_HOLDING_DAYS', str(BacktestParams.MAX_HOLDING_DAYS)))

        # Oracle 数据库（板块筛选用，凭据从 .env 读取，不落代码库）
        self.oracle_dsn = os.getenv('ORACLE_DSN', '')
        self.oracle_user = os.getenv('ORACLE_USER', '')
        self.oracle_password = os.getenv('ORACLE_PASSWORD', '')

        # 确保目录存在
        os.makedirs(os.getenv('DATA_DIR', 'data'), exist_ok=True)
        os.makedirs(os.getenv('LOG_DIR', 'logs'), exist_ok=True)

    def oracle_config(self):
        """返回 Oracle 配置字典"""
        return {
            'dsn': self.oracle_dsn,
            'user': self.oracle_user,
            'password': self.oracle_password
        }
