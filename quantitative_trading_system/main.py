#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
量化交易系统主入口
"""

import logging
import sys
from config.system_config import SystemConfig
from data.acquisition.base_data_source import BaseDataSource
from strategy.base_strategy import BaseStrategy
from backtesting.backtester import Backtester
from execution.order_manager import OrderManager
from risk.risk_controller import RiskController
from performance.performance_calculator import PerformanceCalculator
from visualization.dashboard import Dashboard

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/system.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def main():
    """
    系统主函数
    """
    try:
        logger.info("启动量化交易系统")
        
        # 加载配置
        config = SystemConfig()
        logger.info("配置加载完成")
        
        # 初始化模块
        data_source = BaseDataSource(config)
        strategy = BaseStrategy(config)
        backtester = Backtester(config)
        order_manager = OrderManager(config)
        risk_controller = RiskController(config)
        performance_calculator = PerformanceCalculator(config)
        dashboard = Dashboard(config)
        
        logger.info("模块初始化完成")
        
        # 根据运行模式执行不同操作
        if config.run_mode == 'backtest':
            logger.info("运行回测模式")
            run_backtest(backtester, strategy, data_source)
        elif config.run_mode == 'live':
            logger.info("运行实盘模式")
            run_live_trading(data_source, strategy, order_manager, risk_controller, performance_calculator, dashboard)
        elif config.run_mode == 'paper':
            logger.info("运行纸盘模式")
            run_paper_trading(data_source, strategy, order_manager, risk_controller, performance_calculator, dashboard)
        else:
            logger.error(f"未知运行模式: {config.run_mode}")
            return
            
    except Exception as e:
        logger.error(f"系统启动失败: {str(e)}")
        raise

def run_backtest(backtester, strategy, data_source):
    """
    运行回测
    """
    try:
        # 获取历史数据
        logger.info("获取历史数据")
        data = data_source.get_historical_data()
        
        # 运行回测
        logger.info("开始回测")
        results = backtester.run(strategy, data)
        
        # 生成绩效报告
        logger.info("生成回测报告")
        backtester.generate_report(results)
        
        logger.info("回测完成")
        
    except Exception as e:
        logger.error(f"回测失败: {str(e)}")
        raise

def run_live_trading(data_source, strategy, order_manager, risk_controller, performance_calculator, dashboard):
    """
    运行实盘交易
    """
    try:
        logger.info("连接数据源和交易接口")
        data_source.connect()
        order_manager.connect()
        
        logger.info("启动实时数据订阅")
        data_source.subscribe_market_data()
        
        logger.info("启动交易策略")
        strategy.start()
        
        logger.info("启动风险监控")
        risk_controller.start()
        
        logger.info("启动绩效计算")
        performance_calculator.start()
        
        logger.info("启动可视化面板")
        dashboard.start()
        
        logger.info("实盘交易已启动，按 Ctrl+C 退出")
        
        # 保持系统运行
        while True:
            pass
            
    except KeyboardInterrupt:
        logger.info("收到退出信号，正在关闭系统")
    except Exception as e:
        logger.error(f"实盘交易失败: {str(e)}")
        raise
    finally:
        # 清理资源
        data_source.disconnect()
        order_manager.disconnect()
        strategy.stop()
        risk_controller.stop()
        performance_calculator.stop()
        dashboard.stop()
        
        logger.info("系统已关闭")

def run_paper_trading(data_source, strategy, order_manager, risk_controller, performance_calculator, dashboard):
    """
    运行纸盘交易
    """
    try:
        logger.info("连接数据源")
        data_source.connect()
        
        logger.info("启动实时数据订阅")
        data_source.subscribe_market_data()
        
        logger.info("启动交易策略")
        strategy.start()
        
        logger.info("启动风险监控")
        risk_controller.start()
        
        logger.info("启动绩效计算")
        performance_calculator.start()
        
        logger.info("启动可视化面板")
        dashboard.start()
        
        logger.info("纸盘交易已启动，按 Ctrl+C 退出")
        
        # 保持系统运行
        while True:
            pass
            
    except KeyboardInterrupt:
        logger.info("收到退出信号，正在关闭系统")
    except Exception as e:
        logger.error(f"纸盘交易失败: {str(e)}")
        raise
    finally:
        # 清理资源
        data_source.disconnect()
        strategy.stop()
        risk_controller.stop()
        performance_calculator.stop()
        dashboard.stop()
        
        logger.info("系统已关闭")

if __name__ == "__main__":
    main()
