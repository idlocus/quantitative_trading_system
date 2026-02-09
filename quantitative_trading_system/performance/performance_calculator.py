#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
绩效评估模块
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class PerformanceCalculator:
    """
    绩效计算器类，用于计算策略的绩效指标
    """
    
    def __init__(self, config):
        """
        初始化绩效计算器
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 绩效数据
        self.portfolio_values = []
        self.timestamps = []
        self.trades = []
        self.benchmark_values = []
    
    def start(self):
        """
        启动绩效计算器
        """
        try:
            self.initialize()
            self.logger.info("绩效计算器启动")
        except Exception as e:
            self.logger.error(f"绩效计算器启动失败: {str(e)}")
            raise
    
    def stop(self):
        """
        停止绩效计算器
        """
        self.logger.info("绩效计算器停止")
    
    def initialize(self):
        """
        初始化绩效计算器
        """
        self.portfolio_values = []
        self.timestamps = []
        self.trades = []
        self.benchmark_values = []
    
    def update_portfolio(self, timestamp, portfolio_value, benchmark_value=None):
        """
        更新组合价值
        """
        self.timestamps.append(timestamp)
        self.portfolio_values.append(portfolio_value)
        
        if benchmark_value:
            self.benchmark_values.append(benchmark_value)
    
    def add_trade(self, trade):
        """
        添加交易记录
        """
        self.trades.append(trade)
    
    def calculate_performance(self):
        """
        计算绩效指标
        """
        try:
            if not self.portfolio_values or len(self.portfolio_values) < 2:
                self.logger.warning("组合价值数据不足，无法计算绩效")
                return {}
            
            # 创建绩效数据框
            performance_data = pd.DataFrame({
                'timestamp': self.timestamps,
                'portfolio_value': self.portfolio_values
            })
            
            if self.benchmark_values:
                performance_data['benchmark_value'] = self.benchmark_values
            
            # 计算收益率
            performance_data['portfolio_return'] = performance_data['portfolio_value'].pct_change()
            if 'benchmark_value' in performance_data.columns:
                performance_data['benchmark_return'] = performance_data['benchmark_value'].pct_change()
            
            # 计算累计收益
            performance_data['cumulative_return'] = (1 + performance_data['portfolio_return']).cumprod() - 1
            if 'benchmark_return' in performance_data.columns:
                performance_data['benchmark_cumulative_return'] = (1 + performance_data['benchmark_return']).cumprod() - 1
            
            # 计算绩效指标
            metrics = {
                'total_return': self._calculate_total_return(),
                'annualized_return': self._calculate_annualized_return(),
                'max_drawdown': self._calculate_max_drawdown(),
                'annualized_volatility': self._calculate_annualized_volatility(),
                'sharpe_ratio': self._calculate_sharpe_ratio(),
                'sortino_ratio': self._calculate_sortino_ratio(),
                'calmar_ratio': self._calculate_calmar_ratio(),
                'win_rate': self._calculate_win_rate(),
                'average_win': self._calculate_average_win(),
                'average_loss': self._calculate_average_loss(),
                'profit_factor': self._calculate_profit_factor(),
                'trade_count': len(self.trades),
                'average_holding_period': self._calculate_average_holding_period(),
                'alpha': self._calculate_alpha(),
                'beta': self._calculate_beta()
            }
            
            self.logger.info(f"绩效计算完成: {metrics}")
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"绩效计算失败: {str(e)}")
            raise
    
    def _calculate_total_return(self):
        """
        计算总收益率
        """
        if not self.portfolio_values or len(self.portfolio_values) < 2:
            return 0
        
        return (self.portfolio_values[-1] / self.portfolio_values[0] - 1) * 100
    
    def _calculate_annualized_return(self):
        """
        计算年化收益率
        """
        if not self.portfolio_values or len(self.portfolio_values) < 2:
            return 0
        
        total_return = self.portfolio_values[-1] / self.portfolio_values[0] - 1
        days = (self.timestamps[-1] - self.timestamps[0]).days
        
        if days == 0:
            return 0
        
        annualized_return = (1 + total_return) ** (365 / days) - 1
        return annualized_return * 100
    
    def _calculate_max_drawdown(self):
        """
        计算最大回撤
        """
        if not self.portfolio_values or len(self.portfolio_values) < 2:
            return 0
        
        cumulative = pd.Series(self.portfolio_values)
        peak = cumulative.expanding(min_periods=1).max()
        drawdown = (cumulative - peak) / peak
        max_drawdown = drawdown.min() * 100
        
        return max_drawdown
    
    def _calculate_annualized_volatility(self):
        """
        计算年化波动率
        """
        if not self.portfolio_values or len(self.portfolio_values) < 2:
            return 0
        
        returns = pd.Series(self.portfolio_values).pct_change().dropna()
        
        if len(returns) == 0:
            return 0
        
        daily_volatility = returns.std()
        annualized_volatility = daily_volatility * np.sqrt(252)
        
        return annualized_volatility * 100
    
    def _calculate_sharpe_ratio(self, risk_free_rate=0.0):
        """
        计算夏普比率
        """
        annualized_return = self._calculate_annualized_return() / 100
        annualized_volatility = self._calculate_annualized_volatility() / 100
        
        if annualized_volatility == 0:
            return 0
        
        sharpe_ratio = (annualized_return - risk_free_rate) / annualized_volatility
        
        return sharpe_ratio
    
    def _calculate_sortino_ratio(self, risk_free_rate=0.0, target_return=0):
        """
        计算索提诺比率
        """
        annualized_return = self._calculate_annualized_return() / 100
        
        # 计算下行波动率
        returns = pd.Series(self.portfolio_values).pct_change().dropna()
        downside_returns = returns[returns < target_return]
        
        if len(downside_returns) == 0:
            return 0
        
        daily_downside_volatility = downside_returns.std()
        annualized_downside_volatility = daily_downside_volatility * np.sqrt(252)
        
        if annualized_downside_volatility == 0:
            return 0
        
        sortino_ratio = (annualized_return - risk_free_rate) / annualized_downside_volatility
        
        return sortino_ratio
    
    def _calculate_calmar_ratio(self):
        """
        计算卡玛比率
        """
        annualized_return = self._calculate_annualized_return() / 100
        max_drawdown = abs(self._calculate_max_drawdown() / 100)
        
        if max_drawdown == 0:
            return 0
        
        calmar_ratio = annualized_return / max_drawdown
        
        return calmar_ratio
    
    def _calculate_win_rate(self):
        """
        计算胜率
        """
        if not self.trades:
            return 0
        
        winning_trades = [trade for trade in self.trades if trade.get('profit', 0) > 0]
        win_rate = len(winning_trades) / len(self.trades) * 100
        
        return win_rate
    
    def _calculate_average_win(self):
        """
        计算平均盈利
        """
        winning_trades = [trade for trade in self.trades if trade.get('profit', 0) > 0]
        
        if not winning_trades:
            return 0
        
        average_win = sum(trade.get('profit', 0) for trade in winning_trades) / len(winning_trades)
        
        return average_win
    
    def _calculate_average_loss(self):
        """
        计算平均亏损
        """
        losing_trades = [trade for trade in self.trades if trade.get('profit', 0) < 0]
        
        if not losing_trades:
            return 0
        
        average_loss = sum(abs(trade.get('profit', 0)) for trade in losing_trades) / len(losing_trades)
        
        return average_loss
    
    def _calculate_profit_factor(self):
        """
        计算盈利因子
        """
        winning_trades = [trade for trade in self.trades if trade.get('profit', 0) > 0]
        losing_trades = [trade for trade in self.trades if trade.get('profit', 0) < 0]
        
        if not losing_trades:
            return 0
        
        total_win = sum(trade.get('profit', 0) for trade in winning_trades)
        total_loss = sum(abs(trade.get('profit', 0)) for trade in losing_trades)
        
        if total_loss == 0:
            return 0
        
        profit_factor = total_win / total_loss
        
        return profit_factor
    
    def _calculate_average_holding_period(self):
        """
        计算平均持有周期
        """
        if not self.trades:
            return 0
        
        holding_periods = []
        for trade in self.trades:
            if 'filled_timestamp' in trade and 'timestamp' in trade:
                holding_period = (trade['filled_timestamp'] - trade['timestamp']).total_seconds() / 3600  # 转换为小时
                holding_periods.append(holding_period)
        
        if not holding_periods:
            return 0
        
        average_holding_period = sum(holding_periods) / len(holding_periods)
        
        return average_holding_period
    
    def _calculate_alpha(self):
        """
        计算阿尔法
        """
        if not self.benchmark_values or len(self.benchmark_values) < 2:
            return 0
        
        # 简单线性回归计算阿尔法和贝塔
        returns = pd.Series(self.portfolio_values).pct_change().dropna()
        benchmark_returns = pd.Series(self.benchmark_values).pct_change().dropna()
        
        if len(returns) != len(benchmark_returns):
            return 0
        
        # 确保长度一致
        min_length = min(len(returns), len(benchmark_returns))
        returns = returns[-min_length:]
        benchmark_returns = benchmark_returns[-min_length:]
        
        # 计算协方差和方差
        covariance = np.cov(returns, benchmark_returns)[0, 1]
        benchmark_variance = np.var(benchmark_returns)
        
        if benchmark_variance == 0:
            return 0
        
        # 计算贝塔
        beta = covariance / benchmark_variance
        
        # 计算阿尔法
        average_return = np.mean(returns)
        average_benchmark_return = np.mean(benchmark_returns)
        alpha = average_return - beta * average_benchmark_return
        
        # 年化阿尔法
        annualized_alpha = alpha * 252
        
        return annualized_alpha * 100
    
    def _calculate_beta(self):
        """
        计算贝塔
        """
        if not self.benchmark_values or len(self.benchmark_values) < 2:
            return 0
        
        # 简单线性回归计算贝塔
        returns = pd.Series(self.portfolio_values).pct_change().dropna()
        benchmark_returns = pd.Series(self.benchmark_values).pct_change().dropna()
        
        if len(returns) != len(benchmark_returns):
            return 0
        
        # 确保长度一致
        min_length = min(len(returns), len(benchmark_returns))
        returns = returns[-min_length:]
        benchmark_returns = benchmark_returns[-min_length:]
        
        # 计算协方差和方差
        covariance = np.cov(returns, benchmark_returns)[0, 1]
        benchmark_variance = np.var(benchmark_returns)
        
        if benchmark_variance == 0:
            return 0
        
        # 计算贝塔
        beta = covariance / benchmark_variance
        
        return beta
    
    def generate_report(self, metrics=None):
        """
        生成绩效报告
        """
        try:
            if not metrics:
                metrics = self.calculate_performance()
            
            if not metrics:
                self.logger.warning("绩效数据不足，无法生成报告")
                return
            
            # 打印报告
            print("\n===== 绩效报告 =====")
            print(f"总收益率: {metrics.get('total_return', 0):.2f}%")
            print(f"年化收益率: {metrics.get('annualized_return', 0):.2f}%")
            print(f"最大回撤: {metrics.get('max_drawdown', 0):.2f}%")
            print(f"年化波动率: {metrics.get('annualized_volatility', 0):.2f}%")
            print(f"夏普比率: {metrics.get('sharpe_ratio', 0):.2f}")
            print(f"索提诺比率: {metrics.get('sortino_ratio', 0):.2f}")
            print(f"卡玛比率: {metrics.get('calmar_ratio', 0):.2f}")
            print(f"胜率: {metrics.get('win_rate', 0):.2f}%")
            print(f"平均盈利: {metrics.get('average_win', 0):.2f}")
            print(f"平均亏损: {metrics.get('average_loss', 0):.2f}")
            print(f"盈利因子: {metrics.get('profit_factor', 0):.2f}")
            print(f"交易次数: {metrics.get('trade_count', 0)}")
            print(f"平均持有周期: {metrics.get('average_holding_period', 0):.2f} 小时")
            print(f"阿尔法: {metrics.get('alpha', 0):.2f}%")
            print(f"贝塔: {metrics.get('beta', 0):.2f}")
            print("====================\n")
            
            # 保存报告
            self._save_report(metrics)
            
        except Exception as e:
            self.logger.error(f"生成绩效报告失败: {str(e)}")
            raise
    
    def _save_report(self, metrics):
        """
        保存绩效报告
        """
        import os
        import json
        
        # 确保报告目录存在
        report_dir = 'performance_reports'
        os.makedirs(report_dir, exist_ok=True)
        
        # 生成文件名
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_file = os.path.join(report_dir, f'performance_report_{timestamp}.json')
        
        # 保存报告
        with open(report_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        self.logger.info(f"绩效报告已保存到 {report_file}")
