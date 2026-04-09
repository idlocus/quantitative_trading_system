#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
回测系统
"""

import abc
import logging
import pandas as pd
import numpy as np
from datetime import datetime

class Backtester:
    """
    回测系统类
    """
    
    def __init__(self, config):
        """
        初始化回测系统
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.backtest_config = config.backtest_config
        
        # 回测参数
        self.initial_capital = self.backtest_config.get('initial_capital', 10000)
        self.commission = self.backtest_config.get('commission', 0.001)
        self.slippage = self.backtest_config.get('slippage', 0.0005)
        self.start_date = self.backtest_config.get('start_date')
        self.end_date = self.backtest_config.get('end_date')
        
        # 回测结果
        self.results = {}
        self.equity_curve = pd.DataFrame()
        self.trades = []
    
    def run(self, strategy, data):
        """
        运行回测
        """
        try:
            self.logger.info("开始回测")
            
            # 初始化策略
            strategy.initialize()
            
            # 初始化回测状态
            capital = self.initial_capital
            position = 0
            equity = [capital]
            timestamps = []
            trades = []
            
            # 预处理数据 - 计算所有指标
            self.logger.debug("预处理数据，计算指标")
            strategy.preprocess_data(data)
            
            # 模拟时间序列 - 使用itertuples提高效率
            self.logger.debug(f"开始回测循环，共 {len(data)} 条数据")
            for row in data.itertuples():
                timestamp = row.Index
                current_price = row.close
                
                timestamps.append(timestamp)
                
                # 处理数据 - 直接传递行数据，避免创建DataFrame
                signal = strategy.on_data(row)
                
                # 执行交易
                if signal and signal['type'] != 'hold':
                    trade_result = self._execute_trade(
                        signal['type'],
                        current_price,
                        capital,
                        position,
                        timestamp
                    )
                    
                    if trade_result:
                        capital = trade_result['capital']
                        position = trade_result['position']
                        trades.append(trade_result)
                
                # 计算当前权益
                current_equity = capital + (position * current_price if position != 0 else 0)
                equity.append(current_equity)
            
            # 生成回测结果
            self.equity_curve = pd.DataFrame({
                'timestamp': timestamps,
                'equity': equity[:-1]
            }).set_index('timestamp')
            
            self.trades = trades
            self.results = {
                'initial_capital': self.initial_capital,
                'final_capital': equity[-1],
                'total_return': (equity[-1] / self.initial_capital - 1) * 100,
                'max_drawdown': self._calculate_max_drawdown(self.equity_curve),
                'sharp_ratio': self._calculate_sharp_ratio(self.equity_curve),
                'trade_count': len(trades),
                'win_rate': self._calculate_win_rate(trades),
                'equity_curve': self.equity_curve,
                'trades': trades
            }
            
            self.logger.info(f"回测完成，总收益: {self.results['total_return']:.2f}%")
            
            return self.results
            
        except Exception as e:
            self.logger.error(f"回测失败: {str(e)}")
            raise
    
    def _execute_trade(self, signal_type, price, capital, position, timestamp):
        """
        执行交易
        """
        # 应用滑点
        executed_price = price * (1 + self.slippage) if signal_type == 'buy' else price * (1 - self.slippage)
        
        # 计算交易成本
        cost = executed_price * self.commission
        
        trade_result = {
            'timestamp': timestamp,
            'signal_type': signal_type,
            'price': executed_price,
            'cost': cost,
            'before_capital': capital,
            'before_position': position
        }
        
        if signal_type == 'buy' and position == 0:
            # 买入
            position = capital / (executed_price + cost)
            capital = 0
        elif signal_type == 'sell' and position != 0:
            # 卖出
            capital = position * executed_price - cost
            position = 0
        
        trade_result['capital'] = capital
        trade_result['position'] = position
        trade_result['profit'] = capital + (position * executed_price if position != 0 else 0) - trade_result['before_capital']
        
        return trade_result
    
    def _calculate_max_drawdown(self, equity_curve):
        """
        计算最大回撤
        """
        if equity_curve.empty:
            return 0
        
        # 计算累计收益
        cumulative = equity_curve['equity']
        # 计算峰值
        peak = cumulative.expanding(min_periods=1).max()
        # 计算回撤
        drawdown = (cumulative - peak) / peak
        # 计算最大回撤
        max_drawdown = drawdown.min() * 100
        
        return max_drawdown
    
    def _calculate_sharp_ratio(self, equity_curve, risk_free_rate=0.0):
        """
        计算夏普比率
        """
        if equity_curve.empty:
            return 0
        
        # 计算日收益率
        returns = equity_curve['equity'].pct_change().dropna()
        
        if len(returns) == 0:
            return 0
        
        # 计算年化收益率
        annual_return = (equity_curve['equity'].iloc[-1] / equity_curve['equity'].iloc[0]) ** (252 / len(equity_curve)) - 1
        
        # 计算年化波动率
        annual_volatility = returns.std() * np.sqrt(252)
        
        if annual_volatility == 0:
            return 0
        
        # 计算夏普比率
        sharp_ratio = (annual_return - risk_free_rate) / annual_volatility
        
        return sharp_ratio
    
    def _calculate_win_rate(self, trades):
        """
        计算胜率
        """
        if not trades:
            return 0
        
        winning_trades = [trade for trade in trades if trade['profit'] > 0]
        win_rate = len(winning_trades) / len(trades) * 100
        
        return win_rate
    
    def generate_report(self, results):
        """
        生成回测报告
        """
        try:
            self.logger.info("生成回测报告")
            
            # 打印基本统计信息
            print("\n===== 回测报告 =====")
            print(f"初始资金: ${results['initial_capital']:.2f}")
            print(f"最终资金: ${results['final_capital']:.2f}")
            print(f"总收益: {results['total_return']:.2f}%")
            print(f"最大回撤: {results['max_drawdown']:.2f}%")
            print(f"夏普比率: {results['sharp_ratio']:.2f}")
            print(f"交易次数: {results['trade_count']}")
            print(f"胜率: {results['win_rate']:.2f}%")
            print("====================\n")
            
            # 保存回测结果
            self._save_results(results)
            
        except Exception as e:
            self.logger.error(f"生成报告失败: {str(e)}")
            raise
    
    def _save_results(self, results):
        """
        保存回测结果
        """
        import os
        
        # 确保结果目录存在
        result_dir = 'results/backtests'
        os.makedirs(result_dir, exist_ok=True)
        
        # 生成文件名
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 保存权益曲线
        equity_file = os.path.join(result_dir, f'equity_curve_{timestamp}.csv')
        results['equity_curve'].to_csv(equity_file)
        
        # 保存交易记录
        trades_file = os.path.join(result_dir, f'trades_{timestamp}.csv')
        if results['trades']:
            pd.DataFrame(results['trades']).to_csv(trades_file, index=False)
        
        # 保存统计信息
        stats_file = os.path.join(result_dir, f'stats_{timestamp}.json')
        import json
        with open(stats_file, 'w') as f:
            json.dump({
                'initial_capital': results['initial_capital'],
                'final_capital': results['final_capital'],
                'total_return': results['total_return'],
                'max_drawdown': results['max_drawdown'],
                'sharp_ratio': results['sharp_ratio'],
                'trade_count': results['trade_count'],
                'win_rate': results['win_rate']
            }, f, indent=2)
        
        self.logger.info(f"回测结果已保存到 {result_dir}")
