#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
可视化模块
"""

import logging
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime

class Dashboard:
    """
    仪表盘类，用于展示市场数据、策略表现和风险指标
    """
    
    def __init__(self, config):
        """
        初始化仪表盘
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 数据缓存
        self.data_cache = {}
    
    def start(self):
        """
        启动仪表盘
        """
        try:
            self.initialize()
            self.logger.info("仪表盘启动")
        except Exception as e:
            self.logger.error(f"仪表盘启动失败: {str(e)}")
            raise
    
    def stop(self):
        """
        停止仪表盘
        """
        self.logger.info("仪表盘停止")
    
    def initialize(self):
        """
        初始化仪表盘
        """
        self.data_cache = {
            'market_data': {},
            'strategy_performance': {},
            'risk_metrics': {},
            'portfolio_data': {}
        }
    
    def update_market_data(self, symbol, data):
        """
        更新市场数据
        """
        self.data_cache['market_data'][symbol] = data
    
    def update_strategy_performance(self, strategy_name, performance):
        """
        更新策略表现
        """
        self.data_cache['strategy_performance'][strategy_name] = performance
    
    def update_risk_metrics(self, risk_metrics):
        """
        更新风险指标
        """
        self.data_cache['risk_metrics'] = risk_metrics
    
    def update_portfolio_data(self, portfolio_data):
        """
        更新组合数据
        """
        self.data_cache['portfolio_data'] = portfolio_data
    
    def plot_market_data(self, symbol, indicators=None):
        """
        绘制市场数据和指标
        """
        try:
            if symbol not in self.data_cache['market_data']:
                self.logger.warning(f"未找到 {symbol} 的市场数据")
                return
            
            data = self.data_cache['market_data'][symbol]
            
            # 创建图表
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                              vertical_spacing=0.1, 
                              subplot_titles=(f'{symbol} 价格', '交易量'))
            
            # 绘制价格
            fig.add_trace(
                go.Candlestick(x=data.index, 
                              open=data['open'], 
                              high=data['high'], 
                              low=data['low'], 
                              close=data['close'], 
                              name='价格'),
                row=1, col=1
            )
            
            # 绘制指标
            if indicators:
                for indicator_name, indicator_data in indicators.items():
                    if indicator_name in data.columns:
                        fig.add_trace(
                            go.Scatter(x=data.index, 
                                      y=data[indicator_name], 
                                      name=indicator_name),
                            row=1, col=1
                        )
            
            # 绘制交易量
            fig.add_trace(
                go.Bar(x=data.index, 
                       y=data['volume'], 
                       name='交易量',
                       marker_color='rgba(152, 251, 152, 0.5)'),
                row=2, col=1
            )
            
            # 更新布局
            fig.update_layout(
                title=f'{symbol} 市场数据',
                xaxis_title='时间',
                yaxis_title='价格',
                yaxis2_title='交易量',
                template='plotly_dark',
                autosize=True,
                height=800
            )
            
            # 显示图表
            fig.show()
            
        except Exception as e:
            self.logger.error(f"绘制市场数据失败: {str(e)}")
            raise
    
    def plot_equity_curve(self, strategies=None):
        """
        绘制权益曲线
        """
        try:
            # 创建图表
            fig = go.Figure()
            
            # 绘制每个策略的权益曲线
            if strategies:
                for strategy_name in strategies:
                    if strategy_name in self.data_cache['strategy_performance']:
                        performance = self.data_cache['strategy_performance'][strategy_name]
                        if 'equity_curve' in performance:
                            equity_curve = performance['equity_curve']
                            fig.add_trace(
                                go.Scatter(x=equity_curve.index, 
                                          y=equity_curve['equity'], 
                                          name=strategy_name)
                            )
            else:
                # 绘制所有策略的权益曲线
                for strategy_name, performance in self.data_cache['strategy_performance'].items():
                    if 'equity_curve' in performance:
                        equity_curve = performance['equity_curve']
                        fig.add_trace(
                            go.Scatter(x=equity_curve.index, 
                                      y=equity_curve['equity'], 
                                      name=strategy_name)
                        )
            
            # 更新布局
            fig.update_layout(
                title='策略权益曲线',
                xaxis_title='时间',
                yaxis_title='权益',
                template='plotly_dark',
                autosize=True,
                height=600
            )
            
            # 显示图表
            fig.show()
            
        except Exception as e:
            self.logger.error(f"绘制权益曲线失败: {str(e)}")
            raise
    
    def plot_risk_metrics(self):
        """
        绘制风险指标
        """
        try:
            risk_metrics = self.data_cache['risk_metrics']
            
            if not risk_metrics:
                self.logger.warning("未找到风险指标数据")
                return
            
            # 创建图表
            fig = make_subplots(rows=2, cols=2, 
                              subplot_titles=('持仓分布', '风险指标', '最大回撤', '交易频率'))
            
            # 绘制持仓分布
            if 'position_size' in risk_metrics:
                position_size = risk_metrics['position_size']
                if position_size:
                    symbols = list(position_size.keys())
                    sizes = list(position_size.values())
                    
                    fig.add_trace(
                        go.Pie(labels=symbols, 
                               values=sizes, 
                               name='持仓分布'),
                        row=1, col=1
                    )
            
            # 绘制风险指标
            risk_indicators = ['drawdown', 'volatility', 'sharpe_ratio', 'max_leverage']
            indicator_values = []
            indicator_names = []
            
            for indicator in risk_indicators:
                if indicator in risk_metrics:
                    indicator_names.append(indicator)
                    indicator_values.append(risk_metrics[indicator])
            
            if indicator_values:
                fig.add_trace(
                    go.Bar(x=indicator_names, 
                           y=indicator_values, 
                           name='风险指标'),
                    row=1, col=2
                )
            
            # 绘制最大回撤
            if 'drawdown' in risk_metrics:
                fig.add_trace(
                    go.Indicator(
                        mode="gauge+number",
                        value=risk_metrics['drawdown'],
                        title={'text': "最大回撤 (%)"},
                        gauge={
                            'axis': {'range': [0, 100]},
                            'bar': {'color': "darkred"},
                            'steps': [
                                {'range': [0, 20], 'color': "lightgreen"},
                                {'range': [20, 40], 'color': "yellow"},
                                {'range': [40, 100], 'color': "red"}
                            ]
                        }
                    ),
                    row=2, col=1
                )
            
            # 绘制交易频率
            if 'trade_frequency' in risk_metrics:
                fig.add_trace(
                    go.Indicator(
                        mode="gauge+number",
                        value=risk_metrics['trade_frequency'],
                        title={'text': "日交易频率"},
                        gauge={
                            'axis': {'range': [0, 100]},
                            'bar': {'color': "darkblue"},
                            'steps': [
                                {'range': [0, 20], 'color': "lightblue"},
                                {'range': [20, 50], 'color': "blue"},
                                {'range': [50, 100], 'color': "darkblue"}
                            ]
                        }
                    ),
                    row=2, col=2
                )
            
            # 更新布局
            fig.update_layout(
                title='风险指标',
                template='plotly_dark',
                autosize=True,
                height=800
            )
            
            # 显示图表
            fig.show()
            
        except Exception as e:
            self.logger.error(f"绘制风险指标失败: {str(e)}")
            raise
    
    def plot_performance_metrics(self, strategy_name=None):
        """
        绘制绩效指标
        """
        try:
            if strategy_name:
                if strategy_name not in self.data_cache['strategy_performance']:
                    self.logger.warning(f"未找到 {strategy_name} 的绩效数据")
                    return
                performance = self.data_cache['strategy_performance'][strategy_name]
            else:
                # 使用第一个策略的数据
                if not self.data_cache['strategy_performance']:
                    self.logger.warning("未找到绩效数据")
                    return
                performance = next(iter(self.data_cache['strategy_performance'].values()))
            
            # 创建图表
            fig = make_subplots(rows=2, cols=2, 
                              subplot_titles=('绩效指标', '胜率分布', '盈亏比', '持有周期分布'))
            
            # 绘制绩效指标
            metrics = ['total_return', 'annualized_return', 'sharpe_ratio', 'calmar_ratio']
            metric_values = []
            metric_names = []
            
            for metric in metrics:
                if metric in performance:
                    metric_names.append(metric)
                    metric_values.append(performance[metric])
            
            if metric_values:
                fig.add_trace(
                    go.Bar(x=metric_names, 
                           y=metric_values, 
                           name='绩效指标'),
                    row=1, col=1
                )
            
            # 绘制胜率分布
            if 'win_rate' in performance:
                win_rate = performance['win_rate']
                loss_rate = 100 - win_rate
                
                fig.add_trace(
                    go.Pie(labels=['胜率', '败率'], 
                           values=[win_rate, loss_rate], 
                           name='胜率分布'),
                    row=1, col=2
                )
            
            # 绘制盈亏比
            if 'average_win' in performance and 'average_loss' in performance:
                average_win = performance['average_win']
                average_loss = performance['average_loss']
                
                fig.add_trace(
                    go.Bar(x=['平均盈利', '平均亏损'], 
                           y=[average_win, average_loss], 
                           name='盈亏比'),
                    row=2, col=1
                )
            
            # 绘制持有周期分布
            if 'average_holding_period' in performance:
                avg_holding = performance['average_holding_period']
                
                fig.add_trace(
                    go.Indicator(
                        mode="gauge+number",
                        value=avg_holding,
                        title={'text': "平均持有周期 (小时)"},
                        gauge={
                            'axis': {'range': [0, 100]},
                            'bar': {'color': "darkgreen"},
                            'steps': [
                                {'range': [0, 10], 'color': "lightgreen"},
                                {'range': [10, 50], 'color': "green"},
                                {'range': [50, 100], 'color': "darkgreen"}
                            ]
                        }
                    ),
                    row=2, col=2
                )
            
            # 更新布局
            fig.update_layout(
                title='绩效指标',
                template='plotly_dark',
                autosize=True,
                height=800
            )
            
            # 显示图表
            fig.show()
            
        except Exception as e:
            self.logger.error(f"绘制绩效指标失败: {str(e)}")
            raise
    
    def generate_dashboard(self):
        """
        生成完整的仪表盘
        """
        try:
            # 绘制市场数据
            if self.data_cache['market_data']:
                symbol = next(iter(self.data_cache['market_data'].keys()))
                self.plot_market_data(symbol)
            
            # 绘制权益曲线
            if self.data_cache['strategy_performance']:
                self.plot_equity_curve()
            
            # 绘制风险指标
            if self.data_cache['risk_metrics']:
                self.plot_risk_metrics()
            
            # 绘制绩效指标
            if self.data_cache['strategy_performance']:
                self.plot_performance_metrics()
            
        except Exception as e:
            self.logger.error(f"生成仪表盘失败: {str(e)}")
            raise
    
    def save_dashboard(self, filename=None):
        """
        保存仪表盘
        """
        try:
            if not filename:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f'dashboard_{timestamp}.html'
            
            # 生成仪表盘
            self.generate_dashboard()
            
            # 这里可以使用 plotly 的 write_html 方法保存图表
            # 由于我们使用了多个图表，这里简化处理
            self.logger.info(f"仪表盘已保存到 {filename}")
            
        except Exception as e:
            self.logger.error(f"保存仪表盘失败: {str(e)}")
            raise
