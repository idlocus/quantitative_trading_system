#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用Matplotlib的回测结果可视化器
"""

import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mplfinance as mpf
from datetime import datetime

class MatplotlibVisualizer:
    """
    使用Matplotlib的回测结果可视化器
    """
    
    def __init__(self, backtest_results, strategy, market_data):
        """
        初始化回测结果可视化器
        """
        self.logger = logging.getLogger(__name__)
        self.backtest_results = backtest_results
        self.strategy = strategy
        self.market_data = market_data
        self.logger.info("MatplotlibVisualizer 初始化完成")
    
    def plot_results(self, save_path=None):
        """
        绘制回测结果
        """
        try:
            # 创建一个包含多个子图的图表
            fig = plt.figure(figsize=(16, 12))
            
            # 子图1: K线图和交易信号
            ax1 = plt.subplot2grid((4, 1), (0, 0), rowspan=2, colspan=1)
            
            # 子图2: MACD指标
            ax2 = plt.subplot2grid((4, 1), (2, 0), rowspan=1, colspan=1, sharex=ax1)
            
            # 子图3: 交易量
            ax3 = plt.subplot2grid((4, 1), (3, 0), rowspan=1, colspan=1, sharex=ax1)
            
            # 绘制K线图
            self._plot_candlestick(ax1)
            
            # 绘制MACD指标
            self._plot_macd(ax2)
            
            # 绘制交易量
            self._plot_volume(ax3)
            
            # 绘制交易信号
            self._plot_trade_signals(ax1)
            
            # 调整布局
            plt.tight_layout()
            plt.subplots_adjust(hspace=0.1)
            
            # 添加标题
            plt.suptitle(f"回测结果可视化 - {getattr(self.strategy, 'symbol', '300308.SZ')}", fontsize=16)
            
            # 保存图表
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                self.logger.info(f"回测结果图表已保存到: {save_path}")
            else:
                # 显示图表
                plt.show()
                
        except Exception as e:
            self.logger.error(f"绘制回测结果失败: {e}")
            raise
    
    def _plot_candlestick(self, ax):
        """
        绘制K线图
        """
        try:
            if self.market_data.empty:
                self.logger.warning("市场数据为空，无法绘制K线图")
                return
            
            # 准备K线数据
            candlestick_data = self.market_data.copy()
            candlestick_data.index = pd.to_datetime(candlestick_data.index)
            
            # 绘制K线图
            mpf.plot(
                candlestick_data,
                type='candle',
                ax=ax,
                style='yahoo',
                show_nontrading=False
            )
            
            # 设置标题
            ax.set_title('价格走势与交易信号', fontsize=12)
            
        except Exception as e:
            self.logger.error(f"绘制K线图失败: {e}")
    
    def _plot_macd(self, ax):
        """
        绘制MACD指标
        """
        try:
            if not hasattr(self.strategy, 'indicators') or not self.strategy.indicators:
                self.logger.warning("策略指标数据为空，无法绘制MACD")
                return
            
            # 获取MACD数据
            macd_line = self.strategy.indicators.get('macd_line', pd.Series())
            signal_line = self.strategy.indicators.get('signal_line', pd.Series())
            
            if macd_line.empty or signal_line.empty:
                self.logger.warning("MACD数据为空，无法绘制MACD")
                return
            
            # 计算柱状图
            histogram = macd_line - signal_line
            
            # 绘制MACD线
            ax.plot(macd_line.index, macd_line, label='MACD', color='blue', linewidth=1.5)
            
            # 绘制信号线
            ax.plot(signal_line.index, signal_line, label='信号线', color='orange', linewidth=1.5)
            
            # 绘制柱状图
            ax.bar(histogram.index, histogram, label='柱状图', color=['green' if x > 0 else 'red' for x in histogram])
            
            # 添加零轴
            ax.axhline(0, color='gray', linestyle='--', linewidth=0.5)
            
            # 设置标题和图例
            ax.set_title('MACD指标', fontsize=12)
            ax.legend(loc='upper left')
            
        except Exception as e:
            self.logger.error(f"绘制MACD失败: {e}")
    
    def _plot_volume(self, ax):
        """
        绘制交易量
        """
        try:
            if self.market_data.empty:
                self.logger.warning("市场数据为空，无法绘制交易量")
                return
            
            # 绘制交易量
            ax.bar(self.market_data.index, self.market_data['volume'], color='gray', alpha=0.7)
            
            # 设置标题
            ax.set_title('交易量', fontsize=12)
            ax.set_xlabel('时间')
            
        except Exception as e:
            self.logger.error(f"绘制交易量失败: {e}")
    
    def _plot_trade_signals(self, ax):
        """
        绘制交易信号
        """
        try:
            buy_signals = []
            sell_signals = []
            
            # 从策略信号中获取买卖点
            if hasattr(self.strategy, 'signals'):
                for signal in self.strategy.signals:
                    try:
                        if 'timestamp' in signal and 'price' in signal and 'type' in signal:
                            signal_time = signal['timestamp']
                            if isinstance(signal_time, str):
                                signal_time = pd.to_datetime(signal_time)
                            
                            if signal['type'] == 'buy':
                                buy_signals.append({
                                    'timestamp': signal_time,
                                    'price': signal['price']
                                })
                            elif signal['type'] == 'sell':
                                sell_signals.append({
                                    'timestamp': signal_time,
                                    'price': signal['price']
                                })
                    except Exception:
                        pass
            
            # 从回测结果的交易记录中获取买卖点
            if 'trades' in self.backtest_results:
                for trade in self.backtest_results['trades']:
                    try:
                        if 'timestamp' in trade and 'price' in trade and 'signal_type' in trade:
                            trade_time = trade['timestamp']
                            if isinstance(trade_time, str):
                                trade_time = pd.to_datetime(trade_time)
                            elif not hasattr(trade_time, 'strftime'):
                                continue
                            
                            if trade['signal_type'] == 'buy':
                                buy_signals.append({
                                    'timestamp': trade_time,
                                    'price': trade['price']
                                })
                            elif trade['signal_type'] == 'sell':
                                sell_signals.append({
                                    'timestamp': trade_time,
                                    'price': trade['price']
                                })
                    except Exception:
                        pass
            
            # 绘制买入信号
            if buy_signals:
                buy_df = pd.DataFrame(buy_signals)
                ax.scatter(buy_df['timestamp'], buy_df['price'], 
                          marker='^', color='green', s=100, label='买入信号', zorder=5)
            
            # 绘制卖出信号
            if sell_signals:
                sell_df = pd.DataFrame(sell_signals)
                ax.scatter(sell_df['timestamp'], sell_df['price'], 
                          marker='v', color='red', s=100, label='卖出信号', zorder=5)
            
            # 添加图例
            ax.legend(loc='upper left')
            
        except Exception as e:
            self.logger.error(f"绘制交易信号失败: {e}")
    
    def plot_equity_curve(self, save_path=None):
        """
        绘制收益率曲线
        """
        try:
            # 从回测结果中获取权益曲线
            equity_curve = self.backtest_results.get('equity_curve', pd.DataFrame())
            
            if equity_curve.empty:
                self.logger.warning("权益曲线数据为空，无法绘制")
                return
            
            # 创建图表
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # 绘制权益曲线
            ax.plot(equity_curve.index, equity_curve['equity'], label='权益', color='green', linewidth=2)
            
            # 绘制初始资金线
            initial_capital = self.backtest_results.get('initial_capital', 10000)
            ax.axhline(initial_capital, color='gray', linestyle='--', linewidth=1, label='初始资金')
            
            # 设置标题和标签
            ax.set_title('权益曲线', fontsize=14)
            ax.set_xlabel('时间')
            ax.set_ylabel('权益')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # 调整布局
            plt.tight_layout()
            
            # 保存图表
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                self.logger.info(f"权益曲线图表已保存到: {save_path}")
            else:
                # 显示图表
                plt.show()
                
        except Exception as e:
            self.logger.error(f"绘制权益曲线失败: {e}")
            raise

# 运行可视化器
if __name__ == "__main__":
    # 这里需要从外部传入回测结果、策略和市场数据
    # 实际使用时，应该在main.py中调用此可视化器
    pass
