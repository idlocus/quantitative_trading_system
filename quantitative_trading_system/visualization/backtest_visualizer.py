#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
回测结果可视化器
"""

import logging
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
import numpy as np

class BacktestVisualizer:
    """
    回测结果可视化器
    """
    
    def __init__(self, backtest_results, strategy, market_data):
        """
        初始化回测结果可视化器
        """
        self.logger = logging.getLogger(__name__)
        self.backtest_results = backtest_results
        self.strategy = strategy
        self.market_data = market_data
        # 初始化Dash应用，设置为外部可访问
        self.app = dash.Dash(__name__, external_stylesheets=["https://codepen.io/chriddyp/pen/bWLwgP.css"])
        # 创建布局
        self.layout = self._create_layout()
        # 设置应用布局
        self.app.layout = self.layout
        # 注册回调函数
        self._register_callbacks()
        self.logger.info("BacktestVisualizer 初始化完成")
    
    def _create_layout(self):
        """
        创建布局
        """
        # 处理时间范围选择器的默认值
        from datetime import datetime, timedelta
        start_date = (datetime.now() - timedelta(days=30)).date()
        end_date = datetime.now().date()
        
        # 尝试从市场数据获取时间范围
        if not self.market_data.empty and hasattr(self.market_data.index, 'min'):
            try:
                min_date = self.market_data.index.min()
                max_date = self.market_data.index.max()
                if hasattr(min_date, 'date'):
                    start_date = min_date.date()
                if hasattr(max_date, 'date'):
                    end_date = max_date.date()
            except Exception as e:
                self.logger.warning(f"获取市场数据时间范围失败: {e}")
        # 尝试从回测结果的权益曲线获取时间范围
        elif 'equity_curve' in self.backtest_results:
            equity_curve = self.backtest_results['equity_curve']
            if not equity_curve.empty and hasattr(equity_curve.index, 'min'):
                try:
                    min_date = equity_curve.index.min()
                    max_date = equity_curve.index.max()
                    if hasattr(min_date, 'date'):
                        start_date = min_date.date()
                    if hasattr(max_date, 'date'):
                        end_date = max_date.date()
                except Exception as e:
                    self.logger.warning(f"获取权益曲线时间范围失败: {e}")
        
        return html.Div([
            # 标题
            html.H1("回测结果可视化", style={"text-align": "center", "margin": "20px 0", "color": "#333"}),
            
            # 回测摘要
            html.Div([
                html.H2("回测摘要", style={"color": "#555"}),
                html.Div(id="backtest-summary", className="summary-cards")
            ], style={"margin-bottom": "30px", "padding": "20px", "background-color": "#f5f5f5", "border-radius": "8px"}),
            
            # 时间范围选择
            html.Div([
                html.H3("时间范围选择"),
                dcc.DatePickerRange(
                    id="date-range",
                    start_date=start_date,
                    end_date=end_date,
                    display_format="YYYY-MM-DD",
                    style={"width": "100%", "max-width": "400px"}
                )
            ], style={"margin-bottom": "30px", "padding": "20px", "background-color": "#f5f5f5", "border-radius": "8px"}),
            
            # 价格和指标图表
            html.Div([
                html.H2("价格走势与技术指标", style={"color": "#555"}),
                dcc.Graph(id="price-indicator-chart", style={"height": "800px"})
            ], style={"margin-bottom": "30px", "padding": "20px", "background-color": "#f5f5f5", "border-radius": "8px"}),
            
            # 收益率曲线
            html.Div([
                html.H2("收益率曲线", style={"color": "#555"}),
                dcc.Graph(id="equity-chart", style={"height": "500px"})
            ], style={"margin-bottom": "30px", "padding": "20px", "background-color": "#f5f5f5", "border-radius": "8px"}),
            
            # 交易记录
            html.Div([
                html.H2("交易记录", style={"color": "#555"}),
                html.Div(id="trade-records", style={"overflow-x": "auto"})
            ], style={"margin-bottom": "30px", "padding": "20px", "background-color": "#f5f5f5", "border-radius": "8px"}),
            
            # 信号分析
            html.Div([
                html.H2("信号分析", style={"color": "#555"}),
                html.Div(id="signal-analysis", style={"overflow-x": "auto"})
            ], style={"padding": "20px", "background-color": "#f5f5f5", "border-radius": "8px"})
        ], style={"max-width": "1200px", "margin": "0 auto", "padding": "20px"})
    
    def _register_callbacks(self):
        """
        注册回调函数
        """
        @self.app.callback(
            Output("backtest-summary", "children"),
            Input("date-range", "start_date"),
            Input("date-range", "end_date")
        )
        def update_backtest_summary(start_date, end_date):
            """
            更新回测摘要
            """
            try:
                results = self.backtest_results
                
                summary_items = [
                    html.Div([
                        html.H3("初始资金"),
                        html.P(f"${results.get('initial_capital', 0):.2f}")
                    ], style={"margin-right": "20px", "margin-bottom": "15px", "padding": "15px", "border": "1px solid #ddd", "border-radius": "8px", "background-color": "white", "flex": "1"}),
                    html.Div([
                        html.H3("最终资金"),
                        html.P(f"${results.get('final_capital', 0):.2f}")
                    ], style={"margin-right": "20px", "margin-bottom": "15px", "padding": "15px", "border": "1px solid #ddd", "border-radius": "8px", "background-color": "white", "flex": "1"}),
                    html.Div([
                        html.H3("总收益"),
                        html.P(f"{results.get('total_return', 0):.2f}%", style={"color": "green" if results.get('total_return', 0) >= 0 else "red"})
                    ], style={"margin-right": "20px", "margin-bottom": "15px", "padding": "15px", "border": "1px solid #ddd", "border-radius": "8px", "background-color": "white", "flex": "1"}),
                    html.Div([
                        html.H3("最大回撤"),
                        html.P(f"{results.get('max_drawdown', 0):.2f}%", style={"color": "red"})
                    ], style={"margin-right": "20px", "margin-bottom": "15px", "padding": "15px", "border": "1px solid #ddd", "border-radius": "8px", "background-color": "white", "flex": "1"}),
                    html.Div([
                        html.H3("夏普比率"),
                        html.P(f"{results.get('sharp_ratio', 0):.2f}")
                    ], style={"margin-right": "20px", "margin-bottom": "15px", "padding": "15px", "border": "1px solid #ddd", "border-radius": "8px", "background-color": "white", "flex": "1"}),
                    html.Div([
                        html.H3("交易次数"),
                        html.P(f"{results.get('trade_count', 0)}")
                    ], style={"margin-bottom": "15px", "padding": "15px", "border": "1px solid #ddd", "border-radius": "8px", "background-color": "white", "flex": "1"})
                ]
                
                return html.Div(summary_items, style={"display": "flex", "flex-wrap": "wrap"})
            except Exception as e:
                self.logger.error(f"更新回测摘要失败: {e}")
                return html.P("无法加载回测摘要", style={"color": "red"})
        
        @self.app.callback(
            Output("price-indicator-chart", "figure"),
            Input("date-range", "start_date"),
            Input("date-range", "end_date")
        )
        def update_price_indicator_chart(start_date, end_date):
            """
            更新价格和指标图表
            """
            try:
                # 确保市场数据不为空
                if self.market_data.empty:
                    # 创建空图表
                    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, 
                                      vertical_spacing=0.1, 
                                      subplot_titles=(
                                          "价格走势与交易信号", 
                                          "MACD指标", 
                                          "交易量"
                                      ))
                    
                    # 更新布局
                    fig.update_layout(
                        height=800,
                        template='plotly_white',
                        showlegend=True,
                        title="市场数据为空，请检查数据加载",
                        title_font=dict(size=16, color="#555")
                    )
                    
                    return fig
                
                # 过滤数据
                try:
                    filtered_data = self.market_data[
                        (self.market_data.index >= start_date) & 
                        (self.market_data.index <= end_date)
                    ]
                except Exception:
                    filtered_data = self.market_data
                
                # 创建图表，调整布局使K线图更大
                fig = make_subplots(rows=3, cols=1, shared_xaxes=True, 
                                  vertical_spacing=0.1, 
                                  row_heights=[0.5, 0.25, 0.25],
                                  subplot_titles=(
                                      "价格走势与交易信号", 
                                      "MACD指标", 
                                      "交易量"
                                  ))
                
                # 绘制K线图
                fig.add_trace(
                    go.Candlestick(
                        x=filtered_data.index,
                        open=filtered_data['open'],
                        high=filtered_data['high'],
                        low=filtered_data['low'],
                        close=filtered_data['close'],
                        name='K线',
                        increasing_line_color='#2ca02c',
                        decreasing_line_color='#d62728',
                        increasing_fillcolor='#2ca02c',
                        decreasing_fillcolor='#d62728',
                        line=dict(width=1)
                    ),
                    row=1, col=1
                )
                
                # 标记买卖信号
                buy_signals = []
                sell_signals = []
                
                # 从策略信号中获取买卖点
                if hasattr(self.strategy, 'signals'):
                    for signal in self.strategy.signals:
                        try:
                            if 'timestamp' in signal and 'price' in signal and 'type' in signal:
                                # 检查信号时间是否在过滤范围内
                                signal_time = signal['timestamp']
                                if isinstance(signal_time, str):
                                    signal_time = pd.to_datetime(signal_time)
                                
                                if signal_time >= pd.to_datetime(start_date) and signal_time <= pd.to_datetime(end_date):
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
                                
                                if trade_time >= pd.to_datetime(start_date) and trade_time <= pd.to_datetime(end_date):
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
                    fig.add_trace(
                        go.Scatter(
                            x=buy_df['timestamp'],
                            y=buy_df['price'],
                            mode='markers',
                            name='买入信号',
                            marker=dict(color='#2ca02c', size=10, symbol='triangle-up', line=dict(width=2, color='black'))
                        ),
                        row=1, col=1
                    )
                
                # 绘制卖出信号
                if sell_signals:
                    sell_df = pd.DataFrame(sell_signals)
                    fig.add_trace(
                        go.Scatter(
                            x=sell_df['timestamp'],
                            y=sell_df['price'],
                            mode='markers',
                            name='卖出信号',
                            marker=dict(color='#d62728', size=10, symbol='triangle-down', line=dict(width=2, color='black'))
                        ),
                        row=1, col=1
                    )
                
                # 绘制MACD
                # 尝试从策略中获取MACD数据
                if hasattr(self.strategy, 'indicators') and self.strategy.indicators:
                    try:
                        macd_line = self.strategy.indicators['macd_line']
                        signal_line = self.strategy.indicators['signal_line']
                        
                        # 过滤MACD数据
                        filtered_macd = macd_line[
                            (macd_line.index >= start_date) & 
                            (macd_line.index <= end_date)
                        ]
                        filtered_signal = signal_line[
                            (signal_line.index >= start_date) & 
                            (signal_line.index <= end_date)
                        ]
                        
                        fig.add_trace(
                            go.Scatter(
                                x=filtered_macd.index,
                                y=filtered_macd,
                                name='MACD',
                                line=dict(color='#1f77b4')
                            ),
                            row=2, col=1
                        )
                        fig.add_trace(
                            go.Scatter(
                                x=filtered_signal.index,
                                y=filtered_signal,
                                name='信号线',
                                line=dict(color='#ff7f0e')
                            ),
                            row=2, col=1
                        )
                        
                        # 添加零轴
                        fig.add_trace(
                            go.Scatter(
                                x=filtered_macd.index,
                                y=[0]*len(filtered_macd),
                                name='零轴',
                                line=dict(color='gray', dash='dash', width=1)
                            ),
                            row=2, col=1
                        )
                    except Exception:
                        # 如果获取失败，使用默认值
                        fig.add_trace(
                            go.Scatter(
                                x=filtered_data.index,
                                y=[0]*len(filtered_data),
                                name='MACD (需要计算)',
                                line=dict(color='#1f77b4')
                            ),
                            row=2, col=1
                        )
                else:
                    # 使用默认值
                    fig.add_trace(
                        go.Scatter(
                            x=filtered_data.index,
                            y=[0]*len(filtered_data),
                            name='MACD',
                            line=dict(color='#1f77b4')
                        ),
                        row=2, col=1
                    )
                
                # 绘制交易量
                fig.add_trace(
                    go.Bar(
                        x=filtered_data.index,
                        y=filtered_data['volume'],
                        name='交易量',
                        marker_color='rgba(75, 192, 192, 0.6)'
                    ),
                    row=3, col=1
                )
                
                # 更新布局
                fig.update_layout(
                    height=800,
                    template='plotly_white',
                    showlegend=True,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    font=dict(family="Arial", size=12, color="#333")
                )
                
                # 更新坐标轴
                fig.update_xaxes(title_text="时间", row=3, col=1)
                fig.update_yaxes(title_text="价格", row=1, col=1)
                fig.update_yaxes(title_text="MACD", row=2, col=1)
                fig.update_yaxes(title_text="交易量", row=3, col=1)
                
                return fig
            except Exception as e:
                self.logger.error(f"更新价格指标图表失败: {e}")
                # 创建错误图表
                fig = make_subplots(rows=3, cols=1, shared_xaxes=True, 
                                  vertical_spacing=0.1, 
                                  subplot_titles=(
                                      "价格走势与交易信号", 
                                      "MACD指标", 
                                      "交易量"
                                  ))
                fig.update_layout(
                    height=800,
                    template='plotly_white',
                    showlegend=True,
                    title=f"图表加载失败: {str(e)}",
                    title_font=dict(size=14, color="red")
                )
                return fig
        
        @self.app.callback(
            Output("equity-chart", "figure"),
            Input("date-range", "start_date"),
            Input("date-range", "end_date")
        )
        def update_equity_chart(start_date, end_date):
            """
            更新收益率曲线
            """
            try:
                # 从回测结果中获取权益曲线
                equity_curve = self.backtest_results.get('equity_curve', pd.DataFrame())
                
                if equity_curve.empty:
                    # 如果没有权益曲线，创建一个默认的
                    dates = pd.date_range(start=start_date, end=end_date)
                    equity_curve = pd.DataFrame({
                        'equity': [self.backtest_results.get('initial_capital', 10000)] * len(dates)
                    }, index=dates)
                else:
                    # 过滤数据
                    equity_curve = equity_curve[
                        (equity_curve.index >= start_date) & 
                        (equity_curve.index <= end_date)
                    ]
                
                # 创建图表
                fig = go.Figure()
                
                # 绘制权益曲线
                fig.add_trace(
                    go.Scatter(
                        x=equity_curve.index,
                        y=equity_curve['equity'],
                        name='权益',
                        line=dict(color='#2ca02c', width=2)
                    )
                )
                
                # 绘制初始资金线
                initial_capital = self.backtest_results.get('initial_capital', 10000)
                fig.add_trace(
                    go.Scatter(
                        x=equity_curve.index,
                        y=[initial_capital] * len(equity_curve),
                        name='初始资金',
                        line=dict(color='gray', dash='dash', width=1)
                    )
                )
                
                # 更新布局
                fig.update_layout(
                    title='权益曲线',
                    xaxis_title='时间',
                    yaxis_title='权益',
                    height=500,
                    template='plotly_white',
                    showlegend=True,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    font=dict(family="Arial", size=12, color="#333")
                )
                
                return fig
            except Exception as e:
                self.logger.error(f"更新权益曲线失败: {e}")
                # 创建错误图表
                fig = go.Figure()
                fig.add_trace(
                    go.Scatter(
                        x=pd.date_range(start=start_date, end=end_date),
                        y=[0],
                        name='错误'
                    )
                )
                fig.update_layout(
                    title=f"图表加载失败: {str(e)}",
                    title_font=dict(size=14, color="red"),
                    height=500,
                    template='plotly_white'
                )
                return fig
        
        @self.app.callback(
            Output("trade-records", "children"),
            Input("date-range", "start_date"),
            Input("date-range", "end_date")
        )
        def update_trade_records(start_date, end_date):
            """
            更新交易记录
            """
            try:
                # 过滤交易记录
                filtered_trades = []
                for trade in self.backtest_results.get('trades', []):
                    try:
                        trade_time = trade['timestamp']
                        if isinstance(trade_time, str):
                            trade_date = trade_time.split(' ')[0]
                        elif hasattr(trade_time, 'strftime'):
                            trade_date = trade_time.strftime('%Y-%m-%d')
                        else:
                            continue
                        
                        if start_date <= trade_date <= end_date:
                            filtered_trades.append(trade)
                    except Exception:
                        pass
                
                if not filtered_trades:
                    return html.P("所选时间范围内无交易记录", style={"color": "#666"})
                
                # 创建交易记录表格
                table_header = [
                    html.Th("时间"),
                    html.Th("类型"),
                    html.Th("价格"),
                    html.Th("收益")
                ]
                
                table_rows = []
                for trade in filtered_trades:
                    try:
                        trade_time = trade['timestamp']
                        if isinstance(trade_time, str):
                            time_str = trade_time
                        elif hasattr(trade_time, 'strftime'):
                            time_str = trade_time.strftime('%Y-%m-%d %H:%M:%S')
                        else:
                            time_str = str(trade_time)
                        
                        trade_type = trade.get('signal_type', 'unknown')
                        price = trade.get('price', 0)
                        profit = trade.get('profit', 0)
                        
                        table_rows.append(
                            html.Tr([
                                html.Td(time_str),
                                html.Td(trade_type, style={"color": "green" if trade_type == 'buy' else "red"}),
                                html.Td(f"{price:.2f}"),
                                html.Td(f"{profit:.2f}", style={"color": "green" if profit >= 0 else "red"})
                            ])
                        )
                    except Exception:
                        pass
                
                return html.Table(
                    [
                        html.Thead(html.Tr(table_header, style={"background-color": "#f2f2f2"})),
                        html.Tbody(table_rows)
                    ],
                    style={
                        "border-collapse": "collapse",
                        "width": "100%",
                        "border": "1px solid #ddd",
                        "font-family": "Arial"
                    }
                )
            except Exception as e:
                self.logger.error(f"更新交易记录失败: {e}")
                return html.P("加载交易记录失败", style={"color": "red"})
        
        @self.app.callback(
            Output("signal-analysis", "children"),
            Input("date-range", "start_date"),
            Input("date-range", "end_date")
        )
        def update_signal_analysis(start_date, end_date):
            """
            更新信号分析
            """
            try:
                # 过滤信号
                filtered_signals = []
                if hasattr(self.strategy, 'signals'):
                    for signal in self.strategy.signals:
                        try:
                            signal_time = signal['timestamp']
                            if isinstance(signal_time, str):
                                signal_date = signal_time.split(' ')[0]
                            elif hasattr(signal_time, 'strftime'):
                                signal_date = signal_time.strftime('%Y-%m-%d')
                            else:
                                continue
                            
                            if start_date <= signal_date <= end_date:
                                filtered_signals.append(signal)
                        except Exception:
                            pass
                
                if not filtered_signals:
                    return html.P("所选时间范围内无信号记录", style={"color": "#666"})
                
                # 创建信号分析卡片
                signal_cards = []
                for signal in filtered_signals:
                    try:
                        if signal.get('type') != 'hold':  # 只显示买卖信号
                            signal_time = signal['timestamp']
                            if isinstance(signal_time, str):
                                time_str = signal_time
                            elif hasattr(signal_time, 'strftime'):
                                time_str = signal_time.strftime('%Y-%m-%d %H:%M:%S')
                            else:
                                time_str = str(signal_time)
                            
                            card = html.Div([
                                html.H4(f"{signal['type'].upper()} 信号", style={"color": "green" if signal['type'] == 'buy' else "red"}),
                                html.P(f"时间: {time_str}"),
                                html.P(f"价格: {signal.get('price', 0):.2f}"),
                                html.P(f"原因: {signal.get('reason', '无')}"),
                                html.P("指标值:", style={"margin-top": "10px"}),
                                html.Ul([
                                    html.Li(f"{key}: {value:.4f}") 
                                    for key, value in signal.get('indicators', {}).items()
                                ])
                            ], style={
                                "border": "1px solid #ddd",
                                "padding": "20px",
                                "margin-bottom": "20px",
                                "border-radius": "8px",
                                "background-color": "white",
                                "box-shadow": "0 2px 4px rgba(0,0,0,0.1)"
                            })
                            signal_cards.append(card)
                    except Exception:
                        pass
                
                if not signal_cards:
                    return html.P("所选时间范围内无有效信号记录", style={"color": "#666"})
                
                return html.Div(signal_cards, style={"display": "flex", "flex-wrap": "wrap", "gap": "20px"})
            except Exception as e:
                self.logger.error(f"更新信号分析失败: {e}")
                return html.P("加载信号分析失败", style={"color": "red"})
    
    def run(self, port=8050, debug=True):
        """
        运行应用
        """
        try:
            self.logger.info(f"启动可视化应用，端口: {port}")
            self.logger.info(f"应用布局已设置: {self.app.layout is not None}")
            self.logger.info(f"回调函数已注册: {len(self.app.callback_map) > 0}")
            
            # 使用新的Dash API运行应用
            self.app.run(
                port=port, 
                debug=debug,
                host='0.0.0.0'  # 允许外部访问
            )
        except Exception as e:
            self.logger.error(f"运行可视化应用失败: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            raise

# 运行可视化器
if __name__ == "__main__":
    # 这里需要从外部传入回测结果、策略和市场数据
    # 实际使用时，应该在main.py中调用此可视化器
    pass
