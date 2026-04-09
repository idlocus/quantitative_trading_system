#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
动量轮动策略 V2

策略逻辑：
1. 每日选出技术面最强的1-2只股票
2. 持仓1-3天
3. RSI/MACD指标过滤，避免追高杀跌
4. 严格的止损止盈控制

改进点：
- RSI < 40 超卖区买入，避免追高
- RSI > 70 超买区卖出，锁定利润
- MACD > 0 确认上升趋势
- 市场状态过滤，大盘下跌时不买入
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict

from strategy.base_strategy import BaseStrategy


class MomentumRotationStrategy(BaseStrategy):
    """
    动量轮动策略 V2

    特点：
    - 每日轮动，持仓1-3天
    - RSI/MACD指标过滤，避免追高杀跌
    - 使用技术框架综合评分选股
    - 严格的止损止盈控制
    """

    def __init__(self, config=None):
        # 如果没有配置，使用默认参数
        if config is None:
            config = self._default_config()

        super().__init__(config)

        # 策略参数
        self.max_positions = self.params.get('max_positions', 2)      # 最大持仓数
        self.hold_days_max = self.params.get('hold_days_max', 3)    # 最大持仓天数
        self.stop_loss = self.params.get('stop_loss', 0.03)        # 3%止损
        self.take_profit = self.params.get('take_profit', 0.05)     # 5%止盈
        self.min_rps = self.params.get('min_rps', 70)              # 最低RPS要求
        self.min_composite = self.params.get('min_composite', 50)   # 最低综合评分

        # RSI参数
        self.rsi_buy_max = self.params.get('rsi_buy_max', 40)      # 买入RSI上限（超卖区）
        self.rsi_sell_min = self.params.get('rsi_sell_min', 70)   # 卖出RSI下限（超买区）
        self.rsi_exit_min = self.params.get('rsi_exit_min', 30)   # 止损RSI下限

        # MACD参数
        self.require_macd_positive = self.params.get('require_macd_positive', True)  # 是否要求MACD>0

        # 市场状态过滤
        self.market_filter_enabled = self.params.get('market_filter_enabled', True)
        self.market_min_score = self.params.get('market_min_score', 40)  # 大盘最低评分

        # 持仓状态
        self.positions = {}  # {symbol: {'entry_price': float, 'entry_date': datetime, 'shares': int}}

        # 每日股票评分缓存
        self.daily_scores = {}  # {date: {symbol: {'composite': float, 'rps': float, 'rsi': float, 'macd': float, ...}}}

        # 初始化完成标志
        self.initialized = False

    def _default_config(self):
        """默认配置"""
        return type('Config', (), {
            'strategy_config': {
                'params': {
                    'max_positions': 2,
                    'hold_days_max': 3,
                    'stop_loss': 0.03,
                    'take_profit': 0.05,
                    'min_rps': 70,
                    'min_composite': 50,
                    # RSI参数
                    'rsi_buy_max': 55,          # RSI<55才买入（偏弱区域）
                    'rsi_sell_min': 70,         # RSI>70强制止盈
                    'rsi_exit_min': 30,          # RSI<30不追加止损
                    # MACD参数
                    'require_macd_positive': True,  # 要求MACD>0
                    # 市场过滤
                    'market_filter_enabled': True,
                    'market_min_score': 40,
                }
            }
        })()

    def initialize(self):
        """初始化策略"""
        self.positions = {}
        self.daily_scores = {}
        self.initialized = True
        self.logger.info(f"动量轮动策略V2初始化完成")
        self.logger.info(f"参数: max_positions={self.max_positions}, hold_days={self.hold_days_max}, "
                        f"stop_loss={self.stop_loss}, take_profit={self.take_profit}, "
                        f"rsi_buy_max={self.rsi_buy_max}, rsi_sell_min={self.rsi_sell_min}")

    def set_daily_scores(self, date, scores_dict):
        """
        设置每日股票评分

        Args:
            date: 日期
            scores_dict: {symbol: {'composite': float, 'rps': float, 'rsi': float, 'macd': float, ...}}
        """
        self.daily_scores[date] = scores_dict

    def set_market_regime(self, date, regime, score):
        """
        设置大盘状态

        Args:
            date: 日期
            regime: 市场状态 ('bullish', 'bearish', 'neutral', 'volatile')
            score: 大盘评分
        """
        if not hasattr(self, 'market_data'):
            self.market_data = {}
        self.market_data[date] = {'regime': regime, 'score': score}

    def get_top_stocks(self, date, n=2, exclude=None):
        """
        获取当日评分最高的股票（带指标过滤）

        Args:
            date: 日期
            n: 获取数量
            exclude: 排除的股票列表

        Returns:
            list of (symbol, score, reason) sorted by score descending
        """
        if date not in self.daily_scores:
            return []

        exclude = exclude or []
        scores = self.daily_scores[date]

        # 检查大盘状态
        market_ok = True
        if self.market_filter_enabled and hasattr(self, 'market_data') and date in self.market_data:
            market_score = self.market_data[date].get('score', 50)
            market_regime = self.market_data[date].get('regime', 'neutral')
            if market_regime in ['bearish', 'volatile'] or market_score < self.market_min_score:
                market_ok = False

        # 过滤并排序
        candidates = []
        for symbol, data in scores.items():
            if symbol in exclude:
                continue

            composite = data.get('composite', 0)
            rps = data.get('rps', 0)
            rsi = data.get('rsi', 50)
            macd = data.get('macd', 0)

            # 基本过滤
            if composite < self.min_composite or rps < self.min_rps:
                continue

            # RSI过滤：只在超卖区买入
            if rsi > self.rsi_buy_max:
                continue

            # MACD过滤：确认上升趋势
            if self.require_macd_positive and macd < 0:
                continue

            # 市场过滤
            if not market_ok:
                continue

            candidates.append((symbol, composite, data))

        # 按评分排序
        candidates.sort(key=lambda x: x[1], reverse=True)
        return [(c[0], c[1], c[2]) for c in candidates[:n]]

    def on_data(self, data):
        """
        处理数据，生成交易信号

        Args:
            data: pd.DataFrame with columns [date, open, high, low, close, volume, symbol]

        Returns:
            list of signals
        """
        if not self.initialized:
            self.initialize()

        signals = []
        current_date = data.index[-1] if hasattr(data.index, '__iter__') else None
        current_price = data['close'].iloc[-1]
        symbol = data['symbol'].iloc[-1] if 'symbol' in data.columns else 'UNKNOWN'

        # 获取当前RSI（如果有）
        rsi = None
        macd = None
        if hasattr(data, 'iloc'):
            # 从data中获取RSI和MACD（由回测系统计算）
            if 'rsi' in data.columns:
                rsi = data['rsi'].iloc[-1]
            if 'macd' in data.columns:
                macd = data['macd'].iloc[-1]

        # 1. 检查现有持仓是否需要卖出
        positions_to_close = []
        for pos_symbol, pos_info in list(self.positions.items()):
            entry_price = pos_info['entry_price']
            entry_date = pos_info['entry_date']
            hold_days = (current_date - entry_date).days if entry_date else 0

            # 计算收益率
            ret = (current_price - entry_price) / entry_price

            # 卖出条件检查
            should_sell = False
            reason = ""

            # 止损检查
            if ret <= -self.stop_loss:
                should_sell = True
                reason = f"止损({ret*100:.1f}%)"
            # 止盈检查（优先于到期）
            elif ret >= self.take_profit:
                should_sell = True
                reason = f"止盈({ret*100:.1f}%)"
            # RSI超买检查
            elif rsi is not None and rsi > self.rsi_sell_min:
                should_sell = True
                reason = f"RSI超买({rsi:.1f})"
            # 到期检查
            elif hold_days >= self.hold_days_max:
                # 到期时如果RSI较高则提前止盈
                if rsi is not None and rsi > 60:
                    should_sell = True
                    reason = f"到期止盈RSI({rsi:.1f})"
                else:
                    should_sell = True
                    reason = f"到期({hold_days}天)"

            if should_sell:
                positions_to_close.append((pos_symbol, reason))

        # 生成卖出信号
        for pos_symbol, reason in positions_to_close:
            signals.append({
                'type': 'sell',
                'symbol': pos_symbol,
                'price': current_price,
                'timestamp': current_date,
                'reason': reason,
                'strategy': 'momentum_rotation_v2'
            })
            del self.positions[pos_symbol]

        # 2. 如果有空位，选取新股票
        available_slots = self.max_positions - len(self.positions)
        if available_slots > 0 and current_date is not None:
            # 获取当前持有的股票（排除）
            held_symbols = list(self.positions.keys())

            # 获取评分最高的股票（带指标过滤）
            top_stocks = self.get_top_stocks(current_date, n=available_slots, exclude=held_symbols)

            for symbol, score, data in top_stocks:
                rsi_val = data.get('rsi', 50)
                macd_val = data.get('macd', 0)

                # 买入信号
                signals.append({
                    'type': 'buy',
                    'symbol': symbol,
                    'price': current_price,
                    'timestamp': current_date,
                    'score': score,
                    'rsi': rsi_val,
                    'macd': macd_val,
                    'strategy': 'momentum_rotation_v2'
                })

                # 记录持仓
                self.positions[symbol] = {
                    'entry_price': current_price,
                    'entry_date': current_date,
                    'score': score
                }

        return signals

    def on_order_update(self, order):
        """处理订单更新"""
        pass

    def on_position_update(self, position):
        """处理持仓更新"""
        pass

    def get_positions(self):
        """获取当前持仓"""
        return self.positions

    def get_stats(self):
        """获取策略统计信息"""
        return {
            'max_positions': self.max_positions,
            'hold_days_max': self.hold_days_max,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'min_rps': self.min_rps,
            'min_composite': self.min_composite,
            'rsi_buy_max': self.rsi_buy_max,
            'rsi_sell_min': self.rsi_sell_min,
            'require_macd_positive': self.require_macd_positive,
            'current_positions': len(self.positions)
        }
