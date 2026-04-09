#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
新闻信号生成器

基于新闻情感分析生成交易信号
"""

import logging
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum
import pandas as pd

logger = logging.getLogger(__name__)


class SignalType(Enum):
    """信号类型"""
    STRONG_BUY = "strong_buy"    # 强烈买入
    BUY = "buy"                  # 买入
    HOLD = "hold"                # 持有
    SELL = "sell"                # 卖出
    STRONG_SELL = "strong_sell"  # 强烈卖出


@dataclass
class NewsSignal:
    """新闻信号"""
    symbol: str
    signal_type: SignalType
    sentiment_score: float       # -1 到 1
    confidence: float            # 0 到 1
    news_count: int
    positive_count: int
    negative_count: int
    neutral_count: int = 0      # 中性新闻数
    key_news: List[str] = None  # 关键新闻标题
    timestamp: str = None
    metadata: Dict = None

    def __post_init__(self):
        if self.key_news is None:
            self.key_news = []
        if self.timestamp is None:
            self.timestamp = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')


class NewsSignalGenerator:
    """新闻信号生成器"""

    def __init__(self, news_source=None, config: Dict = None):
        """
        Args:
            news_source: 新闻数据源，默认使用AKShare
            config: 配置字典
        """
        self.config = config or {}

        if news_source is None:
            from data.news import AKShareNewsSource
            self.news_source = AKShareNewsSource()
        else:
            self.news_source = news_source

        # 信号阈值配置
        self.strong_buy_threshold = self.config.get('strong_buy_threshold', 0.5)
        self.buy_threshold = self.config.get('buy_threshold', 0.2)
        self.sell_threshold = self.config.get('sell_threshold', -0.2)
        self.strong_sell_threshold = self.config.get('strong_sell_threshold', -0.5)

        # 最小新闻数量
        self.min_news_count = self.config.get('min_news_count', 3)

    def generate_signal(self, symbol: str) -> NewsSignal:
        """
        生成新闻交易信号

        Args:
            symbol: 股票代码

        Returns:
            NewsSignal: 新闻信号对象
        """
        logger.info(f"为 {symbol} 生成新闻信号")

        # 获取情感分析
        sentiment = self.news_source.get_news_sentiment(symbol)

        score = sentiment['sentiment_score']
        news_count = sentiment['news_count']

        # 获取关键新闻
        news = self.news_source.get_stock_news(symbol, days=7)
        key_news = [n['title'][:50] for n in news[:3]] if news else []

        # 确定信号类型
        if news_count < self.min_news_count:
            signal_type = SignalType.HOLD
            confidence = 0.0
        elif score >= self.strong_buy_threshold:
            signal_type = SignalType.STRONG_BUY
            confidence = min(abs(score) * 2, 1.0)
        elif score >= self.buy_threshold:
            signal_type = SignalType.BUY
            confidence = abs(score)
        elif score <= self.strong_sell_threshold:
            signal_type = SignalType.STRONG_SELL
            confidence = min(abs(score) * 2, 1.0)
        elif score <= self.sell_threshold:
            signal_type = SignalType.SELL
            confidence = abs(score)
        else:
            signal_type = SignalType.HOLD
            confidence = 1 - abs(score)

        signal = NewsSignal(
            symbol=symbol,
            signal_type=signal_type,
            sentiment_score=score,
            confidence=confidence,
            news_count=news_count,
            positive_count=sentiment['positive_count'],
            negative_count=sentiment['negative_count'],
            neutral_count=sentiment['neutral_count'],
            key_news=key_news,
            timestamp=pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
            metadata={'source': 'news_sentiment'}
        )

        logger.info(f"{symbol} 新闻信号: {signal_type.value}, "
                   f"情感评分: {score:.2f}, 新闻数: {news_count}")

        return signal

    def combine_with_technical(
        self,
        news_signal: NewsSignal,
        technical_signal: str = None,
        technical_confidence: float = 0.5
    ) -> Dict:
        """
        结合技术面信号

        Args:
            news_signal: 新闻信号
            technical_signal: 技术信号 ('buy', 'sell', 'hold')
            technical_confidence: 技术信号置信度

        Returns:
            Dict: 组合后的信号
        """
        # 转换为分数
        news_score = news_signal.sentiment_score * news_signal.confidence

        if technical_signal == 'buy':
            tech_score = technical_confidence
        elif technical_signal == 'sell':
            tech_score = -technical_confidence
        else:
            tech_score = 0

        # 加权组合
        weights = {'news': 0.4, 'technical': 0.6}
        combined = news_score * weights['news'] + tech_score * weights['technical']

        # 生成最终信号
        if combined >= 0.3:
            final_signal = SignalType.STRONG_BUY if combined >= 0.5 else SignalType.BUY
        elif combined <= -0.3:
            final_signal = SignalType.STRONG_SELL if combined <= -0.5 else SignalType.SELL
        else:
            final_signal = SignalType.HOLD

        return {
            'symbol': news_signal.symbol,
            'signal': final_signal,
            'combined_score': combined,
            'news_contribution': news_score * weights['news'],
            'technical_contribution': tech_score * weights['technical'],
            'news_signal': news_signal,
            'timestamp': news_signal.timestamp
        }

    def get_signal_description(self, signal: NewsSignal) -> str:
        """获取信号描述"""
        score = signal.sentiment_score
        count = signal.news_count

        if signal.signal_type == SignalType.STRONG_BUY:
            return f"强烈看多: 情感评分 {score:.2f}，{count} 条新闻支持"
        elif signal.signal_type == SignalType.BUY:
            return f"看多: 情感评分 {score:.2f}，{count} 条新闻支持"
        elif signal.signal_type == SignalType.HOLD:
            return f"中性: 情感评分 {score:.2f}，{count} 条新闻"
        elif signal.signal_type == SignalType.SELL:
            return f"看空: 情感评分 {score:.2f}，{count} 条新闻"
        else:
            return f"强烈看空: 情感评分 {score:.2f}，{count} 条新闻警示"
