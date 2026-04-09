#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
市场状态判断模块 (Market Regime Detection)

判断当前市场状态:
1. 上行趋势 (Bullish/Uptrend)
2. 下行趋势 (Bearish/Downtrend)
3. 震荡整理 (Ranging/Consolidation)
4. 高波动状态 (Volatile)

判断依据:
- 价格走势与均线关系
- 动量指标 (RSI, MACD)
- 波动率状态
- 市场宽度 (上涨/下跌家数)
- 趋势强度 (ADX)
- 新高新低指标
- 腾落指标 (AD Line)
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, List
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta


class MarketRegime(Enum):
    """市场状态枚举"""
    BULLISH = "bullish"           # 上行趋势
    BEARISH = "bearish"           # 下行趋势
    NEUTRAL = "neutral"           # 中性/震荡
    VOLATILE = "volatile"         # 高波动状态


class TrendDirection(Enum):
    """趋势方向"""
    STRONG_UP = "strong_up"
    UP = "up"
    NEUTRAL = "neutral"
    DOWN = "down"
    STRONG_DOWN = "strong_down"


@dataclass
class MarketRegimeResult:
    """市场状态结果"""
    regime: MarketRegime
    trend: TrendDirection
    confidence: float              # 0-100, 判断置信度
    strength: float                # 趋势强度 0-100
    momentum: float                # 动量指标 0-100
    volatility: str                 # 'low', 'medium', 'high'
    breadth_score: float            # 市场宽度评分 0-100
    composite_score: float         # 综合评分 0-100
    details: Dict                  # 详细指标
    sentiment: str = "neutral"     # 市场情绪
    risk_level: str = "medium"     # 风险等级


@dataclass
class MarketBreadthResult:
    """市场宽度指标结果"""
    advance_decline_ratio: float   # 上涨/下跌家数比
    ad_line: float                 # 腾落线值
    ad_line_trend: str             # 腾落线趋势
    new_high_new_low: float        # 新高新低指标
    nh_nl_ratio: float            # 新高/新低比
    up_volume_ratio: float         # 上涨成交量占比
    breadth_score: float           # 宽度评分 0-100
    details: Dict = field(default_factory=dict)


class MarketBreadthAnalyzer:
    """
    市场宽度分析器

    分析指标:
    - 上涨/下跌家数比 (A/D Ratio)
    - 腾落线 (Advance-Decline Line)
    - 新高/新低指标 (New High/New Low)
    - 成交量分布
    """

    def __init__(self):
        self.lookback_period = 20  # 回溯周期

    def analyze(self, stock_data_list: List[pd.DataFrame] = None,
                index_data: pd.DataFrame = None) -> MarketBreadthResult:
        """
        分析市场宽度

        Args:
            stock_data_list: 个股数据列表 [pd.DataFrame, ...]
            index_data: 指数数据

        Returns:
            MarketBreadthResult: 市场宽度结果
        """
        if stock_data_list is None and index_data is None:
            # 无法获取市场宽度数据，返回默认值
            return self._default_result()

        # 1. 上涨/下跌家数比
        ad_ratio = self._calculate_ad_ratio(stock_data_list)

        # 2. 腾落线
        ad_line, ad_trend = self._calculate_ad_line(stock_data_list)

        # 3. 新高/新低指标
        nh_nl_ratio, new_high_new_low = self._calculate_nh_nl(stock_data_list)

        # 4. 成交量分布
        up_vol_ratio = self._calculate_volume_distribution(stock_data_list)

        # 5. 计算综合宽度评分
        breadth_score = self._calculate_breadth_score(
            ad_ratio, ad_trend, nh_nl_ratio, up_vol_ratio
        )

        return MarketBreadthResult(
            advance_decline_ratio=ad_ratio,
            ad_line=ad_line,
            ad_line_trend=ad_trend,
            new_high_new_low=new_high_new_low,
            nh_nl_ratio=nh_nl_ratio,
            up_volume_ratio=up_vol_ratio,
            breadth_score=breadth_score,
            details={
                'ad_ratio': ad_ratio,
                'ad_trend': ad_trend,
                'nh_nl_ratio': nh_nl_ratio,
                'up_vol_ratio': up_vol_ratio,
            }
        )

    def _default_result(self) -> MarketBreadthResult:
        """返回默认结果（无数据时）"""
        return MarketBreadthResult(
            advance_decline_ratio=1.0,
            ad_line=0.0,
            ad_line_trend='neutral',
            new_high_new_low=0.0,
            nh_nl_ratio=1.0,
            up_volume_ratio=0.5,
            breadth_score=50.0,
            details={}
        )

    def _calculate_ad_ratio(self, stock_data_list: List[pd.DataFrame]) -> float:
        """计算上涨/下跌家数比"""
        if not stock_data_list:
            return 1.0

        advancing = 0
        declining = 0

        for data in stock_data_list:
            if len(data) < 2:
                continue

            current_close = data['close'].iloc[-1]
            previous_close = data['close'].iloc[-2]

            if current_close > previous_close:
                advancing += 1
            elif current_close < previous_close:
                declining += 1

        if declining == 0:
            return float(advancing) if advancing > 0 else 1.0

        return advancing / declining

    def _calculate_ad_line(self, stock_data_list: List[pd.DataFrame]) -> tuple:
        """
        计算腾落线 (Advance-Decline Line)

        Returns:
            (ad_line_value, trend_direction)
        """
        if not stock_data_list:
            return 0.0, 'neutral'

        # 计算每日净涨跌家数
        daily_net = []
        for i in range(len(stock_data_list[0])):
            net = 0
            for data in stock_data_list:
                if i >= len(data):
                    continue
                if i == 0:
                    continue
                current = data['close'].iloc[i]
                previous = data['close'].iloc[i-1]
                if current > previous:
                    net += 1
                elif current < previous:
                    net -= 1
            daily_net.append(net)

        if not daily_net:
            return 0.0, 'neutral'

        # 计算累计腾落值
        ad_line = sum(daily_net)

        # 判断趋势
        if len(daily_net) >= 5:
            recent_avg = np.mean(daily_net[-5:])
            overall_avg = np.mean(daily_net)
            if recent_avg > overall_avg * 1.2:
                trend = 'up'
            elif recent_avg < overall_avg * 0.8:
                trend = 'down'
            else:
                trend = 'neutral'
        else:
            trend = 'neutral'

        return float(ad_line), trend

    def _calculate_nh_nl(self, stock_data_list: List[pd.DataFrame]) -> tuple:
        """
        计算新高/新低指标

        Returns:
            (nh_nl_ratio, new_high_new_low)
        """
        if not stock_data_list:
            return 1.0, 0.0

        lookback = 20  # 20日新高/新低
        new_highs = 0
        new_lows = 0

        for data in stock_data_list:
            if len(data) < lookback:
                continue

            current_high = data['high'].iloc[-1]
            current_low = data['low'].iloc[-1]

            # 20日最高价和最低价
            high_20d = data['high'].iloc[-lookback:].max()
            low_20d = data['low'].iloc[-lookback:].min()

            if current_high >= high_20d:
                new_highs += 1
            if current_low <= low_20d:
                new_lows += 1

        if new_lows == 0:
            nh_nl_ratio = float(new_highs) if new_highs > 0 else 1.0
        else:
            nh_nl_ratio = new_highs / new_lows

        new_high_new_low = new_highs - new_lows

        return float(nh_nl_ratio), float(new_high_new_low)

    def _calculate_volume_distribution(self, stock_data_list: List[pd.DataFrame]) -> float:
        """计算上涨成交量占比"""
        if not stock_data_list:
            return 0.5

        up_volume = 0
        total_volume = 0

        for data in stock_data_list:
            if len(data) < 2:
                continue

            current_close = data['close'].iloc[-1]
            previous_close = data['close'].iloc[-2]

            if 'volume' in data.columns:
                volume = data['volume'].iloc[-1]
                total_volume += volume

                if current_close > previous_close:
                    up_volume += volume

        if total_volume == 0:
            return 0.5

        return up_volume / total_volume

    def _calculate_breadth_score(self, ad_ratio: float, ad_trend: str,
                                  nh_nl_ratio: float, up_vol_ratio: float) -> float:
        """计算市场宽度综合评分"""
        score = 50.0

        # 上涨/下跌家数比 (权重 30%)
        if ad_ratio >= 2.0:
            score += 20
        elif ad_ratio >= 1.5:
            score += 15
        elif ad_ratio >= 1.2:
            score += 10
        elif ad_ratio >= 1.0:
            score += 5
        elif ad_ratio >= 0.8:
            score -= 5
        elif ad_ratio >= 0.5:
            score -= 15
        else:
            score -= 20

        # 腾落线趋势 (权重 25%)
        if ad_trend == 'up':
            score += 15
        elif ad_trend == 'down':
            score -= 15

        # 新高/新低比 (权重 25%)
        if nh_nl_ratio >= 3.0:
            score += 20
        elif nh_nl_ratio >= 2.0:
            score += 15
        elif nh_nl_ratio >= 1.0:
            score += 5
        elif nh_nl_ratio >= 0.5:
            score -= 10
        else:
            score -= 20

        # 成交量分布 (权重 20%)
        if up_vol_ratio >= 0.6:
            score += 10
        elif up_vol_ratio >= 0.55:
            score += 5
        elif up_vol_ratio >= 0.45:
            score += 0
        elif up_vol_ratio >= 0.4:
            score -= 5
        else:
            score -= 10

        return max(0, min(100, score))


class MarketSentimentAnalyzer:
    """
    市场情绪分析器

    分析市场情绪指标:
    - 恐慌指数 (类似VIX)
    - 资金流向
    - 情绪周期
    """

    def __init__(self):
        self.fear_greed_levels = {
            'extreme_fear': (0, 25),
            'fear': (25, 45),
            'neutral': (45, 55),
            'greed': (55, 75),
            'extreme_greed': (75, 100)
        }

    def analyze(self, price_data: pd.DataFrame, breadth_result: MarketBreadthResult = None) -> Dict:
        """
        分析市场情绪

        Args:
            price_data: 价格数据
            breadth_result: 市场宽度结果

        Returns:
            Dict: 情绪分析结果
        """
        # 1. 价格波动率 (类似VIX)
        fear_greed = self._calculate_fear_greed(price_data)

        # 2. 动量确认
        momentum_confirmation = self._calculate_momentum_confirmation(price_data)

        # 3. 整体情绪判断
        sentiment = self._determine_sentiment(fear_greed, momentum_confirmation, breadth_result)

        # 4. 风险等级
        risk_level = self._determine_risk_level(fear_greed, price_data)

        return {
            'fear_greed_index': fear_greed,
            'fear_greed_level': self._get_fear_greed_level(fear_greed),
            'momentum_confirmation': momentum_confirmation,
            'sentiment': sentiment,
            'risk_level': risk_level,
        }

    def _calculate_fear_greed(self, data: pd.DataFrame) -> float:
        """
        计算恐慌贪婪指数 (0-100)

        基于:
        - 价格波动率 (ATR%)
        - RSI
        - 趋势强度 (ADX)
        """
        close = data['close']
        high = data['high']
        low = data['low']

        # 波动率 (ATR%)
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(14).mean()
        atr_pct = (atr / close * 100).iloc[-1]

        # RSI
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        rsi = (100 - (100 / (1 + rs))).iloc[-1]

        # 趋势强度 (简化)
        ma20 = close.rolling(20).mean()
        ma200 = close.rolling(200).mean()
        trend_pct = (ma20.iloc[-1] / ma200.iloc[-1] - 1) * 100 if not pd.isna(ma200.iloc[-1]) else 0

        # 综合计算 (0-100)
        fear_greed = 50.0

        # 波动率调整 (高波动 = 恐惧)
        if atr_pct > 5:
            fear_greed -= 20
        elif atr_pct > 3:
            fear_greed -= 10
        elif atr_pct < 1:
            fear_greed += 10

        # RSI调整
        if rsi > 70:
            fear_greed += 15  # 过度贪婪
        elif rsi > 60:
            fear_greed += 5
        elif rsi < 30:
            fear_greed -= 15  # 过度恐惧
        elif rsi < 40:
            fear_greed -= 5

        # 趋势调整
        if trend_pct > 10:
            fear_greed += 10
        elif trend_pct < -10:
            fear_greed -= 10

        return max(0, min(100, fear_greed))

    def _calculate_momentum_confirmation(self, data: pd.DataFrame) -> str:
        """动量确认"""
        close = data['close']

        # 短期vs中期动量
        mom_5d = close.iloc[-1] / close.iloc[-5] - 1 if len(close) >= 5 else 0
        mom_20d = close.iloc[-1] / close.iloc[-20] - 1 if len(close) >= 20 else 0

        if mom_5d > 0 and mom_20d > 0:
            return 'confirmed'  # 双重确认上涨
        elif mom_5d < 0 and mom_20d < 0:
            return 'diverging'   # 双重确认下跌
        elif mom_5d > 0 and mom_20d < 0:
            return 'warning'     # 短期反弹但中期仍弱
        elif mom_5d < 0 and mom_20d > 0:
            return 'caution'    # 短期回调但中期仍强
        else:
            return 'neutral'

    def _determine_sentiment(self, fear_greed: float, momentum: str, breadth: MarketBreadthResult = None) -> str:
        """确定市场情绪"""
        if fear_greed < 25:
            sentiment = 'extreme_fear'
        elif fear_greed < 45:
            sentiment = 'fear'
        elif fear_greed < 55:
            sentiment = 'neutral'
        elif fear_greed < 75:
            sentiment = 'greed'
        else:
            sentiment = 'extreme_greed'

        # 如果有市场宽度数据，进一步确认
        if breadth is not None:
            breadth_val = breadth.breadth_score
            if breadth_val < 35 and sentiment in ['greed', 'extreme_greed']:
                sentiment = 'neutral'  # 宽度指标不支持乐观情绪

        return sentiment

    def _determine_risk_level(self, fear_greed: float, data: pd.DataFrame) -> str:
        """确定风险等级"""
        # 基于恐惧指数和价格位置
        close = data['close']
        bb_upper = close.rolling(20).mean() + 2 * close.rolling(20).std()
        bb_lower = close.rolling(20).mean() - 2 * close.rolling(20).std()
        bb_position = (close.iloc[-1] - bb_lower.iloc[-1]) / (bb_upper.iloc[-1] - bb_lower.iloc[-1])

        if fear_greed > 75 or bb_position > 0.95:
            return 'high_risk'
        elif fear_greed < 30 or bb_position < 0.1:
            return 'low_risk'
        else:
            return 'medium_risk'

    def _get_fear_greed_level(self, value: float) -> str:
        """获取恐惧贪婪等级"""
        for level, (low, high) in self.fear_greed_levels.items():
            if low <= value < high:
                return level.replace('_', ' ').title()
        return 'Extreme Greed' if value >= 75 else 'Extreme Fear'


class MarketRegimeAnalyzer:
    """
    市场状态分析器

    使用多种指标综合判断市场状态:
    - 价格均线系统 (SMA/EMA)
    - 动量指标 (RSI, MACD, ADX)
    - 波动率指标 (ATR, Bollinger Bandwidth)
    - 趋势一致性评分
    """

    def __init__(self):
        # 均线配置
        self.ma_config = {
            'short': 20,
            'medium': 60,
            'long': 200
        }
        self.rsi_period = 14
        self.atr_period = 14

    def analyze(self, data: pd.DataFrame, index_data: Optional[pd.DataFrame] = None,
                stock_data_list: List[pd.DataFrame] = None) -> MarketRegimeResult:
        """
        综合分析市场状态

        Args:
            data: 个股或ETF数据 (需要有 close, high, low, open, volume)
            index_data: 可选，对应的指数数据用于市场宽度分析
            stock_data_list: 可选，个股数据列表用于市场宽度分析

        Returns:
            MarketRegimeResult: 市场状态结果
        """
        # 1. 趋势分析
        trend_result = self._analyze_trend(data)

        # 2. 动量分析
        momentum_result = self._analyze_momentum(data)

        # 3. 波动率分析
        volatility_result = self._analyze_volatility(data)

        # 4. 趋势强度分析
        strength_result = self._analyze_strength(data)

        # 5. 市场宽度分析
        breadth_analyzer = MarketBreadthAnalyzer()
        breadth_result = breadth_analyzer.analyze(stock_data_list, index_data)

        # 6. 市场情绪分析
        sentiment_analyzer = MarketSentimentAnalyzer()
        sentiment_result = sentiment_analyzer.analyze(data, breadth_result)

        # 7. 计算综合评分
        composite = self._calculate_composite(
            trend_result, momentum_result, volatility_result, strength_result, breadth_result
        )

        # 8. 确定市场状态
        regime = self._determine_regime(composite, trend_result, volatility_result, breadth_result)

        # 9. 确定趋势方向
        direction = self._determine_direction(trend_result, momentum_result, strength_result)

        details = {
            'trend': trend_result,
            'momentum': momentum_result,
            'volatility': volatility_result,
            'strength': strength_result,
            'breadth': breadth_result.__dict__,
            'sentiment': sentiment_result,
        }

        return MarketRegimeResult(
            regime=regime,
            trend=direction,
            confidence=composite['confidence'],
            strength=composite['strength'],
            momentum=composite['momentum'],
            volatility=volatility_result['status'],
            breadth_score=breadth_result.breadth_score,
            composite_score=composite['overall'],
            details=details,
            sentiment=sentiment_result['sentiment'],
            risk_level=sentiment_result['risk_level']
        )

    def _analyze_trend(self, data: pd.DataFrame) -> Dict:
        """趋势分析 - 基于均线系统"""
        close = data['close']

        # 计算均线
        ma_short = close.rolling(self.ma_config['short']).mean()
        ma_medium = close.rolling(self.ma_config['medium']).mean()
        ma_long = close.rolling(self.ma_config['long']).mean()

        current_price = close.iloc[-1]
        price_20d_ago = close.iloc[-20] if len(close) >= 20 else close.iloc[0]
        price_60d_ago = close.iloc[-60] if len(close) >= 60 else close.iloc[0]

        # 均线趋势评分 (0-100)
        score = 0

        # 价格与均线关系
        if current_price > ma_short.iloc[-1]:
            score += 15
        if current_price > ma_medium.iloc[-1]:
            score += 15
        if current_price > ma_long.iloc[-1]:
            score += 20

        # 均线排列
        if ma_short.iloc[-1] > ma_medium.iloc[-1]:
            score += 15
        if ma_medium.iloc[-1] > ma_long.iloc[-1]:
            score += 15

        # 均线方向
        if ma_short.iloc[-1] > ma_short.iloc[-5]:
            score += 10
        if ma_medium.iloc[-1] > ma_medium.iloc[-20]:
            score += 10

        score = min(100, score)

        # 趋势方向
        if score >= 70:
            direction = 'up'
        elif score >= 50:
            direction = 'neutral_up'
        elif score >= 30:
            direction = 'neutral_down'
        else:
            direction = 'down'

        # 价格动量
        momentum_20d = (current_price / price_20d_ago - 1) * 100 if price_20d_ago != 0 else 0
        momentum_60d = (current_price / price_60d_ago - 1) * 100 if price_60d_ago != 0 else 0

        return {
            'score': score,
            'direction': direction,
            'ma_short': ma_short.iloc[-1],
            'ma_medium': ma_medium.iloc[-1],
            'ma_long': ma_long.iloc[-1],
            'momentum_20d': momentum_20d,
            'momentum_60d': momentum_60d,
            'price_vs_ma200_pct': (current_price / ma_long.iloc[-1] - 1) * 100 if not pd.isna(ma_long.iloc[-1]) else 0
        }

    def _analyze_momentum(self, data: pd.DataFrame) -> Dict:
        """动量分析"""
        close = data['close']
        high = data['high']
        low = data['low']

        # RSI
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(self.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(self.rsi_period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        rsi_value = rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50

        # MACD
        ema_fast = close.ewm(span=12).mean()
        ema_slow = close.ewm(span=26).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=9).mean()
        macd_histogram = macd_line - signal_line

        macd_value = macd_line.iloc[-1]
        macd_signal = signal_line.iloc[-1]
        macd_hist = macd_histogram.iloc[-1]

        # MACD 评分
        macd_score = 50
        if macd_hist > 0:
            macd_score += 25
        if macd_value > macd_signal:
            macd_score += 15
        if macd_value > 0:
            macd_score += 10

        # RSI 评分 (40-60 中性区域)
        if 40 <= rsi_value <= 60:
            rsi_score = 50
        elif rsi_value > 60:
            rsi_score = min(100, 50 + (rsi_value - 60) * 2)
        else:
            rsi_score = max(0, 50 - (40 - rsi_value) * 2)

        # 综合动量评分
        momentum_score = (macd_score * 0.5 + rsi_score * 0.5)

        # 动量方向
        if momentum_score >= 70:
            momentum_direction = 'strong_positive'
        elif momentum_score >= 55:
            momentum_direction = 'positive'
        elif momentum_score >= 45:
            momentum_direction = 'neutral'
        elif momentum_score >= 30:
            momentum_direction = 'negative'
        else:
            momentum_direction = 'strong_negative'

        return {
            'score': momentum_score,
            'direction': momentum_direction,
            'rsi': rsi_value,
            'macd': macd_value,
            'macd_signal': macd_signal,
            'macd_histogram': macd_hist,
        }

    def _analyze_volatility(self, data: pd.DataFrame) -> Dict:
        """波动率分析"""
        close = data['close']
        high = data['high']
        low = data['low']

        # ATR
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(self.atr_period).mean()
        atr_value = atr.iloc[-1] if not pd.isna(atr.iloc[-1]) else 0

        # ATR百分比
        atr_pct = (atr_value / close.iloc[-1]) * 100 if close.iloc[-1] != 0 else 0

        # 布林带宽度
        bb_period = 20
        bb_std = 2
        sma = close.rolling(bb_period).mean()
        std = close.rolling(bb_period).std()
        bb_upper = sma + bb_std * std
        bb_lower = sma - bb_std * std
        bb_bandwidth = ((bb_upper - bb_lower) / sma * 100).iloc[-1] if not pd.isna(((bb_upper - bb_lower) / sma * 100).iloc[-1]) else 0

        # 波动率状态
        if atr_pct > 3 or bb_bandwidth > 10:
            volatility_status = 'high'
        elif atr_pct > 1.5 or bb_bandwidth > 5:
            volatility_status = 'medium'
        else:
            volatility_status = 'low'

        # 波动率评分 (低波动率 = 高评分，因为可能趋势形成)
        volatility_score = max(0, 100 - atr_pct * 10 - bb_bandwidth * 2)

        return {
            'score': volatility_score,
            'status': volatility_status,
            'atr': atr_value,
            'atr_pct': atr_pct,
            'bb_bandwidth': bb_bandwidth,
        }

    def _analyze_strength(self, data: pd.DataFrame) -> Dict:
        """趋势强度分析 - ADX"""
        high = data['high']
        low = data['low']
        close = data['close']

        period = 14

        # +DI 和 -DI
        plus_dm = high.diff()
        minus_dm = -low.diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0

        tr = pd.concat([
            high - low,
            abs(high - close.shift()),
            abs(low - close.shift())
        ], axis=1).max(axis=1)

        atr = tr.rolling(period).mean()

        plus_di = 100 * (plus_dm.rolling(period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(period).mean() / atr)

        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(period).mean()

        adx_value = adx.iloc[-1] if not pd.isna(adx.iloc[-1]) else 0
        plus_di_value = plus_di.iloc[-1] if not pd.isna(plus_di.iloc[-1]) else 0
        minus_di_value = minus_di.iloc[-1] if not pd.isna(minus_di.iloc[-1]) else 0

        # ADX评分 (ADX > 25 表示趋势强)
        if adx_value >= 40:
            strength_score = 80 + min(20, adx_value - 40)
        elif adx_value >= 25:
            strength_score = 50 + (adx_value - 25) * 2
        elif adx_value >= 15:
            strength_score = 25 + (adx_value - 15) * 2
        else:
            strength_score = max(0, 25 - (15 - adx_value))

        # 趋势方向 (基于 +DI vs -DI)
        if plus_di_value > minus_di_value:
            di_direction = 'up'
            di_diff = plus_di_value - minus_di_value
        else:
            di_direction = 'down'
            di_diff = minus_di_value - plus_di_value

        return {
            'score': strength_score,
            'adx': adx_value,
            'plus_di': plus_di_value,
            'minus_di': minus_di_value,
            'direction': di_direction,
            'di_diff': di_diff,
        }

    def _calculate_composite(self, trend: Dict, momentum: Dict, volatility: Dict,
                            strength: Dict, breadth: MarketBreadthResult = None) -> Dict:
        """计算综合评分"""

        # 趋势评分
        trend_score = trend['score']

        # 动量评分
        momentum_score = momentum['score']

        # 趋势强度评分
        strength_score = strength['score']

        # 市场宽度评分
        breadth_score = breadth.breadth_score if breadth else 50.0

        # 综合趋势评分 (均线 + 动量 + 强度 + 宽度)
        combined_trend = (
            trend_score * 0.30 +
            momentum_score * 0.30 +
            strength_score * 0.25 +
            breadth_score * 0.15
        )

        # 置信度 (各指标一致性)
        confidence = 100 - abs(trend_score - momentum_score) / 2 - abs(trend_score - strength_score) / 3
        confidence = max(0, min(100, confidence))

        # 趋势强度评分
        trend_strength = strength_score if strength['direction'] == 'up' else (100 - strength_score)

        # 动量评分
        momentum_final = momentum_score

        return {
            'confidence': confidence,
            'strength': trend_strength,
            'momentum': momentum_final,
            'breadth': breadth_score,
            'overall': combined_trend,
        }

    def _determine_regime(self, composite: Dict, trend: Dict, volatility: Dict,
                        breadth: MarketBreadthResult = None) -> MarketRegime:
        """确定市场状态"""

        overall = composite['overall']
        volatility_status = volatility['status']

        # 高波动状态
        if volatility_status == 'high':
            return MarketRegime.VOLATILE

        # 市场宽度判断
        breadth_score = breadth.breadth_score if breadth else 50.0

        # 调整overall评分，结合宽度
        adjusted_overall = overall * 0.7 + breadth_score * 0.3

        # 基于综合评分判断
        if adjusted_overall >= 65:
            return MarketRegime.BULLISH
        elif adjusted_overall <= 35:
            return MarketRegime.BEARISH
        else:
            return MarketRegime.NEUTRAL

    def _determine_direction(self, trend: Dict, momentum: Dict, strength: Dict) -> TrendDirection:
        """确定趋势方向"""

        trend_score = trend['score']
        momentum_score = momentum['score']
        strength_score = strength['score']

        # 综合评分
        combined = (trend_score + momentum_score + strength_score) / 3

        if combined >= 75:
            return TrendDirection.STRONG_UP
        elif combined >= 55:
            return TrendDirection.UP
        elif combined >= 45:
            return TrendDirection.NEUTRAL
        elif combined >= 30:
            return TrendDirection.DOWN
        else:
            return TrendDirection.STRONG_DOWN


def get_strategy_by_regime(regime: MarketRegime) -> Dict:
    """
    根据市场状态推荐策略

    Args:
        regime: 市场状态

    Returns:
        Dict: 策略建议
    """
    strategies = {
        MarketRegime.BULLISH: {
            'name': '趋势跟踪策略',
            'position': '高仓位 (60-80%)',
            'indicators': ['MA金叉', 'MACD买入', 'RSI>50'],
            'stop_loss': '近期低点下方2-3%',
            'description': '市场处于上升趋势，适合顺势而为，超配动量强劲的个股'
        },
        MarketRegime.BEARISH: {
            'name': '防御策略',
            'position': '低仓位 (20-30%)',
            'indicators': ['MA死叉', 'MACD卖出', 'RSI<50'],
            'stop_loss': '严格止损3-5%',
            'description': '市场处于下降趋势，建议减仓或持有现金，等待机会'
        },
        MarketRegime.NEUTRAL: {
            'name': '区间震荡策略',
            'position': '中低仓位 (30-50%)',
            'indicators': ['布林带上下轨', '支撑位买入', 'RSI 40-60'],
            'stop_loss': '支撑位下方2%',
            'description': '市场无明显趋势，在区间内高抛低吸，注意设置止损'
        },
        MarketRegime.VOLATILE: {
            'name': '谨慎策略',
            'position': '轻仓或观望 (10-20%)',
            'indicators': ['等待突破', '缩短止损', '严格风控'],
            'stop_loss': '非常紧密的1-2%',
            'description': '市场波动剧烈，不宜重仓，等趋势明朗后再操作'
        }
    }

    return strategies.get(regime, strategies[MarketRegime.NEUTRAL])


def generate_market_report(result: MarketRegimeResult) -> str:
    """生成市场状态分析报告"""

    strategy = get_strategy_by_regime(result.regime)

    report = []
    report.append("=" * 60)
    report.append("市场状态分析报告")
    report.append("=" * 60)

    report.append(f"\n【市场状态】")
    report.append(f"  状态: {result.regime.value}")
    report.append(f"  趋势方向: {result.trend.value}")
    report.append(f"  置信度: {result.confidence:.1f}%")
    report.append(f"  市场情绪: {result.sentiment}")
    report.append(f"  风险等级: {result.risk_level}")

    report.append(f"\n【综合评分】")
    report.append(f"  综合评分: {result.composite_score:.1f}/100")
    report.append(f"  趋势强度: {result.strength:.1f}/100")
    report.append(f"  动量指标: {result.momentum:.1f}/100")
    report.append(f"  市场宽度: {result.breadth_score:.1f}/100")

    details = result.details

    report.append(f"\n【趋势分析】")
    report.append(f"  趋势评分: {details['trend']['score']:.1f}/100")
    report.append(f"  20日动量: {details['trend']['momentum_20d']:.2f}%")
    report.append(f"  60日动量: {details['trend']['momentum_60d']:.2f}%")
    report.append(f"  价格vs200日均线: {details['trend']['price_vs_ma200_pct']:.2f}%")

    report.append(f"\n【动量指标】")
    report.append(f"  RSI(14): {details['momentum']['rsi']:.1f}")
    report.append(f"  MACD柱状图: {details['momentum']['macd_histogram']:.4f}")

    report.append(f"\n【波动率】")
    report.append(f"  状态: {result.volatility}")
    report.append(f"  ATR(14): {details['volatility']['atr']:.4f}")
    report.append(f"  ATR%: {details['volatility']['atr_pct']:.2f}%")
    report.append(f"  布林带宽: {details['volatility']['bb_bandwidth']:.2f}%")

    report.append(f"\n【趋势强度 ADX】")
    report.append(f"  ADX: {details['strength']['adx']:.1f}")
    report.append(f"  +DI: {details['strength']['plus_di']:.1f}")
    report.append(f"  -DI: {details['strength']['minus_di']:.1f}")

    # 市场宽度
    if 'breadth' in details and details['breadth']:
        breadth = details['breadth']
        report.append(f"\n【市场宽度】")
        ad_ratio = breadth.get('ad_ratio', 0)
        nh_nl_ratio = breadth.get('nh_nl_ratio', 0)
        up_vol_ratio = breadth.get('up_vol_ratio', 0)
        report.append(f"  A/D家数比: {ad_ratio:.2f}" if isinstance(ad_ratio, (int, float)) else f"  A/D家数比: N/A")
        report.append(f"  腾落线趋势: {breadth.get('ad_trend', 'N/A')}")
        report.append(f"  新高/新低比: {nh_nl_ratio:.2f}" if isinstance(nh_nl_ratio, (int, float)) else f"  新高/新低比: N/A")
        report.append(f"  上涨成交量比: {up_vol_ratio:.2%}" if isinstance(up_vol_ratio, (int, float)) else f"  上涨成交量比: N/A")

    # 市场情绪
    if 'sentiment' in details:
        sent = details['sentiment']
        report.append(f"\n【市场情绪】")
        report.append(f"  恐惧贪婪指数: {sent.get('fear_greed_index', 'N/A'):.1f}")
        report.append(f"  情绪等级: {sent.get('fear_greed_level', 'N/A')}")
        report.append(f"  动量确认: {sent.get('momentum_confirmation', 'N/A')}")

    report.append(f"\n【推荐策略】")
    report.append(f"  策略名称: {strategy['name']}")
    report.append(f"  建议仓位: {strategy['position']}")
    report.append(f"  止损设置: {strategy['stop_loss']}")
    report.append(f"  核心指标: {', '.join(strategy['indicators'])}")
    report.append(f"  说明: {strategy['description']}")

    report.append("\n" + "=" * 60)
    return "\n".join(report)
