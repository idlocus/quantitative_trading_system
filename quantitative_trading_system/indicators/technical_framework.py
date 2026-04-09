#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
技术分析投资框架方法论
基于BofA Technical Primer

框架核心:
1. 趋势识别 (Trend Identification)
2. 动量分析 (Momentum Analysis)
3. 波动率测量 (Volatility Measurement)
4. 支撑阻力 (Support & Resistance)
5. 形态确认 (Pattern Confirmation)
6. 综合信号评分 (Composite Signal Scoring)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple


class TechnicalInvestmentFramework:
    """
    技术分析投资框架

    核心理念:
    - 先判断市场状态(趋势/震荡)
    - 根据状态选择合适的指标
    - 多指标共振确认信号
    - 严格的风险管理
    """

    def __init__(self):
        # 指标配置
        self.ma_config = {
            'short': 10,
            'medium': 50,
            'long': 200
        }
        self.rsi_period = 14
        self.macd_config = {'fast': 12, 'slow': 26, 'signal': 9}
        self.bb_period = 20
        self.bb_std = 2
        self.atr_period = 14

    def analyze(self, data: pd.DataFrame, index_data: pd.DataFrame = None) -> Dict:
        """
        综合技术分析

        Args:
            data: 股票/ETF数据
            index_data: 可选，指数数据用于市场状态判断

        返回:
        {
            'trend': {'direction': 'up/down/neutral', 'strength': 0-100},
            'momentum': {'value': 0-100, 'signal': 'buy/sell/hold'},
            'volatility': {'value': float, 'status': 'low/medium/high'},
            'support_resistance': {'levels': [...], 'nearest_support': float, 'nearest_resistance': float},
            'signals': {'composite': 0-100, 'recommendation': 'buy/sell/hold'},
            'risk': {'stop_loss': float, 'take_profit': float, 'risk_reward_ratio': float},
            'market_regime': {...}  # 市场状态分析结果
        }
        """
        from .trend_indicators import SMA, EMA, MACD, DMI_ADX, GoldenCrossDeathCross
        from .momentum_indicators import RSI
        from .volatility_indicators import BollingerBands, ATR
        from .pattern_indicators import SupportResistanceLevels, FibonacciRetracement
        from .market_regime import MarketRegimeAnalyzer, generate_market_report

        result = {}

        # 1. 趋势分析
        result['trend'] = self._analyze_trend(data)

        # 2. 动量分析
        result['momentum'] = self._analyze_momentum(data)

        # 3. 波动率分析
        result['volatility'] = self._analyze_volatility(data)

        # 4. 支撑阻力
        result['support_resistance'] = self._analyze_support_resistance(data)

        # 5. 综合评分
        result['signals'] = self._generate_signals(data, result)

        # 6. 风险管理
        result['risk'] = self._calculate_risk(data, result)

        # 7. 市场状态判断 (使用指数数据或自身数据)
        market_data = index_data if index_data is not None else data
        market_analyzer = MarketRegimeAnalyzer()
        market_regime_result = market_analyzer.analyze(market_data)

        result['market_regime'] = {
            'regime': market_regime_result.regime.value,
            'trend': market_regime_result.trend.value,
            'confidence': market_regime_result.confidence,
            'strength': market_regime_result.strength,
            'momentum': market_regime_result.momentum,
            'volatility': market_regime_result.volatility,
            'breadth_score': market_regime_result.breadth_score,
            'composite_score': market_regime_result.composite_score,
            'details': market_regime_result.details
        }

        # 8. 根据市场状态调整信号
        result['signals'] = self._adjust_signals_by_regime(result['signals'], result['market_regime'])

        return result

    def _analyze_trend(self, data: pd.DataFrame) -> Dict:
        """趋势分析"""
        from .trend_indicators import SMA, EMA, MACD, DMI_ADX, GoldenCrossDeathCross

        # 移动平均线分析
        ma_short = SMA(data, self.ma_config['short'])
        ma_medium = SMA(data, self.ma_config['medium'])
        ma_long = SMA(data, self.ma_config['long'])

        current_price = data['close'].iloc[-1]

        # 趋势方向
        trend_score = 0
        if current_price > ma_short.iloc[-1]:
            trend_score += 25
        if current_price > ma_medium.iloc[-1]:
            trend_score += 25
        if current_price > ma_long.iloc[-1]:
            trend_score += 25

        if ma_short.iloc[-1] > ma_medium.iloc[-1]:
            trend_score += 12.5
        if ma_medium.iloc[-1] > ma_long.iloc[-1]:
            trend_score += 12.5

        # MACD分析
        macd = MACD(data, fast_period=12, slow_period=26, signal_period=9)
        macd_trend = 0
        if macd['macd'].iloc[-1] > macd['signal'].iloc[-1]:
            macd_trend = 25
        if macd['macd'].iloc[-1] > 0:
            macd_trend += 25

        trend_score = min(100, (trend_score + macd_trend) / 2)

        # 确定趋势方向
        if trend_score >= 70:
            direction = 'strong_up'
        elif trend_score >= 55:
            direction = 'up'
        elif trend_score >= 45:
            direction = 'neutral'
        elif trend_score >= 30:
            direction = 'down'
        else:
            direction = 'strong_down'

        # DMI趋势确认
        dmi = DMI_ADX(data)
        adx_strength = dmi['adx'].iloc[-1] if not pd.isna(dmi['adx'].iloc[-1]) else 0

        return {
            'direction': direction,
            'strength': trend_score,
            'adx': adx_strength,
            'ma_short': ma_short.iloc[-1],
            'ma_medium': ma_medium.iloc[-1],
            'ma_long': ma_long.iloc[-1],
            'current_price': current_price
        }

    def _analyze_momentum(self, data: pd.DataFrame) -> Dict:
        """动量分析"""
        from .momentum_indicators import RSI, Stochastic, RPS

        # RSI
        rsi = RSI(data, self.rsi_period)
        rsi_value = rsi.iloc[-1]

        # Stochastic
        stoch = Stochastic(data)
        stoch_k = stoch['k'].iloc[-1]
        stoch_d = stoch['d'].iloc[-1]

        # RPS (Relative Price Strength)
        rps = RPS(data, period=20)
        rps_value = rps.iloc[-1] if not pd.isna(rps.iloc[-1]) else 50

        # 动量评分
        momentum_score = 0

        # RSI评分
        if 40 <= rsi_value <= 60:
            momentum_score += 20  # 中性区域
        elif rsi_value > 60:
            momentum_score += 10 + (rsi_value - 60)  # 偏多但不过度
        elif rsi_value < 40:
            momentum_score += 10 + (40 - rsi_value)  # 偏空但不过度

        # Stochastic交叉
        if stoch_k > stoch_d and stoch_k < 80:
            momentum_score += 20

        # RPS评分 (权重: 15分)
        # RPS > 80: 强势股加分
        # RPS < 20: 弱势股减分
        if rps_value >= 80:
            momentum_score += 15
        elif rps_value >= 60:
            momentum_score += 10
        elif rps_value >= 50:
            momentum_score += 5
        elif rps_value >= 40:
            momentum_score += 0
        elif rps_value >= 20:
            momentum_score -= 5
        else:
            momentum_score -= 10

        momentum_score = max(0, min(100, momentum_score))

        # 信号
        if momentum_score >= 70:
            signal = 'strong_buy'
        elif momentum_score >= 55:
            signal = 'buy'
        elif momentum_score >= 45:
            signal = 'neutral'
        elif momentum_score >= 30:
            signal = 'sell'
        else:
            signal = 'strong_sell'

        return {
            'value': momentum_score,
            'signal': signal,
            'rsi': rsi_value,
            'stoch_k': stoch_k,
            'stoch_d': stoch_d,
            'rps': rps_value
        }

    def _analyze_volatility(self, data: pd.DataFrame) -> Dict:
        """波动率分析"""
        from .volatility_indicators import BollingerBands, ATR

        bb = BollingerBands(data, self.bb_period, self.bb_std)
        atr = ATR(data, self.atr_period)

        bb_upper = bb['upper'].iloc[-1]
        bb_lower = bb['lower'].iloc[-1]
        bb_percent = bb['percent_b'].iloc[-1]
        bb_bandwidth = bb['bandwidth'].iloc[-1]
        atr_value = atr.iloc[-1]

        current_price = data['close'].iloc[-1]

        # ATR状态 (相对于价格)
        atr_percent = (atr_value / current_price) * 100

        # 布林带状态
        if bb_percent > 0.9:
            volatility_status = 'high'  # 价格接近上轨
        elif bb_percent < 0.1:
            volatility_status = 'high'  # 价格接近下轨
        elif bb_bandwidth < 2:
            volatility_status = 'low'  # 布林带收缩
        elif bb_bandwidth > 5:
            volatility_status = 'high'
        else:
            volatility_status = 'medium'

        # 波动率评分 (低波动率 = 高评分，因为可能突破)
        vol_score = 100 - min(100, bb_bandwidth * 10)

        return {
            'value': vol_score,
            'status': volatility_status,
            'atr': atr_value,
            'atr_percent': atr_percent,
            'bb_percent': bb_percent,
            'bb_bandwidth': bb_bandwidth,
            'bb_upper': bb_upper,
            'bb_lower': bb_lower
        }

    def _analyze_support_resistance(self, data: pd.DataFrame) -> Dict:
        """支撑阻力分析"""
        from .pattern_indicators import SupportResistanceLevels, FibonacciRetracement

        current_price = data['close'].iloc[-1]
        high_50 = data['high'].rolling(50).max().iloc[-1]
        low_50 = data['low'].rolling(50).min().iloc[-1]

        # 简单支撑阻力
        resistance = high_50
        support = low_50

        # Fibonacci回撤位
        fib_levels = FibonacciRetracement(high_50, low_50)

        # 距离支撑阻力的百分比
        dist_to_resistance = (resistance - current_price) / current_price * 100
        dist_to_support = (current_price - support) / current_price * 100

        return {
            'resistance': resistance,
            'support': support,
            'dist_to_resistance': dist_to_resistance,
            'dist_to_support': dist_to_support,
            'fib_levels': fib_levels,
            'mid_point': (resistance + support) / 2
        }

    def _generate_signals(self, data: pd.DataFrame, analysis: Dict) -> Dict:
        """生成综合信号"""
        trend_score = analysis['trend']['strength']
        momentum_score = analysis['momentum']['value']
        volatility_score = analysis['volatility']['value']

        # 加权综合评分
        weights = {'trend': 0.4, 'momentum': 0.35, 'volatility': 0.25}
        composite = (
            trend_score * weights['trend'] +
            momentum_score * weights['momentum'] +
            volatility_score * weights['volatility']
        )

        # 信号判定
        if composite >= 75:
            recommendation = 'strong_buy'
        elif composite >= 60:
            recommendation = 'buy'
        elif composite >= 45:
            recommendation = 'hold'
        elif composite >= 30:
            recommendation = 'sell'
        else:
            recommendation = 'strong_sell'

        return {
            'composite': composite,
            'recommendation': recommendation,
            'trend_contribution': trend_score * weights['trend'],
            'momentum_contribution': momentum_score * weights['momentum'],
            'volatility_contribution': volatility_score * weights['volatility']
        }

    def _calculate_risk(self, data: pd.DataFrame, analysis: Dict) -> Dict:
        """计算风险管理参数"""
        from .volatility_indicators import ATR

        current_price = data['close'].iloc[-1]
        atr = ATR(data, self.atr_period).iloc[-1]

        # 动态止损 (2x ATR)
        stop_loss = current_price - 2 * atr

        # 止盈目标 (2:1风险收益比)
        risk = current_price - stop_loss
        take_profit = current_price + 2 * risk

        # 支撑位作为备选止损
        sr_support = analysis['support_resistance']['support']
        if sr_support > stop_loss:
            stop_loss = sr_support

        # 计算风险收益比
        risk_reward_ratio = (take_profit - current_price) / (current_price - stop_loss) if stop_loss != current_price else 0

        # 计算推荐仓位
        position_size = 0.5  # 默认50%仓位
        if (current_price - stop_loss) > 0:
            max_loss_percent = 0.02  # 最大损失2%
            risk_per_share = current_price - stop_loss
            position_size = min(1.0, (current_price * max_loss_percent) / risk_per_share)

        return {
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'risk_reward_ratio': risk_reward_ratio,
            'atr': atr,
            'position_size_recommendation': position_size
        }

    def _calculate_position_size(self, data: pd.DataFrame, analysis: Dict) -> float:
        """计算推荐仓位"""
        from .volatility_indicators import ATR

        current_price = data['close'].iloc[-1]
        atr = ATR(data, self.atr_period).iloc[-1]

        # 最大损失控制在2%
        max_loss_percent = 0.02
        risk_per_share = current_price - 2 * atr

        if risk_per_share > 0:
            position_size = (current_price * max_loss_percent) / risk_per_share
            return min(1.0, position_size)  # 最大100%仓位
        return 0.5  # 默认50%仓位

    def _adjust_signals_by_regime(self, signals: Dict, market_regime: Dict) -> Dict:
        """
        根据市场状态调整信号

        在市场状态为熊市或高波动时，降低买入信号的强度
        在市场状态为牛市时，增强买入信号的强度
        """
        regime = market_regime['regime']
        composite = signals['composite']
        recommendation = signals['recommendation']

        adjustment = 0
        adjusted_recommendation = recommendation

        # 根据市场状态调整评分
        if regime == 'bearish':
            adjustment = -15  # 熊市降低评分
        elif regime == 'bullish':
            adjustment = +10  # 牛市增强评分
        elif regime == 'volatile':
            adjustment = -20  # 高波动降低评分
        elif regime == 'neutral':
            adjustment = 0   # 中性不变

        adjusted_composite = max(0, min(100, composite + adjustment))

        # 重新判定信号
        if adjusted_composite >= 75:
            adjusted_recommendation = 'strong_buy'
        elif adjusted_composite >= 60:
            adjusted_recommendation = 'buy'
        elif adjusted_composite >= 45:
            adjusted_recommendation = 'hold'
        elif adjusted_composite >= 30:
            adjusted_recommendation = 'sell'
        else:
            adjusted_recommendation = 'strong_sell'

        # 添加市场状态调整信息
        signals['composite_adjusted'] = adjusted_composite
        signals['recommendation_adjusted'] = adjusted_recommendation
        signals['market_regime_adjustment'] = adjustment
        signals['original_composite'] = composite
        signals['market_regime'] = regime

        return signals


def generate_framework_report(analysis: Dict) -> str:
    """生成投资框架分析报告"""
    report = []
    report.append("=" * 60)
    report.append("技术分析投资框架报告")
    report.append("=" * 60)

    # 趋势
    report.append(f"\n【趋势分析】")
    report.append(f"  趋势方向: {analysis['trend']['direction']}")
    report.append(f"  趋势强度: {analysis['trend']['strength']:.1f}/100")
    report.append(f"  ADX指标: {analysis['trend']['adx']:.1f}")
    report.append(f"  当前价格: ¥{analysis['trend']['current_price']:.2f}")
    report.append(f"  均线状态: 短期{analysis['trend']['ma_short']:.2f}/中期{analysis['trend']['ma_medium']:.2f}/长期{analysis['trend']['ma_long']:.2f}")

    # 动量
    report.append(f"\n【动量分析】")
    report.append(f"  动量评分: {analysis['momentum']['value']:.1f}/100")
    report.append(f"  信号: {analysis['momentum']['signal']}")
    report.append(f"  RSI(14): {analysis['momentum']['rsi']:.1f}")
    report.append(f"  Stochastic %K: {analysis['momentum']['stoch_k']:.1f}, %D: {analysis['momentum']['stoch_d']:.1f}")

    # 波动率
    report.append(f"\n【波动率分析】")
    report.append(f"  ATR: {analysis['volatility']['atr']:.2f} ({analysis['volatility']['atr_percent']:.2f}%)")
    report.append(f"  状态: {analysis['volatility']['status']}")
    report.append(f"  布林带%: {analysis['volatility']['bb_percent']:.2f}")
    report.append(f"  布林带宽: {analysis['volatility']['bb_bandwidth']:.2f}")

    # 支撑阻力
    report.append(f"\n【支撑阻力】")
    report.append(f"  阻力位: ¥{analysis['support_resistance']['resistance']:.2f} (距离{analysis['support_resistance']['dist_to_resistance']:.1f}%)")
    report.append(f"  支撑位: ¥{analysis['support_resistance']['support']:.2f} (距离{analysis['support_resistance']['dist_to_support']:.1f}%)")

    # 综合信号
    report.append(f"\n【综合信号】")
    report.append(f"  综合评分: {analysis['signals']['composite']:.1f}/100")
    report.append(f"  建议: {analysis['signals']['recommendation']}")
    report.append(f"  趋势贡献: {analysis['signals']['trend_contribution']:.1f}")
    report.append(f"  动量贡献: {analysis['signals']['momentum_contribution']:.1f}")
    report.append(f"  波动率贡献: {analysis['signals']['volatility_contribution']:.1f}")

    # 市场状态
    if 'market_regime' in analysis:
        mr = analysis['market_regime']
        report.append(f"\n【市场状态】")
        report.append(f"  市场状态: {mr['regime']}")
        report.append(f"  趋势方向: {mr['trend']}")
        report.append(f"  置信度: {mr['confidence']:.1f}%")
        report.append(f"  市场评分: {mr['composite_score']:.1f}/100")
        if 'composite_adjusted' in analysis['signals']:
            report.append(f"  调整后评分: {analysis['signals']['composite_adjusted']:.1f}/100")
            report.append(f"  调整后建议: {analysis['signals']['recommendation_adjusted']}")
            report.append(f"  调整幅度: {analysis['signals']['market_regime_adjustment']:+.0f}")

    # 风险管理
    report.append(f"\n【风险管理】")
    report.append(f"  建议止损: ¥{analysis['risk']['stop_loss']:.2f}")
    report.append(f"  建议止盈: ¥{analysis['risk']['take_profit']:.2f}")
    report.append(f"  风险收益比: {analysis['risk']['risk_reward_ratio']:.2f}:1")
    report.append(f"  推荐仓位: {analysis['risk']['position_size_recommendation']*100:.0f}%")

    report.append("\n" + "=" * 60)
    return "\n".join(report)
