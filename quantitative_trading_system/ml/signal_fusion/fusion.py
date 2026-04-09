#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
信号融合模块

将深度学习信号与现有技术框架信号、市场情绪信号进行融合。
"""

import logging
from typing import Dict, Optional, Any
from datetime import datetime
import numpy as np

from ..inference.predictor import DLPredictor

logger = logging.getLogger(__name__)


class SignalType:
    """信号类型常量"""
    STRONG_BUY = 'strong_buy'
    BUY = 'buy'
    HOLD = 'hold'
    SELL = 'sell'
    STRONG_SELL = 'strong_sell'


class DLSignalFusion:
    """
    深度学习信号融合器

    将DL模型预测、技术框架评分、新闻情感信号进行加权融合，
    输出最终交易信号。
    """

    def __init__(
        self,
        model_path: str,
        weights: Optional[Dict[str, float]] = None,
        device: str = 'auto',
        sequence_length: int = 60
    ):
        """
        Args:
            model_path: LSTM模型路径
            weights: 融合权重，{'dl': 0.3, 'technical': 0.45, 'news': 0.25}
            device: 推理设备
            sequence_length: 序列长度
        """
        self.dl_predictor = DLPredictor(model_path, device=device, sequence_length=sequence_length)

        # 默认权重
        self.weights = weights or {
            'dl': 0.30,
            'technical': 0.45,
            'news': 0.25
        }

        logger.info(f"信号融合器初始化完成，权重: {self.weights}")

    def generate_signal(
        self,
        symbol: str,
        price_data,  # pd.DataFrame
        technical_framework_output: Optional[Dict] = None,
        news_signal: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """
        生成融合信号

        Args:
            symbol: 股票代码
            price_data: 60日OHLCV数据
            technical_framework_output: 技术框架输出 (来自TechnicalInvestmentFramework)
                {
                    'signals': {'composite': 0~100, 'recommendation': 'buy'...},
                    'trend': {'strength': 0~100},
                    'momentum': {'value': ...},
                    'volatility': {'value': ...}
                }
            news_signal: 新闻信号 (来自NewsSignalGenerator)
                {
                    'sentiment_score': -1~1,
                    'confidence': 0~1,
                    'signal_type': SignalType enum
                }

        Returns:
            融合后的信号字典
        """
        # 1. DL模型预测
        dl_result = self.dl_predictor.predict(price_data)

        dl_score = dl_result['direction_score'] * 100  # 转成 -100~100
        dl_confidence = dl_result['confidence']
        dl_contribution = dl_score * self.weights['dl']

        # 2. 技术框架评分
        if technical_framework_output:
            tech_composite = technical_framework_output.get('signals', {}).get('composite', 50)
            # 技术框架0-100映射到-100~100
            tech_score = (tech_composite - 50) * 2
        else:
            tech_score = 0.0
            tech_composite = 50

        tech_contribution = tech_score * self.weights['technical']

        # 3. 新闻情感评分
        if news_signal:
            news_score = news_signal.get('sentiment_score', 0) * 100
            news_confidence = news_signal.get('confidence', 0)
        else:
            news_score = 0.0
            news_confidence = 0.0

        news_contribution = news_score * self.weights['news']

        # 4. 加权融合
        combined = dl_contribution + tech_contribution + news_contribution

        # 5. 生成最终信号
        final_signal = self._score_to_signal(combined)

        result = {
            'symbol': symbol,
            'timestamp': datetime.now().isoformat(),
            'prediction': final_signal,
            'combined_score': round(combined, 2),

            # DL部分
            'dl_prediction': dl_result['prediction'],
            'dl_confidence': round(dl_confidence, 4),
            'dl_score': round(dl_score, 2),
            'dl_contribution': round(dl_contribution, 2),
            'dl_probabilities': dl_result['probabilities'],

            # 技术框架部分
            'technical_score': round(tech_score, 2),
            'technical_contribution': round(tech_contribution, 2),
            'technical_composite': tech_composite,

            # 新闻部分
            'news_score': round(news_score, 2),
            'news_contribution': round(news_contribution, 2),
            'news_confidence': round(news_confidence, 4),

            # 各部分贡献明细
            'contributions': {
                'dl': round(dl_contribution, 2),
                'technical': round(tech_contribution, 2),
                'news': round(news_contribution, 2)
            },

            # 信号强度
            'signal_strength': self._get_signal_strength(combined),

            # 建议
            'recommendation': self._get_recommendation(final_signal, dl_confidence)
        }

        logger.info(
            f"{symbol} 融合信号: {final_signal} (综合得分={combined:.1f}), "
            f"DL={dl_result['prediction']}({dl_confidence:.2f}), "
            f"Tech={tech_composite:.0f}, News={news_score:.1f}"
        )

        return result

    def _score_to_signal(self, score: float) -> str:
        """将综合得分转换为信号类型"""
        if score >= 20:
            return SignalType.STRONG_BUY
        elif score >= 5:
            return SignalType.BUY
        elif score <= -20:
            return SignalType.STRONG_SELL
        elif score <= -5:
            return SignalType.SELL
        else:
            return SignalType.HOLD

    def _get_signal_strength(self, score: float) -> str:
        """获取信号强度描述"""
        abs_score = abs(score)
        if abs_score >= 40:
            return 'very_strong'
        elif abs_score >= 20:
            return 'strong'
        elif abs_score >= 10:
            return 'moderate'
        else:
            return 'weak'

    def _get_recommendation(self, signal: str, dl_confidence: float) -> Dict[str, Any]:
        """生成具体操作建议"""
        recommendations = {
            SignalType.STRONG_BUY: {
                'action': '买入',
                'position': '可建仓30-50%',
                'rationale': '多个信号源共振看多'
            },
            SignalType.BUY: {
                'action': '轻仓买入',
                'position': '可建仓10-20%',
                'rationale': '偏多信号'
            },
            SignalType.HOLD: {
                'action': '观望',
                'position': '暂不操作',
                'rationale': '信号不明确'
            },
            SignalType.SELL: {
                'action': '减仓',
                'position': '建议降至半仓以下',
                'rationale': '偏空信号'
            },
            SignalType.STRONG_SELL: {
                'action': '清仓/做空',
                'position': '建议清仓',
                'rationale': '多个信号源共振看空'
            }
        }

        rec = recommendations.get(signal, recommendations[SignalType.HOLD])

        # 如果DL置信度高，在建议中强调
        if dl_confidence > 0.85 and signal in [SignalType.STRONG_BUY, SignalType.STRONG_SELL]:
            rec['rationale'] = rec['rationale'] + '（DL模型高置信度确认）'

        return rec

    def get_fusion_params(self) -> Dict[str, Any]:
        """获取融合参数（供回测用）"""
        return {
            'weights': self.weights,
            'signal_thresholds': {
                'strong_buy': 20,
                'buy': 5,
                'sell': -5,
                'strong_sell': -20
            }
        }
