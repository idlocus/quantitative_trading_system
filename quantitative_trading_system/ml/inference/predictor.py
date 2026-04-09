#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
推理接口模块

提供模型推理的简洁接口。
"""

import logging
from typing import Dict, Optional, Tuple, Any
from datetime import datetime
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from ..models.lstm_predictor import load_model, LSTMPredictor
from ..data.dataset import InferenceDataset
from ..data.preprocessor import DataPreprocessor, build_feature_df

logger = logging.getLogger(__name__)


class DLPredictor:
    """
    深度学习推理器

    提供简洁的推理接口，用于单只股票的次日走势预测。
    """

    def __init__(
        self,
        model_path: str,
        device: str = 'auto',
        sequence_length: int = 60
    ):
        """
        Args:
            model_path: 模型文件路径 (.pt)
            device: 推理设备
            sequence_length: 序列窗口长度
        """
        self.device = device if device != 'auto' else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.sequence_length = sequence_length

        self.model = load_model(model_path, self.device)
        self.model.eval()
        logger.info(f"模型加载完成，设备: {self.device}")

    def predict(
        self,
        price_data: pd.DataFrame,
        technical_indicators: Optional[Dict] = None,
        fundamental_factors: Optional[Dict] = None,
        market_breadth: Optional[Dict] = None,
        money_flow: Optional[Dict] = None,
        news_sentiment: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """
        预测单只股票次日走势

        Args:
            price_data: 60日OHLCV数据
            technical_indicators: 技术指标字典
            fundamental_factors: 基本面因子字典
            market_breadth: 市场宽度字典
            money_flow: 资金流字典
            news_sentiment: 新闻情感字典

        Returns:
            预测结果字典
        """
        # 构建特征
        feature_df = build_feature_df(
            price_data=price_data,
            technical_indicators=technical_indicators,
            fundamental_factors=fundamental_factors,
            market_breadth=market_breadth,
            money_flow=money_flow,
            news_sentiment=news_sentiment,
        )

        # 取最后sequence_length行作为输入
        if len(feature_df) < self.sequence_length:
            logger.warning(f"数据长度不足，需要{self.sequence_length}，实际{len(feature_df)}")
            # padding
            pad_len = self.sequence_length - len(feature_df)
            feature_df = pd.concat([
                pd.DataFrame(np.zeros((pad_len, feature_df.shape[1])), columns=feature_df.columns),
                feature_df
            ], ignore_index=True)

        X = feature_df.values[-self.sequence_length:]
        X = X.reshape(1, self.sequence_length, -1)  # (1, seq_len, n_features)

        # 推理
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            outputs = self.model(X_tensor)
            proba = torch.softmax(outputs, dim=-1).cpu().numpy()[0]

        # 解析结果
        labels = ['down', 'flat', 'up']
        pred_label = labels[np.argmax(proba)]
        confidence = float(np.max(proba))

        # 计算方向强度: (-1, 1) 范围
        direction_score = proba[2] - proba[0]  # up概率 - down概率

        result = {
            'prediction': pred_label,
            'confidence': confidence,
            'probabilities': {
                'down': float(proba[0]),
                'flat': float(proba[1]),
                'up': float(proba[2])
            },
            'direction_score': float(direction_score),  # -1到1之间
            'timestamp': datetime.now().isoformat(),
            'model_info': {
                'input_features': X.shape[2],
                'sequence_length': self.sequence_length
            }
        }

        logger.info(
            f"预测结果: {pred_label} (置信度: {confidence:.2f}), "
            f"方向得分: {direction_score:.2f}"
        )

        return result

    def predict_batch(
        self,
        symbol_data_list: list,
        preprocessor: Optional[DataPreprocessor] = None
    ) -> list:
        """
        批量推理

        Args:
            symbol_data_list: [(symbol, price_data), ...]
            preprocessor: 数据预处理器（需与训练时一致）

        Returns:
            预测结果列表
        """
        if preprocessor is None:
            raise ValueError("批量推理需要提供预处理器")

        results = []

        for symbol, price_data in symbol_data_list:
            try:
                feature_df = build_feature_df(price_data)
                feature_values = preprocessor.transform(feature_df)

                X = feature_values[-self.sequence_length:]
                X = X.reshape(1, self.sequence_length, -1)

                with torch.no_grad():
                    X_tensor = torch.FloatTensor(X).to(self.device)
                    outputs = self.model(X_tensor)
                    proba = torch.softmax(outputs, dim=-1).cpu().numpy()[0]

                labels = ['down', 'flat', 'up']
                results.append({
                    'symbol': symbol,
                    'prediction': labels[np.argmax(proba)],
                    'confidence': float(np.max(proba)),
                    'probabilities': {
                        'down': float(proba[0]),
                        'flat': float(proba[1]),
                        'up': float(proba[2])
                    }
                })

            except Exception as e:
                logger.error(f"预测 {symbol} 时出错: {e}")
                results.append({
                    'symbol': symbol,
                    'prediction': 'error',
                    'error': str(e)
                })

        return results


def quick_predict(
    model_path: str,
    price_data: pd.DataFrame,
    device: str = 'auto'
) -> Dict[str, Any]:
    """
    快捷预测函数

    Args:
        model_path: 模型路径
        price_data: 60日OHLCV数据
        device: 设备

    Returns:
        预测结果
    """
    predictor = DLPredictor(model_path, device=device)
    return predictor.predict(price_data)
