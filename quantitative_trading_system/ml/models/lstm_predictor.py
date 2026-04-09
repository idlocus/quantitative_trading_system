#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LSTM 预测模型

双层LSTM，用于预测次日价格走势（涨/跌/震荡三分类）。
"""

import logging
from typing import Optional, Tuple
import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class LSTMPredictor(nn.Module):
    """
    双层LSTM价格走势分类器

    输入: (batch_size, seq_len, n_features) 60日 × 特征维度
    输出: (batch_size, 3) 涨/跌/震荡三类概率
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        num_classes: int = 3
    ):
        """
        Args:
            input_size: 输入特征维度
            hidden_size: LSTM隐藏层维度
            num_layers: LSTM层数
            dropout: Dropout比例
            num_classes: 分类数（默认3: 跌/震荡/涨）
        """
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # 双层LSTM
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False  # 单向LSTM
        )

        # 层归一化
        self.layer_norm = nn.LayerNorm(hidden_size)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # 全连接分类器
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_classes)
        )

        logger.info(
            f"LSTM模型: input={input_size}, hidden={hidden_size}, "
            f"layers={num_layers}, dropout={dropout}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 输入张量, shape: (batch_size, seq_len, input_size)

        Returns:
            输出张量, shape: (batch_size, num_classes)
        """
        # LSTM输出
        # lstm_out: (batch_size, seq_len, hidden_size)
        # hidden: (num_layers, batch_size, hidden_size)
        lstm_out, (hidden, cell) = self.lstm(x)

        # 只取最后一个时间步的输出
        # hidden: (num_layers, batch_size, hidden_size) -> 取最后一层
        last_hidden = hidden[-1]  # (batch_size, hidden_size)

        # 层归一化
        normed = self.layer_norm(last_hidden)

        # Dropout
        dropped = self.dropout(normed)

        # 全连接
        output = self.fc(dropped)  # (batch_size, num_classes)

        return output

    def predict_proba(self, x: torch.Tensor) -> np.ndarray:
        """
        推理并返回概率

        Args:
            x: 输入张量

        Returns:
            概率数组, shape: (batch_size, num_classes)
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            proba = torch.softmax(logits, dim=-1)
        return proba.cpu().numpy()

    def predict(self, x: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        """
        推理并返回预测类别和置信度

        Args:
            x: 输入张量

        Returns:
            (预测类别, 置信度)
        """
        proba = self.predict_proba(x)
        preds = np.argmax(proba, axis=-1)
        confidence = np.max(proba, axis=-1)
        return preds, confidence

    @staticmethod
    def count_parameters(model: nn.Module) -> int:
        """统计模型参数量"""
        return sum(p.numel() for p in model.parameters() if p.requires_grad)


def create_model(
    input_size: int,
    hidden_size: int = 128,
    num_layers: int = 2,
    dropout: float = 0.3,
    num_classes: int = 3
) -> LSTMPredictor:
    """
    工厂函数：创建LSTM模型

    Args:
        input_size: 输入特征维度
        hidden_size: 隐藏层维度
        num_layers: LSTM层数
        dropout: Dropout比例
        num_classes: 分类数

    Returns:
        LSTMPredictor实例
    """
    model = LSTMPredictor(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        num_classes=num_classes
    )

    n_params = LSTMPredictor.count_parameters(model)
    logger.info(f"模型创建完成，参数量: {n_params:,}")

    return model


def save_model(model: LSTMPredictor, path: str) -> None:
    """保存模型到文件"""
    torch.save({
        'model_state_dict': model.state_dict(),
        'input_size': model.input_size,
        'hidden_size': model.hidden_size,
        'num_layers': model.num_layers,
    }, path)
    logger.info(f"模型已保存到: {path}")


def load_model(path: str, device: str = 'cpu') -> LSTMPredictor:
    """从文件加载模型"""
    checkpoint = torch.load(path, map_location=device)
    model = LSTMPredictor(
        input_size=checkpoint['input_size'],
        hidden_size=checkpoint['hidden_size'],
        num_layers=checkpoint['num_layers']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    logger.info(f"模型已从: {path} 加载")
    return model
