#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据集定义模块

用于PyTorch DataLoader的Dataset封装。
"""

import logging
from typing import Optional, List
import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class StockDataset(Dataset):
    """股票时序数据集"""

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        class_weights: Optional[np.ndarray] = None
    ):
        """
        Args:
            X: 特征序列数组，shape: (n_samples, seq_len, n_features)
            y: 标签数组，shape: (n_samples,)
            class_weights: 类别权重，用于处理不平衡
        """
        assert X.shape[0] == len(y), "X和y的样本数不匹配"

        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
        self.class_weights = torch.FloatTensor(class_weights) if class_weights is not None else None

        logger.info(
            f"Dataset: {len(self)} 样本, "
            f"特征shape: {self.X.shape}, "
            f"标签分布: {np.bincount(y)}"
        )

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]

    def get_class_weights(self) -> Optional[torch.Tensor]:
        """返回类别权重"""
        return self.class_weights


class InferenceDataset(Dataset):
    """推理数据集（无标签）"""

    def __init__(self, X: np.ndarray):
        """
        Args:
            X: 特征序列数组，shape: (n_samples, seq_len, n_features)
        """
        self.X = torch.FloatTensor(X)
        logger.info(f"InferenceDataset: {len(self)} 样本")

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx]


def compute_class_weights(y: np.ndarray) -> np.ndarray:
    """
    计算类别权重（基于样本频率的逆）

    Args:
        y: 标签数组

    Returns:
        类别权重数组，长度为类别数
    """
    counts = np.bincount(y)
    total = len(y)
    n_classes = len(counts)

    # 权重 = 总样本数 / (类别数 × 该类样本数)
    weights = total / (n_classes * counts + 1e-8)

    # 归一化
    weights = weights / weights.sum() * n_classes

    logger.info(f"类别权重: {weights}")

    return weights.astype(np.float32)
