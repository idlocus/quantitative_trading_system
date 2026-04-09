#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型评估模块

计算分类模型的各类评估指标。
"""

import logging
from typing import Dict, Any, Tuple
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_recall_fscore_support,
)

logger = logging.getLogger(__name__)


class Evaluator:
    """模型评估器"""

    def __init__(self, model: torch.nn.Module, device: str = 'auto'):
        """
        Args:
            model: 已训练的LSTM模型
            device: 评估设备
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device == 'auto' else torch.device(device)
        self.model = model.to(self.device)
        self.model.eval()

        self.class_names = ['down', 'flat', 'up']

    @torch.no_grad()
    def evaluate(self, test_loader: DataLoader) -> Dict[str, Any]:
        """
        在测试集上评估模型

        Args:
            test_loader: 测试数据加载器

        Returns:
            评估指标字典
        """
        all_preds = []
        all_labels = []
        all_probas = []

        for batch_X, batch_y in test_loader:
            batch_X = batch_X.to(self.device)
            batch_y = batch_y.to(self.device)

            outputs = self.model(batch_X)
            proba = torch.softmax(outputs, dim=-1)

            _, predicted = torch.max(outputs.data, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())
            all_probas.extend(proba.cpu().numpy())

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probas = np.array(all_probas)

        return self._compute_metrics(all_labels, all_preds, all_probas)

    @torch.no_grad()
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        推理预测

        Args:
            X: 输入特征数组, shape: (n_samples, seq_len, n_features)

        Returns:
            (预测类别, 预测概率)
        """
        X_tensor = torch.FloatTensor(X).to(self.device)

        self.model.eval()
        outputs = self.model(X_tensor)
        proba = torch.softmax(outputs, dim=-1).cpu().numpy()
        preds = np.argmax(proba, axis=-1)

        return preds, proba

    def _compute_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: np.ndarray
    ) -> Dict[str, Any]:
        """计算所有评估指标"""
        accuracy = accuracy_score(y_true, y_pred)

        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='weighted', zero_division=0
        )

        # 各类别指标
        per_class = precision_recall_fscore_support(
            y_true, y_pred, average=None, zero_division=0
        )

        # 混淆矩阵
        cm = confusion_matrix(y_true, y_pred)

        # 每类样本数
        support = np.bincount(y_true, minlength=3)

        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': cm.tolist(),
            'per_class': {
                name: {
                    'precision': per_class[0][i],
                    'recall': per_class[1][i],
                    'f1': per_class[2][i],
                    'support': int(support[i])
                }
                for i, name in enumerate(self.class_names)
            }
        }

        return metrics

    def print_report(self, metrics: Dict[str, Any]) -> None:
        """打印评估报告"""
        print("\n" + "=" * 50)
        print("模型评估报告")
        print("=" * 50)
        print(f"准确率 (Accuracy):  {metrics['accuracy']:.4f}")
        print(f"精确率 (Precision): {metrics['precision']:.4f}")
        print(f"召回率 (Recall):    {metrics['recall']:.4f}")
        print(f"F1分数 (F1):        {metrics['f1']:.4f}")

        print("\n各类别详情:")
        for name, stats in metrics['per_class'].items():
            print(f"  [{name}]")
            print(f"    Precision: {stats['precision']:.4f}")
            print(f"    Recall:    {stats['recall']:.4f}")
            print(f"    F1:        {stats['f1']:.4f}")
            print(f"    Support:   {stats['support']}")

        print("\n混淆矩阵:")
        cm = np.array(metrics['confusion_matrix'])
        header = "        " + " ".join(f"{n:>8}" for n in self.class_names)
        print(header)
        for i, row in enumerate(cm):
            print(f"  {self.class_names[i]:<8}" + " ".join(f"{v:>8}" for v in row))

        print("=" * 50)


def evaluate_model(
    model: torch.nn.Module,
    test_loader: DataLoader,
    device: str = 'auto',
    verbose: bool = True
) -> Dict[str, Any]:
    """
    快捷评估函数

    Args:
        model: LSTM模型
        test_loader: 测试数据加载器
        device: 设备
        verbose: 是否打印报告

    Returns:
        评估指标
    """
    evaluator = Evaluator(model, device)
    metrics = evaluator.evaluate(test_loader)

    if verbose:
        evaluator.print_report(metrics)

    return metrics
