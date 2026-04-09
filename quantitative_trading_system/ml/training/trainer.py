#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
训练循环模块

包含训练器类，支持早停、最优模型保存、学习率调度。
"""

import logging
import os
from typing import Optional, Dict, Any
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from ..models.lstm_predictor import LSTMPredictor, save_model

logger = logging.getLogger(__name__)


class Trainer:
    """LSTM模型训练器"""

    def __init__(
        self,
        model: LSTMPredictor,
        device: str = 'auto',
        checkpoint_dir: str = 'ml/models',
    ):
        """
        Args:
            model: LSTMPredictor模型实例
            device: 训练设备 ('auto', 'cpu', 'cuda')
            checkpoint_dir: 模型保存目录
        """
        self.device = self._get_device(device)
        self.model = model.to(self.device)
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)

        self.best_val_acc = 0.0
        self.best_val_loss = float('inf')
        self.patience_counter = 0

    def _get_device(self, device: str) -> torch.device:
        if device == 'auto':
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return torch.device(device)

    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        criterion = nn.CrossEntropyLoss()

        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(self.device)
            batch_y = batch_y.to(self.device)

            # 前向传播
            outputs = self.model(batch_X)
            loss = criterion(outputs, batch_y)

            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item() * len(batch_y)
            _, predicted = torch.max(outputs.data, 1)
            total += len(batch_y)
            correct += (predicted == batch_y).sum().item()

        avg_loss = total_loss / total
        accuracy = correct / total

        return {'loss': avg_loss, 'accuracy': accuracy}

    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """验证"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        criterion = nn.CrossEntropyLoss()

        for batch_X, batch_y in val_loader:
            batch_X = batch_X.to(self.device)
            batch_y = batch_y.to(self.device)

            outputs = self.model(batch_X)
            loss = criterion(outputs, batch_y)

            total_loss += loss.item() * len(batch_y)
            _, predicted = torch.max(outputs.data, 1)
            total += len(batch_y)
            correct += (predicted == batch_y).sum().item()

        avg_loss = total_loss / total
        accuracy = correct / total

        return {'loss': avg_loss, 'accuracy': accuracy}

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 50,
        lr: float = 0.001,
        patience: int = 5,
        min_delta: float = 0.001,
    ) -> Dict[str, Any]:
        """
        训练模型

        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            epochs: 最大训练轮数
            lr: 学习率
            patience: 早停耐心值
            min_delta: 验证损失最小改善阈值

        Returns:
            训练历史记录
        """
        self.optimizer = Adam(self.model.parameters(), lr=lr)
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=2, verbose=True
        )

        history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': []
        }

        self.patience_counter = 0
        self.best_val_acc = 0.0
        self.best_val_loss = float('inf')

        logger.info(f"开始训练，设备: {self.device}, 总epochs: {epochs}")

        for epoch in range(epochs):
            # 训练
            train_metrics = self.train_epoch(train_loader)
            val_metrics = self.validate(val_loader)

            # 记录历史
            history['train_loss'].append(train_metrics['loss'])
            history['train_acc'].append(train_metrics['accuracy'])
            history['val_loss'].append(val_metrics['loss'])
            history['val_acc'].append(val_metrics['accuracy'])

            # 学习率调整
            self.scheduler.step(val_metrics['loss'])

            # 日志输出
            logger.info(
                f"Epoch {epoch+1}/{epochs} | "
                f"Train: loss={train_metrics['loss']:.4f}, acc={train_metrics['accuracy']:.4f} | "
                f"Val: loss={val_metrics['loss']:.4f}, acc={val_metrics['accuracy']:.4f}"
            )

            # 早停检查 - 按验证准确率
            if val_metrics['accuracy'] > self.best_val_acc + min_delta:
                self.best_val_acc = val_metrics['accuracy']
                self.patience_counter = 0
                self._save_checkpoint('best_model.pt')
                logger.info(f"✓ 保存最优模型 (val_acc={val_metrics['accuracy']:.4f})")
            else:
                self.patience_counter += 1
                if self.patience_counter >= patience:
                    logger.info(f"早停触发于 epoch {epoch+1}")
                    break

        logger.info(f"训练完成，最佳验证准确率: {self.best_val_acc:.4f}")
        return history

    def _save_checkpoint(self, filename: str) -> None:
        """保存检查点"""
        path = os.path.join(self.checkpoint_dir, filename)
        save_model(self.model, path)


def train_with_class_weights(
    model: LSTMPredictor,
    train_loader: DataLoader,
    val_loader: DataLoader,
    class_weights: np.ndarray,
    device: str = 'auto',
    epochs: int = 50,
    lr: float = 0.001,
    patience: int = 5,
    checkpoint_dir: str = 'ml/models',
) -> Dict[str, Any]:
    """
    使用类别权重的快捷训练函数

    Args:
        model: LSTM模型
        train_loader: 训练数据
        val_loader: 验证数据
        class_weights: 类别权重
        device: 设备
        epochs: 最大轮数
        lr: 学习率
        patience: 早停耐心值
        checkpoint_dir: 模型保存目录

    Returns:
        训练历史
    """
    trainer = Trainer(model, device=device, checkpoint_dir=checkpoint_dir)

    # 使用加权损失
    trainer.optimizer = Adam(model.parameters(), lr=lr)
    trainer.scheduler = ReduceLROnPlateau(
        trainer.optimizer, mode='min', factor=0.5, patience=2
    )

    # 训练循环（带类别权重）
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }

    weights_tensor = torch.FloatTensor(class_weights).to(trainer.device)
    criterion = nn.CrossEntropyLoss(weight=weights_tensor)

    trainer.model.train()
    trainer.patience_counter = 0
    trainer.best_val_acc = 0.0

    for epoch in range(epochs):
        # 训练
        trainer.model.train()
        total_loss, correct, total = 0.0, 0, 0

        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(trainer.device)
            batch_y = batch_y.to(trainer.device)

            outputs = trainer.model(batch_X)
            loss = criterion(outputs, batch_y)

            trainer.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            trainer.optimizer.step()

            total_loss += loss.item() * len(batch_y)
            _, predicted = torch.max(outputs.data, 1)
            total += len(batch_y)
            correct += (predicted == batch_y).sum().item()

        train_loss = total_loss / total
        train_acc = correct / total

        # 验证
        trainer.model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(trainer.device)
                batch_y = batch_y.to(trainer.device)
                outputs = trainer.model(batch_X)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item() * len(batch_y)
                _, predicted = torch.max(outputs.data, 1)
                val_total += len(batch_y)
                val_correct += (predicted == batch_y).sum().item()

        val_loss = val_loss / val_total
        val_acc = val_correct / val_total

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        trainer.scheduler.step(val_loss)

        logger.info(
            f"Epoch {epoch+1}/{epochs} | "
            f"Train: loss={train_loss:.4f}, acc={train_acc:.4f} | "
            f"Val: loss={val_loss:.4f}, acc={val_acc:.4f}"
        )

        # 早停
        if val_acc > trainer.best_val_acc + 0.001:
            trainer.best_val_acc = val_acc
            trainer.patience_counter = 0
            trainer._save_checkpoint('best_model.pt')
        else:
            trainer.patience_counter += 1
            if trainer.patience_counter >= patience:
                logger.info(f"早停于 epoch {epoch+1}")
                break

    return history
