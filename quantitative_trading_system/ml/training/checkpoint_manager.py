#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检查点管理器 - 管理增量滚动训练的检查点保存与恢复
"""

import os
import json
import torch
import logging
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class TrainingState:
    """训练状态快照"""
    batch_index: int           # 当前批次索引
    epoch: int                 # 当前epoch
    best_val_loss: float       # 最佳验证损失
    patience_counter: int       # 早停计数器
    total_batches: int         # 总批次数
    stocks_per_batch: int       # 每批股票数
    completed_batches: list     # 已完成的批次列表
    timestamp: str              # 创建时间


class CheckpointManager:
    """检查点管理器"""

    def __init__(
        self,
        checkpoint_dir: str,
        model_name: str = "lstm_wind",
        total_batches: int = 12,
        stocks_per_batch: int = 500,
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name
        self.total_batches = total_batches
        self.stocks_per_batch = stocks_per_batch
        self.state_file = self.checkpoint_dir / "training_state.json"

    def get_batch_checkpoint_path(self, batch_index: int) -> Path:
        """获取指定批次的检查点路径"""
        return self.checkpoint_dir / f"{self.model_name}_batch_{batch_index}.pt"

    def get_final_model_path(self) -> Path:
        """获取最终模型路径"""
        return self.checkpoint_dir / f"{self.model_name}_final.pt"

    def save_checkpoint(
        self,
        batch_index: int,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        best_val_loss: float,
        patience_counter: int,
        scheduler: Optional[Any] = None,
        **extra_data
    ) -> Path:
        """保存检查点"""
        checkpoint_path = self.get_batch_checkpoint_path(batch_index)

        checkpoint = {
            'model_state_dict': model.state_dict(),
            'epoch': epoch,
            'best_val_loss': best_val_loss,
            'patience_counter': patience_counter,
            'batch_index': batch_index,
            'timestamp': datetime.now().isoformat(),
        }

        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()

        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()

        checkpoint.update(extra_data)

        torch.save(checkpoint, checkpoint_path)
        logger.info(f"检查点已保存: {checkpoint_path}")

        self._save_training_state(batch_index, epoch, best_val_loss, patience_counter)

        return checkpoint_path

    def load_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
        """加载检查点"""
        try:
            if not os.path.exists(checkpoint_path):
                raise FileNotFoundError(f"检查点文件不存在: {checkpoint_path}")

            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            logger.info(f"检查点已加载: {checkpoint_path}")
            return checkpoint
        except Exception as e:
            logger.error(f"加载检查点失败: {e}")
            raise

    def load_latest_checkpoint(self) -> Optional[Dict[str, Any]]:
        """加载最新的批次检查点"""
        state = self.load_training_state()
        if state is None:
            return None

        completed = state.get('completed_batches', [])
        if not completed:
            return None

        last_batch = max(completed)
        checkpoint_path = self.get_batch_checkpoint_path(last_batch)
        if not checkpoint_path.exists():
            return None

        return self.load_checkpoint(str(checkpoint_path))

    def _save_training_state(
        self,
        batch_index: int,
        epoch: int,
        best_val_loss: float,
        patience_counter: int
    ):
        """保存训练状态"""
        existing_state = self.load_training_state() or {
            'completed_batches': [],
            'total_batches': self.total_batches,
            'stocks_per_batch': self.stocks_per_batch,
        }

        if batch_index not in existing_state['completed_batches']:
            existing_state['completed_batches'].append(batch_index)

        existing_state['batch_index'] = batch_index
        existing_state['epoch'] = epoch
        existing_state['best_val_loss'] = best_val_loss
        existing_state['patience_counter'] = patience_counter
        existing_state['timestamp'] = datetime.now().isoformat()

        try:
            with open(self.state_file, 'w') as f:
                json.dump(existing_state, f, indent=2)
        except Exception as e:
            logger.error(f"保存训练状态失败: {e}")
            raise

    def load_training_state(self) -> Optional[Dict[str, Any]]:
        """加载训练状态"""
        if not self.state_file.exists():
            return None

        try:
            with open(self.state_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"加载训练状态失败: {e}")
            return None

    def get_next_batch_index(self) -> int:
        """获取下一个要训练的批次索引"""
        state = self.load_training_state()
        if state is None:
            return 1

        completed = state.get('completed_batches', [])
        total_batches = state.get('total_batches', self.total_batches)

        for i in range(1, total_batches + 1):
            if i not in completed:
                return i

        return -1  # 所有批次已完成

    def is_training_complete(self) -> bool:
        """检查是否所有批次都已完成"""
        state = self.load_training_state()
        if state is None:
            return False

        completed = set(state.get('completed_batches', []))
        total = state.get('total_batches', self.total_batches)
        expected = set(range(1, total + 1))

        return completed >= expected
