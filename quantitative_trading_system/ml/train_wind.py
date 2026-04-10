#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
全A股增量滚动训练脚本

从Wind数据库获取全量A股数据，分批训练LSTM模型。
每批训练完成保存检查点，支持崩溃恢复。

用法:
    python ml/train_wind.py                        # 从头开始训练
    python ml/train_wind.py --resume               # 从上次中断处恢复
    python ml/train_wind.py --batch-num 5          # 从第5批开始
    python ml/train_wind.py --batches 6            # 只训练6批
"""

import argparse
import logging
import os
import sys
import gc
import json
from datetime import datetime

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import numpy as np
import torch
from torch.utils.data import DataLoader

from ml.data.batch_loader import WindBatchLoader
from ml.data.dataset import StockDataset, compute_class_weights
from ml.data.preprocessor import prepare_training_data
from ml.models.lstm_predictor import create_model, save_model
from ml.training.trainer import train_with_class_weights
from ml.training.evaluator import evaluate_model
from ml.training.checkpoint_manager import CheckpointManager

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description='全A股增量滚动训练')
    parser.add_argument('--epochs', type=int, default=30, help='每批训练轮数')
    parser.add_argument('--batch-size', type=int, default=32, help='批量大小')
    parser.add_argument('--stocks', type=int, default=6000, help='训练股票总数')
    parser.add_argument('--stocks-per-batch', type=int, default=500, help='每批股票数')
    parser.add_argument('--total-batches', type=int, default=12, help='总批次数')
    parser.add_argument('--days', type=int, default=1250, help='历史数据天数(约5年)')
    parser.add_argument('--lr', type=float, default=0.001, help='学习率')
    parser.add_argument('--up-threshold', type=float, default=0.01, help='上涨阈值')
    parser.add_argument('--down-threshold', type=float, default=-0.01, help='下跌阈值')
    parser.add_argument('--output-dir', type=str, default='ml/models', help='输出目录')
    parser.add_argument('--resume', action='store_true', help='从上次中断处恢复')
    parser.add_argument('--batch-num', type=int, default=None, help='从指定批次开始')
    parser.add_argument('--patience', type=int, default=5, help='早停patience')
    return parser.parse_args()


def log_resource_usage():
    """记录当前资源使用情况"""
    try:
        import psutil
        process = psutil.Process()
        mem_info = process.memory_info()
        logger.info(f"内存使用: {mem_info.rss / 1024 / 1024:.1f} MB")
    except ImportError:
        pass

    if torch.cuda.is_available():
        logger.info(f"GPU显存: {torch.cuda.memory_allocated() / 1024 / 1024:.1f} MB")


def train_single_batch(
    batch_index: int,
    symbols: list,
    args: argparse.Namespace,
    batch_loader: WindBatchLoader,
    checkpoint_manager: CheckpointManager,
):
    """训练单个批次"""
    logger.info("=" * 60)
    logger.info(f"开始训练批次 {batch_index}/{args.total_batches}")
    logger.info(f"股票数量: {len(symbols)}")
    log_resource_usage()

    # 加载该批数据
    def progress_callback(current, total):
        if current % 100 == 0 or current == total:
            logger.info(f"数据加载进度: {current}/{total}")

    result = batch_loader.load_batch(symbols, progress_callback)

    if result.loaded_count < 10:
        logger.error(f"批次 {batch_index} 加载数据太少，跳过")
        del result
        gc.collect()
        return False

    # 准备训练数据
    logger.info(f"准备训练数据 ({result.loaded_count} 只股票)...")
    X_train, y_train, X_val, y_val, X_test, y_test = prepare_training_data(
        result.symbol_data,
        sequence_length=60,
        up_threshold=args.up_threshold,
        down_threshold=args.down_threshold,
        train_ratio=0.7
    )

    # 释放原始数据内存
    del result.symbol_data
    gc.collect()

    if len(X_train) < 100:
        logger.error(f"批次 {batch_index} 训练样本太少，跳过")
        return False

    logger.info(f"样本数: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")
    log_resource_usage()

    # 创建数据集
    class_weights = compute_class_weights(y_train)
    train_dataset = StockDataset(X_train, y_train, class_weights)
    val_dataset = StockDataset(X_val, y_val)
    test_dataset = StockDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    # 创建模型
    n_features = X_train.shape[2]
    model = create_model(input_size=n_features, hidden_size=128, num_layers=2, dropout=0.3, num_classes=3)

    # 训练
    logger.info(f"开始训练 (epochs={args.epochs})...")
    history = train_with_class_weights(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        class_weights=class_weights,
        epochs=args.epochs,
        lr=args.lr,
        patience=args.patience,
        checkpoint_dir=args.output_dir
    )

    # 评估
    logger.info(f"批次 {batch_index} 测试集评估...")
    metrics = evaluate_model(model, test_loader, verbose=True)

    # 保存该批检查点
    best_val_loss = min(history.get('val_loss', [float('inf')]))
    checkpoint_path = checkpoint_manager.save_checkpoint(
        batch_index=batch_index,
        model=model,
        optimizer=None,
        epoch=args.epochs,
        best_val_loss=best_val_loss,
        patience_counter=0,
        metrics={k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                 for k, v in metrics.items()},
        n_samples=len(X_train) + len(X_val) + len(X_test)
    )

    # 释放内存
    del X_train, y_train, X_val, y_val, X_test, y_test
    del train_dataset, val_dataset, test_dataset
    del train_loader, val_loader, test_loader
    del model
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    log_resource_usage()
    logger.info(f"批次 {batch_index} 训练完成")

    return True


def main():
    args = parse_args()

    logger.info("=" * 60)
    logger.info("全A股增量滚动训练")
    logger.info("=" * 60)
    logger.info(f"参数配置:")
    logger.info(f"  总股票数: {args.stocks}")
    logger.info(f"  每批股票数: {args.stocks_per_batch}")
    logger.info(f"  总批次数: {args.total_batches}")
    logger.info(f"  历史天数: {args.days}")
    logger.info(f"  涨跌阈值: {args.up_threshold}/{args.down_threshold}")
    logger.info(f"  学习率: {args.lr}")
    logger.info(f"  输出目录: {args.output_dir}")

    os.makedirs(args.output_dir, exist_ok=True)

    # 初始化组件
    batch_loader = WindBatchLoader(days=args.days)
    checkpoint_manager = CheckpointManager(
        args.output_dir,
        model_name="lstm_wind",
        total_batches=args.total_batches,
        stocks_per_batch=args.stocks_per_batch
    )

    # 获取全量股票列表
    logger.info("获取全量A股列表...")
    all_stocks = batch_loader.get_all_stocks()
    all_stocks = all_stocks[:args.stocks]  # 限制数量
    logger.info(f"待训练股票总数: {len(all_stocks)}")

    # 确定要训练的批次
    if args.resume:
        start_batch = checkpoint_manager.get_next_batch_index()
        if start_batch == -1:
            logger.info("所有批次已完成，无需恢复")
            return
        logger.info(f"从检查点恢复，从批次 {start_batch} 开始")
    elif args.batch_num is not None:
        start_batch = args.batch_num
        logger.info(f"用户指定从批次 {start_batch} 开始")
    else:
        start_batch = 1
        logger.info("从头开始训练")

    # 创建批次
    batches = batch_loader.create_batches(
        all_stocks,
        batch_size=args.stocks_per_batch,
        shuffle=True
    )

    # 限制总批次数
    if args.total_batches < len(batches):
        batches = batches[:args.total_batches]

    total_actual_batches = len(batches)
    logger.info(f"实际训练批次数: {total_actual_batches}")

    # 循环训练
    for i, batch_symbols in enumerate(batches):
        batch_index = i + 1

        if batch_index < start_batch:
            logger.info(f"跳过批次 {batch_index} (已训练)")
            continue

        success = train_single_batch(
            batch_index=batch_index,
            symbols=batch_symbols,
            args=args,
            batch_loader=batch_loader,
            checkpoint_manager=checkpoint_manager
        )

        if not success:
            logger.warning(f"批次 {batch_index} 训练失败，继续下一批")

    # 训练完成，检查所有批次
    if checkpoint_manager.is_training_complete() or start_batch == 1:
        logger.info("=" * 60)
        logger.info("所有批次训练完成!")
        logger.info(f"最终模型: {args.output_dir}/lstm_wind_final.pt")
        logger.info("=" * 60)

        # 汇总训练历史
        state = checkpoint_manager.load_training_state()
        if state:
            summary = {
                'total_batches': total_actual_batches,
                'completed_batches': state.get('completed_batches', []),
                'training_complete': True,
                'timestamp': datetime.now().isoformat()
            }
            summary_path = os.path.join(args.output_dir, 'training_summary.json')
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2)
            logger.info(f"训练摘要已保存: {summary_path}")
    else:
        logger.warning("训练未完成，部分批次失败")


if __name__ == '__main__':
    main()