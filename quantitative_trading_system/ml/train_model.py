#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型训练脚本

用法:
    python ml/train_model.py                    # 默认配置训练
    python ml/train_model.py --epochs 100      # 指定epoch数
    python ml/train_model.py --stock-list 300750.SZ,600519.SH  # 指定股票池
"""

import argparse
import logging
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.data.dataset import StockDataset, compute_class_weights
from ml.data.preprocessor import prepare_training_data
from ml.models.lstm_predictor import create_model, save_model
from ml.training.trainer import train_with_class_weights
from ml.training.evaluator import evaluate_model

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_stock_data(symbols: list, days: int = 500) -> dict:
    """
    加载股票数据

    Args:
        symbols: 股票代码列表
        days: 加载天数

    Returns:
        {symbol: DataFrame} 字典
    """
    data = {}

    # 优先使用Wind数据源
    try:
        import wind
        logger.info("使用Wind数据源")

        for symbol in symbols:
            try:
                # 获取日线数据
                end_date = datetime.now().strftime('%Y%m%d')
                start_date = pd.Timestamp(end_date) - pd.Timedelta(days=days)
                start_date = start_date.strftime('%Y%m%d')

                # 尝试获取数据
                ws = wind.Component(sector='A')
                df = ws.history(
                    symbol, start_date, end_date,
                    'date,open,high,low,close,volume',
                    'skipnull=0'
                )
                if df is not None and not df.empty:
                    data[symbol] = df
                    logger.info(f"{symbol}: {len(df)} 条数据")
            except Exception as e:
                logger.warning(f"{symbol} Wind数据获取失败: {e}")

    except ImportError:
        pass

    # 备用: AKShare
    if not data:
        logger.info("使用AKShare数据源")
        try:
            import akshare as ak

            for symbol in symbols:
                try:
                    # 转换代码格式
                    if symbol.endswith('.SZ') or symbol.endswith('.SH'):
                        code = symbol.replace('.SZ', '').replace('.SH', '')
                        market = 'sz' if symbol.endswith('.SZ') else 'sh'
                        ak_symbol = f"{market}{code}"
                    else:
                        ak_symbol = symbol

                    df = ak.stock_zh_a_hist(symbol=code if not symbol.startswith(('0', '3', '6')) else symbol[-6:],
                                           period='daily', adjust='qfq')
                    if df is not None and not df.empty:
                        df = df.rename(columns={
                            '日期': 'date', '开盘': 'open', '收盘': 'close',
                            '最高': 'high', '最低': 'low', '成交量': 'volume'
                        })
                        df['date'] = pd.to_datetime(df['date'])
                        df = df.set_index('date').tail(days)
                        data[symbol] = df
                        logger.info(f"{symbol}: {len(df)} 条数据")
                except Exception as e:
                    logger.warning(f"{symbol} AKShare获取失败: {e}")

        except ImportError:
            logger.error("请安装 akshare: pip install akshare")

    return data


def create_mock_data(symbols: list, n_days: int = 500) -> dict:
    """
    创建模拟数据用于测试（当没有真实数据时）

    Args:
        symbols: 股票代码列表
        n_days: 天数

    Returns:
        模拟数据字典
    """
    import random

    data = {}
    for symbol in symbols:
        dates = pd.date_range(end=datetime.now(), periods=n_days, freq='B')

        # 生成随机游走价格
        np.random.seed(hash(symbol) % 2**32)
        base_price = 10 + np.random.rand() * 90
        returns = np.random.randn(n_days) * 0.02
        prices = base_price * np.exp(np.cumsum(returns))

        df = pd.DataFrame({
            'date': dates,
            'open': prices * (1 + np.random.randn(n_days) * 0.005),
            'high': prices * (1 + np.abs(np.random.randn(n_days)) * 0.01),
            'low': prices * (1 - np.abs(np.random.randn(n_days)) * 0.01),
            'close': prices,
            'volume': np.random.randint(1e6, 1e8, n_days)
        })
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')

        data[symbol] = df
        logger.info(f"{symbol} (模拟): {len(df)} 条数据")

    return data


def main():
    parser = argparse.ArgumentParser(description='训练LSTM价格预测模型')
    parser.add_argument('--symbols', type=str, default='000001.SZ,000002.SZ,000858.SZ,600000.SH,600519.SH,600036.SH,601318.SH,601398.SH,000333.SZ,300750.SZ',
                       help='逗号分隔的股票代码列表')
    parser.add_argument('--days', type=int, default=500, help='数据天数')
    parser.add_argument('--epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--lr', type=float, default=0.001, help='学习率')
    parser.add_argument('--batch-size', type=int, default=32, help='批次大小')
    parser.add_argument('--hidden-size', type=int, default=128, help='LSTM隐藏层维度')
    parser.add_argument('--sequence-length', type=int, default=60, help='序列长度')
    parser.add_argument('--patience', type=int, default=5, help='早停耐心值')
    parser.add_argument('--output-dir', type=str, default='ml/models', help='模型输出目录')
    parser.add_argument('--mock', action='store_true', help='使用模拟数据')
    args = parser.parse_args()

    logger.info("=" * 50)
    logger.info("LSTM模型训练")
    logger.info("=" * 50)

    # 解析股票列表
    symbols = [s.strip() for s in args.symbols.split(',')]
    logger.info(f"股票池: {symbols}")

    # 加载数据
    if args.mock:
        logger.info("使用模拟数据（--mock）")
        symbol_data = create_mock_data(symbols, args.days)
    else:
        logger.info(f"加载最近 {args.days} 天数据")
        symbol_data = load_stock_data(symbols, args.days)

    if not symbol_data:
        logger.error("没有可用数据，退出")
        return

    # 准备训练数据
    logger.info("准备训练数据...")
    try:
        X_train, y_train, X_val, y_val, X_test, y_test = prepare_training_data(
            symbol_data,
            sequence_length=args.sequence_length,
            up_threshold=0.005,
            down_threshold=-0.005,
            train_ratio=0.7  # 70%训练, 10%验证, 20%测试
        )
    except ValueError as e:
        logger.error(f"数据准备失败: {e}")
        return

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
    model = create_model(
        input_size=n_features,
        hidden_size=args.hidden_size,
        num_layers=2,
        dropout=0.3,
        num_classes=3
    )

    # 训练
    logger.info("开始训练...")
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
    logger.info("\n在测试集上评估...")
    os.makedirs(args.output_dir, exist_ok=True)
    metrics = evaluate_model(model, test_loader, verbose=True)

    # 保存最终模型（不是最优模型）
    final_model_path = os.path.join(args.output_dir, 'lstm_final.pt')
    save_model(model, final_model_path)

    # 保存训练历史
    history_path = os.path.join(args.output_dir, 'training_history.npz')
    np.savez(history_path, **history)
    logger.info(f"训练历史已保存到: {history_path}")

    # 保存指标
    import json
    metrics_path = os.path.join(args.output_dir, 'metrics.json')
    # 转换numpy类型为Python类型
    metrics_serializable = {
        k: v if not isinstance(v, (np.ndarray, np.floating)) else float(v)
        for k, v in metrics.items()
    }
    with open(metrics_path, 'w') as f:
        json.dump(metrics_serializable, f, indent=2)
    logger.info(f"指标已保存到: {metrics_path}")

    logger.info("\n训练完成！")
    logger.info(f"最优模型: {args.output_dir}/best_model.pt")
    logger.info(f"最终模型: {final_model_path}")


if __name__ == '__main__':
    main()
