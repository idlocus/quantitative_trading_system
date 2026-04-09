#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据预处理模块

负责将原始行情数据、基本面因子、技术指标等转换为模型可用的特征向量。
"""

import logging
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """数据预处理器"""

    def __init__(self, scaler: Optional[StandardScaler] = None):
        """
        Args:
            scaler: 预训练的标准化器，如果是None则新创建
        """
        self.scaler = scaler or StandardScaler()
        self._fitted = False
        self._feature_names: List[str] = []

    def fit(self, df: pd.DataFrame) -> 'DataPreprocessor':
        """
        在训练集上拟合标准化器

        Args:
            df: 包含所有特征的DataFrame
        """
        self.scaler.fit(df.values)
        self._feature_names = list(df.columns)
        self._fitted = True
        logger.info(f"预处理器已拟合，特征数: {len(self._feature_names)}")
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        转换数据

        Args:
            df: 待转换的特征DataFrame

        Returns:
            标准化后的numpy数组
        """
        if not self._fitted:
            raise ValueError("预处理器尚未拟合，请先调用fit()")
        return self.scaler.transform(df.values)

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        """拟合并转换"""
        self.fit(df)
        return self.transform(df)

    def inverse_transform(self, arr: np.ndarray) -> np.ndarray:
        """逆标准化"""
        return self.scaler.inverse_transform(arr)

    @property
    def feature_names(self) -> List[str]:
        return self._feature_names

    @property
    def n_features(self) -> int:
        return len(self._feature_names)


def build_feature_df(
    price_data: pd.DataFrame,
    technical_indicators: Optional[Dict] = None,
    fundamental_factors: Optional[Dict] = None,
    market_breadth: Optional[Dict] = None,
    money_flow: Optional[Dict] = None,
    news_sentiment: Optional[Dict] = None,
) -> pd.DataFrame:
    """
    从多源数据构建特征DataFrame

    Args:
        price_data: OHLCV数据，index为日期，columns为[open, high, low, close, volume]
        technical_indicators: 技术指标字典
        fundamental_factors: 基本面因子字典
        market_breadth: 市场宽度字典
        money_flow: 资金流字典
        news_sentiment: 新闻情感字典

    Returns:
        特征DataFrame，index为日期，columns为特征名
    """
    features = {}

    # 1. 价格/成交量特征 (最后一行即最新值)
    if price_data is not None and not price_data.empty:
        close = price_data['close']
        high = price_data['high']
        low = price_data['low']
        volume = price_data['volume']
        open_price = price_data['open']

        # 基础价格比
        features['close_open_ratio'] = close / open_price - 1
        features['high_low_ratio'] = (high - low) / close
        features['volume_ratio'] = volume / volume.rolling(20).mean()

        # 均线偏离度
        for window in [5, 10, 20, 60]:
            if len(close) >= window:
                ma = close.rolling(window).mean()
                features[f'price_ma{window}_ratio'] = close / ma - 1

        # 波动率
        features['volatility_20'] = close.pct_change().rolling(20).std()
        features['volatility_60'] = close.pct_change().rolling(60).std()

        # 价格动量
        for period in [5, 10, 20]:
            features[f'return_{period}d'] = close.pct_change(period)
            features[f'high_{period}d'] = close.rolling(period).max()
            features[f'low_{period}d'] = close.rolling(period).min()

        # 相对位置
        features['price_position'] = (close - low.rolling(60).min()) / (
            high.rolling(60).max() - low.rolling(60).min() + 1e-8
        )

    # 2. 技术指标
    if technical_indicators:
        features.update({f'tech_{k}': v for k, v in technical_indicators.items() if v is not None})

    # 3. 基本面因子
    if fundamental_factors:
        features.update({f'fund_{k}': v for k, v in fundamental_factors.items() if v is not None})

    # 4. 市场宽度
    if market_breadth:
        features.update({f'breadth_{k}': v for k, v in market_breadth.items() if v is not None})

    # 5. 资金流
    if money_flow:
        features.update({f'flow_{k}': v for k, v in money_flow.items() if v is not None})

    # 6. 新闻情感
    if news_sentiment:
        features['news_sentiment'] = news_sentiment.get('sentiment_score', 0)
        features['news_confidence'] = news_sentiment.get('confidence', 0)
        features['news_count'] = news_sentiment.get('news_count', 0)

    df = pd.DataFrame(features)

    # 处理缺失值和无穷值
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(0)

    return df


def create_sequence_features(
    feature_df: pd.DataFrame,
    sequence_length: int = 60
) -> np.ndarray:
    """
    将特征DataFrame转换为序列张量

    Args:
        feature_df: 特征DataFrame
        sequence_length: 序列长度

    Returns:
        numpy数组，shape: (n_samples, sequence_length, n_features)
    """
    sequences = []
    n_samples = len(feature_df) - sequence_length

    if n_samples <= 0:
        raise ValueError(
            f"数据长度 {len(feature_df)} 小于序列长度 {sequence_length}"
        )

    values = feature_df.values

    for i in range(n_samples):
        seq = values[i:i + sequence_length]
        sequences.append(seq)

    return np.array(sequences)


def create_labels(
    price_data: pd.DataFrame,
    up_threshold: float = 0.005,
    down_threshold: float = -0.005
) -> np.ndarray:
    """
    从价格数据创建标签

    Args:
        price_data: OHLCV数据
        up_threshold: 上涨阈值（默认0.5%）
        down_threshold: 下跌阈值（默认-0.5%）

    Returns:
        标签数组: 0=跌, 1=震荡, 2=涨
    """
    close = price_data['close']
    returns = close.pct_change().shift(-1)  # 次日收益率

    labels = np.zeros(len(returns), dtype=np.int64)

    # 涨: 收益 > up_threshold
    labels[returns > up_threshold] = 2

    # 跌: 收益 < down_threshold
    labels[returns < down_threshold] = 0

    # 震荡: 其他
    # 0保持为震荡

    # 最后一行的标签设为1（震荡），因为没有次日数据
    labels[-1] = 1

    return labels


def prepare_training_data(
    symbol_data: Dict[str, pd.DataFrame],
    sequence_length: int = 60,
    up_threshold: float = 0.005,
    down_threshold: float = -0.005,
    train_ratio: float = 0.8,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    准备多股票训练数据

    Args:
        symbol_data: {symbol: price_data} 字典
        sequence_length: 序列窗口长度
        up_threshold: 上涨阈值
        down_threshold: 下跌阈值
        train_ratio: 训练集比例

    Returns:
        (X_train, y_train, X_val, y_val, X_test, y_test)
    """
    all_features = []
    all_labels = []
    all_indices = []  # 记录每条数据属于哪个股票

    preprocessor = DataPreprocessor()

    # 对每只股票处理
    for symbol, price_data in symbol_data.items():
        try:
            # 构建特征
            feature_df = build_feature_df(price_data)

            if len(feature_df) < sequence_length + 20:
                logger.warning(f"{symbol} 数据长度不足，跳过")
                continue

            # 拟合并标准化
            feature_df_scaled = preprocessor.fit_transform(feature_df)

            # 创建序列
            sequences = create_sequence_features(
                pd.DataFrame(feature_df_scaled, columns=feature_df.columns),
                sequence_length
            )

            # 创建标签
            labels = create_labels(price_data, up_threshold, down_threshold)
            # 取与序列对应的标签部分
            labels_aligned = labels[sequence_length:]

            all_features.append(sequences)
            all_labels.append(labels_aligned)
            all_indices.extend([symbol] * len(sequences))

            logger.info(f"{symbol}: {len(sequences)} 样本")

        except Exception as e:
            logger.error(f"处理 {symbol} 时出错: {e}")
            continue

    # 合并所有股票数据
    X = np.vstack(all_features)
    y = np.concatenate(all_labels)

    logger.info(f"总样本数: {len(X)}, 特征维度: {X.shape[2] if len(X.shape) > 2 else 0}")
    logger.info(f"标签分布: 跌={np.sum(y==0)}, 震荡={np.sum(y==1)}, 涨={np.sum(y==2)}")

    # 按时间顺序划分（不用shuffle，保持时序）
    n = len(X)
    n_train = int(n * train_ratio)
    n_val = int(n * (train_ratio + 0.1))  # 10%验证

    X_train, y_train = X[:n_train], y[:n_train]
    X_val, y_val = X[n_train:n_val], y[n_train:n_val]
    X_test, y_test = X[n_val:], y[n_val:]

    logger.info(f"训练集: {len(X_train)}, 验证集: {len(X_val)}, 测试集: {len(X_test)}")

    return X_train, y_train, X_val, y_val, X_test, y_test
