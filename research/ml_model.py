#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ML 日频信号模型 —— LightGBM 滚动训练/预测

核心思路：
1. 用日频因子库（50+个量价因子）作为特征
2. 横截面排名标准化（消除量纲差异）
3. 滚动训练：用过去 N 个月训练，预测未来 M 天收益
4. 输出：每日全市场预测收益排名 → Top N 买入
"""

from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd

from utils.logger import get_logger

logger = get_logger(__name__)


class DailyMLModel:
    """LightGBM 日频截面预测模型"""

    def __init__(self, train_months: int = 24, pred_horizon: int = 5,
                 top_n: int = 50, n_estimators: int = 300):
        """
        Args:
            train_months: 训练窗口（月）
            pred_horizon: 预测未来几天收益
            top_n: 选前N只
            n_estimators: LightGBM 树数量
        """
        self.train_months = train_months
        self.pred_horizon = pred_horizon
        self.top_n = top_n
        self.n_estimators = n_estimators
        self.model = None
        self.feature_names = []

    def prepare_features(self, factors: Dict[str, pd.DataFrame],
                        ref_date: pd.Timestamp) -> pd.DataFrame:
        """
        构建某日的横截面特征矩阵

        Returns:
            DataFrame (symbols × features)
        """
        rows = {}
        for name, mat in factors.items():
            if ref_date in mat.index:
                rows[name] = mat.loc[ref_date]
        if not rows:
            return pd.DataFrame()
        features = pd.DataFrame(rows)
        # 横截面排名标准化（消除量纲差异）
        features = features.rank(axis=0, pct=True)
        self.feature_names = list(features.columns)
        return features

    def prepare_target(self, close: pd.DataFrame, ref_date: pd.Timestamp) -> pd.Series:
        """构建目标变量：未来 N 日收益"""
        idx = close.index.get_loc(ref_date)
        if idx + self.pred_horizon >= len(close):
            return pd.Series(dtype=float)
        future = close.iloc[idx + self.pred_horizon]
        current = close.iloc[idx]
        return future / current - 1

    def train(self, factors: Dict[str, pd.DataFrame], close: pd.DataFrame,
              end_date: pd.Timestamp) -> bool:
        """
        滚动训练：用 end_date 之前 train_months 的数据训练

        Returns:
            是否训练成功
        """
        try:
            from sklearn.ensemble import HistGradientBoostingRegressor
        except ImportError:
            logger.warning("sklearn not available")
            return False

        start_date = end_date - pd.DateOffset(months=self.train_months)
        dates = [d for d in close.index if start_date <= d <= end_date][::5]  # 每5天采样

        X_all, y_all = [], []
        for d in dates:
            feat = self.prepare_features(factors, d)
            target = self.prepare_target(close, d)
            if feat.empty or target.empty:
                continue
            common = feat.index.intersection(target.index)
            if len(common) < 50:
                continue
            X_all.append(feat.loc[common].fillna(0.5))
            y_all.append(target.loc[common])

        if not X_all:
            logger.warning("no training data")
            return False

        X = pd.concat(X_all, axis=0)
        y = pd.concat(y_all, axis=0)

        # 清理数据：去除NaN/Inf，LightGBM 原生代码对脏数据敏感
        mask = np.isfinite(y.values) & ~np.isnan(X.values).any(axis=1)
        X, y = X[mask], y[mask]

        # 过滤极端值
        lo, hi = y.quantile(0.01), y.quantile(0.99)
        mask2 = y.between(lo, hi)
        X, y = X[mask2], y[mask2]

        # 确保数据类型
        X = X.astype(np.float32)
        y = y.astype(np.float64)

        if len(X) < 100:
            logger.warning(f"too few clean samples: {len(X)}")
            return False

        # 限制样本量（Windows LightGBM 大数据集可能崩溃）
        MAX_SAMPLES = 50000
        if len(X) > MAX_SAMPLES:
            idx = np.random.RandomState(42).choice(len(X), MAX_SAMPLES, replace=False)
            X, y = X.iloc[idx], y.iloc[idx]

        self.model = HistGradientBoostingRegressor(
            max_iter=self.n_estimators,
            max_depth=5,
            learning_rate=0.05,
            random_state=42,
        )
        self.model.fit(X, y)
        logger.info(f"[ML] trained on {len(X)} samples, {len(self.feature_names)} features")
        return True

    def predict(self, factors: Dict[str, pd.DataFrame],
                ref_date: pd.Timestamp) -> Optional[pd.Series]:
        """
        预测某日的未来收益排名

        Returns:
            Series (symbol → predicted_score)，高分=预测收益高
        """
        if self.model is None:
            return None
        feat = self.prepare_features(factors, ref_date)
        if feat.empty:
            return None
        X = feat.fillna(0.5)
        pred = self.model.predict(X)
        return pd.Series(pred, index=X.index).sort_values(ascending=False)

    def feature_importance(self) -> pd.Series:
        """特征重要性"""
        if self.model is None:
            return pd.Series()
        # HistGradientBoosting 没有直接 feature_importances_，用 permutation importance
        try:
            from sklearn.inspection import permutation_importance
            # 简单返回 None（滚动训练太慢不适合每次算 permutation）
            return pd.Series(dtype=float)
        except Exception:
            return pd.Series(dtype=float)
