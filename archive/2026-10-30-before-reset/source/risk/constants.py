#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Risk and backtest constants.

All magic numbers are centralized here. Do not hardcode numeric values elsewhere.
"""

from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class RiskParams:
    """Risk management parameters."""
    STOP_LOSS: float = 0.03          # -3% 止损
    TAKE_PROFIT: float = 0.06        # +6% 止盈
    MAX_POSITION_SIZE: float = 0.20  # 单笔最大仓位 20%
    MAX_TOTAL_EXPOSURE: float = 0.80  # 总仓位不超过 80%
    SCORE_TIERS: List[Tuple[int, float]] = None  # (min_score, weight)

    def __post_init__(self):
        if self.SCORE_TIERS is None:
            self.SCORE_TIERS = [
                (80, 0.20),
                (70, 0.15),
                (60, 0.10),
            ]

    def score_to_weight(self, score: int) -> float:
        """Convert signal score to position weight."""
        for min_score, weight in self.SCORE_TIERS:
            if score >= min_score:
                return weight
        return 0.05  # minimum 5% for scores below 60


@dataclass
class BacktestParams:
    """Backtesting parameters."""
    TRADING_DAYS_PER_YEAR: int = 252
    RISK_FREE_RATE: float = 0.0
    COMMISSION: float = 0.0003      # 万3手续费
    SLIPPAGE: float = 0.0           # 单边滑点比例（0.001 = 0.1%）
    DEFAULT_INITIAL_CAPITAL: float = 1000000.0
    MAX_POSITIONS: int = 5
    TRAILING_STOP: float = 0.05     # 移动止损：从持仓期最高收盘价回撤 5% 触发
    MAX_HOLDING_DAYS: int = 20      # 时间止损：最长持仓交易日数（0 = 关闭）
    COOLDOWN_DAYS: int = 5          # 止损后同标的冷却交易日数，期间禁止回买（0 = 关闭）
    EXIT_MA_PERIOD: int = 20        # 均线破位退出的均线周期
    BENCHMARK_SYMBOL: str = "000300.SH"  # 基准指数（默认沪深300）
    PRELOAD_LOOKBACK_DAYS: int = 200     # K线预加载额外回溯的自然日数（指标暖机用）
    MARKET_INDEX_SYMBOL: str = "000985.SH"  # 市场状态（ADX）计算用的指数
    RPS_PERIOD: int = 20                 # 横截面RPS的回看交易日数（欧奈尔常用50/120，短线取20）

    # 自适应退出：趋势市场（ADX>30）放宽止损，避免趋势股被紧止损打断
    TRENDING_TRAILING_MULTIPLIER: float = 1.6   # 趋势时移动止损回撤阈值 ×1.6
    TRENDING_HOLDING_MULTIPLIER: float = 2.0    # 趋势时时间止损天数 ×2

    # A股交易现实约束
    STAMP_TAX: float = 0.0005          # 印花税（卖出单边，2023-08 起 0.05%）
    TRANSFER_FEE: float = 0.00001      # 过户费（双边，万0.1）
    ENFORCE_PRICE_LIMITS: bool = True  # 涨跌停不可成交约束
    LIMIT_UP_TOLERANCE: float = 0.002  # 涨跌停判定容差（收盘价距涨停价 0.2% 以内视为封板）


@dataclass
class IndicatorParams:
    """Indicator calculation parameters."""
    MIN_DATA_LENGTH: int = 60      # 最小数据长度
    LOOKBACK_DAYS: int = 120       # 历史数据回溯天数
    PATTERN_TOLERANCE: float = 0.02  # 价格聚类容差 (2%)

    # 连续强度评分的满强度尺度（用于把"满足程度"映射到 0-1）
    STRENGTH_MA_PREMIUM: float = 0.05       # 收盘价高于均线 5% 视为满强度
    STRENGTH_MA_CROSS_SPREAD: float = 0.02  # MA5/MA20 分离 2% 视为满强度
    STRENGTH_ADX_EXCESS: float = 20.0       # ADX 超出阈值 20 点视为满强度
    STRENGTH_RPS_EXCESS: float = 20.0       # 横截面RPS 超出阈值 20 个百分位视为满强度