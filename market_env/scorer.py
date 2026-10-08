#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
市场环境综合评分器（四维评分 + 季节性因子）

四个维度：
1. 市场宽度（A/D breadth）
2. 估值分位（PE/PB historical percentile）
3. 宏观增长（PMI、GDP、CPI 等）
4. 先行指数（社融+M2+PMI 合成）

季节性因子：总统周期 + 月度规律 + Sell in May 效应

四维等权综合评分: market_score = Σ 各维度分 × 25%
仓位系数: base_factor × seasonal_factor
"""

from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING
import pandas as pd

from utils.logger import get_logger
from .base import IMarketEnvProvider

if TYPE_CHECKING:
    from .macro_db import MacroDB

logger = get_logger(__name__)


@dataclass
class MarketEnvScore:
    """市场环境评分结果（四维 + 季节性）"""
    market_score: float = 50          # 0-100，综合市场评分
    position_factor: float = 1.0      # 0.0-1.0，建议仓位系数

    # 信号
    breadth_signal: str = 'neutral'  # 'expanding' | 'contracting' | 'neutral'
    valuation_signal: str = 'fair'    # 'cheap' | 'fair' | 'expensive'
    macro_signal: str = 'neutral'     # 'expanding' | 'contracting' | 'neutral'
    leading_signal: str = 'neutral'   # 'expanding' | 'contracting' | 'neutral'
    seasonal_signal: str = 'neutral'  # 'favorable' | 'neutral' | 'unfavorable'

    # 各维度分 (0-100)
    breadth_score: float = 50
    valuation_score: float = 50
    macro_score: float = 50
    leading_score: float = 50
    seasonal_factor: float = 1.0       # 0.8-1.2，季节性调整因子

    detail: dict = field(default_factory=dict)


class MarketEnvScorer:
    """
    宏观市场环境评分器（四维 + 季节性）

    数据源通过 IMarketEnvProvider 注入；
    宏观数据库通过 MacroDB 注入（可选；未注入时若 provider 支持
    fetch_indicator 则自动装配，宏观/领先两维才能生效）。
    季节性因子通过 seasonal.py 计算。
    """

    def __init__(self, provider: IMarketEnvProvider,
                 macro_db: "MacroDB" = None):
        self.provider = provider
        if macro_db is None and hasattr(provider, 'fetch_indicator'):
            # provider 支持宏观指标序列查询时自动装配（保证四维评分完整）
            from .macro_db import MacroDB
            macro_db = MacroDB(provider)
        self.macro_db = macro_db
        self._score_cache: dict = {}  # date -> MarketEnvScore（回测逐日复用）

    def score(self, date: str, breadth_df: Optional[pd.DataFrame] = None) -> MarketEnvScore:
        """
        评估给定日期的市场环境（时点正确：所有数据均过滤到 ≤ date）

        Args:
            date: 日期，格式 yyyyMMdd
            breadth_df: 可选的外部宽度数据（columns: advance/decline），
                        由引擎从缓存K线计算，优先于 provider

        Returns:
            MarketEnvScore
        """
        cache_key = date
        if cache_key in self._score_cache:
            return self._score_cache[cache_key]

        # 1. 市场宽度（权重 25%）—— 外部注入优先
        if breadth_df is not None:
            breadth_signal, breadth_score = self._eval_breadth(breadth_df)
        else:
            breadth_signal, breadth_score = self._eval_breadth(
                self.provider.get_market_breadth(date)
            )

        # 2. 估值分位（权重 25%）
        val_data = self.provider.get_market_valuation('000985.SH', date)
        valuation_signal, val_score = self._eval_valuation(val_data)

        # 3. 宏观增长维（权重 25%）—— 传入 date 保证时点正确
        if self.macro_db is not None:
            macro_score, macro_signal = self._eval_macro(date)
            syzm_score, leading_signal = self._eval_syzm(date)
        else:
            macro_score, macro_signal = 50, 'neutral'
            syzm_score, leading_signal = 50, 'neutral'

        # 4. 季节性因子（调整仓位系数）
        seasonal_factor, seasonal_signal = self._eval_seasonal(date)

        # 5. 综合评分（四维等权）
        market_score = int(
            breadth_score * 0.25 +
            val_score * 0.25 +
            macro_score * 0.25 +
            syzm_score * 0.25
        )
        market_score = max(0, min(100, market_score))

        # 6. 仓位系数（base_factor × 季节性因子）
        base_factor = self._score_to_position(market_score)
        position_factor = round(base_factor * seasonal_factor, 2)
        position_factor = max(0.2, min(1.0, position_factor))

        detail = {
            'breadth_signal': breadth_signal,
            'breadth_score': breadth_score,
            'valuation_signal': valuation_signal,
            'valuation_score': val_score,
            'macro_signal': macro_signal,
            'macro_score': macro_score,
            'leading_signal': leading_signal,
            'leading_score': syzm_score,
            'seasonal_signal': seasonal_signal,
            'seasonal_factor': seasonal_factor,
            'date': date,
        }

        result = MarketEnvScore(
            market_score=market_score,
            position_factor=position_factor,
            breadth_signal=breadth_signal,
            valuation_signal=valuation_signal,
            macro_signal=macro_signal,
            leading_signal=leading_signal,
            seasonal_signal=seasonal_signal,
            breadth_score=breadth_score,
            valuation_score=val_score,
            macro_score=macro_score,
            leading_score=syzm_score,
            seasonal_factor=seasonal_factor,
            detail=detail,
        )
        self._score_cache[cache_key] = result
        return result

    def _eval_breadth(self, df: Optional[pd.DataFrame]):
        """
        评估市场宽度。

        Returns:
            (signal: str, score: float 0-100)
        """
        if df is None or len(df) < 5:
            return 'neutral', 50

        ad_ratio = (df['advance'] / df['decline'].replace(0, 1)).iloc[-5:]
        avg_ratio = ad_ratio.mean()

        if avg_ratio > 1.2:
            return 'expanding', 75
        elif avg_ratio < 0.8:
            return 'contracting', 30
        else:
            return 'neutral', 50

    def _eval_valuation(self, val_data: Optional[dict]):
        """
        评估估值分位。

        Returns:
            (signal: str, score: float 0-100)
        """
        if val_data is None:
            return 'fair', 50

        pe_pct = val_data.get('pe_percentile', 50)
        pb_pct = val_data.get('pb_percentile', 50)
        avg_pct = (pe_pct + pb_pct) / 2

        if avg_pct < 30:
            return 'cheap', 80
        elif avg_pct > 70:
            return 'expensive', 25
        else:
            return 'fair', 55

    def _eval_macro(self, date: str) -> tuple[float, str]:
        """
        评估宏观增长维度（PMI + GDP + CPI），时点过滤到 ≤ date

        Returns:
            (score: float 0-100, signal: str)
        """
        if self.macro_db is None:
            return 50, 'neutral'

        try:
            return self.macro_db.get_macro_score(end=date)
        except Exception as e:
            logger.warning(f"_eval_macro failed: {e}")
            return 50, 'neutral'

    def _eval_syzm(self, date: str) -> tuple[float, str]:
        """
        评估先行合成指数，时点过滤到 ≤ date

        Returns:
            (score: float 0-100, signal: str)
        """
        if self.macro_db is None:
            return 50, 'neutral'

        try:
            return self.macro_db.get_syzm(end=date)
        except Exception as e:
            logger.warning(f"_eval_syzm failed: {e}")
            return 50, 'neutral'

    def _eval_seasonal(self, date: str) -> tuple[float, str]:
        """
        评估季节性因子。

        Returns:
            (seasonal_factor: float 0.8-1.2, signal: str)
        """
        try:
            from .seasonal import get_seasonal_factor
            return get_seasonal_factor(date)
        except Exception as e:
            logger.warning(f"_eval_seasonal failed: {e}")
            return 1.0, 'neutral'

    def _score_to_position(self, market_score: int) -> float:
        """将市场评分转换为建议仓位系数"""
        if market_score >= 80:
            return 1.0
        elif market_score >= 65:
            return 0.8
        elif market_score >= 50:
            return 0.6
        elif market_score >= 35:
            return 0.4
        else:
            return 0.2
