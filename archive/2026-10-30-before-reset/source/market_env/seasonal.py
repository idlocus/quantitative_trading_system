#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
季节性因子 - A股实证月度规律

基于沪深300指数 2002-2026 共 296 个月度样本的实证统计（2026-09 计算）：
- 用中位数而非均值（均值被 2006/2014 等极端年份拉偏，中位数稳健）
- 胜率本身对极端值免疫，与中位数互补

实证结论（与美股 S&P 规律明显不同）：
- 1月/2月最强（春季躁动/春节效应，2月胜率 72%）
- 12月次强（年末机构排名行情，胜率 62.5%）
- 8月最弱（中位数 -0.51%，胜率 44%）
- 3月/10月偏弱

已移除原实现中的美国总统周期（对A股无因果基础）与美股月度表。

季节性因子范围 0.8-1.2，集成到 MarketEnvScorer 仓位系数（base × seasonal）。
"""

from datetime import datetime
from typing import Tuple, Dict, Optional

import pandas as pd

from utils.logger import get_logger

logger = get_logger(__name__)


# =============================================================================
# A股月度实证统计（沪深300 2002-2026；median=月收益中位数，win_rate=上涨概率）
# 可通过 refresh_monthly_stats() 从 akshare 重新计算更新
# =============================================================================

ASHARE_MONTHLY_STATS: Dict[int, Dict[str, float]] = {
    1:  {'median': 0.0200, 'win_rate': 0.542},
    2:  {'median': 0.0191, 'win_rate': 0.720},
    3:  {'median': -0.0007, 'win_rate': 0.480},
    4:  {'median': 0.0106, 'win_rate': 0.560},
    5:  {'median': 0.0041, 'win_rate': 0.560},
    6:  {'median': 0.0040, 'win_rate': 0.520},
    7:  {'median': 0.0048, 'win_rate': 0.600},
    8:  {'median': -0.0051, 'win_rate': 0.440},
    9:  {'median': 0.0039, 'win_rate': 0.560},
    10: {'median': 0.0044, 'win_rate': 0.500},
    11: {'median': 0.0063, 'win_rate': 0.542},
    12: {'median': 0.0122, 'win_rate': 0.625},
}

# 统计来源（用于追溯）
STATS_SOURCE = {
    'index': 'sh000300',
    'start': '2002-01',
    'end': '2026-09',
    'samples': 296,
}

# 综合公式权重：中位数收益 60% + 胜率偏离 40%
W_RETURN = 0.6
W_WINRATE = 0.4
RETURN_SCALE = 10.0   # 月中位数收益放大系数（0.02 → 0.2 分）


def get_seasonal_factor(date_str: str = None, dt: datetime = None) -> Tuple[float, str]:
    """
    计算A股季节性因子。

    综合"月收益中位数"与"月胜率偏离"两个实证统计，
    无总统周期/美股数据。

    Args:
        date_str: 日期字符串 (yyyyMMdd)
        dt: datetime 对象（二选一）

    Returns:
        (seasonal_factor, signal)
        - seasonal_factor: 0.8-1.2，>1 表示季节性有利
        - signal: "favorable" / "neutral" / "unfavorable"
    """
    if dt is None:
        if date_str:
            dt = datetime.strptime(date_str, "%Y%m%d")
        else:
            dt = datetime.now()

    month = dt.month
    stats = ASHARE_MONTHLY_STATS.get(month, {'median': 0.0, 'win_rate': 0.5})

    # 综合得分：月中位数收益（放大）+ 胜率对 50% 的偏离（小数尺度，二者量纲对齐）
    score = 0.0
    score += W_RETURN * stats['median'] * RETURN_SCALE
    score += W_WINRATE * (stats['win_rate'] - 0.5)

    # 归一化到 0.8-1.2
    seasonal_factor = max(0.8, min(1.2, 1.0 + score))

    if seasonal_factor >= 1.1:
        signal = "favorable"
    elif seasonal_factor <= 0.9:
        signal = "unfavorable"
    else:
        signal = "neutral"

    logger.debug(
        f"[seasonal] {dt.strftime('%Y-%m')} factor={seasonal_factor:.3f} signal={signal} "
        f"(median={stats['median']:+.4f}, win_rate={stats['win_rate']:.2f})"
    )
    return seasonal_factor, signal


def refresh_monthly_stats(index_symbol: str = 'sh000300') -> bool:
    """
    从 akshare 重新计算月度统计并更新 ASHARE_MONTHLY_STATS

    用中位数 + 胜率，避免极端月份主导。失败时保留现有表。

    Args:
        index_symbol: 新浪指数代码（默认沪深300）

    Returns:
        是否成功更新
    """
    try:
        import akshare as ak

        df = ak.stock_zh_index_daily(symbol=index_symbol)
        if df is None or df.empty:
            return False

        df['date'] = pd.to_datetime(df['date'])
        monthly_ret = (
            df.set_index('date')['close']
            .resample('ME').last()
            .pct_change().dropna()
        )
        if len(monthly_ret) < 60:  # 至少5年样本
            logger.warning(f"[refresh_monthly_stats] 样本不足: {len(monthly_ret)}")
            return False

        grouped = monthly_ret.groupby(monthly_ret.index.month)
        new_stats = {
            int(month): {
                'median': float(grouped.median()[month]),
                'win_rate': float((g > 0).mean()),
            }
            for month, g in grouped
        }
        if len(new_stats) < 12:
            logger.warning(f"[refresh_monthly_stats] 月份覆盖不全: {sorted(new_stats)}")
            return False

        ASHARE_MONTHLY_STATS.clear()
        ASHARE_MONTHLY_STATS.update(new_stats)
        STATS_SOURCE.update({
            'index': index_symbol,
            'start': str(monthly_ret.index.min().strftime('%Y-%m')),
            'end': str(monthly_ret.index.max().strftime('%Y-%m')),
            'samples': len(monthly_ret),
        })
        logger.info(f"[refresh_monthly_stats] updated from {index_symbol}: "
                    f"{STATS_SOURCE['start']} ~ {STATS_SOURCE['end']} ({len(monthly_ret)} 样本)")
        return True
    except Exception as e:
        logger.warning(f"[refresh_monthly_stats] failed: {e}")
        return False
