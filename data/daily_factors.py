#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
日频量价因子库 —— 从日线 OHLCV+成交额 构建 50+ 个价格/成交量因子

因子分类：
  momentum  — 动量类（多周期收益）
  reversal  — 反转类（短期超卖、缺口回补）
  volume    — 成交量类（量比、量价背离）
  vol       — 波动率类（已实现波动率、ATR）
  structure — 价格结构类（收盘位置、K线形态）
  ma        — 均线类（均线偏离、均线交叉）
  relative  — 相对强度类（vs指数、RPS）
  signal    — 复合信号类（条件组合）

所有因子输出 (dates × symbols) 矩阵，可直接接入 FactorEvaluator / ML 模型。
"""

from typing import Dict

import numpy as np
import pandas as pd

from utils.logger import get_logger

logger = get_logger(__name__)


def build_all_factors(close: pd.DataFrame, high: pd.DataFrame, low: pd.DataFrame,
                      open_: pd.DataFrame, volume: pd.DataFrame, amount: pd.DataFrame,
                      bench_close: pd.Series = None) -> Dict[str, pd.DataFrame]:
    """
    构建全部日频因子

    Args:
        close/high/low/open_/volume/amount: (dates × symbols) 矩阵
        bench_close: 基准收盘价 Series（可选，用于相对强度）

    Returns:
        {factor_name: matrix} 字典
    """
    factors = {}
    builders = [
        _momentum_factors, _reversal_factors, _volume_factors,
        _volatility_factors, _structure_factors, _ma_factors,
    ]
    for build in builders:
        try:
            factors.update(build(close, high, low, open_, volume, amount))
        except Exception as e:
            logger.warning(f"factor builder {build.__name__} failed: {e}")

    if bench_close is not None:
        try:
            factors.update(_relative_factors(close, bench_close))
        except Exception as e:
            logger.warning(f"relative factors failed: {e}")

    try:
        factors.update(_signal_factors(close, volume, factors))
    except Exception as e:
        logger.warning(f"signal factors failed: {e}")

    logger.info(f"[daily_factors] built {len(factors)} factors")
    return factors


# =============================================================================
# 动量类
# =============================================================================

def _momentum_factors(close, high, low, open_, volume, amount):
    """多周期动量"""
    f = {}
    for n in [1, 3, 5, 10, 20, 60]:
        f[f'mom_{n}d'] = close / close.shift(n) - 1
    # 动量加速度（5日动量 - 前5日动量）
    f['mom_accel'] = (close / close.shift(5) - 1) - (close.shift(5) / close.shift(10) - 1)
    # 动量质量：收益/波动
    ret = close.pct_change()
    f['mom_sharpe_10'] = ret.rolling(10).mean() / (ret.rolling(10).std() + 1e-9)
    return f


# =============================================================================
# 反转类
# =============================================================================

def _reversal_factors(close, high, low, open_, volume, amount):
    """短期反转信号"""
    f = {}
    # 短期超卖（取负 = 跌越多值越大）
    f['rev_1d'] = -(close / close.shift(1) - 1)
    f['rev_3d'] = -(close / close.shift(3) - 1)
    f['rev_5d'] = -(close / close.shift(5) - 1)
    # 隔夜缺口回补
    gap = open_ / close.shift(1) - 1
    f['gap_fade'] = -gap  # 低开越多，反弹潜力越大
    # 连续下跌天数
    down = (close < close.shift(1)).astype(int)
    f['consec_down'] = down.rolling(5).sum()
    # 偏离近期均值
    f['dev_ma5'] = (close / close.rolling(5).mean() - 1) * -1  # 低于MA5越多越好
    return f


# =============================================================================
# 成交量类
# =============================================================================

def _volume_factors(close, high, low, open_, volume, amount):
    """成交量信号"""
    f = {}
    # 量比
    f['vol_ratio_5_20'] = volume.rolling(5).mean() / (volume.rolling(20).mean() + 1e-9)
    f['vol_ratio_1_20'] = volume / (volume.rolling(20).mean() + 1e-9)
    # 量缩价稳（缩量回调）
    f['vol_shrink'] = -(volume.rolling(3).mean() / (volume.rolling(20).mean() + 1e-9))
    # 量价相关性（5日）
    ret = close.pct_change()
    vol_chg = volume.pct_change()
    f['pv_corr_5'] = ret.rolling(5).corr(vol_chg)
    # 金额比（大资金活跃度）
    f['amt_ratio'] = amount.rolling(5).mean() / (amount.rolling(20).mean() + 1e-9)
    # 换手率代理（金额/收盘价 ≈ 换手量）
    f['turnover_proxy'] = amount / (close + 1e-9)
    # 量能异常
    f['vol_spike'] = (volume > volume.rolling(20).mean() * 2).astype(float)
    return f


# =============================================================================
# 波动率类
# =============================================================================

def _volatility_factors(close, high, low, open_, volume, amount):
    """波动率信号"""
    f = {}
    ret = close.pct_change()
    f['rvol_10'] = ret.rolling(10).std()
    f['rvol_20'] = ret.rolling(20).std()
    f['vol_ratio'] = ret.rolling(5).std() / (ret.rolling(20).std() + 1e-9)
    # ATR（避免 pd.concat 跨 DataFrame 布局问题）
    tr = np.maximum(high.values - low.values,
                    np.maximum(np.abs(high.values - close.shift().values),
                               np.abs(low.values - close.shift().values)))
    tr = pd.DataFrame(tr, index=close.index, columns=close.columns)
    f['atr_pct'] = tr.rolling(14).mean() / (close + 1e-9)
    f['vol_squeeze'] = -(ret.rolling(5).std() / (ret.rolling(20).std() + 1e-9))
    return f


# =============================================================================
# 价格结构类
# =============================================================================

def _structure_factors(close, high, low, open_, volume, amount):
    """K线结构信号"""
    f = {}
    rng = high - low + 1e-9
    f['close_pos'] = (close - low) / rng
    max_co = pd.DataFrame(np.maximum(close.values, open_.values), index=close.index, columns=close.columns)
    min_co = pd.DataFrame(np.minimum(close.values, open_.values), index=close.index, columns=close.columns)
    f['upper_shadow'] = (high - max_co) / rng
    f['lower_shadow'] = (min_co - low) / rng
    f['body_ratio'] = (close - open_).abs() / rng
    f['is_bull'] = (close > open_).astype(float)
    f['dist_high_20'] = close / high.rolling(20).max() - 1
    f['dist_low_20'] = close / low.rolling(20).min() - 1
    f['breakout_20'] = (close > high.rolling(20).max().shift(1)).astype(float)
    return f


# =============================================================================
# 均线类
# =============================================================================

def _ma_factors(close, high, low, open_, volume, amount):
    """均线偏离和交叉"""
    f = {}
    for n in [5, 10, 20, 60]:
        f[f'dist_ma{n}'] = close / close.rolling(n).mean() - 1
    # 均线多头排列
    ma5 = close.rolling(5).mean()
    ma20 = close.rolling(20).mean()
    ma60 = close.rolling(60).mean()
    f['ma_bull'] = ((ma5 > ma20) & (ma20 > ma60)).astype(float)
    # 均线斜率（MA20的5日变化率）
    f['ma20_slope'] = ma20 / ma20.shift(5) - 1
    return f


# =============================================================================
# 相对强度类
# =============================================================================

def _relative_factors(close, bench_close):
    """相对基准的强度"""
    f = {}
    bench = bench_close.reindex(close.index).ffill()
    for n in [5, 20]:
        stock_ret = close / close.shift(n) - 1
        bench_ret = bench / bench.shift(n) - 1
        f[f'rel_bench_{n}d'] = stock_ret.sub(bench_ret, axis=0)
    # 横截面RPS（20日收益在全市场的百分位排名）
    ret_20 = close / close.shift(20) - 1
    f['rps_20'] = ret_20.rank(axis=1, pct=True)
    return f


# =============================================================================
# 复合信号类
# =============================================================================

def _signal_factors(close, volume, existing):
    """条件组合信号"""
    f = {}
    # 放量反转（跌多了 + 放量 = 聪明钱进场）
    if 'rev_5d' in existing and 'vol_ratio_1_20' in existing:
        f['vol_reversal'] = existing['rev_5d'] * (existing['vol_ratio_1_20'] > 1.2)
    # 缩量回调（涨趋势中的缩量回调 = 洗盘）
    if 'mom_20d' in existing and 'vol_shrink' in existing:
        f['quiet_pullback'] = (existing['mom_20d'] > 0) * existing['vol_shrink']
    # 波动率收缩 + 突破
    if 'vol_squeeze' in existing and 'breakout_20' in existing:
        f['squeeze_breakout'] = existing['vol_squeeze'] * existing['breakout_20']
    # 强势 + 收盘高位（日内强势确认）
    if 'mom_5d' in existing and 'close_pos' in existing:
        f['strong_close'] = existing['mom_5d'] * existing['close_pos']
    return f
