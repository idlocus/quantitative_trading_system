#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
中期趋势维度实验 —— 在质量过滤+价值基础上加入动量

假设：价值和动量跨市场互补（Asness et al. "Value and Momentum Everywhere"），
检验在质量价值策略上叠加6个月趋势（剔最近1个月）是否改善净收益/回撤/行业集中度。

预注册条件：
  1. 趋势版本必须跑赢无趋势版本（净收益）
  2. 不能显著恶化回撤
  3. 不能显著恶化行业集中度
  如果趋势不改善，不加——保持简单。
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import date

import numpy as np
import pandas as pd

from data.layered import LayeredDataSource
from data.base.data_loader import DataLoader
from portfolio import composite_from_matrices
from portfolio.daily_replay import DailyReplayBacktester, ReplayConfig
from trading import config as chain

ERAS = [('fullmarket_2022_2026', date(2021, 11, 1), date(2026, 10, 9), date(2022, 1, 1))]


def build_momentum_matrix(cache, d0, d1):
    """中期趋势矩阵——委托生产唯一实现 trading/strategy._build_momentum_matrix
    （2026-10-15 依赖方向修正：实验引用生产，不再被生产引用）"""
    from trading.strategy import _build_momentum_matrix
    return _build_momentum_matrix(cache, d0, d1)


def combine_value_momentum(value_score, momentum, momentum_weight=0.3):
    """
    价值+动量合成：value_weight=1-momentum_weight
    各自先做截面标准化（z-score），避免量纲差异。
    """
    common_dates = value_score.index.intersection(momentum.index)
    common_cols = value_score.columns.intersection(momentum.columns)
    if len(common_dates) == 0:
        return value_score

    combined = value_score.copy()
    for dt in common_dates:
        v = value_score.loc[dt].reindex(common_cols)
        m = momentum.loc[dt].reindex(common_cols)

        # 截面z-score
        v_z = (v - v.mean()) / (v.std() + 1e-10)
        m_z = (m - m.mean()) / (m.std() + 1e-10)

        # 加权合成（动量方向=正，高动量=高分）
        vw = 1.0 - momentum_weight
        combined.loc[dt, common_cols] = vw * v_z + momentum_weight * m_z

    return combined


def main():
    from experiments.quality_value_candidate import (
        build_value_score, build_quality_filter, apply_quality_filter,
        monthly_signal)

    ds = LayeredDataSource()
    loader = DataLoader.__new__(DataLoader)
    loader.data_source_type = 'layered'
    loader.data_source = ds
    loader._cache = None

    era, d0, d1, ts = ERAS[0]
    print(f"===== 趋势维度实验 @ {era} =====\n")
    pool = sorted(ds.list_symbols(as_of=ts))
    cache = loader.preload_klines(pool + ['000300.SH'], d0, d1)._cache

    # 构建质量+价值分数
    print("构建质量过滤+价值分数...", flush=True)
    value_score = build_value_score(ds, pool, d0, d1)
    value_score = value_score[value_score.columns.intersection(cache.keys())]
    quality = build_quality_filter(ds, pool, d0, d1)
    score_qv = apply_quality_filter(value_score, quality)

    # 构建动量矩阵
    print("构建动量矩阵...", flush=True)
    momentum = build_momentum_matrix(cache, d0, d1)
    print(f"动量矩阵 {momentum.shape[0]}日 × {momentum.shape[1]}只", flush=True)

    # 实验组
    experiments = [
        ('M0', '质量+价值（无趋势，基线）', score_qv),
        ('M1', '质量+价值+趋势(权重0.2)', combine_value_momentum(score_qv, momentum, 0.2)),
        ('M2', '质量+价值+趋势(权重0.3)', combine_value_momentum(score_qv, momentum, 0.3)),
        ('M3', '质量+价值+趋势(权重0.5)', combine_value_momentum(score_qv, momentum, 0.5)),
        # 纯趋势对照（检验趋势本身是否有alpha）
        ('M4', '纯趋势(无价值，质量过滤)', combine_value_momentum(
            pd.DataFrame(0.0, index=score_qv.index, columns=score_qv.columns),
            momentum, 1.0)),
    ]

    results = {}
    for name, desc, score in experiments:
        score_m = monthly_signal(score)
        cfg = ReplayConfig(
            top_n=chain.TOP_N, band=chain.BUFFER_BAND,
            use_risk_gate=True,
            min_listing_days=chain.MIN_LISTING_DAYS,
            min_daily_turnover=chain.MIN_DAILY_TURNOVER,
        )
        rp = DailyReplayBacktester(cache, cfg)
        r = rp.run(score_m, start_date=str(ts), end_date=str(d1))
        results[name] = r
        cost = r.total_cost / r.initial_capital
        print(f"  {name} {desc:>28s}: 净{r.total_return:+.1%} 毛{r.total_return + cost:+.1%} "
              f"夏普{r.sharpe:.2f} 回撤{r.max_drawdown:.1%} "
              f"{'⚠终态' if r.terminal_date else '运行至期末'}", flush=True)

    # 汇总
    print(f"\n{'='*60}")
    print("趋势维度汇总")
    print(f"{'='*60}")
    print(f"{'实验':>4s} {'描述':>28s} {'净收益':>7s} {'夏普':>5s} {'回撤':>6s} {'vs基线':>7s}")
    base = results['M0']
    for name, desc, _ in experiments:
        r = results[name]
        diff = r.total_return - base.total_return
        print(f"  {name}  {desc:>28s} {r.total_return:>6.1%} {r.sharpe:>5.2f} "
              f"{r.max_drawdown:>5.1%} {diff:>+6.1%}")

    # 判定
    print(f"\n判定：")
    for name, desc, _ in experiments[1:4]:  # M1-M3
        r = results[name]
        beat = '✅' if r.total_return > base.total_return else '❌'
        dd_ok = '✅' if r.max_drawdown > base.max_drawdown - 0.02 else '❌'
        print(f"  {name}: 跑赢基线{beat} 回撤可控{dd_ok}")


if __name__ == '__main__':
    main()
