#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
参数稳定性检查 —— 质量过滤+价值策略在邻近参数下的表现

预注册条件：
  如果只有在特定参数组合下才盈利（如仅Top100/月度），说明是过拟合。
  相邻参数（Top50/75/125/150、双周/月度/季度）都应有合理表现。

数据复用 quality_value_candidate.py 的构建逻辑。
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import date

import pandas as pd

from data.layered import LayeredDataSource
from data.base.data_loader import DataLoader
from portfolio.daily_replay import DailyReplayBacktester, ReplayConfig
from trading import config as chain

ERAS = [('fullmarket_2022_2026', date(2021, 11, 1), date(2026, 10, 9), date(2022, 1, 1))]


def quarterly_signal(score_mx):
    """只在每季度第一个交易日更新信号"""
    dates = score_mx.index
    qt_dates = set()
    last_q = None
    for d in dates:
        q = (d.year, (d.month - 1) // 3)
        if q != last_q:
            qt_dates.add(d)
            last_q = q
    return score_mx.loc[sorted(qt_dates)]


def weekly_signal(score_mx):
    """每两周更新一次"""
    dates = score_mx.index
    wk_dates = set()
    for i, d in enumerate(dates):
        if i % 10 == 0:  # 约每两周（10个交易日）
            wk_dates.add(d)
    return score_mx.loc[sorted(wk_dates)]


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
    print(f"===== 参数稳定性检查 @ {era} =====\n")
    pool = sorted(ds.list_symbols(as_of=ts))
    cache = loader.preload_klines(pool + ['000300.SH'], d0, d1)._cache

    # 构建质量+价值分数（一次构建，多组使用）
    print("构建质量过滤+价值分数...", flush=True)
    value_score = build_value_score(ds, pool, d0, d1)
    value_score = value_score[value_score.columns.intersection(cache.keys())]
    quality = build_quality_filter(ds, pool, d0, d1)
    score = apply_quality_filter(value_score, quality)

    # 三种信号频率
    score_biweekly = weekly_signal(score)
    score_monthly = monthly_signal(score)
    score_quarterly = quarterly_signal(score)

    # 参数网格
    configs = []
    # 持仓数变化（月度调仓固定）
    for top_n in [50, 75, 100, 125, 150]:
        configs.append((f'Top{top_n}_月度', top_n, score_monthly, chain.BUFFER_BAND))
    # 调仓频率变化（Top100固定）
    configs.append(('Top100_双周', 100, score_biweekly, chain.BUFFER_BAND))
    configs.append(('Top100_月度', 100, score_monthly, chain.BUFFER_BAND))
    configs.append(('Top100_季度', 100, score_quarterly, chain.BUFFER_BAND))
    # 缓冲带变化（Top100月度固定）
    for band in [0.20, 0.30, 0.50]:
        configs.append((f'Top100_月度_band{band:.0%}', 100, score_monthly, band))

    # 运行
    results = []
    for name, top_n, score_sig, band in configs:
        cfg = ReplayConfig(
            top_n=top_n, band=band,
            use_risk_gate=True,
            min_listing_days=chain.MIN_LISTING_DAYS,
            min_daily_turnover=chain.MIN_DAILY_TURNOVER,
        )
        rp = DailyReplayBacktester(cache, cfg)
        r = rp.run(score_sig, start_date=str(ts), end_date=str(d1))
        cost = r.total_cost / r.initial_capital
        results.append({
            'name': name, 'top_n': top_n, 'band': band,
            'net': r.total_return, 'gross': r.total_return + cost,
            'sharpe': r.sharpe, 'mdd': r.max_drawdown,
            'cost': cost, 'stops': r.n_stop_sells,
            'terminal': r.terminal_date,
        })
        print(f"  {name:>24s}: 净{r.total_return:+.1%} 毛{r.total_return + cost:+.1%} "
              f"夏普{r.sharpe:.2f} 回撤{r.max_drawdown:.1%} "
              f"{'⚠终态' + r.terminal_date[:7] if r.terminal_date else '运行至期末'}",
              flush=True)

    # 汇总
    print(f"\n{'='*60}")
    print("参数稳定性汇总")
    print(f"{'='*60}")
    print(f"{'配置':>24s} {'净收益':>7s} {'毛收益':>7s} {'夏普':>5s} {'回撤':>6s} {'终态':s}")
    for r in results:
        term = r['terminal'][:7] if r['terminal'] else '无'
        print(f"  {r['name']:>22s} {r['net']:>6.1%} {r['gross']:>6.1%} "
              f"{r['sharpe']:>5.2f} {r['mdd']:>5.1%}  {term}")

    # 稳定性判定
    positive = [r for r in results if r['net'] > 0]
    print(f"\n稳定性判定：{len(positive)}/{len(results)} 组净收益为正")
    if len(positive) >= len(results) * 0.7:
        print("✅ 策略在邻近参数下普遍盈利——不是过拟合")
    else:
        print("⚠️ 部分参数组合亏损——需检查是否依赖特定参数")
        losing = [r for r in results if r['net'] <= 0]
        for r in losing:
            print(f"  亏损: {r['name']} ({r['net']:+.1%})")

    # 收益分布
    nets = [r['net'] for r in results]
    print(f"\n净收益分布: min={min(nets):+.1%} median={sorted(nets)[len(nets)//2]:+.1%} "
          f"max={max(nets):+.1%}")
    print(f"标准差: {pd.Series(nets).std():.1%}")


if __name__ == '__main__':
    main()
