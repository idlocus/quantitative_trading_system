#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模式一实验：基本面筛选 + 反转择时
基线: 5因子 → Top30（等权，月调仓）
实验: 5因子 → Top100 候选池 → 60日反转择时重排 → Top30
两段年代消融验证
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datetime import date
import numpy as np
import pandas as pd

from data.layered import LayeredDataSource
from data.base.data_loader import DataLoader
from data.fundamentals import WindFundamentalProvider, build_fundamental_matrix
from data.fundamental_pit import PITFinancialProvider
from portfolio import PortfolioBacktester, PortfolioConfig, composite_from_matrices
from research.artifacts import save_artifact, portfolio_payload

DIRECTIONS = {'ep': 1.0, 'bp': 1.0, 'sp': 1.0, 'size': -1.0, 'leverage': -1.0}
ERAS = [('timing_oos_2017_2021', date(2016,11,1), date(2021,12,31), date(2017,1,1)),
        ('timing_fullmarket_2022_2026', date(2021,11,1), date(2026,9,24), date(2022,1,1))]
BENCH = '000300.SH'
REVERSAL_PERIOD = 60  # 60日反转
CANDIDATE_POOL = 100   # 候选池大小
TIMING_WEIGHT = 0.30   # 反转择时权重（30%）


def build_timing_score(fund_score, close_mx, ref_date):
    """
    基本面分数 + 反转择时 → 最终排名

    fund_score: 综合分矩阵（dates × symbols）
    close_mx: 收盘价矩阵
    反转信号: 60日收益越低（跌越多），择时加分越高
    """
    ts = pd.Timestamp(ref_date)
    valid = fund_score.index[fund_score.index <= ts]
    if len(valid) == 0:
        return None
    latest = valid[-1]

    # 1) 基本面排名
    fund_rank = fund_score.loc[latest].rank(ascending=False, pct=True)  # 高分=低rank值

    # 2) 反转排名（60日收益越低越好 → rank ascending=True）
    idx = close_mx.index.searchsorted(latest, side='right') - 1
    if idx < REVERSAL_PERIOD:
        return None
    ret_60d = close_mx.iloc[idx] / close_mx.iloc[idx - REVERSAL_PERIOD] - 1
    reversal_rank = ret_60d.rank(ascending=True, pct=True)  # 跌得多=低rank值=好

    # 3) 加权组合（rank值越低越好）
    combined = (1 - TIMING_WEIGHT) * fund_rank + TIMING_WEIGHT * reversal_rank
    # 回测器选最高分，反转使高值=好（基本面好+近期跌）
    return (1 - combined).sort_values(ascending=False)


def bench_series(bt, r):
    b = bt.close_mx[BENCH].loc[r.equity_curve.index].ffill()
    return {'benchmark_series': {'dates': [str(x)[:10] for x in b.index],
            'values': [round(float(v/b.iloc[0]*r.initial_capital),2) for v in b.values]}}


def main():
    ds = LayeredDataSource()
    loader = DataLoader.__new__(DataLoader)
    loader.data_source_type = 'layered'; loader.data_source = ds; loader._cache = None
    prov = WindFundamentalProvider(ds)
    pit = PITFinancialProvider(ds)

    for era, d0, d1, ts in ERAS:
        print(f"===== {era} =====", flush=True)
        pool = sorted(ds.list_symbols(as_of=ts))
        cache = loader.preload_klines(pool + [BENCH], d0, d1)._cache

        # 基本面因子矩阵
        mats = {}
        for f in ['ep','bp','sp','size']:
            fm = build_fundamental_matrix(prov, f, pool, d0, d1)
            if not fm.empty: mats[f] = fm
        lev = pit.get_factor_matrix('leverage', pool, d0, d1)
        if not lev.empty: mats['leverage'] = lev
        fund_score = composite_from_matrices(mats, directions=DIRECTIONS, experimental=True)

        # 收盘价矩阵
        bt_base = PortfolioBacktester(cache, PortfolioConfig(benchmark_symbol=BENCH))
        close_mx = bt_base.close_mx

        # 构造择时版目标持仓序列
        from portfolio.construction import select_portfolio
        timing_scores = {}
        for d in fund_score.index[::5]:  # 每5个交易日检查一次（双周频率）
            combined = build_timing_score(fund_score, close_mx, d)
            if combined is not None:
                timing_scores[d] = combined

        # 转换为日期→Series 的 score matrix
        timing_mx = pd.DataFrame(timing_scores).T
        timing_mx = timing_mx[sorted(timing_mx.columns)]
        # 重索引到日频并前向填充（确保任何调仓日都有分数）
        daily_idx = close_mx.index[close_mx.index <= timing_mx.index[-1]]
        timing_mx = timing_mx.reindex(daily_idx).ffill()

        combos = [
            ('基线5因子', fund_score, 30),       # 原始策略
            ('择时版', timing_mx, 30),            # 筛选+择时
        ]

        for label, score, topn in combos:
            bt = PortfolioBacktester(cache, PortfolioConfig(
                top_n=topn, rebalance_days=20, slippage=0.001,
                benchmark_symbol=BENCH))
            r = bt.run(score)
            save_artifact('portfolio', f'{era}_{label[:2]}',
                          portfolio_payload(r, label=label, extra=bench_series(bt, r)))
            print(f"  {label}: {r.total_return:+.1%} 夏普{r.sharpe:.2f} "
                  f"回撤{r.max_drawdown:.1%} 交易{len(r.trade_log)}", flush=True)


if __name__ == '__main__':
    main()
