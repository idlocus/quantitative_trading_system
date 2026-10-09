#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
前视偏差验证：原始信号 vs T-1信号
如果两者差异小（<2pp），则前视偏差可忽略
如果差异大，需要修正回测框架
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datetime import date
import pandas as pd

from data.layered import LayeredDataSource
from data.base.data_loader import DataLoader
from data.fundamentals import WindFundamentalProvider, build_fundamental_matrix
from data.fundamental_pit import PITFinancialProvider
from portfolio import PortfolioBacktester, PortfolioConfig, composite_from_matrices
from research.artifacts import save_artifact, portfolio_payload

DIRECTIONS = {'ep': 1.0, 'bp': 1.0, 'sp': 1.0, 'size': -1.0, 'leverage': -1.0}
ERAS = [('bias_oos_2017_2021', date(2016,11,1), date(2021,12,31), date(2017,1,1)),
        ('bias_fullmarket_2022_2026', date(2021,11,1), date(2026,9,24), date(2022,1,1))]
BENCH = '000300.SH'


def bench_extra(bt, r):
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

        mats = {}
        for f in ['ep','bp','sp','size']:
            fm = build_fundamental_matrix(prov, f, pool, d0, d1)
            if not fm.empty: mats[f] = fm
        lev = pit.get_factor_matrix('leverage', pool, d0, d1)
        if not lev.empty: mats['leverage'] = lev

        original = composite_from_matrices(mats, directions=DIRECTIONS)

        # T-1 信号：shift(1) 使每个日期用的是前一日的因子值
        shifted = original.shift(1)

        for label, score in [('原始T+0', original), ('延迟T-1', shifted)]:
            bt = PortfolioBacktester(cache, PortfolioConfig(
                top_n=100, rebalance_days=20, slippage=0.001,
                benchmark_symbol=BENCH))
            r = bt.run(score)
            print(f"  {label}: {r.total_return:+.1%} 夏普{r.sharpe:.2f} "
                  f"回撤{r.max_drawdown:.1%} 换手{r.avg_turnover:.0%}%", flush=True)


if __name__ == '__main__':
    main()
