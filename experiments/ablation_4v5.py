#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""五因子消融(-leverage)：4因子基线 vs +低杠杆因子，两年代（验收通过的因子组合级复检）"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datetime import date
import pandas as pd

from data.layered import LayeredDataSource
from data.base.data_loader import DataLoader
from data.fundamentals import WindFundamentalProvider, build_fundamental_matrix
from data.fundamental_pit import PITFinancialProvider
from portfolio import PortfolioBacktester, PortfolioConfig, composite_from_matrices
from research.risk_model import style_exposures, neutralize
from research.artifacts import save_artifact, portfolio_payload

D4 = {'ep': 1.0, 'bp': 1.0, 'sp': 1.0, 'size': -1.0}
ERAS = [('oos_2017_2021', date(2016,11,1), date(2021,12,31), date(2017,1,1)),
        ('fullmarket_2022_2026', date(2021,11,1), date(2026,9,24), date(2022,1,1))]
BENCH = '000300.SH'

def main():
    ds = LayeredDataSource()
    loader = DataLoader.__new__(DataLoader); loader.data_source_type='layered'
    loader.data_source = ds; loader._cache = None
    prov = WindFundamentalProvider(ds); pit = PITFinancialProvider(ds)

    for era, d0, d1, ts in ERAS:
        print(f"===== {era} =====", flush=True)
        pool = sorted(ds.list_symbols(as_of=ts))  # DYNAMIC UNIVERSE: 起始日PIT池+60日上市过滤由backtester动态处理
        cache = loader.preload_klines(pool + [BENCH], d0, d1)._cache
        mats = {}
        for f in D4:
            fm = build_fundamental_matrix(prov, f, pool, d0, d1)
            if not fm.empty: mats[f] = fm.loc[pd.Timestamp(d0):pd.Timestamp(d1)]
        lev = pit.get_factor_matrix('leverage', pool, d0, d1).loc[pd.Timestamp(d0):pd.Timestamp(d1)]

        combos = [('4因子基线', mats, D4),
                  ('5因子生产', {**mats, 'leverage': lev}, {**D4, 'leverage': -1.0})]
        for label, m, dirs in combos:
            score = composite_from_matrices(m, directions=dirs)
            bt = PortfolioBacktester(cache, PortfolioConfig(top_n=100, rebalance_days=20,
                                     slippage=0.001, benchmark_symbol=BENCH))
            r = bt.run(score, start_date=str(ts))
            b = bt.close_mx[BENCH].loc[r.equity_curve.index].ffill()
            extra = {'benchmark_series': {'dates': [str(d)[:10] for d in b.index],
                     'values': [round(float(v/b.iloc[0]*r.initial_capital),2) for v in b.values]}}
            save_artifact('portfolio', f'composite_v5_{era}_{label[:4]}',
                          portfolio_payload(r, label=label, extra=extra))
            print(f"  {label}: {r.total_return:+.1%} 基准{r.benchmark_return:+.1%} 夏普{r.sharpe:.2f}", flush=True)

if __name__ == '__main__':
    main()
