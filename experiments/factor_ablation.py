#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
因子贡献消融实验：单因子组合 vs 合成组合 vs 等权对照（两个年代）
产出 portfolio_ablation_<era>_<factor>.json 工件，仪表盘自动展示
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import date
import pandas as pd

from data.layered import LayeredDataSource
from data.base.data_loader import DataLoader
from data.fundamentals import WindFundamentalProvider, build_fundamental_matrix
from portfolio import PortfolioBacktester, PortfolioConfig, composite_from_matrices
from research.artifacts import save_artifact, portfolio_payload

DIRECTIONS = {'ep': 1.0, 'bp': 1.0, 'sp': 1.0, 'size': -1.0}
ERAS = [('oos_2017_2021', date(2016, 11, 1), date(2021, 12, 31), date(2017, 1, 1)),
        ('fullmarket_2022_2026', date(2021, 11, 1), date(2026, 9, 24), date(2022, 1, 1))]
BENCH = '000300.SH'


def bench_extra(bt, r):
    b = bt.close_mx[BENCH].loc[r.equity_curve.index].ffill()
    return {'benchmark_series': {'dates': [str(d)[:10] for d in b.index],
                                 'values': [round(float(v / b.iloc[0] * r.initial_capital), 2) for v in b.values]}}


def main():
    ds = LayeredDataSource()
    loader = DataLoader.__new__(DataLoader)
    loader.data_source_type = 'layered'
    loader.data_source = ds
    loader._cache = None
    provider = WindFundamentalProvider(ds)

    for era, dstart, dend, test_start in ERAS:
        print(f"===== {era} =====", flush=True)
        pool = sorted(ds.list_symbols(as_of=test_start))
        cache = loader.preload_klines(pool + [BENCH], dstart, dend)._cache
        mats = {}
        for f in DIRECTIONS:
            fm = build_fundamental_matrix(provider, f, pool, dstart, dend)
            if not fm.empty:
                mats[f] = fm.loc[pd.Timestamp(dstart):pd.Timestamp(dend)]

        def run(score, label, topn=100):
            bt = PortfolioBacktester(cache, PortfolioConfig(
                top_n=topn, rebalance_days=20, slippage=0.001, benchmark_symbol=BENCH))
            r = bt.run(score, start_date=str(ts))
            save_artifact('portfolio', f'ablation_{era}_{label}',
                          portfolio_payload(r, label=label, extra=bench_extra(bt, r)))
            print(f"  {label:10s}: {r.total_return:+.1%}", flush=True)
            return r

        # 各单因子（Top100，方向与合成一致）
        for f, fm in mats.items():
            run(fm, f)
        # 合成（4因子）
        run(composite_from_matrices(mats, directions=DIRECTIONS), '合成组合')
        # 等权对照
        idx = list(mats.values())[0].loc[pd.Timestamp(test_start):].index
        cols = list(mats.values())[0].columns
        run(pd.DataFrame(1.0, index=idx, columns=cols), '等权', topn=len(pool))


if __name__ == '__main__':
    main()
