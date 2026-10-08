#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""重建关键实验并保存结构化工件（供研究仪表盘渲染）——全部走本地缓存/DB，零配额"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datetime import date
import numpy as np
import pandas as pd

from data.layered import LayeredDataSource
from data.base.data_loader import DataLoader
from data.fundamentals import WindFundamentalProvider, build_fundamental_matrix
from portfolio import PortfolioBacktester, PortfolioConfig, composite_from_matrices
from research.factor_eval import FactorEvaluator
from research.artifacts import save_artifact, portfolio_payload, factor_payload

DIRECTIONS = {'ep': 1.0, 'bp': 1.0, 'sp': 1.0, 'size': -1.0}
ERAS = [('oos_2017_2021', date(2016, 11, 1), date(2021, 12, 31), date(2017, 1, 1), None),
        ('fullmarket_2022_2026', date(2021, 11, 1), date(2026, 9, 24), date(2022, 1, 1), None),
        ('enhanced_2017_2021', date(2016, 11, 1), date(2021, 12, 31), date(2017, 1, 1), '000300.SH'),
        ('enhanced_2022_2026', date(2021, 11, 1), date(2026, 9, 24), date(2022, 1, 1), '000300.SH')]


def bench_extra(bt, r, initial):
    b = bt.close_mx[r.config.benchmark_symbol].loc[r.equity_curve.index].ffill()
    return {'benchmark_series': {'dates': [str(d)[:10] for d in b.index],
                                 'values': [round(float(v / b.iloc[0] * initial), 2) for v in b.values]}}


def main():
    ds = LayeredDataSource()
    loader = DataLoader.__new__(DataLoader)
    loader.data_source_type = 'layered'
    loader.data_source = ds
    loader._cache = None
    provider = WindFundamentalProvider(ds)
    BENCH = '000300.SH'

    for name, dstart, dend, test_start, member_idx in ERAS:
        print(f"== {name} ==", flush=True)
        if member_idx:
            pool = ds.get_index_members(member_idx, as_of=test_start)
        else:
            pool = sorted(ds.list_symbols(as_of=test_start))
        cache = loader.preload_klines(pool + [BENCH], dstart, dend)._cache
        matrices = {}
        for f in DIRECTIONS:
            fm = build_fundamental_matrix(provider, f, pool, dstart, dend)
            if not fm.empty:
                matrices[f] = fm.loc[pd.Timestamp(dstart):pd.Timestamp(dend)]
        score = composite_from_matrices(matrices, directions=DIRECTIONS)

        for label, sc, topn in [('Top100', score, 100),
                                ('Top500', score, min(500, len(pool))),
                                ('等权', None, len(pool))]:
            if sc is None:
                idx = list(matrices.values())[0].loc[pd.Timestamp(test_start):].index
                cols = list(matrices.values())[0].columns
                sc = pd.DataFrame(1.0, index=idx, columns=cols)
            bt = PortfolioBacktester(cache, PortfolioConfig(
                top_n=topn, rebalance_days=20, slippage=0.001, benchmark_symbol=BENCH))
            r = bt.run(sc)
            extra = bench_extra(bt, r, r.initial_capital) if r.benchmark_return is not None else {}
            path = save_artifact('portfolio', f"{name}_{label}",
                                 portfolio_payload(r, label=label, extra=extra))
            print(f"  {label}: 总收益{r.total_return:+.1%} → {os.path.basename(path)}", flush=True)

        # 因子工件：bp@60d（本时代区间）
        ev = FactorEvaluator(cache)
        if 'bp' in matrices:
            fres = ev.evaluate_matrix(matrices['bp'], name='bp', period=60)
            if fres:
                save_artifact('factor', f"bp60_{name}", factor_payload(fres))
                print(f"  bp@60d RankIC={fres.rankic_mean:+.4f} 已存", flush=True)


if __name__ == '__main__':
    main()
