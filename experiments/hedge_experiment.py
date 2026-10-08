#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase B：优化器指增重试 + 市场中性模拟
两年代 × {等权Top100, 优化器加权Top100(风格约束)} × {多头, β对冲中性}
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datetime import date
import pandas as pd

from data.layered import LayeredDataSource
from data.base.data_loader import DataLoader
from data.fundamentals import WindFundamentalProvider, build_fundamental_matrix
from portfolio import PortfolioBacktester, PortfolioConfig, composite_from_matrices
from portfolio.optimizer import solve_weights, OptimizerConfig
from portfolio.hedged import beta_hedge
from research.risk_model import style_exposures
from research.artifacts import save_artifact, portfolio_payload

D4 = {'ep': 1.0, 'bp': 1.0, 'sp': 1.0, 'size': -1.0}
ERAS = [('oos_2017_2021', date(2016,11,1), date(2021,12,31), date(2017,1,1)),
        ('fullmarket_2022_2026', date(2021,11,1), date(2026,9,24), date(2022,1,1))]
BENCH = '000300.SH'


def main():
    ds = LayeredDataSource()
    loader = DataLoader.__new__(DataLoader); loader.data_source_type='layered'
    loader.data_source = ds; loader._cache = None
    prov = WindFundamentalProvider(ds)

    for era, d0, d1, ts in ERAS:
        print(f"===== {era} =====", flush=True)
        pool = sorted(ds.list_symbols(as_of=ts))
        cache = loader.preload_klines(pool + [BENCH], d0, d1)._cache
        mats = {}
        for f in D4:
            fm = build_fundamental_matrix(prov, f, pool, d0, d1)
            if not fm.empty: mats[f] = fm.loc[pd.Timestamp(d0):pd.Timestamp(d1)]
        score = composite_from_matrices(mats, directions=D4)

        bt_base = PortfolioBacktester(cache, PortfolioConfig(benchmark_symbol=BENCH))
        close_mx = bt_base.close_mx
        rets = close_mx.pct_change()
        mv = prov.get_field_matrix('mkt_cap_ard', pool, d0, d1)
        expo = style_exposures(close_mx, mv)

        for label, use_opt in [('等权', False), ('优化器', True)]:
            state = {'prev': {}}
            def w_func(targets, d, _opt=use_opt, _st=state):
                if not _opt:
                    from portfolio.construction import equal_weights
                    w = equal_weights(targets, 0.03)
                    _st['prev'] = w
                    return w
                d = pd.Timestamp(d)
                hist = rets.loc[:d].iloc[-120:]
                exp_d = pd.DataFrame({f: m.loc[d] for f, m in expo.items()})
                w = solve_weights(
                    pd.Series(1.0, index=targets),
                    hist, style_exp=exp_d,
                    bench_style={'size': 0.0, 'momentum': 0.0, 'volatility': 0.0},
                    prev_weights=_st['prev'],
                    config=OptimizerConfig(max_weight=0.03, style_budget=0.3))
                _st['prev'] = w
                return w

            bt = PortfolioBacktester(cache, PortfolioConfig(
                top_n=100, rebalance_days=20, slippage=0.001, benchmark_symbol=BENCH))
            r = bt.run(score, weight_func=w_func)
            bench = close_mx[BENCH].loc[r.equity_curve.index].ffill()
            bex = {'benchmark_series': {'dates': [str(x)[:10] for x in bench.index],
                   'values': [round(float(v/bench.iloc[0]*r.initial_capital),2) for v in bench.values]}}
            save_artifact('portfolio', f'phaseB_{era}_{label}', portfolio_payload(r, label=label, extra=bex))
            print(f"  {label}多头: {r.total_return:+.1%} 夏普{r.sharpe:.2f}", flush=True)

            h = beta_hedge(r.equity_curve, bench, beta_window=60, cost_annual=0.06)
            hex_ = dict(bex)
            save_artifact('portfolio', f'phaseB中性_{era}_{label}', {
                **portfolio_payload(r, label=f'中性-{label}'),
                'total_return': h.total_return, 'annual_return': h.annual_return,
                'sharpe': h.sharpe, 'max_drawdown': h.max_drawdown,
                'benchmark_return': None, 'market_correlation': h.market_correlation,
                'hedged_cost': h.hedged_cost_total,
                'equity': {'dates': [str(x)[:10] for x in h.nav.index],
                           'values': [round(float(v*r.initial_capital),2) for v in h.nav.values]},
                **{k: v for k, v in hex_.items() if k == 'benchmark_series'}})
            print(f"  {label}中性: {h.total_return:+.1%} 夏普{h.sharpe:.2f} "
                  f"市场相关{h.market_correlation:+.2f} 回撤{h.max_drawdown:.1%}", flush=True)


if __name__ == '__main__':
    main()
