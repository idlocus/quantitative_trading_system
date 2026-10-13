#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
主链 v5 工件生成 —— 新语义（T-1信号/上市天数/流动性/ST/缓冲带/成本冲击/风控闸门）
的完整收益结果，两年代各生成两口径：

  composite_v5_{era}_5因子生 ：升级版 PortfolioBacktester（20日近似 + 全部新约束，
                                含市场冲击/参与率；与 v4 同引擎家族可横向对比）
  replay_v5_{era}_5因子生    ：DailyReplayBacktester（严格生产语义：逐日重放
                                build_orders/RiskGate/PaperBroker/Ledger）

v4 工件保留不动（webapp 无 v5 时回退 v4 的逻辑不变）。
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
from portfolio.daily_replay import DailyReplayBacktester, ReplayConfig
from research.artifacts import save_artifact, portfolio_payload
from trading import config as chain

D5 = chain.PRODUCTION_FACTORS
ERAS = [('oos_2017_2021', date(2016, 11, 1), date(2021, 12, 31), date(2017, 1, 1)),
        ('fullmarket_2022_2026', date(2021, 11, 1), date(2026, 10, 9), date(2022, 1, 1))]
BENCHES = ['000300.SH', '000905.SH', '000852.SH']


def build_score(ds, pool, d0, d1):
    prov = WindFundamentalProvider(ds)
    pit = PITFinancialProvider(ds)
    mats = {}
    for f in ['ep', 'bp', 'sp', 'size']:
        fm = build_fundamental_matrix(prov, f, pool, d0, d1)
        if not fm.empty:
            mats[f] = fm.loc[pd.Timestamp(d0):pd.Timestamp(d1)]
    lev = pit.get_factor_matrix('leverage', pool, d0, d1)
    if not lev.empty:
        lev = lev.loc[pd.Timestamp(d0):pd.Timestamp(d1)]
        mats['leverage'] = lev
    return composite_from_matrices(mats, directions=D5)


def bench_series_payload(bt, result):
    b = bt.close_mx['000300.SH'].loc[result.equity_curve.index].ffill()
    return {'benchmark_series': {'dates': [str(d)[:10] for d in b.index],
            'values': [round(float(v / b.iloc[0] * result.initial_capital), 2)
                       for v in b.values]}}


def main():
    args = sys.argv[1:]
    only = args[0].split('=')[1] if args and args[0].startswith('--era=') else None
    part = args[1].split('=')[1] if len(args) > 1 and args[1].startswith('--part=') else \
        (args[0].split('=')[1] if args and args[0].startswith('--part=') else None)
    ds = LayeredDataSource()
    loader = DataLoader.__new__(DataLoader)
    loader.data_source_type = 'layered'
    loader.data_source = ds
    loader._cache = None

    for era, d0, d1, ts in ERAS:
        if only and era != only:
            continue
        print(f"===== {era} =====", flush=True)
        pool = sorted(ds.list_symbols(as_of=ts))
        print(f"  PIT池 {len(pool)} 只 | warm-up {d0} ~ {d1} | 统计起 {ts}", flush=True)
        cache = loader.preload_klines(pool + BENCHES, d0, d1)._cache

        # --part=c 跳过因子矩阵与A/B，仅重生成可交易等权对照（对照用常数分数）
        if part == 'c':
            from portfolio.tradable_benchmark import tradable_equal_weight
            cfg = PortfolioConfig(top_n=chain.TOP_N, rebalance_days=20)
            try:
                from data.st_provider import STProvider
                idx = pd.DatetimeIndex(pd.to_datetime(
                    next(iter(cache.values()))['date']))
                cfg = PortfolioConfig(top_n=chain.TOP_N, rebalance_days=20,
                                      st_matrix=STProvider(ds).st_matrix(idx, pool))
            except Exception as e:
                print(f"  警告：ST矩阵不可用（{e}）", flush=True)
            cbt, rC, cap_scale = tradable_equal_weight(cache, cfg,
                                                       start_date=str(ts), end_date=str(d1))
            save_artifact('portfolio', f'equaltradable_v5_{era}_等权',
                          portfolio_payload(rC, label=f'可交易等权对照(同约束,资金×{cap_scale})',
                                            extra=bench_series_payload(cbt, rC)))
            print(f"  [C等权可交易] {rC.total_return:+.1%} 夏普{rC.sharpe:.2f} "
                  f"回撤{rC.max_drawdown:.1%} 成本{rC.total_cost / rC.initial_capital:.2%} "
                  f"持仓{rC.config.top_n}只 资金×{cap_scale}", flush=True)
            continue

        score = build_score(ds, pool, d0, d1)
        score = score[score.columns.intersection(cache.keys())]
        print(f"  因子矩阵 {score.shape[0]}日 × {score.shape[1]}只", flush=True)

        # ST 矩阵（PIT）；不可用时与 CLI 同安全口径（st_data_required → 闸门禁买）
        st_matrix = None
        st_required = False
        try:
            from data.st_provider import STProvider
            idx = pd.DatetimeIndex(pd.to_datetime(
                next(iter(cache.values()))['date']))
            st_matrix = STProvider(ds).st_matrix(idx, pool)
            print(f"  ST矩阵就绪（{int(st_matrix.values.sum())} 标的日）", flush=True)
        except Exception as e:
            st_required = True
            st_matrix = None
            print(f"  警告：ST矩阵不可用（{e}），重放按生产安全口径禁买", flush=True)

        # ── 口径A：升级版近似回测（含市场冲击/参与率，全约束链） ──
        if part != 'b':      # --part=b 仅重生成重放工件（审计明细更新）
            cfgA = PortfolioConfig(top_n=chain.TOP_N, rebalance_days=20,
                                   st_matrix=st_matrix,
                                   risk_limits=__import__(
                                       'portfolio.risk_manager', fromlist=['RiskLimits']
                                   ).RiskLimits())
            bt = PortfolioBacktester(cache, cfgA)
            rA = bt.run(score, start_date=str(ts))
            save_artifact('portfolio', f'composite_v5_{era}_5因子生',
                          portfolio_payload(rA, label='5因子生产(主链v5)',
                                            extra=bench_series_payload(bt, rA)))
            print(f"  [A近似] {rA.total_return:+.1%} 基准{rA.benchmark_return:+.1%} "
                  f"夏普{rA.sharpe:.2f} 回撤{rA.max_drawdown:.1%} "
                  f"风控事件{rA.risk_events}", flush=True)

        # ── 口径B：每日重放（严格生产语义） ──
        cfgB = ReplayConfig(top_n=chain.TOP_N, st_matrix=st_matrix,
                            use_risk_gate=True, st_data_required=st_required)
        rp = DailyReplayBacktester(cache, cfgB)
        rB = rp.run(score, start_date=str(ts), end_date=str(d1))
        save_artifact('portfolio', f'replay_v5_{era}_5因子生',
                      portfolio_payload(rB, label='5因子生产(每日重放v5)',
                                        extra=bench_series_payload(rp, rB)))
        print(f"  [B重放] {rB.total_return:+.1%} 基准{rB.benchmark_return:+.1%} "
              f"夏普{rB.sharpe:.2f} 回撤{rB.max_drawdown:.1%} "
              f"成本{rB.total_cost / rB.initial_capital:.2%} "
              f"止损{rB.n_stop_sells}笔 停机{rB.n_halt_days}天", flush=True)
        # ── 口径C：可交易等权对照（同约束真实回测：过滤/涨跌停/成本/冲击/缓冲带/现金底线） ──
        # ── 口径C：可交易等权对照（同约束真实回测：过滤/涨跌停/成本/冲击/缓冲带/现金底线） ──
        if part != 'b':
            from portfolio.tradable_benchmark import tradable_equal_weight
            cbt, rC, cap_scale = tradable_equal_weight(cache, cfgA, start_date=str(ts),
                                                       end_date=str(d1))
            save_artifact('portfolio', f'equaltradable_v5_{era}_等权',
                          portfolio_payload(rC, label=f'可交易等权对照(同约束,资金×{cap_scale})',
                                            extra=bench_series_payload(cbt, rC)))
            print(f"  [C等权可交易] {rC.total_return:+.1%} 夏普{rC.sharpe:.2f} "
                  f"回撤{rC.max_drawdown:.1%} 成本{rC.total_cost / rC.initial_capital:.2%} "
                  f"持仓{rC.config.top_n}只 资金×{cap_scale} 风控事件{rC.risk_events}", flush=True)

        if part != 'b':      # part=b 时 rA 未跑，A/B差异行跳过
            print(f"  A/B差异: {rA.total_return - rB.total_return:+.1%}", flush=True)


if __name__ == '__main__':
    main()
