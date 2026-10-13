#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
每日重放回测 CLI —— 严格生产语义的收益验证

与 run_portfolio_backtest.py（每N日调仓近似）同数据、同因子，但逐日重放生产链路：
每日信号 → 缓冲带订单 → 风控闸门 → Paper撮合 → 账本。

用法:
    python run_daily_replay.py --universe 500 --years 3
    python run_daily_replay.py --no-gate                 # 关闸门对照
    python run_daily_replay.py --compare                 # 同数据再跑5日近似回测，输出A/B对比
因子固定为生产5因子（ep/bp/sp/size/leverage，trading/config.PRODUCTION_FACTORS）。
"""

import argparse
import os
import sys
from datetime import date, timedelta

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

from data.layered import LayeredDataSource
from data.fundamentals import WindFundamentalProvider, build_fundamental_matrix
from data.fundamental_pit import PITFinancialProvider
from live.signals import _make_loader
from portfolio.construction import composite_from_matrices
from portfolio.daily_replay import DailyReplayBacktester, ReplayConfig
from trading import config as chain


def build_score_matrix(ds, pool, start, end):
    """生产5因子矩阵（与 live/signals.generate_target 同构造）"""
    loader = _make_loader(ds)
    cache = loader.preload_klines(pool, start, end)._cache
    prov = WindFundamentalProvider(ds)
    pit = PITFinancialProvider(ds)

    mats = {}
    for f in ['ep', 'bp', 'sp', 'size']:
        fm = build_fundamental_matrix(prov, f, pool, start, end)
        if not fm.empty:
            mats[f] = fm
    lev = pit.get_factor_matrix('leverage', pool, start, end)
    if not lev.empty:
        mats['leverage'] = lev
    score_mx = composite_from_matrices(mats, directions=chain.PRODUCTION_FACTORS)
    # 分数列 ∩ K线池：选股只能在有行情的池内（否则闸门价格覆盖检查必然停机）
    score_mx = score_mx[score_mx.columns.intersection(cache.keys())]
    return score_mx, cache


def main():
    parser = argparse.ArgumentParser(description='Daily-replay backtest (strict production semantics)')
    parser.add_argument('--universe', type=int, default=500, help='PIT 池规模（代码序前N）')
    parser.add_argument('--years', type=float, default=3.0)
    parser.add_argument('--top', type=int, default=chain.TOP_N)
    parser.add_argument('--capital', type=float, default=1_000_000.0)
    parser.add_argument('--no-gate', action='store_true', help='关闭风控闸门（对照）')
    parser.add_argument('--restart', action='store_true',
                        help='硬限清仓后自动重启（默认停机等待人工决策；研究对照用）')
    parser.add_argument('--compare', action='store_true', help='同数据再跑5日近似回测做A/B对比')
    args = parser.parse_args()

    ds = LayeredDataSource()
    end = date.today()
    start = end - timedelta(days=int(args.years * 365))
    warmup = start - timedelta(days=chain.WARMUP_DAYS)
    pool = sorted(ds.list_symbols(as_of=start))[:args.universe]
    preload = pool + ['000300.SH', '000905.SH', '000852.SH']
    print(f"PIT universe: {len(pool)} 只 | {start} ~ {end}（warm-up {warmup} 起）")

    score_mx, cache = build_score_matrix(ds, preload, warmup, end)
    if score_mx.empty:
        print("因子矩阵为空（检查数据）")
        return
    print(f"因子矩阵: {score_mx.shape[0]} 日 × {score_mx.shape[1]} 只")

    # ST 矩阵（PIT，ASHAREST；不可用时降级为无ST处理并警告）
    st_matrix = None
    st_required = False
    try:
        from data.st_provider import STProvider
        idx = pd.DatetimeIndex(pd.to_datetime(next(iter(cache.values()))['date']))
        st_matrix = STProvider(ds).st_matrix(idx, list(pool))
        n_st = int(st_matrix.values.sum())
        print(f"ST矩阵: {n_st} 个标的日处于ST状态")
    except Exception as e:
        st_required = True   # 数据缺失→闸门禁买，与实盘 run_daily 同安全口径
        print(f"警告：ST数据不可用，本次回测按非ST撮合并禁止新买入（{e}）")

    from live.risk_gate import GateConfig
    cfg = ReplayConfig(top_n=args.top, initial_capital=args.capital,
                       use_risk_gate=not args.no_gate, st_matrix=st_matrix,
                       st_data_required=st_required,
                       gate_config=GateConfig(restart_after_liquidation=args.restart)
                       if not args.no_gate else None)
    print("逐日重放生产链路（build_orders → RiskGate → PaperBroker → Ledger）...")
    replay = DailyReplayBacktester(cache, cfg)
    result = replay.run(score_mx, start_date=start.isoformat())
    print(result.summary_markdown())

    if args.compare:
        from portfolio.backtester import PortfolioBacktester, PortfolioConfig
        bt = PortfolioBacktester(cache, PortfolioConfig(top_n=args.top,
                                                        st_matrix=st_matrix))
        bt_result = bt.run(score_mx, start_date=start.isoformat())
        print("\n## A/B 对比：每日重放（生产语义） vs 每5日近似回测\n")
        print(f"| 指标 | 每日重放 | 5日近似 |")
        print(f"|------|----------|---------|")
        print(f"| 总收益 | {result.total_return:+.2%} | {bt_result.total_return:+.2%} |")
        print(f"| 年化 | {result.annual_return:+.2%} | {bt_result.annual_return:+.2%} |")
        print(f"| 夏普 | {result.sharpe:.2f} | {bt_result.sharpe:.2f} |")
        print(f"| 最大回撤 | {result.max_drawdown:.2%} | {bt_result.max_drawdown:.2%} |")
        print(f"| 累计成本 | {result.total_cost:,.0f} | {bt_result.total_cost:,.0f} |")
        print(f"| 调仓次数/有成交日 | {result.n_trade_days} | {bt_result.n_rebalances} |")
        if result.n_stop_sells:
            print(f"\n注意：重放触发 L4 止损卖出 {result.n_stop_sells} 笔（5日近似回测无此语义）")


if __name__ == '__main__':
    main()
