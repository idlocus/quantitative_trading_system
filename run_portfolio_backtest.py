#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
组合回测 CLI —— 因子驱动 TopN 组合（Phase 3：向指增形态收敛）

用法:
    # 反转组合（MFI+RSI+CCI 反向），PIT 池前500只，Top50，5日调仓
    python run_portfolio_backtest.py --universe 500 --top 50 --rebalance 5 \
        --factors mfi,rsi,cci --direction negative

    # 对照组：全池等权（不做因子选择）
    python run_portfolio_backtest.py --universe 500 --top 500 --rebalance 5 \
        --factors mfi,rsi,cci --direction negative --all-equal
"""

import argparse
import os
import sys
from datetime import date, timedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.layered import LayeredDataSource
from data.base.data_loader import DataLoader
from indicators.registry import IndicatorRegistry
from portfolio import (PortfolioBacktester, PortfolioConfig,
                       composite_score, equal_weights)
from utils.logger import get_logger

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description='Factor-driven TopN portfolio backtest')
    parser.add_argument('--universe', type=int, default=500, help='PIT 池规模（代码序前N，默认500）')
    parser.add_argument('--top', type=int, default=50, help='目标持仓数（默认50）')
    parser.add_argument('--rebalance', type=int, default=5, help='调仓周期交易日（默认5）')
    parser.add_argument('--years', type=float, default=3.0, help='回看年数（默认3）')
    parser.add_argument('--factors', default='mfi,rsi,cci', help='逗号分隔的因子注册名')
    parser.add_argument('--direction', choices=['positive', 'negative'], default='negative',
                        help='因子方向：negative=反转（买低动量）')
    parser.add_argument('--weight', choices=['equal', 'inverse_vol'], default='equal')
    parser.add_argument('--slippage', type=float, default=0.001)
    parser.add_argument('--all-equal', action='store_true',
                        help='对照组：不选股，全池等权（top=universe 时等同）')
    parser.add_argument('--fundamental', default=None,
                        help='基本面因子模式：逗号分隔（ep/bp/sp/dp/size），与 --factors 互斥')
    parser.add_argument('--band', default=None,
                        help='选股分数带 lo,hi（如 0.2,0.8 规避极端尾部；默认不过滤）')
    parser.add_argument('--save-report', action='store_true', help='保存 Markdown 报告')
    args = parser.parse_args()

    # 1. 数据：Wind PIT 池（含退市股）+ 基准指数
    ds = LayeredDataSource()
    end = date.today()
    start = end - timedelta(days=int(args.years * 365))
    pit = sorted(ds.list_symbols(as_of=start))[:args.universe]
    preload_symbols = pit + ['000300.SH']  # 基准指数
    print(f"Wind PIT universe: {len(pit)} 只 | {start} ~ {end}")

    loader = DataLoader.__new__(DataLoader)
    loader.data_source_type = 'layered'
    loader.data_source = ds
    loader._cache = None
    cache_obj = loader.preload_klines(preload_symbols, start, end)
    cache = cache_obj._cache
    print(f"已缓存: {len(cache)} 只")

    # 2. 因子合成（direction=negative → 买因子值低的，即反转）
    if args.fundamental:
        # 基本面模式：外部矩阵合成（ep/bp/sp/dp/size）
        from data.fundamentals import (WindFundamentalProvider, FUNDAMENTAL_FACTORS,
                                       build_fundamental_matrix)
        from portfolio import composite_from_matrices
        fnames = [f.strip() for f in args.fundamental.split(',') if f.strip()]
        for f in fnames:
            if f not in FUNDAMENTAL_FACTORS:
                print(f"未知基本面因子 '{f}'。可用: {list(FUNDAMENTAL_FACTORS)}")
                sys.exit(1)
        provider = WindFundamentalProvider(ds)
        matrices, directions = {}, {}
        for f in fnames:
            field, _, default_dir, desc = FUNDAMENTAL_FACTORS[f]
            print(f"构建基本面因子 {f}（{desc}）...")
            fm = build_fundamental_matrix(provider, f, pit, start, end)
            if not fm.empty:
                matrices[f] = fm
                directions[f] = default_dir
        print(f"合成综合分: {list(matrices)} directions={directions} ...")
        score_mx = composite_from_matrices(matrices, directions=directions)
    else:
        factor_names = [f.strip() for f in args.factors.split(',') if f.strip()]
        for f in factor_names:
            if f not in IndicatorRegistry.list_all():
                print(f"未知因子 '{f}'。可用: {IndicatorRegistry.list_all()}")
                sys.exit(1)
        factor_funcs = {f: IndicatorRegistry.get(f) for f in factor_names}
        direction = -1.0 if args.direction == 'negative' else 1.0
        directions = {f: direction for f in factor_names}

        print(f"合成综合分: {factor_names} direction={args.direction} ...")
        score_mx = composite_score(cache, factor_funcs, directions=directions)
    if score_mx.empty:
        print("综合分矩阵为空")
        sys.exit(1)
    print(f"综合分矩阵: {score_mx.shape[0]} 天 × {score_mx.shape[1]} 只")

    # 3. 组合回测
    score_band = None
    if args.band:
        lo, hi = [float(x) for x in args.band.split(',')]
        score_band = (lo, hi)
        print(f"选股分数带: [{lo}, {hi}]")

    if args.weight == 'inverse_vol':
        from portfolio import inverse_vol_weights
        bt = PortfolioBacktester(cache, PortfolioConfig(
            top_n=args.top, rebalance_days=args.rebalance,
            slippage=args.slippage, weight_method=args.weight,
        ))
        result = bt.run(score_mx, weight_func=lambda syms, d: inverse_vol_weights(
            bt.close_mx, syms, d, max_weight=0.05), score_band=score_band)
    else:
        bt = PortfolioBacktester(cache, PortfolioConfig(
            top_n=args.top, rebalance_days=args.rebalance,
            slippage=args.slippage,
        ))
        result = bt.run(score_mx, score_band=score_band)

    print(result.summary_markdown())

    if args.save_report:
        report_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
        os.makedirs(report_dir, exist_ok=True)
        from datetime import datetime
        path = os.path.join(report_dir,
                            f"portfolio_{'+'.join(factor_names)}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md")
        with open(path, 'w', encoding='utf-8') as f:
            f.write(result.summary_markdown())
        print(f"报告已保存: {path}")


if __name__ == '__main__':
    main()
