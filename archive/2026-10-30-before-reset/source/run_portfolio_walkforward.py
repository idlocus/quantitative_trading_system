#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
组合滚动样本外验证（walk-forward）

规则（无后视）：
- 测试窗口 = 每个自然年（2022..2026）
- 因子选择只用测试窗开始前 3 年的数据：候选=全部基本面因子，
  按 60 日 RankIC 的 |IR| 取前二，方向 = RankIC 符号；无有效因子则等权
- 同窗口跑等权对照与沪深300基准

用法: python run_portfolio_walkforward.py --universe 500 --top 100
"""

import argparse
import os
import sys
from datetime import date, timedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

from data.layered import LayeredDataSource
from data.base.data_loader import DataLoader
from data.fundamentals import (WindFundamentalProvider, FUNDAMENTAL_FACTORS,
                               build_fundamental_matrix)
from portfolio import PortfolioBacktester, PortfolioConfig, composite_from_matrices
from research.factor_eval import FactorEvaluator
from utils.logger import get_logger

logger = get_logger(__name__)


def run_fold(cache, close_mx, score_mx, cfg, start, end):
    """在 [start, end] 窗口上跑组合，返回区间收益率"""
    fold_score = score_mx.loc[start:end]
    if fold_score.empty:
        return None
    bt = PortfolioBacktester(cache, PortfolioConfig(**cfg))
    result = bt.run(fold_score)
    return result


def main():
    parser = argparse.ArgumentParser(description='Walk-forward portfolio validation')
    parser.add_argument('--universe', type=int, default=500)
    parser.add_argument('--top', type=int, default=100)
    parser.add_argument('--rebalance', type=int, default=20)
    parser.add_argument('--eval-years', type=int, default=3)
    parser.add_argument('--folds', default='2022,2023,2024,2025,2026')
    parser.add_argument('--ic-period', type=int, default=60)
    args = parser.parse_args()

    folds = [int(y) for y in args.folds.split(',')]
    data_start = date(folds[0] - args.eval_years - 1, 1, 1)
    data_end = date.today()

    ds = LayeredDataSource()
    pit = sorted(ds.list_symbols(as_of=date(folds[0], 1, 1)))[:args.universe]
    print(f"PIT universe@{folds[0]}-01-01: {len(pit)} 只 | 数据 {data_start} ~ {data_end}")

    loader = DataLoader.__new__(DataLoader)
    loader.data_source_type = 'layered'
    loader.data_source = ds
    loader._cache = None
    preload = pit + ['000300.SH']
    cache_obj = loader.preload_klines(preload, data_start, data_end)
    cache = cache_obj._cache
    print(f"K线已缓存: {len(cache)} 只")

    provider = WindFundamentalProvider(ds)
    matrices = {}
    for fname in FUNDAMENTAL_FACTORS:
        fm = build_fundamental_matrix(provider, fname, pit, data_start, data_end)
        if not fm.empty:
            matrices[fname] = fm
            print(f"因子矩阵 {fname}: {fm.shape}")

    evaluator = FactorEvaluator(cache)
    cfg_base = dict(top_n=args.top, rebalance_days=args.rebalance, slippage=0.001)
    ctrl_cfg = dict(top_n=len(pit), rebalance_days=args.rebalance, slippage=0.001)

    rows = []
    cum_strat = cum_ctrl = cum_bench = 1.0
    for year in folds:
        fold_start, fold_end = date(year, 1, 1), min(date(year, 12, 31), data_end)
        eval_start = date(year - args.eval_years, 1, 1)
        eval_end = date(year - 1, 12, 31)

        # 1. 因子选择（只用 eval 窗口）
        scored = []
        for fname, fm in matrices.items():
            r = evaluator.evaluate_matrix(fm, name=fname, period=args.ic_period,
                                          start=eval_start.strftime('%Y-%m-%d'),
                                          end=eval_end.strftime('%Y-%m-%d'))
            if r is not None and abs(r.rankic_ir) > 0.05:
                scored.append((fname, r.rankic_mean, abs(r.rankic_ir)))
        scored.sort(key=lambda x: -x[2])
        chosen = scored[:2]
        selection = ', '.join(f"{n}({'+' if ic > 0 else '-'})" for n, ic, _ in chosen) or '等权(无有效因子)'

        # 2. 构建综合分并跑测试窗
        if chosen:
            dirs = {n: (1.0 if ic > 0 else -1.0) for n, ic, _ in chosen}
            sub = {n: matrices[n].loc[fold_start:fold_end] for n, _, _ in chosen}
            score_mx = composite_from_matrices(sub, directions=dirs)
        else:
            score_mx = None

        # 基准（同窗口沪深300）
        bench_ret = None
        if '000300.SH' in evaluator.close_mx.columns:
            cm = evaluator.close_mx['000300.SH'].loc[str(year)].dropna()
            if len(cm) >= 2:
                bench_ret = cm.iloc[-1] / cm.iloc[0] - 1

        if score_mx is not None and not score_mx.empty:
            res = run_fold(cache, None, score_mx, cfg_base, fold_start, fold_end)
            strat_ret = res.total_return if res else None
        else:
            strat_ret = None

        # 等权对照（全1分数矩阵 → 全池入选）
        any_mx = matrices[list(matrices)[0]]
        ctrl_full = pd.DataFrame(1.0, index=any_mx.loc[fold_start:fold_end].index,
                                 columns=any_mx.columns)
        res_ctrl = run_fold(cache, None, ctrl_full, ctrl_cfg, fold_start, fold_end)
        ctrl_ret = res_ctrl.total_return if res_ctrl else None

        if strat_ret is not None: cum_strat *= (1 + strat_ret)
        if ctrl_ret is not None: cum_ctrl *= (1 + ctrl_ret)
        if bench_ret is not None: cum_bench *= (1 + bench_ret)

        rows.append({
            'year': year, 'factors': selection,
            'strategy': strat_ret, 'equal_weight': ctrl_ret, 'benchmark': bench_ret,
            'excess_vs_ew': (strat_ret - ctrl_ret) if (strat_ret is not None and ctrl_ret is not None) else None,
        })
        print(f"{year}: {selection} | 策略={f'{strat_ret:+.1%}' if strat_ret is not None else 'N/A'} "
              f"等权={f'{ctrl_ret:+.1%}' if ctrl_ret is not None else 'N/A'} "
              f"基准={f'{bench_ret:+.1%}' if bench_ret is not None else 'N/A'}")

    df = pd.DataFrame(rows)
    print("\n" + "=" * 88)
    print(df.to_string(index=False, float_format=lambda x: f'{x:+.4f}'))
    print("=" * 88)
    print(f"累计: 策略 {cum_strat-1:+.1%} | 等权 {cum_ctrl-1:+.1%} | 基准 {cum_bench-1:+.1%}")

    report_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
    os.makedirs(report_dir, exist_ok=True)
    from datetime import datetime
    path = os.path.join(report_dir, f"walkforward_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md")
    with open(path, 'w', encoding='utf-8') as f:
        f.write(f"# 组合滚动样本外验证（PIT池{len(pit)}只，{folds[0]}-{folds[-1]}）\n\n")
        f.write(df.to_markdown(index=False, floatfmt='+.4f'))
        f.write(f"\n\n**累计**: 策略 {cum_strat-1:+.2%} | 等权 {cum_ctrl-1:+.2%} | 基准 {cum_bench-1:+.2%}\n")
    print(f"报告: {path}")


if __name__ == '__main__':
    main()
