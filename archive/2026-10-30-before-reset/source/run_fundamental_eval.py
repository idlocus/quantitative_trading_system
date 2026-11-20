#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基本面因子评价 CLI —— 价值/红利/市值因子在 PIT 池上的 IC 与分层

用法:
    python run_fundamental_eval.py --universe 500 --years 3 --period 10 20 60
"""

import argparse
import os
import sys
from datetime import date, timedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd

from data.layered import LayeredDataSource
from data.base.data_loader import DataLoader
from data.fundamentals import (WindFundamentalProvider, FUNDAMENTAL_FACTORS,
                               build_fundamental_matrix)
from research.factor_eval import FactorEvaluator
from utils.logger import get_logger

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description='Fundamental factor evaluation')
    parser.add_argument('--universe', type=int, default=500)
    parser.add_argument('--years', type=float, default=3.0)
    parser.add_argument('--period', type=int, nargs='+', default=[10, 20, 60],
                        help='前瞻期（交易日），可多个')
    parser.add_argument('--quantiles', type=int, default=5)
    args = parser.parse_args()

    ds = LayeredDataSource()
    end = date.today()
    start = end - timedelta(days=int(args.years * 365))
    pit = sorted(ds.list_symbols(as_of=start))[:args.universe]
    print(f"Wind PIT universe: {len(pit)} 只 | {start} ~ {end}")

    loader = DataLoader.__new__(DataLoader)
    loader.data_source_type = 'layered'
    loader.data_source = ds
    loader._cache = None
    cache_obj = loader.preload_klines(pit, start, end)
    cache = cache_obj._cache
    print(f"K线已缓存: {len(cache)} 只")

    provider = WindFundamentalProvider(ds)
    evaluator = FactorEvaluator(cache)

    rows = []
    for fname, (field, _, default_dir, desc) in FUNDAMENTAL_FACTORS.items():
        print(f"\n===== {fname}（{desc}，字段 {field}）=====")
        try:
            fm = build_fundamental_matrix(provider, fname, pit, start, end)
        except Exception as e:
            print(f"  构建失败: {e}")
            continue
        if fm.empty:
            print("  矩阵为空，跳过")
            continue
        for period in args.period:
            result = evaluator.evaluate_matrix(fm, name=f"{fname}@{period}d",
                                               period=period, quantiles=args.quantiles)
            if result is None:
                print(f"  {period}日: 数据不足")
                continue
            q_rets = result.quantile_mean_ret
            q_hi = q_rets.get(max(q_rets), 0) if q_rets else 0
            q_lo = q_rets.get(min(q_rets), 0) if q_rets else 0
            rows.append({
                'factor': fname, 'period': period,
                'rankic': result.rankic_mean, 'ir': result.rankic_ir,
                'ls_annual': result.long_short_annual,
                'Q1': q_lo, 'Q5': q_hi,
                'turnover': result.turnover,
            })
            print(f"  {period:3d}日 RankIC={result.rankic_mean:+.4f} IR={result.rankic_ir:+.2f} "
                  f"多空年化={result.long_short_annual:+.1%} "
                  f"Q1={q_lo:+.3%} Q5={q_hi:+.3%}")

    if rows:
        df = pd.DataFrame(rows)
        print("\n" + "=" * 80)
        print(df.to_string(index=False, float_format=lambda x: f'{x:+.4f}'))
        report_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
        os.makedirs(report_dir, exist_ok=True)
        from datetime import datetime
        path = os.path.join(report_dir, f"fundamental_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md")
        with open(path, 'w', encoding='utf-8') as f:
            f.write(f"# 基本面因子评价（PIT 池 {len(pit)} 只，{start} ~ {end}）\n\n")
            f.write("> caveat: Wind 日频估值字段按当日已知财报计算，存在少量回填风险\n\n")
            f.write(df.to_markdown(index=False, floatfmt='+.4f'))
        print(f"\n报告已保存: {path}")


if __name__ == '__main__':
    main()
