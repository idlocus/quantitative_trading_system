#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
因子评价 CLI —— 任何指标进入策略前，先测量预测力

用法:
    python run_factor_eval.py --factor rsi --period 5
    python run_factor_eval.py --factor rsi --period 1 5 10 20   # 多前瞻期对比
    python run_factor_eval.py --factor macd --quantiles 5 --start 2022-01-01

输出: 控制台摘要 + results/factor_eval_*.md
"""

import argparse
import os
import sys
from datetime import date, timedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.akshare import AKShareDataSource
from data.base.data_loader import DataLoader
from indicators.registry import IndicatorRegistry
from research.factor_eval import FactorEvaluator
from utils.logger import get_logger

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description='Factor evaluation (IC / layers / turnover)')
    parser.add_argument('--factor', required=True, help='指标注册名（如 rsi / macd / atr / obv）')
    parser.add_argument('--period', type=int, nargs='+', default=[5],
                        help='前瞻收益期数（交易日），可多个（默认 5）')
    parser.add_argument('--quantiles', type=int, default=5, help='分层数（默认5）')
    parser.add_argument('--universe', type=int, default=100,
                        help='股票池大小：全市场代码序前N只（默认100，无偏但非随机）')
    parser.add_argument('--years', type=float, default=3.0, help='回看年数（默认3）')
    parser.add_argument('--start', default=None, help='评价开始日期 yyyy-MM-dd')
    parser.add_argument('--end', default=None, help='评价结束日期 yyyy-MM-dd')
    parser.add_argument('--source', choices=['akshare', 'wind'], default='akshare',
                        help='数据源（wind = 本机WindPy，PIT股票池含退市股）')
    args = parser.parse_args()

    # 可用指标列表提示
    available = IndicatorRegistry.list_all()
    if args.factor not in available:
        print(f"未知指标 '{args.factor}'。可用: {available}")
        sys.exit(1)

    # 数据：无偏抽样（代码序前N）+ 回看窗口
    end = date.today()
    start = end - timedelta(days=int(args.years * 365))

    if args.source == 'wind':
        from data.layered import LayeredDataSource
        ds = LayeredDataSource()
        loader = DataLoader.__new__(DataLoader)
        loader.data_source_type = 'layered'
        loader.data_source = ds
        loader._cache = None
        # PIT 全A：评价起始日的时点成分（含后来退市的股票，消除幸存者偏差）
        pit_symbols = ds.list_symbols(as_of=start)
        symbols = sorted(pit_symbols)[:args.universe]
        src_desc = f"Wind PIT 全A（含退市）代码序前 {len(symbols)} 只"
    else:
        ds = AKShareDataSource()
        loader = DataLoader.__new__(DataLoader)
        loader.data_source_type = 'akshare'
        loader.data_source = ds
        loader._cache = None
        symbols = sorted(ds.list_symbols())[:args.universe]
        src_desc = f"akshare 代码序前 {len(symbols)} 只（幸存者偏差 caveat）"

    print(f"因子: {args.factor} | 股票池: {src_desc} | 区间: {start} ~ {end}")

    cache_obj = loader.preload_klines(symbols, start, end)
    cache = cache_obj._cache  # symbol -> DataFrame 原始缓存
    print(f"已缓存: {len(cache)} 只")

    factor_func = IndicatorRegistry.get(args.factor)
    evaluator = FactorEvaluator(cache)

    md = [f"# 因子评价报告 - {args.factor}", ""]
    md.append(f"- 股票池: 全市场代码序前 {len(symbols)} 只（akshare 现存股票，含幸存者偏差）")
    md.append(f"- 区间: {start} ~ {end} | 分层数: {args.quantiles}\n")

    for period in args.period:
        print(f"\n评价中: {args.factor} @ {period}日前瞻 ...")
        result = evaluator.evaluate(factor_func, name=args.factor,
                                    period=period, quantiles=args.quantiles,
                                    start=args.start, end=args.end)
        if result is None:
            print(f"  数据不足，跳过 period={period}")
            continue
        print(result.summary_markdown())
        md.append(result.summary_markdown())
        md.append("")

    if len(md) > 4:
        report_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
        os.makedirs(report_dir, exist_ok=True)
        from datetime import datetime as dt
        path = os.path.join(report_dir, f"factor_eval_{args.factor}_{dt.now().strftime('%Y%m%d_%H%M%S')}.md")
        with open(path, 'w', encoding='utf-8') as f:
            f.write("\n".join(md))
        print(f"\n报告已保存: {path}")


if __name__ == '__main__':
    main()
