#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量因子筛选：预加载一次 Wind PIT 数据，逐个测量指标库中的因子预测力
输出按 |RankIC| 排序的筛选报告（淘汰无效因子、保留有预测力的）

用法: python ai-created/factor_screen.py [universe_size] [years]
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datetime import date, timedelta

import pandas as pd

from data.wind import WindDataSource
from data.base.data_loader import DataLoader
from indicators.registry import IndicatorRegistry
from research.factor_eval import FactorEvaluator
from utils.logger import get_logger

logger = get_logger(__name__)

# 参与筛选的因子（注册名 → 说明）；排除形态类（非时序函数）与已弃用项
SCREEN_FACTORS = [
    'rsi', 'roc', 'cci', 'atr', 'std', 'obv', 'ad_line', 'cmf', 'mfi', 'vwap',
    'macd', 'sma', 'ema', 'bb', 'stochastic', 'keltner', 'donchian', 'goldcross',
]


def main(n_universe=100, years=3.0, period=5):
    ds = WindDataSource()
    end = date.today()
    start = end - timedelta(days=int(years * 365))

    pit = sorted(ds.list_symbols(as_of=start))[:n_universe]
    print(f"Wind PIT universe: {len(pit)} 只 | {start} ~ {end} | 前瞻期 {period} 日")

    loader = DataLoader.__new__(DataLoader)
    loader.data_source_type = 'wind'
    loader.data_source = ds
    loader._cache = None
    cache_obj = loader.preload_klines(pit, start, end)
    cache = cache_obj._cache
    print(f"已缓存: {len(cache)} 只\n")

    evaluator = FactorEvaluator(cache)
    rows = []
    for name in SCREEN_FACTORS:
        try:
            func = IndicatorRegistry.get(name)
        except KeyError:
            continue
        result = evaluator.evaluate(func, name=name, period=period, quantiles=5)
        if result is None:
            print(f"  {name:12s} 数据不足，跳过")
            continue
        rows.append({
            'factor': name,
            'rankic': result.rankic_mean,
            'rankic_ir': result.rankic_ir,
            'ic': result.ic_mean,
            'ls_annual': result.long_short_annual,
            'monotonic': '单调' if _is_monotonic(result) else '不单调',
            'turnover': result.turnover,
        })
        print(f"  {name:12s} RankIC={result.rankic_mean:+.4f} IR={result.rankic_ir:+.2f} "
              f"多空年化={result.long_short_annual:+.1%}")

    if not rows:
        print("无有效结果")
        return

    df = pd.DataFrame(rows).sort_values('rankic', key=lambda s: s.abs(), ascending=False)

    print("\n" + "=" * 76)
    print(f"因子筛选排名（按 |RankIC| 降序，前瞻 {period} 日，PIT 池 {len(cache)} 只）")
    print("=" * 76)
    print(df.to_string(index=False, float_format=lambda x: f'{x:+.4f}'))

    report_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
    os.makedirs(report_dir, exist_ok=True)
    from datetime import datetime
    path = os.path.join(report_dir, f"factor_screen_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md")
    with open(path, 'w', encoding='utf-8') as f:
        f.write(f"# 批量因子筛选（Wind PIT 池 {len(pit)} 只，{start} ~ {end}，前瞻 {period} 日）\n\n")
        f.write(df.to_markdown(index=False, floatfmt='+.4f'))
    print(f"\n报告已保存: {path}")


def _is_monotonic(result) -> bool:
    rets = [result.quantile_mean_ret[q] for q in sorted(result.quantile_mean_ret)]
    return (all(rets[i] <= rets[i + 1] for i in range(len(rets) - 1)) or
            all(rets[i] >= rets[i + 1] for i in range(len(rets) - 1)))


if __name__ == '__main__':
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    y = float(sys.argv[2]) if len(sys.argv) > 2 else 3.0
    main(n, y)
