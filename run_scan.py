#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
每日选股入口 — 短线选股扫描

用法:
    python run_scan.py                          # 默认: 沪深300池 + pure_trend_follow
    python run_scan.py --strategy oversold_bounce --top 10
    python run_scan.py --universe zz500         # 中证500池
    python run_scan.py --universe all           # 全市场（较慢）
    python run_scan.py --market-env             # 附带市场环境评分

输出: 控制台表格 + results/daily_scan_YYYYMMDD_HHMMSS.md
"""

import argparse
import sys
import os
from datetime import date, timedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd

from data.akshare import AKShareDataSource
from selection import StockScanner, load_strategy_from_yaml
from risk.constants import IndicatorParams
from utils.logger import get_logger

logger = get_logger(__name__)

# 指数池配置：universe 名 → 中证指数代码
UNIVERSE_INDEX = {
    'hs300': '000300',   # 沪深300
    'zz500': '000905',   # 中证500
    'zz1000': '000852',  # 中证1000
}


def load_universe(universe: str) -> list:
    """
    加载股票池

    Args:
        universe: hs300/zz500/zz1000 指数成分池，或 all 全市场

    Returns:
        标准代码列表（600519.SH 形式）
    """
    ds = AKShareDataSource()
    if universe == 'all':
        return ds.list_symbols()

    index_code = UNIVERSE_INDEX.get(universe)
    if index_code is None:
        raise ValueError(f"Unknown universe '{universe}'. Available: {list(UNIVERSE_INDEX)} + all")

    import akshare as ak
    df = ak.index_stock_cons_csindex(symbol=index_code)
    code_col = '成分券代码' if '成分券代码' in df.columns else df.columns[4]
    symbols = []
    for code in df[code_col].astype(str):
        if code.startswith('6'):
            symbols.append(code + '.SH')
        elif code.startswith(('4', '8')):
            symbols.append(code + '.BJ')
        else:
            symbols.append(code + '.SZ')
    return symbols


def scan(universe_symbols: list, strategy_name: str, top_n: int,
         with_market_env: bool) -> pd.DataFrame:
    """
    执行扫描

    Returns:
        DataFrame: symbol, score, 条件明细
    """
    ds = AKShareDataSource()
    scanner = StockScanner(ds)

    yaml_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             'selection', 'strategies.yml')
    config = load_strategy_from_yaml(strategy_name, yaml_path)
    print(f"策略: {config.name} ({strategy_name})")
    print(f"股票池: {len(universe_symbols)} 只 | 条件数: {len(config.indicators)} | "
          f"min_score: {config.min_score} | 扫描日期: {date.today()}")

    results = scanner.scan(universe_symbols, config, ref_date=date.today())

    # 市场环境评分（可选）
    env_lines = []
    if with_market_env:
        try:
            from market_env import AkShareMarketEnvProvider
            from market_env.scorer import MarketEnvScorer
            scorer = MarketEnvScorer(AkShareMarketEnvProvider())
            env = scorer.score(date.today().strftime('%Y%m%d'))
            env_lines = [
                f"市场综合分: {env.market_score} | 建议仓位系数: {env.position_factor}",
                f"宽度: {env.breadth_signal} | 估值: {env.valuation_signal} | "
                f"宏观: {env.macro_signal} | 先行: {env.leading_signal} | 季节性: {env.seasonal_signal}",
            ]
            for line in env_lines:
                print(f"  {line}")
        except Exception as e:
            logger.warning(f"Market env scoring failed: {e}")

    print(f"\n通过筛选: {len(results)} 只，展示前 {min(top_n, len(results))} 只:")
    return results[:top_n], env_lines


def to_markdown(picks, strategy_name, universe, env_lines) -> str:
    """扫描结果转 Markdown"""
    lines = [f"# 每日选股 - {date.today()}", ""]
    lines.append(f"- 策略: `{strategy_name}` | 股票池: `{universe}`")
    if env_lines:
        lines.append(f"- {env_lines[0]}")
        lines.append(f"- {env_lines[1]}")
    lines.append("")
    if not picks:
        lines.append("*今日无标的通过筛选*")
        return "\n".join(lines)

    lines.append("| 排名 | 代码 | 评分 | 满足条件（含强度） |")
    lines.append("|------|------|------|--------------------|")
    for i, s in enumerate(picks, 1):
        conds = "; ".join(
            f"[{c['indicator']}] {c['operator']} {c['threshold']} (强度={c['strength']})"
            for c in s.conditions
        )
        lines.append(f"| {i} | {s.symbol} | {s.score} | {conds} |")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description='Daily stock scan')
    parser.add_argument('--strategy', default='pure_trend_follow',
                        help='策略名（strategies.yml，默认 pure_trend_follow）')
    parser.add_argument('--top', type=int, default=5, help='展示前N只（默认5）')
    parser.add_argument('--universe', default='hs300',
                        choices=list(UNIVERSE_INDEX) + ['all'],
                        help='股票池（默认 hs300）')
    parser.add_argument('--market-env', action='store_true', help='附带市场环境评分')
    args = parser.parse_args()

    universe_symbols = load_universe(args.universe)
    picks, env_lines = scan(universe_symbols, args.strategy, args.top, args.market_env)

    if picks:
        print(f"\n{'排名':<4}{'代码':<12}{'评分':<6}满足条件")
        print("-" * 80)
        for i, s in enumerate(picks, 1):
            conds = "; ".join(
                f"[{c['indicator']}] {c['operator']} {c['threshold']} (强度={c['strength']})"
                for c in s.conditions
            )
            print(f"{i:<4}{s.symbol:<12}{s.score:<6}{conds}")
    else:
        print("\n今日无标的通过筛选")

    # 保存 Markdown
    report_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
    os.makedirs(report_dir, exist_ok=True)
    from datetime import datetime
    md_path = os.path.join(
        report_dir,
        f"daily_scan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    )
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(to_markdown(picks, args.strategy, args.universe, env_lines))
    print(f"\n报告已保存: {md_path}")


if __name__ == '__main__':
    main()
