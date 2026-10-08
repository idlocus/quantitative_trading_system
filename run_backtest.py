#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
量化交易系统 - 回测入口

用法: python run_backtest.py [--strategy STRATEGY_NAME]
"""

import argparse
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.system_config import SystemConfig
from backtesting.runner import BacktestRunner
from utils.logger import get_logger

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description='Run quantitative trading backtest')
    parser.add_argument(
        '--strategy',
        default='momentum_with_sector_filter',
        help='Strategy name in strategies.yml (default: momentum_with_sector_filter)'
    )
    parser.add_argument(
        '--sector-filter',
        action='store_true',
        help='Enable SW industry sector filter'
    )
    parser.add_argument(
        '--top-sectors',
        type=int,
        default=10,
        help='Number of top sectors to filter (default: 10)'
    )
    parser.add_argument(
        '--rolling',
        action='store_true',
        help='Run rolling-window backtest (IS/OOS validation)'
    )
    parser.add_argument(
        '--rolling-window',
        type=int,
        default=252,
        help='Rolling train window in trading days (default: 252)'
    )
    parser.add_argument(
        '--oos-ratio',
        type=float,
        default=0.2,
        help='Out-of-sample ratio of the window (default: 0.2)'
    )
    args = parser.parse_args()

    config = SystemConfig()
    runner = BacktestRunner(config)

    # 板块筛选
    allowed_symbols = None
    if args.sector_filter or config.oracle_dsn:
        oracle_cfg = config.oracle_config()
        if oracle_cfg['dsn'] and oracle_cfg['user'] and oracle_cfg['password']:
            allowed_symbols = runner.compute_sector_allowed_symbols(top_n=args.top_sectors)
            if not allowed_symbols:
                logger.info("Sector filter: disabled (failed to get data)")
                allowed_symbols = None
        else:
            logger.info("Sector filter: skipped (ORACLE_* env vars not set)")

    # 运行回测
    if args.rolling:
        result = runner.run_rolling(
            args.strategy,
            allowed_symbols=allowed_symbols,
            train_window_days=args.rolling_window,
            oos_ratio=args.oos_ratio
        )
    else:
        result = runner.run(args.strategy, allowed_symbols=allowed_symbols)

    # 打印结果摘要
    print("\n" + "=" * 60)
    print("Backtest Results")
    print("=" * 60)
    print(f"Initial capital:  {result.initial_capital:,.0f}")
    print(f"Final capital:    {result.final_capital:,.0f}")
    print(f"Total return:     {result.total_return*100:+.1f}%")
    print(f"Annual return:    {result.annual_return*100:+.1f}%")
    print(f"Sharpe ratio:     {result.sharpe_ratio:.2f}")
    print(f"Max drawdown:     {result.max_drawdown*100:.1f}%")
    print(f"Win rate:         {result.win_rate*100:.1f}%")
    print(f"Trade count:      {result.trade_count}")
    print(f"Avg holding days: {result.avg_holding_days:.1f}")
    print("=" * 60)


if __name__ == "__main__":
    main()