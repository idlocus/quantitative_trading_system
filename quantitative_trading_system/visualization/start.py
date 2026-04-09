#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
启动回测结果可视化界面

用法: python start.py [股票代码]
示例: python start.py 300308.SZ
默认: 300308.SZ
"""

import os
import sys
import json
import logging
from datetime import datetime

import pandas as pd

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from visualization.backtest_visualizer import BacktestVisualizer
from strategy.trend_following import TrendFollowingStrategy
from config.system_config import SystemConfig

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_latest_backtest_results():
    """加载最新的回测结果"""
    result_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'results', 'backtests'
    )

    if not os.path.exists(result_dir):
        logger.error(f"回测结果目录不存在: {result_dir}")
        return None

    # 获取所有统计文件
    stats_files = sorted(
        [f for f in os.listdir(result_dir) if f.startswith('stats_') and f.endswith('.json')]
    )

    if not stats_files:
        logger.error("没有找到回测结果文件")
        return None

    # 使用最新的结果
    latest_stats = stats_files[-1]
    timestamp = latest_stats.replace('stats_', '').replace('.json', '')

    logger.info(f"加载回测结果: {timestamp}")

    # 加载统计数据
    with open(os.path.join(result_dir, f'stats_{timestamp}.json'), 'r') as f:
        results = json.load(f)

    # 加载权益曲线
    equity_curve = pd.read_csv(
        os.path.join(result_dir, f'equity_curve_{timestamp}.csv'),
        index_col=0,
        parse_dates=True
    )
    results['equity_curve'] = equity_curve

    # 加载交易记录
    trades = pd.read_csv(os.path.join(result_dir, f'trades_{timestamp}.csv'))
    results['trades'] = trades.to_dict('records')

    # 转换timestamp为datetime
    for trade in results['trades']:
        if isinstance(trade.get('timestamp'), str):
            trade['timestamp'] = pd.to_datetime(trade['timestamp'])

    logger.info(f"加载成功: {len(equity_curve)} 条权益数据, {len(trades)} 条交易记录")

    return results


def load_market_data(symbol='300308.SZ'):
    """从数据源加载市场数据"""
    config = SystemConfig()
    from data.market import WindMarketSource

    try:
        data_source = WindMarketSource(config)
        data = data_source.get_historical_data(symbol=symbol, timeframe='1d')
        if data is not None and not data.empty:
            data.index = pd.to_datetime(data.index)
            logger.info(f"从Wind加载市场数据成功: {symbol}, {len(data)} 条")
            return data
    except Exception as e:
        logger.warning(f"从Wind加载失败: {e}")

    # 备用：从CSV加载
    csv_paths = [
        os.path.join(os.path.dirname(os.path.dirname(__file__)), f'data_cache/{symbol}_two_years_data.csv'),
        os.path.join(os.path.dirname(os.path.dirname(__file__)), f'{symbol}_two_years_data.csv'),
        os.path.join(os.path.dirname(os.path.dirname(__file__)), f'data_cache/{symbol}_recent_data.csv'),
    ]

    for csv_path in csv_paths:
        if os.path.exists(csv_path):
            data = pd.read_csv(csv_path, index_col=0, parse_dates=True)
            logger.info(f"从CSV加载市场数据成功: {len(data)} 条")
            return data

    logger.warning("无法加载市场数据，使用空数据")
    return pd.DataFrame()


def main():
    # 从命令行参数获取股票代码
    symbol = sys.argv[1] if len(sys.argv) > 1 else '300308.SZ'

    logger.info(f"{'='*60}")
    logger.info(f"启动回测结果可视化 - {symbol}")
    logger.info(f"{'='*60}")

    try:
        # 加载回测结果
        results = load_latest_backtest_results()
        if not results:
            logger.error("加载回测结果失败")
            return

        # 加载市场数据
        market_data = load_market_data(symbol)

        # 初始化策略
        strategy = TrendFollowingStrategy(SystemConfig())
        strategy.symbol = symbol
        if not market_data.empty:
            strategy.preprocess_data(market_data)

        # 打印回测统计
        total_return = results.get('total_return', 0)
        trade_count = results.get('trade_count', 0)
        sharpe = results.get('sharpe_ratio', 0)
        max_dd = results.get('max_drawdown', 0)

        logger.info(f"\n回测统计:")
        logger.info(f"  总收益: {total_return:+.2f}%")
        logger.info(f"  交易次数: {trade_count}")
        logger.info(f"  夏普比率: {sharpe:.2f}")
        logger.info(f"  最大回撤: {max_dd:.2f}%")

        # 启动可视化器
        logger.info(f"\n启动可视化服务...")
        logger.info(f"请在浏览器中访问: http://127.0.0.1:8050/")

        visualizer = BacktestVisualizer(results, strategy, market_data)
        visualizer.run(port=8050, debug=False)

    except Exception as e:
        logger.error(f"启动可视化界面失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
