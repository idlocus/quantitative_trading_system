#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
回测运行器 - 编排回测运行、报告生成和数据库持久化
"""

from datetime import datetime, timedelta
from typing import Optional, List
import os

import pandas as pd

from data.base import DataLoader
from data import WindDBDataSource
from selection import StockScanner, load_strategy_from_yaml
from strategy import MomentumStrategy, AdaptiveStrategy, MultiStrategy
from backtesting import BacktestEngine, BacktestConfig, MarketData
from backtesting.reporter import BacktestReporter
from risk.constants import RiskParams, BacktestParams
from utils.logger import get_logger

logger = get_logger(__name__)


class BacktestRunner:
    """
    回测运行器。

    封装回测的完整流程：数据加载 → 选股 → 回测 → 报告生成 → 数据库持久化。
    """

    def __init__(self, config: 'SystemConfig'):
        """
        初始化运行器

        Args:
            config: SystemConfig 实例
        """
        self.config = config
        self._loader: Optional[DataLoader] = None

    def run(self, strategy_name: str, allowed_symbols: Optional[set] = None) -> 'BacktestResult':
        """
        运行完整回测流程

        Args:
            strategy_name: 策略名称（用于从 YAML 加载配置）
            allowed_symbols: 可选的交易标的范围

        Returns:
            BacktestResult 回测结果
        """
        ctx = self._prepare(strategy_name, allowed_symbols)

        logger.info("Starting backtest...")
        result = ctx['engine'].run(ctx['strategy'], ctx['market_data'])
        logger.info(f"Backtest complete: {result.trade_count} trades")

        # 生成报告并保存到数据库
        self._generate_reports(result)
        self._save_result(result, ctx['start_date'], ctx['end_date'], ctx['strategy_config'])

        return result

    def run_rolling(
        self,
        strategy_name: str,
        allowed_symbols: Optional[set] = None,
        train_window_days: int = 252,
        oos_ratio: float = 0.2
    ) -> 'BacktestResult':
        """
        运行滚动窗口回测：每年样本内训练 + 样本外测试

        Args:
            strategy_name: 策略名称
            allowed_symbols: 可选的交易标的范围
            train_window_days: 训练窗口天数（默认252 = 1年）
            oos_ratio: 样本外比例（默认0.2 = 20%）

        Returns:
            BacktestResult，含 phase_results 列出各阶段表现
        """
        ctx = self._prepare(strategy_name, allowed_symbols)

        logger.info(f"Starting rolling backtest (train_window={train_window_days}, oos_ratio={oos_ratio})...")
        result = ctx['engine'].run_rolling(
            ctx['strategy'], ctx['market_data'],
            train_window_days=train_window_days,
            oos_ratio=oos_ratio
        )
        logger.info(f"Rolling backtest complete: {result.trade_count} total trades, {len(result.phase_results)} phases")

        # 生成报告并保存到数据库
        self._generate_reports_rolling(result)
        self._save_result(result, ctx['start_date'], ctx['end_date'], ctx['strategy_config'])

        return result

    def _prepare(self, strategy_name: str, allowed_symbols: Optional[set]) -> dict:
        """
        回测前置准备（run / run_rolling 共用）

        Returns:
            dict: {loader, symbols, cache, strategy_config, market_data,
                   strategy, engine, start_date, end_date}
        """
        # 1. 初始化数据加载器
        if self.config.data_source_type == 'winddb':
            oracle_cfg = self.config.oracle_config()
            self._loader = DataLoader.from_winddb(
                oracle_cfg['dsn'],
                oracle_cfg['user'],
                oracle_cfg['password']
            )
        else:
            self._loader = DataLoader(self.config.data_source_type)

        # 2. 加载股票列表（PIT：回测起始日的时点成分，含后来退市的股票）与策略配置
        start_date = datetime.strptime(self.config.start_date, '%Y-%m-%d').date()
        end_date = datetime.strptime(self.config.end_date, '%Y-%m-%d').date()
        symbols = self._loader.load_symbols(as_of=start_date)
        logger.info(f"Loaded {len(symbols)} symbols")
        yaml_path = self._find_strategies_yaml()
        strategy_config = load_strategy_from_yaml(strategy_name, yaml_path)

        # 3. 回测配置（需先确定基准代码，供预加载；参数从常量与 SystemConfig 读取）
        backtest_config = BacktestConfig(
            initial_capital=self.config.initial_capital,
            commission=self.config.commission,
            slippage=self.config.slippage,
            stop_loss=self.config.stop_loss,
            take_profit=self.config.take_profit,
            trailing_stop=self.config.trailing_stop,
            max_holding_days=self.config.max_holding_days,
            max_positions=strategy_config.max_positions
        )

        # 4. 预加载 K 线数据
        #    - 额外回溯一段历史，保证指标暖机（RSI/ADX 等需要回看数据）
        #    - 纳入基准指数与市场状态指数（基准对比 / ADX 市场信号用）
        preload_start = start_date - timedelta(days=BacktestParams.PRELOAD_LOOKBACK_DAYS)
        extra_symbols = {
            backtest_config.benchmark_symbol,
            BacktestParams.MARKET_INDEX_SYMBOL,
        } - {'', None}
        preload_symbols = list(set(symbols) | extra_symbols)
        cache = self._loader.preload_klines(preload_symbols, preload_start, end_date)
        missing_idx = extra_symbols - set(cache.get_tradable_symbols(None))  # 缓存内实际有的标的
        if missing_idx:
            logger.warning(f"Index data unavailable (benchmark/ADX degraded): {sorted(missing_idx)}")

        # 5. 创建策略（支持多种策略类型）
        strategy = self._create_strategy(strategy_name, strategy_config, cache)
        logger.info(f"Strategy '{strategy_name}' created: {type(strategy).__name__}")

        # 6. 准备市场数据（交易日历优先从缓存数据推导，排除节假日；
        #    暖机数据不进入交易日历）
        market_data = MarketData(cache)
        trading_dates = self._build_trading_calendar(cache, start_date, end_date)
        market_data.add_trading_dates(trading_dates)
        market_data._symbols = symbols

        # 实例化市场环境提供者（akshare 免费数据源）
        market_env_provider = None
        try:
            from market_env import AkShareMarketEnvProvider
            market_env_provider = AkShareMarketEnvProvider()
            logger.info("AkShareMarketEnvProvider initialized for market environment scoring")
        except Exception as e:
            logger.warning(f"Failed to init market_env_provider, backtest will run without it: {e}")

        engine = BacktestEngine(backtest_config, market_env_provider=market_env_provider)
        engine._allowed_symbols = allowed_symbols

        return {
            'loader': self._loader,
            'symbols': symbols,
            'cache': cache,
            'strategy_config': strategy_config,
            'market_data': market_data,
            'strategy': strategy,
            'engine': engine,
            'start_date': start_date,
            'end_date': end_date,
        }

    def _build_trading_calendar(self, cache, start_date, end_date) -> list:
        """
        构建交易日历

        优先从缓存K线日期并集推导（真实交易日，排除节假日）；
        缓存为空时回退到工作日历（bdate_range）。
        """
        trading_dates = []
        if hasattr(cache, 'get_trading_dates'):
            trading_dates = cache.get_trading_dates(start_date, end_date)
        if trading_dates:
            logger.info(f"Trading calendar derived from cached data: {len(trading_dates)} days")
            return trading_dates  # add_trading_date 内部统一归一化为 date
        logger.warning("Empty cache, falling back to bdate_range calendar (may include holidays)")
        return [d.date() for d in pd.bdate_range(start=start_date, end=end_date)]

    def _create_strategy(self, strategy_name: str, strategy_config, cache):
        """
        根据策略名称创建对应的策略对象

        支持的策略类型：
        - adaptive:      AdaptiveStrategy（市场环境自适应）
        - multi_strategy: MultiStrategy（多策略组合）
        - 其他:          MomentumStrategy（单策略）
        """
        if strategy_name == "adaptive":
            return AdaptiveStrategy(strategy_name, cache)
        elif strategy_name == "multi_strategy":
            return MultiStrategy(strategy_name, cache)
        else:
            return MomentumStrategy(strategy_config, cache)

    def _find_strategies_yaml(self) -> str:
        """查找 strategies.yml 文件路径"""
        import os
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        return os.path.join(base, 'selection', 'strategies.yml')

    def _generate_reports(self, result: 'BacktestResult'):
        """生成 Markdown 和 HTML 报告"""
        reporter = BacktestReporter()
        report_dir = self._get_report_dir()
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        md_path = os.path.join(report_dir, f"backtest_report_{timestamp}.md")
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(reporter.generate_markdown(result))
        logger.info(f"Markdown report saved: {md_path}")

        html_path = os.path.join(report_dir, f"backtest_report_{timestamp}.html")
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(reporter.generate_html(result))
        logger.info(f"HTML report saved: {html_path}")

    def _generate_reports_rolling(self, result: 'BacktestResult'):
        """生成滚动窗口回测报告（含各阶段对比）"""
        reporter = BacktestReporter()
        report_dir = self._get_report_dir()
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # 整体报告
        md_path = os.path.join(report_dir, f"rolling_report_{timestamp}.md")
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(reporter.generate_markdown(result))
        logger.info(f"Markdown report saved: {md_path}")

        html_path = os.path.join(report_dir, f"rolling_report_{timestamp}.html")
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(reporter.generate_html(result))
        logger.info(f"HTML report saved: {html_path}")

        # 各阶段对比报告
        phase_lines = []
        for ph in result.phase_results:
            phase_lines.append(
                f"| {ph.phase_name} | {ph.start_date} | {ph.end_date} | "
                f"{ph.total_return:.2%} | {ph.sharpe_ratio:.2f} | {ph.max_drawdown:.2%} | {ph.trade_count} |"
            )

        phase_md = os.path.join(report_dir, f"rolling_phases_{timestamp}.md")
        with open(phase_md, 'w', encoding='utf-8') as f:
            f.write("# 滚动窗口回测 - 各阶段表现\n\n")
            f.write("| 阶段 | 开始 | 结束 | 收益率 | 夏普比率 | 最大回撤 | 交易次数 |\n")
            f.write("|------|------|------|--------|---------|---------|----------|\n")
            for line in phase_lines:
                f.write(line + "\n")
        logger.info(f"Phase report saved: {phase_md}")

    def _save_result(self, result: 'BacktestResult', start_date, end_date, strategy_config):
        """保存回测结果到数据库"""
        from webapp.database import BacktestDB

        db_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'webapp', 'backtest.db'
        )
        db = BacktestDB(db_path)

        # 计算交易 PnL
        trades_with_pnl = self._compute_trades_pnl(result.trades)

        backtest_data = {
            'name': f"回测_{datetime.now().strftime('%Y%m%d_%H%M')}",
            'start_date': start_date.strftime('%Y-%m-%d'),
            'end_date': end_date.strftime('%Y-%m-%d'),
            'initial_capital': result.initial_capital,
            'final_capital': result.final_capital,
            'total_return': result.total_return,
            'annual_return': result.annual_return,
            'sharpe_ratio': result.sharpe_ratio,
            'max_drawdown': result.max_drawdown,
            'win_rate': result.win_rate,
            'trade_count': result.trade_count,
            'equity_curve': result.equity_curve.tolist() if hasattr(result.equity_curve, 'tolist') else list(result.equity_curve),
            'trades': trades_with_pnl,
            'benchmark_return': result.benchmark_return,
            'benchmark_symbol': result.benchmark_symbol
        }

        backtest_id = db.save_backtest(backtest_data)
        logger.info(f"Backtest saved to database with ID: {backtest_id}")

    def _compute_trades_pnl(self, trades: List) -> List[dict]:
        """计算每笔交易的盈亏"""
        result = []
        position_book = {}
        for trade in trades:
            t = {
                'symbol': trade.symbol,
                'action': trade.action,
                'date': trade.date.strftime('%Y-%m-%d'),
                'price': trade.price,
                'quantity': trade.quantity,
                'commission': trade.commission,
                'taxes': getattr(trade, 'taxes', 0.0),
                'reason': trade.reason,
                'signal_score': trade.signal_score,
                'conditions': getattr(trade, 'conditions', []),
                'pnl': None
            }
            if trade.action == 'BUY':
                position_book[trade.symbol] = {
                    'entry_date': trade.date,
                    'entry_price': trade.price,
                    'quantity': trade.quantity,
                    'cost': trade.commission + getattr(trade, 'taxes', 0.0),
                }
            else:
                if trade.symbol in position_book:
                    pos = position_book[trade.symbol]
                    pnl = ((trade.price - pos['entry_price']) * trade.quantity
                           - trade.commission - trade.taxes - pos['cost'])
                    t['pnl'] = pnl
                    del position_book[trade.symbol]
                else:
                    t['pnl'] = 0
            result.append(t)
        return result

    def _get_report_dir(self) -> str:
        """获取报告输出目录"""
        report_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'results'
        )
        os.makedirs(report_dir, exist_ok=True)
        return report_dir

    def compute_sector_allowed_symbols(self, top_n: int = 10) -> Optional[set]:
        """从 Wind Oracle DB 计算板块筛选后的股票集合"""
        oracle_cfg = self.config.oracle_config()
        if not oracle_cfg['dsn'] or not oracle_cfg['user'] or not oracle_cfg['password']:
            logger.warning("Oracle not configured, sector filter disabled")
            return None

        try:
            industry_ds = WindDBDataSource(
                oracle_cfg['dsn'],
                oracle_cfg['user'],
                oracle_cfg['password']
            )

            # 获取所有申万二级行业
            industries = industry_ds.get_sw_industry_constituents()

            # 筛选强势板块：5日涨幅>0 AND MA5>MA20 AND 成交量>20日均量
            valid_sectors = {}
            for ind in industries:
                ind_code = ind['industry_code']
                momentum = industry_ds.get_industry_momentum(ind_code)
                if momentum is None:
                    continue
                if (momentum['return_5d'] > 0 and
                    momentum['ma5_above_ma20'] and
                    momentum['volume_ratio'] > 1.0):
                    valid_sectors[ind_code] = {
                        'industry_name': ind['industry_name'],
                        'symbols': ind['symbols'],
                        'return_5d': momentum['return_5d'],
                        'volume_ratio': momentum['volume_ratio'],
                    }

            # 按5日涨幅排序，取前N名
            sorted_sectors = sorted(
                valid_sectors.items(),
                key=lambda x: x[1]['return_5d'],
                reverse=True
            )[:top_n]

            allowed = set()
            for _, info in sorted_sectors:
                allowed.update(info['symbols'])

            industry_ds.close()
            logger.info(f"Sector filter: {len(allowed)} stocks eligible")
            return allowed
        except Exception as e:
            logger.error(f"Sector filter failed: {e}")
            return None
