#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# DEPRECATED (2026-10-13): research legacy, not main chain. See docs/README.md

"""
回测引擎 - 核心回测逻辑
"""

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import List, Dict, Optional

import pandas as pd
import numpy as np

from risk.manager import Position, RiskManager
from risk.constants import RiskParams, BacktestParams
from strategy import Strategy
from selection.config import ScoredStock


@dataclass
class BacktestConfig:
    """回测配置（默认值来自 BacktestParams）"""
    initial_capital: float = BacktestParams.DEFAULT_INITIAL_CAPITAL
    commission: float = BacktestParams.COMMISSION
    slippage: float = BacktestParams.SLIPPAGE
    stop_loss: float = RiskParams.STOP_LOSS
    take_profit: float = RiskParams.TAKE_PROFIT
    max_positions: int = BacktestParams.MAX_POSITIONS
    trailing_stop: float = BacktestParams.TRAILING_STOP          # 移动止损（0 = 关闭）
    max_holding_days: int = BacktestParams.MAX_HOLDING_DAYS      # 时间止损（0 = 关闭）
    cooldown_days: int = BacktestParams.COOLDOWN_DAYS            # 止损冷却（0 = 关闭）
    exit_on_ma_break: bool = False                               # 跌破均线信号退出
    exit_ma_period: int = BacktestParams.EXIT_MA_PERIOD
    benchmark_symbol: str = BacktestParams.BENCHMARK_SYMBOL       # 基准指数（空 = 关闭）
    adaptive_exits: bool = True                                   # 趋势市场自适应放宽止损
    enforce_price_limits: bool = BacktestParams.ENFORCE_PRICE_LIMITS  # 涨跌停不可成交


@dataclass
class Trade:
    """交易记录"""
    id: int
    symbol: str
    action: str              # "BUY" / "SELL"
    date: date
    price: float
    quantity: int
    commission: float
    reason: str              # 触发原因
    conditions: List[Dict]   # 当时的指标状态
    signal_score: int        # 信号评分
    phase: str = "IS"        # "IS"=样本内(in-sample) / "OOS"=样本外(out-of-sample)
    taxes: float = 0.0       # 税费（印花税+过户费，卖出为主）


@dataclass
class BacktestPhaseResult:
    """滚动窗口中单个阶段的结果"""
    phase_name: str           # e.g. "2020_IS", "2020_OOS"
    start_date: str
    end_date: str
    initial_capital: float
    final_capital: float
    total_return: float
    annual_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    trade_count: int
    equity_curve: pd.Series
    trades: List[Trade]


@dataclass
class BacktestResult:
    """回测结果"""
    initial_capital: float
    final_capital: float
    total_return: float
    annual_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    trade_count: int
    avg_holding_days: float
    equity_curve: pd.Series
    trades: List[Trade]
    positions: Dict[str, Position]  # 当前持仓
    benchmark_return: Optional[float] = None  # 基准区间收益（无数据时为 None）
    benchmark_symbol: str = ""                 # 实际使用的基准代码
    phase_results: List['BacktestPhaseResult'] = None  # 滚动窗口各阶段结果

    def __post_init__(self):
        if self.phase_results is None:
            self.phase_results = []


class MarketData:
    """
    市场数据接口 (用于回测)

    回测时需要提供以下方法:
    - trading_dates: List[date] - 交易日期列表
    - get_close(symbol: str, date: date) -> float - 获取收盘价
    """

    def __init__(self, data_source=None):
        """
        初始化市场数据

        Args:
            data_source: 数据源对象，需提供 get_close(symbol, date) 方法
        """
        self.data_source = data_source
        self._trading_dates = []
        self._sorted_dates = None   # 惰性缓存的排序结果
        self._date_index = None     # 惰性构建的 date -> idx 映射（O(1) 查询）
        self._price_cache = {}

    def add_trading_date(self, trading_date):
        """添加交易日期（接受 date / datetime / pd.Timestamp，统一归一为 date）"""
        if isinstance(trading_date, pd.Timestamp):
            trading_date = trading_date.date()
        elif isinstance(trading_date, datetime):
            trading_date = trading_date.date()
        self._trading_dates.append(trading_date)
        self._sorted_dates = None  # 使缓存失效
        self._date_index = None

    def add_trading_dates(self, dates):
        """批量添加交易日期"""
        for d in dates:
            self.add_trading_date(d)

    @property
    def trading_dates(self) -> List[date]:
        """获取交易日期列表（排序后缓存，重复调用 O(1)）"""
        if self._sorted_dates is None:
            self._sorted_dates = sorted(set(self._trading_dates))
            self._date_index = {d: i for i, d in enumerate(self._sorted_dates)}
        return self._sorted_dates

    def date_index(self, trading_date) -> Optional[int]:
        """O(1) 查询交易日期在日历中的位置；不在日历中返回 None"""
        self.trading_dates  # 确保索引已构建
        return self._date_index.get(trading_date)

    def get_close(self, symbol: str, trading_date: date) -> float:
        """
        获取收盘价

        Args:
            symbol: 股票代码
            trading_date: 交易日期

        Returns:
            收盘价
        """
        cache_key = (symbol, trading_date)
        if cache_key in self._price_cache:
            return self._price_cache[cache_key]

        if self.data_source is not None and hasattr(self.data_source, 'get_close'):
            price = self.data_source.get_close(symbol, trading_date)
        else:
            price = 0.0

        self._price_cache[cache_key] = price
        return price

    def get_tradable_symbols(self, date) -> List[str]:
        """获取可交易标的列表"""
        return getattr(self, '_symbols', [])


class BacktestEngine:
    """回测引擎"""

    def __init__(self, config: BacktestConfig,
                 market_env_provider=None,
                 sector_provider=None):
        """
        初始化回测引擎

        Args:
            config: 回测配置
            market_env_provider: IMarketEnvProvider 实现，可选
            sector_provider: ISectorProvider 实现，可选
        """
        self.config = config
        self.risk_manager = RiskManager(
            params=RiskParams(
                STOP_LOSS=config.stop_loss,
                TAKE_PROFIT=config.take_profit
            )
        )
        self.trades = []
        self.positions = {}
        self.cash = config.initial_capital
        self.equity_curve = []
        self._equity_dates = []                 # 权益曲线对应的交易日
        self._trade_id = 0
        self._position_entry_dates = {}  # 记录持仓入场日期
        self._position_peaks = {}        # 持仓期间最高收盘价（移动止损用）
        self._cooldown_until = {}        # 止损冷却：symbol -> 可回买的日历idx+1
        self._last_trading_date = None   # 最后交易日（期末估值用）
        self._benchmark_curve = []       # 基准收盘价序列
        self._data_source = None
        self._market_env_provider = market_env_provider
        self._sector_provider = sector_provider
        self._market_env_scorer = None
        self._sector_rotator = None
        self._current_market_env = None
        self._current_allowed_sectors = None
        self._breadth_series = None       # （已由 _close_matrix 取代，保留兼容）
        self._close_matrix = None         # 缓存K线收盘价矩阵（宽度/RPS共用）
        self._current_constraint = None   # 当日 ScanConstraint（自适应退出读取市场状态）
        self._logger = None  # lazy init
        self._current_phase = "IS"  # "IS" or "OOS" for rolling window

        if market_env_provider is not None:
            from market_env import MarketEnvScorer
            self._market_env_scorer = MarketEnvScorer(market_env_provider)
        if sector_provider is not None:
            from sector import SectorRotator
            self._sector_rotator = SectorRotator(sector_provider)

    @property
    def logger(self):
        if self._logger is None:
            from utils.logger import get_logger
            self._logger = get_logger(__name__)
        return self._logger

    def run(self, strategy: Strategy, data: MarketData) -> BacktestResult:
        """
        运行回测

        Args:
            strategy: 策略对象
            data: 市场数据

        Returns:
            BacktestResult 回测结果
        """
        # Set data_source on risk_manager so it can get real prices
        self._data_source = data.data_source
        self.risk_manager.data_source = data.data_source

        self._data = data
        self._current_phase = "IS"
        for trading_date in data.trading_dates:
            self._run_single_day(trading_date, strategy, data)

        return self._generate_result()

    def run_rolling(
        self,
        strategy: Strategy,
        data: MarketData,
        train_window_days: int = 252,
        oos_ratio: float = 0.2
    ) -> 'BacktestResult':
        """
        滚动窗口回测：每年样本内训练 + 样本外测试

        Args:
            strategy: 策略对象
            data: 市场数据
            train_window_days: 训练窗口天数（默认252 = 1年）
            oos_ratio: 样本外比例（默认0.2 = 20%）

        Returns:
            BacktestResult，含 phase_results 列出各阶段表现
        """
        self._data_source = data.data_source
        self.risk_manager.data_source = data.data_source
        self._data = data

        trading_dates = data.trading_dates
        if len(trading_dates) == 0:
            return self._generate_result()

        step_days = int(train_window_days * (1 - oos_ratio))  # IS步长

        all_phase_results: List[BacktestPhaseResult] = []
        cursor = 0

        while cursor + train_window_days <= len(trading_dates):
            train_end = cursor + step_days        # IS截止
            oos_end = cursor + train_window_days  # OOS截止

            is_dates = trading_dates[cursor:train_end]
            oos_dates = trading_dates[train_end:oos_end]

            year_label = is_dates[0].strftime("%Y")

            # ── 阶段1: IS 样本内 ──────────────────────────────────────
            self._reset_state()
            self._current_phase = "IS"
            for td in is_dates:
                self._run_single_day(td, strategy, data)
            is_result = self._generate_result()

            is_phase = BacktestPhaseResult(
                phase_name=f"{year_label}_IS",
                start_date=is_dates[0].strftime("%Y-%m-%d"),
                end_date=is_dates[-1].strftime("%Y-%m-%d"),
                initial_capital=is_result.initial_capital,
                final_capital=is_result.final_capital,
                total_return=is_result.total_return,
                annual_return=is_result.annual_return,
                sharpe_ratio=is_result.sharpe_ratio,
                max_drawdown=is_result.max_drawdown,
                win_rate=is_result.win_rate,
                trade_count=is_result.trade_count,
                equity_curve=is_result.equity_curve,
                trades=is_result.trades
            )
            all_phase_results.append(is_phase)

            # ── 阶段2: OOS 样本外（使用相同策略参数，不重新训练）────────
            self._reset_state()
            self._current_phase = "OOS"
            for td in oos_dates:
                self._run_single_day(td, strategy, data)
            oos_result = self._generate_result()

            oos_phase = BacktestPhaseResult(
                phase_name=f"{year_label}_OOS",
                start_date=oos_dates[0].strftime("%Y-%m-%d"),
                end_date=oos_dates[-1].strftime("%Y-%m-%d"),
                initial_capital=oos_result.initial_capital,
                final_capital=oos_result.final_capital,
                total_return=oos_result.total_return,
                annual_return=oos_result.annual_return,
                sharpe_ratio=oos_result.sharpe_ratio,
                max_drawdown=oos_result.max_drawdown,
                win_rate=oos_result.win_rate,
                trade_count=oos_result.trade_count,
                equity_curve=oos_result.equity_curve,
                trades=oos_result.trades
            )
            all_phase_results.append(oos_phase)

            self.logger.info(
                f"[Rolling] {year_label}_IS: ret={is_result.total_return:.2%} "
                f"Sharpe={is_result.sharpe_ratio:.2f} | "
                f"{year_label}_OOS: ret={oos_result.total_return:.2%} "
                f"Sharpe={oos_result.sharpe_ratio:.2f}"
            )

            cursor += step_days

        # 整体结果（最后一个窗口后的累积）
        final_result = self._generate_result()
        final_result.phase_results = all_phase_results
        return final_result

    def _reset_state(self):
        """重置回测状态（用于滚动窗口各阶段切换）"""
        self.trades = []
        self.positions = {}
        self.cash = self.config.initial_capital
        self.equity_curve = []
        self._equity_dates = []
        self._trade_id = 0
        self._position_entry_dates = {}
        self._position_peaks = {}
        self._cooldown_until = {}
        self._last_trading_date = None
        self._benchmark_curve = []
        self._current_market_env = None
        self._current_allowed_sectors = None
        self._current_constraint = None

    def _run_single_day(self, trading_date, strategy: Strategy, data: MarketData):
        """运行单个交易日（市场环境 → 选股 → 买入 → 退出检查 → 权益更新）"""
        date_str = trading_date.strftime("%Y%m%d")

        self._current_market_env = None
        self._current_allowed_sectors = None

        if self._market_env_scorer is not None:
            try:
                market_env = self._market_env_scorer.score(
                    date_str, breadth_df=self._compute_breadth(trading_date)
                )
                self._current_market_env = market_env
                self.logger.info(
                    f"[{date_str}] market_score={market_env.market_score}, "
                    f"position_factor={market_env.position_factor}, "
                    f"breadth={market_env.breadth_signal}"
                )
            except Exception as e:
                self.logger.warning(f"[{date_str}] market_env scoring failed: {e}")

        if self._sector_rotator is not None:
            try:
                sectors = self._sector_rotator.get_top_sectors(date_str)
                self._current_allowed_sectors = [s.sector_code for s in sectors]
            except Exception as e:
                self.logger.warning(f"[{date_str}] sector rotation failed: {e}")

        constraint = self._build_constraint(date_str)
        self._current_constraint = constraint

        # 1. 选股
        signals = strategy.select(
            trading_date, data,
            allowed_symbols=getattr(self, '_allowed_symbols', None),
            constraint=constraint
        )

        # 2. 仓位分配（市场环境仓位系数缩配资金：市场冷时降低敞口）
        position_factor = getattr(constraint, 'position_factor', 1.0) or 1.0
        new_positions = self.risk_manager.allocate(
            signals, self.cash * position_factor, config=None, trading_date=trading_date
        )

        # 3. 执行买入（含止损冷却检查 + 涨停不可买约束）
        current_idx = data.date_index(trading_date)
        for pos in new_positions:
            if pos.symbol in self.positions:
                continue
            if len(self.positions) >= self.config.max_positions:
                continue
            if self._in_cooldown(pos.symbol, current_idx):
                continue
            if self._is_at_price_limit(pos.symbol, data, trading_date, current_idx, 'up'):
                continue  # 收盘封涨停，现实中买不进
            self._execute_buy(pos, trading_date)

        # 4. 检查退出条件（固定/ATR止损、止盈、移动止损、时间止损、信号退出）
        self._check_exit_conditions(trading_date, data)

        # 5. 更新权益
        self._update_equity(trading_date, data)

    def _in_cooldown(self, symbol: str, current_idx: Optional[int]) -> bool:
        """止损后冷却期内禁止回买同一标的"""
        if self.config.cooldown_days <= 0 or current_idx is None:
            return False
        until = self._cooldown_until.get(symbol)
        return until is not None and current_idx <= until

    @staticmethod
    def _price_limit_ratio(symbol: str) -> Optional[float]:
        """
        A股涨跌停幅度（按代码板块；None = 无限制，如港股）

        主板 10%；创业板(30/301)与科创板(688) 20%；北交所(4/8/92开头) 30%。
        ST ±5% 需要证券名称数据，暂按板块基数近似（回测偏保守）。
        """
        if not symbol or '.' not in symbol:
            return None
        code, suffix = symbol.split('.')
        if suffix.upper() == 'HK':
            return None  # 港股无涨跌停
        if code.startswith(('688',)):
            return 0.20
        if code.startswith(('300', '301')):
            return 0.20
        if code.startswith(('4', '8', '92')):
            return 0.30
        return 0.10

    def _is_at_price_limit(self, symbol: str, data: MarketData,
                           trading_date: date, current_idx: Optional[int],
                           direction: str) -> bool:
        """
        判断当日收盘是否封死涨/跌停（不可成交）

        近似规则：收盘价距涨/跌停价不超过 LIMIT_UP_TOLERANCE 视为封板。
        保守偏向：宁可错过（不买），不可虚构成交。
        """
        if not self.config.enforce_price_limits:
            return False
        limit = self._price_limit_ratio(symbol)
        if limit is None or current_idx is None or current_idx == 0:
            return False

        prev_date = data.trading_dates[current_idx - 1]
        prev_close = data.get_close(symbol, prev_date)
        current_price = data.get_close(symbol, trading_date)
        if prev_close <= 0 or current_price <= 0:
            return False

        tol = prev_close * BacktestParams.LIMIT_UP_TOLERANCE
        if direction == 'up':
            return current_price >= prev_close * (1 + limit) - tol
        return current_price <= prev_close * (1 - limit) + tol

    def _compute_breadth(self, trading_date: date) -> Optional[pd.DataFrame]:
        """
        从缓存K线计算市场宽度（涨/跌家数序列，截断到 ≤ trading_date）

        时点正确（只用历史数据）、零网络成本；缓存K线不足 20 只时返回 None。
        """
        if len(getattr(self._data_source, '_cache', {})) < 20:
            return None
        close_mx = self._get_close_matrix()
        if close_mx is None:
            return None
        pct = close_mx.pct_change()
        breadth = pd.DataFrame({
            'advance': (pct > 0).sum(axis=1),
            'decline': (pct < 0).sum(axis=1),
        })
        mask = breadth.index <= pd.Timestamp(trading_date)
        return breadth.loc[mask] if mask.any() else None

    def _compute_rps_rank(self, trading_date: date) -> Optional[Dict[str, float]]:
        """
        横截面 RPS：各股票 N 日收益在全市场的百分位排名（0-100，欧奈尔语义）

        从缓存价格矩阵向量化计算，时点正确；基准/市场指数不参与排名。
        """
        close_mx = self._get_close_matrix()
        if close_mx is None:
            return None

        # 排除指数（基准/市场状态指数不是选股标的）
        exclude = {self.config.benchmark_symbol, BacktestParams.MARKET_INDEX_SYMBOL} - {'', None}
        cols = [c for c in close_mx.columns if c not in exclude]
        if len(cols) < 10:
            return None

        idx = close_mx.index.searchsorted(pd.Timestamp(trading_date), side='right') - 1
        if idx < BacktestParams.RPS_PERIOD:
            return None

        end_row = close_mx.iloc[idx][cols]
        start_row = close_mx.iloc[idx - BacktestParams.RPS_PERIOD][cols]
        with np.errstate(divide='ignore', invalid='ignore'):
            ret = end_row / start_row - 1.0
        ret = ret.replace([np.inf, -np.inf], np.nan).dropna()
        if len(ret) < 10:
            return None

        # 百分位排名 ×100：收益最高的股票 RPS=100
        return (ret.rank(pct=True) * 100).round(1).to_dict()

    def _get_close_matrix(self) -> Optional[pd.DataFrame]:
        """缓存K线的收盘价矩阵（dates × symbols），惰性构建一次（规模检查由调用方负责）"""
        if self._close_matrix is None:
            cache = getattr(self._data_source, '_cache', None)
            if not cache:
                return None
            try:
                self._close_matrix = pd.DataFrame({
                    sym: df.set_index('date')['close'].astype(float)
                    for sym, df in cache.items()
                    if not df.empty and 'close' in df.columns
                }).sort_index()
            except Exception:
                return None
        return self._close_matrix if not self._close_matrix.empty else None

    def _build_constraint(self, date_str: str):
        """构建当日 ScanConstraint（含市场环境信号 + ADX市场状态）"""
        from selection.config import ScanConstraint
        from datetime import timedelta
        from indicators.trend_indicators import DMI_ADX

        constraint = ScanConstraint()

        if self._current_market_env is not None:
            env = self._current_market_env
            constraint.min_market_score = max(0, int(env.market_score))
            constraint.position_factor = env.position_factor
            constraint.market_score = int(env.market_score)
            # 市场环境信号（供 AdaptiveStrategy 动态选策略用）
            constraint.breadth_signal = env.breadth_signal
            constraint.valuation_signal = env.valuation_signal
            constraint.macro_signal = env.macro_signal
            constraint.leading_signal = env.leading_signal
            constraint.seasonal_signal = env.seasonal_signal

        if self._sector_rotator is not None and self._current_allowed_sectors:
            constraint.allowed_sectors = self._current_allowed_sectors

        # 计算 ADX 市场状态（从市场指数K线计算，如中证全指 000985.SH）
        if self._data_source is not None:
            try:
                ref_date = datetime.strptime(date_str, "%Y%m%d").date()
                index_data = self._data_source.get_kline(
                    BacktestParams.MARKET_INDEX_SYMBOL,
                    ref_date - timedelta(days=60),
                    ref_date
                )
                if index_data is not None and len(index_data) > 14:
                    dmi_result = DMI_ADX(index_data, period=14)
                    adx = dmi_result['adx'].iloc[-1]
                    if not pd.isna(adx):
                        if adx > 30:
                            constraint.adx_signal = 'trending'
                        elif adx < 20:
                            constraint.adx_signal = 'ranging'
                        else:
                            constraint.adx_signal = 'neutral'
            except Exception:
                pass  # ADX 计算失败不影响其他功能

        # 横截面 RPS（全市场 N 日收益百分位排名，供 rps_rank_gt 操作符使用）
        try:
            ref_date = datetime.strptime(date_str, "%Y%m%d").date()
            constraint.rps_rank = self._compute_rps_rank(ref_date)
        except Exception:
            pass  # RPS 计算失败不影响其他功能

        return constraint

    def _execute_buy(self, position: Position, trading_date: date):
        """执行买入（买入价含滑点；费用=佣金+过户费）"""
        exec_price = position.entry_price * (1 + self.config.slippage)
        position.entry_price = exec_price  # 实际成本基准
        cost = exec_price * position.quantity
        commission = cost * self.config.commission
        taxes = cost * BacktestParams.TRANSFER_FEE  # 过户费（双边）
        total_cost = cost + commission + taxes

        if total_cost > self.cash:
            # 资金不足，调整数量
            position.quantity = int(
                self.cash / (exec_price * (1 + self.config.commission + BacktestParams.TRANSFER_FEE))
            )
            cost = exec_price * position.quantity
            commission = cost * self.config.commission
            taxes = cost * BacktestParams.TRANSFER_FEE
            total_cost = cost + commission + taxes

        if position.quantity <= 0:
            return

        self.cash -= total_cost
        self.positions[position.symbol] = position
        self._position_entry_dates[position.symbol] = trading_date
        self._position_peaks[position.symbol] = exec_price  # 峰值初始化为入场价

        self.trades.append(Trade(
            id=self._trade_id,
            symbol=position.symbol,
            action="BUY",
            date=trading_date,
            price=exec_price,
            quantity=position.quantity,
            commission=commission,
            reason="买入信号",
            conditions=position.conditions,
            signal_score=position.signal_score,
            phase=self._current_phase,
            taxes=taxes
        ))
        self._trade_id += 1

    def _execute_sell(self, symbol: str, trading_date: date, price: float, reason: str):
        """执行卖出（卖出价含滑点；费用=佣金+过户费+印花税）"""
        position = self.positions[symbol]
        exec_price = price * (1 - self.config.slippage)
        sell_value = exec_price * position.quantity
        commission = sell_value * self.config.commission
        taxes = sell_value * BacktestParams.TRANSFER_FEE       # 过户费
        taxes += sell_value * BacktestParams.STAMP_TAX         # 印花税（卖出单边）
        net_value = sell_value - commission - taxes

        self.trades.append(Trade(
            id=self._trade_id,
            symbol=symbol,
            action="SELL",
            date=trading_date,
            price=exec_price,
            quantity=position.quantity,
            commission=commission,
            reason=reason,
            conditions=position.conditions,
            signal_score=position.signal_score,
            phase=self._current_phase,
            taxes=taxes
        ))
        self._trade_id += 1

        # 止损类卖出触发冷却期（防止次日 churn 式回买）
        if reason in ("stop_loss", "trailing_stop") and self.config.cooldown_days > 0:
            sell_idx = self._data.date_index(trading_date)
            if sell_idx is not None:
                self._cooldown_until[symbol] = sell_idx + self.config.cooldown_days

        self.cash += net_value
        del self.positions[symbol]
        del self._position_entry_dates[symbol]
        self._position_peaks.pop(symbol, None)

    def _check_exit_conditions(self, trading_date: date, data: MarketData):
        """
        检查退出条件（T+1规则：当天买入的股票不能当天卖出）

        退出优先级：
        1. 时间止损（持仓超过 max_holding_days 个交易日）
        2. 移动止损（从持仓期最高收盘回撤 trailing_stop）
        3. 信号退出（可选：收盘跌破 exit_ma_period 日均线）
        4. 风控止损/止盈（固定 + ATR 动态）
        """
        current_idx = data.date_index(trading_date)
        # 自适应退出：按当日市场状态取有效参数（趋势市场放宽，避免趋势股被紧止损打断）
        effective_trailing = self._effective_trailing_stop()
        effective_max_hold = self._effective_max_holding_days()

        for symbol in list(self.positions.keys()):
            position = self.positions[symbol]
            entry_date = self._position_entry_dates.get(symbol)

            # T+1规则：只有持仓超过1个交易日才能卖出
            entry_idx = data.date_index(entry_date) if entry_date is not None else None
            if entry_idx is not None and current_idx is not None and current_idx <= entry_idx:
                continue

            current_price = data.get_close(symbol, trading_date)
            if current_price <= 0:
                continue  # 无有效价格（停牌等），跳过退出检查

            # 跌停不可卖：当日封死跌停则延后退出（仓位保留，次日再判）
            if self._is_at_price_limit(symbol, data, trading_date, current_idx, 'down'):
                continue

            # ── 1. 时间止损 ──
            if (effective_max_hold > 0 and entry_idx is not None
                    and current_idx is not None
                    and (current_idx - entry_idx) >= effective_max_hold):
                self._execute_sell(symbol, trading_date, current_price, "time_stop")
                continue

            # ── 2. 移动止损 ──
            peak = self._position_peaks.get(symbol, position.entry_price)
            if current_price > peak:
                peak = current_price
                self._position_peaks[symbol] = peak
            if (effective_trailing > 0 and peak > 0
                    and current_price <= peak * (1 - effective_trailing)):
                self._execute_sell(symbol, trading_date, current_price, "trailing_stop")
                continue

            # ── 3. 信号退出：跌破均线 ──
            if self.config.exit_on_ma_break and self._check_ma_break(symbol, trading_date, current_price):
                self._execute_sell(symbol, trading_date, current_price, "signal_exit_ma")
                continue

            # ── 4. 风控止损/止盈 ──
            exit_reason = self.risk_manager.check_exit(position, current_price)
            if exit_reason:
                self._execute_sell(symbol, trading_date, current_price, exit_reason)

    def _is_trending_market(self) -> bool:
        """当日市场是否处于趋势状态（ADX 信号）"""
        constraint = self._current_constraint
        return constraint is not None and constraint.is_adx_trending()

    def _effective_trailing_stop(self) -> float:
        """有效移动止损阈值：趋势市场放宽（×TRENDING_TRAILING_MULTIPLIER）"""
        base = self.config.trailing_stop
        if base <= 0 or not self.config.adaptive_exits:
            return base
        if self._is_trending_market():
            return base * BacktestParams.TRENDING_TRAILING_MULTIPLIER
        return base

    def _effective_max_holding_days(self) -> int:
        """有效时间止损天数：趋势市场放宽（×TRENDING_HOLDING_MULTIPLIER）"""
        base = self.config.max_holding_days
        if base <= 0 or not self.config.adaptive_exits:
            return base
        if self._is_trending_market():
            return int(base * BacktestParams.TRENDING_HOLDING_MULTIPLIER)
        return base

    def _check_ma_break(self, symbol: str, trading_date: date, current_price: float) -> bool:
        """信号退出：收盘价跌破 exit_ma_period 日均线"""
        if self._data_source is None:
            return False
        try:
            kline = self._data_source.get_kline(
                symbol,
                trading_date - timedelta(days=self.config.exit_ma_period * 2),
                trading_date
            )
            if kline is None or len(kline) < self.config.exit_ma_period:
                return False
            ma = kline['close'].rolling(self.config.exit_ma_period).mean().iloc[-1]
            if pd.isna(ma):
                return False
            return current_price < ma
        except Exception:
            return False

    def _update_equity(self, trading_date: date, data: MarketData):
        """更新权益曲线和基准曲线"""
        position_value = 0.0
        for symbol, position in self.positions.items():
            current_price = data.get_close(symbol, trading_date)
            position_value += current_price * position.quantity

        total_equity = self.cash + position_value
        self.equity_curve.append(total_equity)
        self._equity_dates.append(trading_date)
        self._last_trading_date = trading_date

        # 基准曲线（买入持有基准指数）
        if self.config.benchmark_symbol:
            try:
                bench_close = data.get_close(self.config.benchmark_symbol, trading_date)
            except Exception:
                bench_close = 0.0
            if bench_close > 0:
                self._benchmark_curve.append(bench_close)

    def _generate_result(self) -> BacktestResult:
        """生成回测结果"""
        initial_capital = self.config.initial_capital
        # 期末持仓按最后交易日收盘价估值（而非入场价）
        if self._last_trading_date is not None:
            final_capital = self.cash + sum(
                self._data.get_close(symbol, self._last_trading_date) * pos.quantity
                for symbol, pos in self.positions.items()
            )
        else:
            final_capital = self.cash

        # 计算总收益率
        total_return = (final_capital - initial_capital) / initial_capital

        # 计算年化收益率
        trading_days = len(self.equity_curve)
        if trading_days > 0:
            annual_return = (final_capital / initial_capital) ** (BacktestParams.TRADING_DAYS_PER_YEAR / trading_days) - 1
        else:
            annual_return = 0.0

        # 计算夏普比率
        sharpe_ratio = self._calculate_sharpe_ratio()

        # 计算最大回撤
        max_drawdown = self._calculate_max_drawdown()

        # 计算胜率
        win_rate = self._calculate_win_rate()

        # 计算平均持仓天数
        avg_holding_days = self._calculate_avg_holding_days()

        # 基准区间收益（买入持有）
        benchmark_return = None
        if len(self._benchmark_curve) >= 2 and self._benchmark_curve[0] > 0:
            benchmark_return = self._benchmark_curve[-1] / self._benchmark_curve[0] - 1

        # 创建权益曲线 Series（以交易日为索引）
        equity_series = pd.Series(self.equity_curve, index=self._equity_dates)

        return BacktestResult(
            initial_capital=initial_capital,
            final_capital=final_capital,
            total_return=total_return,
            annual_return=annual_return,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            trade_count=len(self.trades),
            avg_holding_days=avg_holding_days,
            equity_curve=equity_series,
            trades=self.trades,
            positions=self.positions.copy(),
            benchmark_return=benchmark_return,
            benchmark_symbol=self.config.benchmark_symbol if benchmark_return is not None else ""
        )

    def _calculate_sharpe_ratio(self, risk_free_rate: float = None) -> float:
        """计算夏普比率"""
        if risk_free_rate is None:
            risk_free_rate = BacktestParams.RISK_FREE_RATE
        if len(self.equity_curve) < 2:
            return 0.0

        equity_series = pd.Series(self.equity_curve)
        returns = equity_series.pct_change().dropna()

        if len(returns) == 0 or returns.std() == 0:
            return 0.0

        annual_return = returns.mean() * BacktestParams.TRADING_DAYS_PER_YEAR
        annual_volatility = returns.std() * np.sqrt(BacktestParams.TRADING_DAYS_PER_YEAR)

        if annual_volatility == 0:
            return 0.0

        sharpe_ratio = (annual_return - risk_free_rate) / annual_volatility
        return sharpe_ratio

    def _calculate_max_drawdown(self) -> float:
        """计算最大回撤"""
        if len(self.equity_curve) == 0:
            return 0.0

        equity_series = pd.Series(self.equity_curve)
        cumulative = equity_series
        peak = cumulative.expanding(min_periods=1).max()
        drawdown = (cumulative - peak) / peak

        max_drawdown = drawdown.min()
        return max_drawdown if not pd.isna(max_drawdown) else 0.0

    def _iter_round_trips(self):
        """单次遍历配对每笔卖出对应的最近买入（round trip）"""
        last_buy = {}
        for t in self.trades:
            if t.action == "BUY":
                last_buy[t.symbol] = t
            elif t.symbol in last_buy:
                yield last_buy.pop(t.symbol), t

    def _calculate_win_rate(self) -> float:
        """计算胜率 (盈利交易 / 总平仓交易)"""
        sell_count = sum(1 for t in self.trades if t.action == "SELL")
        if sell_count == 0:
            return 0.0

        winning_trades = 0
        for buy, sell in self._iter_round_trips():
            profit = ((sell.price - buy.price) * sell.quantity
                      - buy.commission - buy.taxes
                      - sell.commission - sell.taxes)
            if profit > 0:
                winning_trades += 1

        return winning_trades / sell_count

    def _calculate_avg_holding_days(self) -> float:
        """计算平均持仓天数"""
        holding_days_list = [
            (sell.date - buy.date).days
            for buy, sell in self._iter_round_trips()
        ]
        return np.mean(holding_days_list) if holding_days_list else 0.0
