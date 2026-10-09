#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
组合回测器 —— TopN 分散持仓、周期调仓、真实成本与涨跌停约束

与 backtesting/engine 的差异（产品形态不同）：
- 单票止损/冷却不适用：组合通过分散与调仓控制风险，而非个股止损
- 调仓日按目标权重差额成交；涨停不买（跳过）、跌停不卖（顺延）
- 逐日 mark-to-market 权益曲线、换手率与成本统计、基准对比

全部计算向量化（收盘价矩阵），全市场（5000+只）规模可用。
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from risk.constants import BacktestParams
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class PortfolioConfig:
    top_n: int = 50                    # 目标持仓数
    rebalance_days: int = 5            # 调仓周期（交易日）
    initial_capital: float = 1_000_000.0
    commission: float = 0.0003         # 佣金（双边）
    stamp_tax: float = 0.0005          # 印花税（卖出）
    transfer_fee: float = 0.00001      # 过户费（双边）
    slippage: float = 0.001            # 单边滑点
    enforce_price_limits: bool = True
    max_weight: float = 0.05           # 单票权重上限
    weight_method: str = 'equal'       # equal | inverse_vol
    benchmark_symbol: str = '000300.SH'
    limit_tolerance: float = 0.002     # 涨跌停判定容差
    buffer_band: float = 0.0           # 权重缓冲带（0=全量换仓，0.2=与生产一致的20%缓冲）
    max_participation: float = 0.10    # 单日最大成交量占比（10%=不超过日均量10%）
    min_daily_turnover: float = 500e4  # 最低日均成交额（500万，流动性过滤）
    impact_coeff: float = 0.10         # 市场冲击系数（sqrt模型）
    min_listing_days: int = 60         # 上市天数过滤（需≥N个交易日数据才可入选）


@dataclass
class PortfolioResult:
    config: PortfolioConfig
    n_dates: int
    n_rebalances: int
    initial_capital: float
    final_capital: float
    total_return: float
    annual_return: float
    sharpe: float
    max_drawdown: float
    avg_turnover: float                # 单边平均换手率/次调仓
    total_cost: float                  # 累计交易成本（含税费滑点）
    benchmark_return: Optional[float]
    equity_curve: pd.Series = None
    holdings_history: List[Dict] = field(default_factory=list)
    # 回测过程明细（供过程回放）
    daily_holdings: Dict = field(default_factory=dict)   # {date: [[sym, weight], ...]}
    daily_contrib: Dict = field(default_factory=dict)    # {date: [[sym, 贡献], ...]}
    trade_log: List = field(default_factory=list)        # [date, sym, side, price, qty]

    def summary_markdown(self) -> str:
        lines = ["## 组合回测结果\n"]
        lines.append(f"- 持仓数: Top{self.config.top_n} | 调仓: 每{self.config.rebalance_days}个交易日"
                     f" | 加权: {self.config.weight_method}")
        lines.append(f"- 区间交易日: {self.n_dates} | 调仓次数: {self.n_rebalances}")
        lines.append(f"| 指标 | 值 |")
        lines.append("|------|------|")
        lines.append(f"| 总收益 | {self.total_return:+.2%} |")
        if self.benchmark_return is not None:
            lines.append(f"| 基准（{self.config.benchmark_symbol}） | {self.benchmark_return:+.2%} |")
            lines.append(f"| 超额收益 | {self.total_return - self.benchmark_return:+.2%} |")
        lines.append(f"| 年化收益 | {self.annual_return:+.2%} |")
        lines.append(f"| 夏普 | {self.sharpe:.2f} |")
        lines.append(f"| 最大回撤 | {self.max_drawdown:.2%} |")
        lines.append(f"| 平均换手/次 | {self.avg_turnover:.1%} |")
        lines.append(f"| 累计成本 | {self.total_cost:,.0f} ({self.total_cost / self.initial_capital:.2%} 本金) |")
        return "\n".join(lines)


class PortfolioBacktester:
    """TopN 组合回测器（向量化，接受日频综合分矩阵驱动调仓）"""

    def __init__(self, cache: Dict[str, pd.DataFrame], config: PortfolioConfig):
        self.config = config
        close = {}
        volume = {}
        for sym, df in cache.items():
            if df is not None and not df.empty:
                idx = pd.DatetimeIndex(pd.to_datetime(df['date'])).as_unit('ns')
                close[sym] = pd.Series(pd.to_numeric(df['close'], errors='coerce').values, index=idx)
                if 'volume' in df.columns:
                    volume[sym] = pd.Series(pd.to_numeric(df['volume'], errors='coerce').values, index=idx)
        self.close_mx = pd.DataFrame(close).sort_index()
        self.prev_close_mx = self.close_mx.shift(1)
        # 成交量矩阵（用于流动性过滤和冲击成本）
        self.volume_mx = pd.DataFrame(volume).reindex(self.close_mx.index).sort_index()
        # 20日均成交额（价格×成交量，用于流动性判定）
        if not self.volume_mx.empty:
            self.avg_turnover_20d = (self.close_mx * self.volume_mx).rolling(20).mean()
        else:
            self.avg_turnover_20d = pd.DataFrame(index=self.close_mx.index, columns=self.close_mx.columns)
        # 上市天数矩阵（累计有效数据天数，用于上市天数过滤）
        # 动态处理：新上市股在达到 min_listing_days 前不可选，退市股数据变NaN后自然排除
        self.data_age = self.close_mx.notna().cumsum()

    # ---- 涨跌停判定（向量化）----

    def _limit_up_mask(self, date) -> pd.Series:
        """当日收盘封涨停的股票（不可买）"""
        if not self.config.enforce_price_limits:
            return pd.Series(dtype=bool)
        cur = self.close_mx.loc[date]
        prev = self.prev_close_mx.loc[date]
        limit = self._limit_ratios(cur.index)
        tol = prev * self.config.limit_tolerance
        return cur >= (prev * (1 + limit) - tol).where(limit.notna(), -np.inf)

    def _limit_down_mask(self, date) -> pd.Series:
        """当日收盘封跌停的股票（不可卖）"""
        if not self.config.enforce_price_limits:
            return pd.Series(dtype=bool)
        cur = self.close_mx.loc[date]
        prev = self.prev_close_mx.loc[date]
        limit = self._limit_ratios(cur.index)
        tol = prev * self.config.limit_tolerance
        return cur <= (prev * (1 - limit) + tol).where(limit.notna(), np.inf)

    @staticmethod
    def _limit_ratios(symbols) -> pd.Series:
        """按代码板块的涨跌停幅度（与 BacktestEngine._price_limit_ratio 同规则）"""
        def ratio(sym):
            if not isinstance(sym, str) or '.' not in sym:
                return np.nan
            code, suffix = sym.split('.')
            if suffix.upper() == 'HK':
                return np.nan
            if code.startswith(('688', '300', '301')):
                return 0.20
            if code.startswith(('4', '8', '92')):
                return 0.30
            return 0.10
        return pd.Series({s: ratio(s) for s in symbols})

    def _liquidity_mask(self, date) -> pd.Series:
        """流动性不足的股票（20日均成交额低于阈值，不可买）"""
        cfg = self.config
        if cfg.min_daily_turnover <= 0 or self.avg_turnover_20d.empty:
            return pd.Series(dtype=bool)
        try:
            avg = self.avg_turnover_20d.loc[date]
            return avg < cfg.min_daily_turnover
        except KeyError:
            return pd.Series(dtype=bool)

    def _max_buy_amount(self, sym: str, date) -> float:
        """单票当日最大可买入金额（受成交量约束）"""
        cfg = self.config
        if cfg.max_participation <= 0:
            return float('inf')
        try:
            avg_turn = self.avg_turnover_20d.loc[date, sym]
            if pd.isna(avg_turn) or avg_turn <= 0:
                return 0.0
            return float(avg_turn) * cfg.max_participation
        except KeyError:
            return 0.0

    def _market_impact(self, sym: str, date, trade_amount: float) -> float:
        """市场冲击成本：冲击比例 ∝ sqrt(交易额/日均成交额)"""
        cfg = self.config
        if cfg.impact_coeff <= 0 or trade_amount <= 0:
            return 0.0
        try:
            avg_turn = self.avg_turnover_20d.loc[date, sym]
            if pd.isna(avg_turn) or avg_turn <= 0:
                return cfg.slippage  # 无数据时退化为固定滑点
            participation = trade_amount / avg_turn
            return cfg.slippage + cfg.impact_coeff * np.sqrt(participation)
        except KeyError:
            return cfg.slippage

    # ---- 主流程 ----

    def run(self, score_mx: pd.DataFrame,
            weight_func=None,
            score_band: Optional[tuple] = None,
            start_date: Optional[str] = None) -> PortfolioResult:
        """
        运行组合回测

        Args:
            score_mx: 综合分矩阵（dates × symbols，越高越值得买入；时点已保证 ≤t）
            weight_func: 可选自定义权重函数 (symbols, date) -> {sym: w}
            score_band: 可选 (lo, hi) 选股分数带（规避极端尾部，见 select_portfolio）
            start_date: 可选回测起始日（'YYYY-MM-DD'），截断 warm-up 区间，确保只统计目标时段
        """
        cfg = self.config
        # 前视偏差修正：将信号 shift(1)，确保调仓日 T 使用 T-1 的因子排名
        score_mx = score_mx.shift(1)
        # 起始日截断：去除 warm-up 前置区间
        if start_date:
            score_mx = score_mx.loc[start_date:]
        dates = self.close_mx.index
        dates = dates[(dates >= score_mx.index.min()) & (dates <= score_mx.index.max())]
        process_holdings, process_contrib, trade_log = {}, {}, []

        cash = cfg.initial_capital
        holdings: Dict[str, float] = {}      # symbol -> 市值（按成本记权重，逐日随价格漂移）
        shares: Dict[str, float] = {}        # symbol -> 股数
        pending_sells: List[str] = []        # 跌停未卖出的顺延清单
        equity_curve = {}
        total_cost = 0.0
        turnover_list = []
        n_rebalances = 0
        holdings_history = []

        weight_func = weight_func or self._default_weight_func

        rebalance_idx = set()
        step = cfg.rebalance_days
        i = 0
        while i < len(dates):
            rebalance_idx.add(i)
            i += step

        for idx, t in enumerate(dates):
            prices = self.close_mx.loc[t]

            # 1. 执行顺延卖出（跌停解除后优先清仓）
            if pending_sells:
                down_mask = self._limit_down_mask(t)
                still_blocked = []
                for sym in pending_sells:
                    if sym not in shares:
                        continue
                    if not down_mask.get(sym, False) and not pd.isna(prices.get(sym, np.nan)):
                        px = prices[sym] * (1 - cfg.slippage)
                        value = shares[sym] * px
                        cost = value * (cfg.commission + cfg.stamp_tax + cfg.transfer_fee)
                        cash += value - cost
                        total_cost += cost
                        del shares[sym]
                    else:
                        still_blocked.append(sym)
                pending_sells = still_blocked

            # 2. 调仓日：构造目标组合并交易差额
            if idx in rebalance_idx and t in score_mx.index:
                scores = score_mx.loc[t]
                up_mask = self._limit_up_mask(t)
                liq_mask = self._liquidity_mask(t)
                # 排除当日涨停（不可买）、流动性不足、已无价格数据的股票
                buyable_excl = set(up_mask[up_mask == True].index) | set(liq_mask[liq_mask == True].index)  # noqa: E712
                targets = self._select(scores, buyable_excl, score_band, date=t)
                target_w = weight_func(targets, t)

                # 目标市值
                total_equity = cash + sum(
                    shares[s] * prices[s] for s in shares if not pd.isna(prices.get(s, np.nan))
                )
                target_value = {s: total_equity * w for s, w in target_w.items()}

                # 换手率（调仓前权重 vs 目标权重，单边）
                cur_value = {
                    s: shares[s] * prices[s] for s in shares
                    if not pd.isna(prices.get(s, np.nan))
                }
                all_syms = set(cur_value) | set(target_value)
                denom = max(total_equity, 1e-9)
                turnover = sum(
                    abs(target_value.get(s, 0.0) - cur_value.get(s, 0.0)) for s in all_syms
                ) / denom / 2
                turnover_list.append(turnover)

                # 卖出：目标中不存在或需减仓的（缓冲带内不调）
                down_mask = self._limit_down_mask(t)
                target_w_each = 1.0 / max(len(target_value), 1) if target_value else 0
                for s in list(shares.keys()):
                    cur_w = cur_value.get(s, 0.0) / denom if denom > 0 else 0
                    if s not in target_value:
                        # 不在目标 → 卖出
                        need_sell = True
                    elif cfg.buffer_band > 0 and cur_w > 0:
                        # 在目标但检查是否超配（超过缓冲带上限不调）
                        tw = target_value[s] / denom if denom > 0 else 0
                        need_sell = cur_w > tw * (1 + cfg.buffer_band)
                    else:
                        need_sell = target_value[s] < cur_value.get(s, 0.0) * 0.999
                    if need_sell:
                        if down_mask.get(s, False) or pd.isna(prices.get(s, np.nan)):
                            pending_sells.append(s)  # 跌停顺延
                            continue
                        px = prices[s] * (1 - cfg.slippage)
                        value = shares[s] * px
                        cost = value * (cfg.commission + cfg.stamp_tax + cfg.transfer_fee)
                        cash += value - cost
                        total_cost += cost
                        trade_log.append([str(t)[:10], s, 'SELL', round(px, 3), int(shares[s])])
                        del shares[s]

                # 买入：新目标与加仓（缓冲带内低配不调）
                for s, tv in target_value.items():
                    cur_v = shares.get(s, 0.0) * prices.get(s, 0.0)
                    if pd.isna(prices.get(s, np.nan)) or tv <= cur_v * 1.001:
                        continue  # 无价格或已达标
                    if cfg.buffer_band > 0 and s in shares:
                        cur_w_s = cur_v / denom if denom > 0 else 0
                        tw_s = tv / denom if denom > 0 else 0
                        if cur_w_s >= tw_s * (1 - cfg.buffer_band):
                            continue  # 在缓冲带内，不调
                    buy_value = min(tv - cur_v, cash)
                    if buy_value <= 0:
                        continue
                    # 成交量约束：单日买入不超过日均成交额的 max_participation
                    max_amt = self._max_buy_amount(s, t)
                    buy_value = min(buy_value, max_amt)
                    if buy_value <= 0:
                        continue
                    # 市场冲击滑点（参与率越高，滑点越大）
                    impact_slip = self._market_impact(s, t, buy_value)
                    px = prices[s] * (1 + impact_slip)
                    cost_rate = cfg.commission + cfg.transfer_fee
                    # A股最小交易单位：整手（100股）
                    qty = int((buy_value / (px * (1 + cost_rate))) // 100) * 100
                    if qty <= 0:
                        continue  # 不足一手，跳过
                    actual_cost = qty * px * cost_rate
                    actual_value = qty * px + actual_cost
                    if actual_value > cash:
                        qty -= 100  # 减一手
                        if qty <= 0:
                            continue
                        actual_cost = qty * px * cost_rate
                        actual_value = qty * px + actual_cost
                    trade_log.append([str(t)[:10], s, 'BUY', round(px, 3), int(qty)])
                    shares[s] = shares.get(s, 0.0) + qty
                    cash -= actual_value
                    total_cost += actual_cost

                n_rebalances += 1
                holdings_history.append({
                    'date': t, 'n_holdings': len(shares),
                    'symbols': sorted(shares.keys())[:10],
                })

            # 3. 逐日权益 + 过程记录
            pos_value = 0.0
            day_hold, day_contrib = [], []
            prev_close = self.close_mx.shift(1).loc[t]
            for s2, q2 in shares.items():
                p2 = prices.get(s2, np.nan)
                if pd.isna(p2):
                    continue
                pos_value += q2 * p2
                day_hold.append([s2, round(q2 * p2, 0)])
                pp2 = prev_close.get(s2, np.nan)
                if not pd.isna(pp2) and pp2 > 0:
                    day_contrib.append([s2, round(q2 * (p2 - pp2), 0)])
            eq_total = cash + pos_value
            process_holdings[str(t)[:10]] = sorted(day_hold, key=lambda x: -x[1])
            process_contrib[str(t)[:10]] = day_contrib
            equity_curve[t] = eq_total

        # ---- 绩效 ----
        equity = pd.Series(equity_curve).sort_index()
        final_capital = float(equity.iloc[-1])
        total_return = final_capital / cfg.initial_capital - 1
        n_days = len(equity)
        annual_return = (1 + total_return) ** (BacktestParams.TRADING_DAYS_PER_YEAR / max(n_days, 1)) - 1
        rets = equity.pct_change().dropna()
        sharpe = float((rets.mean() * BacktestParams.TRADING_DAYS_PER_YEAR) /
                       (rets.std() * np.sqrt(BacktestParams.TRADING_DAYS_PER_YEAR))) if rets.std() > 0 else 0.0
        dd = (equity / equity.cummax() - 1).min()

        benchmark_return = None
        if cfg.benchmark_symbol in self.close_mx.columns:
            bench = self.close_mx[cfg.benchmark_symbol].loc[dates[0]:dates[-1]].dropna()
            if len(bench) >= 2 and bench.iloc[0] > 0:
                benchmark_return = float(bench.iloc[-1] / bench.iloc[0] - 1)

        return PortfolioResult(
            config=cfg,
            n_dates=n_days,
            n_rebalances=n_rebalances,
            initial_capital=cfg.initial_capital,
            final_capital=final_capital,
            total_return=float(total_return),
            annual_return=float(annual_return),
            sharpe=sharpe,
            max_drawdown=float(dd),
            avg_turnover=float(np.mean(turnover_list)) if turnover_list else 0.0,
            total_cost=float(total_cost),
            benchmark_return=benchmark_return,
            equity_curve=equity,
            holdings_history=holdings_history,
            daily_holdings=process_holdings,
            daily_contrib=process_contrib,
            trade_log=trade_log,
        )

    def _select(self, scores: pd.Series, exclude: set,
                score_band: Optional[tuple] = None,
                date=None) -> List[str]:
        """调仓日选股（排除涨停/流动性不足/上市不足N日/无分标的）"""
        from .construction import select_portfolio

        # 上市天数过滤：需≥N个交易日数据（动态处理新上市和退市）
        if self.config.min_listing_days > 0 and date is not None:
            try:
                age = self.data_age.loc[date]
                too_new = set(age[age < self.config.min_listing_days].index)
                exclude = exclude | too_new
            except KeyError:
                pass

        return select_portfolio(scores, self.config.top_n,
                                exclude_symbols=exclude, band=score_band)

    def _default_weight_func(self, symbols, date=None):
        from .construction import equal_weights
        return equal_weights(symbols, self.config.max_weight)
