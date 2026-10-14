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
from trading import config as chain
from trading import rules
from trading.rules import limit_ratio_series
from .risk_manager import PortfolioRiskManager, RiskLimits
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class PortfolioConfig:
    top_n: int = chain.TOP_N              # 目标持仓数（主链默认=生产100）
    rebalance_days: int = chain.REBALANCE_DAYS_BT  # 调仓周期（交易日）
    initial_capital: float = 1_000_000.0
    commission: float = chain.COMMISSION    # 佣金（双边）
    stamp_tax: float = chain.STAMP_TAX      # 印花税（卖出）
    transfer_fee: float = chain.TRANSFER_FEE  # 过户费（双边）
    slippage: float = chain.SLIPPAGE        # 单边滑点
    enforce_price_limits: bool = True
    max_weight: float = chain.MAX_WEIGHT    # 单票权重上限
    weight_method: str = 'equal'            # equal | inverse_vol
    benchmark_symbol: str = '000300.SH'
    limit_tolerance: float = chain.LIMIT_TOLERANCE  # 涨跌停判定容差
    buffer_band: float = chain.BUFFER_BAND  # 权重缓冲带（主链默认=生产20%；复现旧全量换仓回测显式传0）
    max_participation: float = chain.MAX_PARTICIPATION  # 单日最大成交量占比
    min_daily_turnover: float = chain.MIN_DAILY_TURNOVER  # 最低日均成交额（流动性过滤）
    impact_coeff: float = chain.IMPACT_COEFF  # 市场冲击系数（sqrt模型）
    min_listing_days: int = chain.MIN_LISTING_DAYS  # 上市天数过滤
    risk_limits: RiskLimits = field(default_factory=RiskLimits)  # 风控约束
    industry_map: Dict[str, str] = field(default_factory=dict)
    # 行业映射 {symbol: 行业名}——提供后 max_sector_weight 生效；缺省时行业约束不启用
    st_matrix: Optional[pd.DataFrame] = None
    # dates×symbols 的 ST 布尔矩阵（data/st_provider.STProvider.st_matrix）——
    # 提供后：主板ST涨跌幅按±5%判定、ST股不入选；缺省时按非ST处理（旧行为）


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
    # 统一基准组：多基准对比（消除"超额收益"口径歧义）
    benchmark_suite: Dict[str, Optional[float]] = field(default_factory=dict)
    # {name: 区间收益}，如 {'沪深300': 0.12, '中证500': 0.08, '等权对照': 0.15}
    benchmark_curves: Dict[str, pd.Series] = field(default_factory=dict)
    holdings_history: List[Dict] = field(default_factory=list)
    # 回测过程明细（供过程回放）
    daily_holdings: Dict = field(default_factory=dict)   # {date: [[sym, weight], ...]}
    daily_contrib: Dict = field(default_factory=dict)    # {date: [[sym, 贡献], ...]}
    trade_log: List = field(default_factory=list)        # [date, sym, side, price, qty]
    risk_events: Dict[str, int] = field(default_factory=dict)
    # 生效的风控事件计数：turnover_capped / market_cb_days / sector_capped

    def summary_markdown(self) -> str:
        from trading.config import BACKTEST_ONLY, LIVE_ONLY
        lines = ["## 组合回测结果\n"]
        lines.append(f"> ⚠️ **口径声明**：本回测含 {BACKTEST_ONLY}，不含 {LIVE_ONLY}。"
                     "实盘风控集合不同，收益/回撤/夏普不可直接对比实盘。\n")
        lines.append(f"- 持仓数: Top{self.config.top_n} | 调仓: 每{self.config.rebalance_days}个交易日"
                     f" | 加权: {self.config.weight_method}")
        lines.append(f"- 区间交易日: {self.n_dates} | 调仓次数: {self.n_rebalances}")
        lines.append(f"| 指标 | 值 |")
        lines.append("|------|------|")
        lines.append(f"| 总收益 | {self.total_return:+.2%} |")
        for bname, bret in self.benchmark_suite.items():
            lines.append(f"| 基准：{bname} | {bret:+.2%} | 超额 {self.total_return - bret:+.2%} |")
        lines.append(f"| 年化收益 | {self.annual_return:+.2%} |")
        lines.append(f"| 夏普 | {self.sharpe:.2f} |")
        lines.append(f"| 最大回撤 | {self.max_drawdown:.2%} |")
        lines.append(f"| 平均换手/次 | {self.avg_turnover:.1%} |")
        lines.append(f"| 累计成本 | {self.total_cost:,.0f} ({self.total_cost / self.initial_capital:.2%} 本金) |")
        if self.risk_events:
            lines.append("- 风控事件：" + "、".join(f"{k}×{v}" for k, v in self.risk_events.items()))
        return "\n".join(lines)


class PortfolioBacktester:
    """TopN 组合回测器（向量化，接受日频综合分矩阵驱动调仓）"""

    def __init__(self, cache: Dict[str, pd.DataFrame], config: PortfolioConfig):
        self.config = config
        close = {}          # 不复权（涨跌停判定用——与真实市场价一致）
        adjclose = {}       # 前复权（净值/收益计算用——跨除权连续）
        volume = {}
        amount = {}
        for sym, df in cache.items():
            if df is not None and not df.empty:
                idx = pd.DatetimeIndex(pd.to_datetime(df['date'])).as_unit('ns')
                close[sym] = pd.Series(pd.to_numeric(df['close'], errors='coerce').values, index=idx)
                if 'adjclose' in df.columns:
                    adjclose[sym] = pd.Series(pd.to_numeric(df['adjclose'], errors='coerce').values, index=idx)
                else:
                    adjclose[sym] = close[sym]     # 无复权价时退化（不跨除权场景OK）
                if 'volume' in df.columns:
                    volume[sym] = pd.Series(pd.to_numeric(df['volume'], errors='coerce').values, index=idx)
                if 'amount' in df.columns:
                    amount[sym] = pd.Series(pd.to_numeric(df['amount'], errors='coerce').values, index=idx)
        # 净值/收益矩阵用复权价；涨跌停矩阵用不复权价
        self.close_mx = pd.DataFrame(adjclose).sort_index()     # 收益计算（复权）
        self.prev_close_mx = self.close_mx.shift(1)
        self.raw_close_mx = pd.DataFrame(close).sort_index()    # 涨跌停判定（不复权）
        self._check_bench_continuity()
        # 风险管理器（多层级风控）
        self.risk_manager = PortfolioRiskManager(config.risk_limits)
        # ST 状态矩阵（PIT，主板±5%与选股排除）
        self.st_matrix = config.st_matrix
        if self.st_matrix is not None:
            self.st_matrix = self.st_matrix.copy()
            self.st_matrix.index = pd.DatetimeIndex(
                pd.to_datetime(self.st_matrix.index)).as_unit('ns')   # 单位与close_mx对齐
            self.st_matrix = self.st_matrix.reindex(
                index=self.close_mx.index.intersection(self.st_matrix.index),
                columns=self.close_mx.columns.intersection(self.st_matrix.columns))
        # 行业映射（提供后行业集中度约束生效）
        self.industry_map = config.industry_map or {}
        if config.risk_limits.max_sector_weight > 0 and not self.industry_map:
            logger.info("[Risk] 行业集中度约束未启用：未提供 PortfolioConfig.industry_map")
        # 成交量矩阵（用于流动性过滤和冲击成本）
        self.volume_mx = pd.DataFrame(volume).reindex(self.close_mx.index).sort_index()
        # 20日均成交额（元）：经由 semantics.turnover_matrix 统一口径
        from data.semantics import turnover_matrix
        amount_mx = pd.DataFrame(amount).reindex(self.close_mx.index).sort_index()
        turnover_src = turnover_matrix(self.close_mx, self.volume_mx, amount_mx)
        self.avg_turnover_20d = turnover_src.rolling(20).mean()
        # 上市天数矩阵（累计有效数据天数，用于上市天数过滤）
        # 动态处理：新上市股在达到 min_listing_days 前不可选，退市股数据变NaN后自然排除
        self.data_age = self.close_mx.notna().cumsum()

    def _check_bench_continuity(self):
        """基准连续性防线：|单日涨跌|>20% 即多源拼接断层（2024-09-09曾现-35%假跳变，
        触发过649天假停机）。发现即大声告警——结果不可信，先重建指数缓存
        （scripts/rebuild_index_cache.py）。"""
        b = self.close_mx.get(self.config.benchmark_symbol)
        if b is None or len(b) < 3:
            return
        chg = b.pct_change().abs()
        jumps = chg[chg > 0.20]
        if len(jumps):
            dates = [str(d)[:10] for d in jumps.index]
            logger.error(f"[Bench] 基准{self.config.benchmark_symbol}存在{len(jumps)}处"
                         f">20%假跳变（{dates[:3]}...）——数据断层，回测/熔断结果不可信！"
                         f"运行 scripts/rebuild_index_cache.py 重建")

    # ---- 涨跌停判定（向量化，ST感知）----

    def _st_set_at(self, date) -> set:
        """date 日处于ST状态的股票集合（无矩阵时空集=按非ST处理）"""
        if self.st_matrix is None or date not in self.st_matrix.index:
            return set()
        row = self.st_matrix.loc[date]
        return set(row[row].index)

    def _limit_up_mask(self, date) -> pd.Series:
        """当日收盘封涨停的股票（不可买）；主板ST按±5%（用不复权价判定——与真实市场一致）"""
        if not self.config.enforce_price_limits:
            return pd.Series(dtype=bool)
        cur = self.raw_close_mx.loc[date]
        prev = self.raw_close_mx.shift(1).loc[date]
        limit = limit_ratio_series(cur.index, st_symbols=self._st_set_at(date))
        tol = prev * self.config.limit_tolerance
        return cur >= (prev * (1 + limit) - tol).where(limit.notna(), -np.inf)

    def _limit_down_mask(self, date) -> pd.Series:
        """当日收盘封跌停的股票（不可卖）；主板ST按±5%（用不复权价判定）"""
        if not self.config.enforce_price_limits:
            return pd.Series(dtype=bool)
        cur = self.raw_close_mx.loc[date]
        prev = self.raw_close_mx.shift(1).loc[date]
        limit = limit_ratio_series(cur.index, st_symbols=self._st_set_at(date))
        tol = prev * self.config.limit_tolerance
        return cur <= (prev * (1 - limit) + tol).where(limit.notna(), np.inf)

    @staticmethod
    def _limit_ratios(symbols) -> pd.Series:
        """按代码板块的涨跌停幅度（主链共享规则；PIT的ST感知版本见 _limit_*_mask）"""
        return limit_ratio_series(symbols)

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
        risk_events: Dict[str, int] = {}   # turnover_capped / market_cb_days / sector_capped

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
                        px = rules.round_tick(sym, float(prices[sym]) * (1 - cfg.slippage))
                        value = shares[sym] * px
                        cost = value * (cfg.commission + cfg.stamp_tax + cfg.transfer_fee)
                        cash += value - cost
                        total_cost += cost
                        trade_log.append([str(t)[:10], sym, 'SELL', round(px, 3), int(shares[sym])])
                        del shares[sym]
                    else:
                        still_blocked.append(sym)
                pending_sells = still_blocked

            # 2. 市场熔断：基准单日跌幅≥熔断线 → 当日禁买（卖出照常，与生产闸门no_buy一致）
            no_buy_today = False
            if cfg.benchmark_symbol in self.close_mx.columns:
                b1 = prices.get(cfg.benchmark_symbol, np.nan)
                b0 = self.prev_close_mx.loc[t].get(cfg.benchmark_symbol, np.nan)
                if not pd.isna(b1) and not pd.isna(b0) and b0 > 0:
                    if self.risk_manager.check_market_circuit_breaker(float(b1) / float(b0) - 1):
                        no_buy_today = True
                        risk_events['market_cb_days'] = risk_events.get('market_cb_days', 0) + 1

            # 3. 调仓日：构造目标组合并交易差额
            if idx in rebalance_idx and t in score_mx.index:
                scores = score_mx.loc[t]
                up_mask = self._limit_up_mask(t)
                liq_mask = self._liquidity_mask(t)
                # 排除当日涨停（不可买）、流动性不足、已无价格数据的股票
                buyable_excl = set(up_mask[up_mask == True].index) | set(liq_mask[liq_mask == True].index)  # noqa: E712
                targets = self._select(scores, buyable_excl, score_band, date=t)
                target_w = weight_func(targets, t)
                # 风控约束链：回撤降仓/单票上限/现金底线 → 行业集中度 → 换手上限
                target_w = self.risk_manager.adjust_weights(target_w)
                n0 = len(self.risk_manager.state.active_constraints)
                target_w = self.risk_manager.cap_sector_weights(target_w, self.industry_map)
                if len(self.risk_manager.state.active_constraints) > n0:
                    risk_events['sector_capped'] = risk_events.get('sector_capped', 0) + 1

                # 目标市值与当前权重
                total_equity = cash + sum(
                    shares[s] * prices[s] for s in shares if not pd.isna(prices.get(s, np.nan))
                )
                cur_value = {
                    s: shares[s] * prices[s] for s in shares
                    if not pd.isna(prices.get(s, np.nan))
                }
                denom = max(total_equity, 1e-9)
                cur_w_map = {s: v / denom for s, v in cur_value.items()}

                # 换手上限：目标隐含换手超限 → 目标权重向当前权重回缩（实际换手=上限）
                allowed, _tn = self.risk_manager.check_turnover(cur_w_map, target_w)
                if not allowed:
                    risk_events['turnover_capped'] = risk_events.get('turnover_capped', 0) + 1
                    target_w = self.risk_manager.blend_for_turnover(cur_w_map, target_w)
                target_value = {s: total_equity * w for s, w in target_w.items()}

                # 换手率（回缩后的实际值，单边）
                all_syms = set(cur_value) | set(target_value)
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
                        px = rules.round_tick(s, float(prices[s]) * (1 - cfg.slippage))
                        value = shares[s] * px
                        cost = value * (cfg.commission + cfg.stamp_tax + cfg.transfer_fee)
                        cash += value - cost
                        total_cost += cost
                        trade_log.append([str(t)[:10], s, 'SELL', round(px, 3), int(shares[s])])
                        del shares[s]

                # 买入：新目标与加仓（缓冲带内低配不调）；市场熔断日禁买
                if not no_buy_today:
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
                        px = rules.round_tick(s, float(prices[s]) * (1 + impact_slip))
                        cost_rate = cfg.commission + cfg.transfer_fee
                        # 申报数量：科创板≥200且1股递增；北交所≥100且1股递增；其余100整手
                        qty = rules.round_buy_qty(
                            s, int(buy_value / (px * (1 + cost_rate))))
                        if qty <= 0:
                            continue  # 不足最低申报，跳过
                        actual_cost = qty * px * cost_rate
                        actual_value = qty * px + actual_cost
                        if actual_value > cash:
                            # 资金不足：按申报规则重算可承受数量（科创板/北交所1股递增，
                            # 此前固定 -=100 会产生非法数量或仍超资金）
                            qty = rules.round_buy_qty(
                                s, int(cash / (px * (1 + cost_rate))))
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

            # 3. 逐日权益 + 过程记录（顺序：先估值→风控检查→必要时清仓→记录最终净值）
            pos_value = 0.0
            day_hold, day_contrib = [], []
            prev_close = self.close_mx.shift(1).loc[t]
            for s2, q2 in shares.items():
                p2 = prices.get(s2, np.nan)
                if pd.isna(p2):
                    continue
                pos_value += q2 * p2
                day_contrib.append([s2, round(q2 * (p2 - prev_close.get(s2, np.nan))
                                           if not pd.isna(prev_close.get(s2, np.nan)) else 0, 0)])
            eq_pre = cash + pos_value

            # 每日更新风险状态（回撤/波动率/熔断检查）
            self.risk_manager.update(eq_pre, t)
            if self.risk_manager.state.is_halted:
                # 硬限触发：清仓所有持仓（先完成清仓与费用，再记录当日净值；
                # 跌停/停牌不可卖 → 顺延 pending_sells，之后每个交易日继续尝试）
                down_mask = self._limit_down_mask(t)
                for s2 in list(shares.keys()):
                    p2 = prices.get(s2, np.nan)
                    if pd.isna(p2) or down_mask.get(s2, False):
                        pending_sells.append(s2)   # 跌停顺延，不虚构可成交
                        continue
                    px = rules.round_tick(s2, float(p2) * (1 - cfg.slippage))
                    value = shares[s2] * px
                    cost = value * (cfg.commission + cfg.stamp_tax + cfg.transfer_fee)
                    trade_log.append([str(t)[:10], s2, 'SELL', round(px, 3), int(shares[s2])])
                    cash += value - cost
                    total_cost += cost
                    del shares[s2]

            # 清仓后的最终持仓估值（跌停顺延部分仍按市值计）
            pos_value = 0.0
            for s2, q2 in shares.items():
                p2 = prices.get(s2, np.nan)
                if not pd.isna(p2):
                    pos_value += q2 * p2
                    day_hold.append([s2, round(q2 * p2, 0)])
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

        # 统一基准组：沪深300 + 中证500 + 中证1000 + 同池等权
        benchmark_suite = {}
        benchmark_curves = {}
        bench_idx = {'沪深300': '000300.SH', '中证500': '000905.SH', '中证1000': '000852.SH'}
        for name, code in bench_idx.items():
            if code in self.close_mx.columns:
                bench = self.close_mx[code].loc[dates[0]:dates[-1]].dropna()
                if len(bench) >= 2 and bench.iloc[0] > 0:
                    benchmark_suite[name] = float(bench.iloc[-1] / bench.iloc[0] - 1)
                    benchmark_curves[name] = bench / bench.iloc[0]
        # 同池等权统计参照（不可交易口径：无过滤/成本/再平衡——仅统计对照，
        # 严格可交易对照见 portfolio/tradable_benchmark.py）
        pool_cols = [c for c in self.close_mx.columns if c not in bench_idx.values()]
        if pool_cols:
            pool_ret = self.close_mx[pool_cols].loc[dates[0]:dates[-1]].pct_change().mean(axis=1)
            ew_curve = (1 + pool_ret.fillna(0)).cumprod()
            if len(ew_curve) >= 2:
                benchmark_suite['等权统计参照'] = float(ew_curve.iloc[-1] - 1)
                benchmark_curves['等权统计参照'] = ew_curve

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
            benchmark_suite=benchmark_suite,
            benchmark_curves=benchmark_curves,
            equity_curve=equity,
            holdings_history=holdings_history,
            daily_holdings=process_holdings,
            daily_contrib=process_contrib,
            trade_log=trade_log,
            risk_events=risk_events,
        )

    def _select(self, scores: pd.Series, exclude: set,
                score_band: Optional[tuple] = None,
                date=None) -> List[str]:
        """调仓日选股（排除涨停/流动性不足/上市不足N日/ST/无分标的）"""
        from .construction import select_portfolio

        # 上市天数过滤：需≥N个交易日数据（动态处理新上市和退市）
        if self.config.min_listing_days > 0 and date is not None:
            try:
                age = self.data_age.loc[date]
                too_new = set(age[age < self.config.min_listing_days].index)
                exclude = exclude | too_new
            except KeyError:
                pass

        # ST 过滤（PIT）：主板ST涨跌幅±5%、退市风险高，不入选
        if date is not None:
            exclude = exclude | self._st_set_at(date)

        return select_portfolio(scores, self.config.top_n,
                                exclude_symbols=exclude, band=score_band)

    def _default_weight_func(self, symbols, date=None):
        from .construction import equal_weights
        return equal_weights(symbols, self.config.max_weight)
