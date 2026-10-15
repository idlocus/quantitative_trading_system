#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
每日重放回测器 —— 严格复刻生产策略语义（回测即生产的时间机器）

与 PortfolioBacktester（每N日调仓的近似口径）不同，本回测器逐日重放生产链路，
每个环节直接 import 生产组件，不重写第二套逻辑：

    T-1 收盘信号（PIT 过滤：可交易/上市天数/流动性，与 generate_target 同口径）
      → live.signals.build_orders（缓冲带，与生产同一函数）
      → live.risk_gate.RiskGate（L2停机/L3熔断/L4止损，与生产同一实现）
      → live.broker.PaperBroker（同费用/板块涨跌停/整手）
      → live.ledger.Ledger（同入账去重/净值序列，逐日 mark_to_market）

语义要点（正是与每N日回测的差异所在）：
- 每日排名变化：跌出 TopN 的持仓**当日立即卖出**（缓冲带只管权重漂移，不管进出）
- 新进 TopN 的股票当日买入
- L4 止损每日检查，强制卖单逐日注入（忠实复刻生产行为，含"止损次日若仍在
  目标会被回买"——这正是需要本工具才能暴露的语义）
"""

import os
import tempfile
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from live.broker import PaperBroker
from live.ledger import Ledger
from live.risk_gate import RiskGate
from live.signals import build_orders
from trading import config as chain
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ReplayConfig:
    top_n: int = chain.TOP_N
    band: float = chain.BUFFER_BAND
    initial_capital: float = 1_000_000.0
    min_listing_days: int = chain.MIN_LISTING_DAYS
    min_daily_turnover: float = chain.MIN_DAILY_TURNOVER
    use_risk_gate: bool = True          # 生产默认开闸门；关闭仅供对照
    benchmark_symbol: str = '000300.SH'
    rebalance_days: int = 1             # 每日重放（工件payload字段兼容）
    st_matrix: Optional[pd.DataFrame] = None
    # dates×symbols ST布尔矩阵（STProvider.st_matrix）：选股排除+主板ST按±5%撮合
    st_data_required: bool = False
    # True=ST数据不可用（矩阵缺失）时禁买（与生产闸门同安全口径）；
    # False=研究对照模式（无ST矩阵按非ST处理，历史默认）
    gate_config: object = None
    # RiskGate.GateConfig 覆盖（如 restart_after_liquidation）；None=生产默认


@dataclass
class ReplayResult:
    config: ReplayConfig
    n_dates: int
    initial_capital: float
    final_capital: float
    total_return: float
    annual_return: float
    sharpe: float
    max_drawdown: float
    n_rebalances: int                  # 有成交日数（工件payload字段兼容）
    n_trade_days: int                   # 有成交的交易日数
    total_cost: float
    avg_turnover: float                 # 单边平均换手/有成交日（工件payload兼容）
    n_stop_sells: int                   # L4 止损强制卖出笔数
    n_halt_days: int                    # L2 停机天数
    n_no_buy_days: int                  # L3 禁买天数
    benchmark_return: Optional[float] = None   # 工件payload兼容（沪深300）
    equity_curve: pd.Series = None
    trades: List[dict] = field(default_factory=list)
    benchmark_suite: Dict[str, Optional[float]] = field(default_factory=dict)
    daily_holdings: Dict = field(default_factory=dict)   # {date: [[sym, 市值], ...]}（审计）
    daily_contrib: Dict = field(default_factory=dict)    # {date: [[sym, 贡献], ...]}（审计）
    trade_log: List = field(default_factory=list)        # [[date, sym, side, price, qty]]（审计）
    terminal_note: Optional[str] = None                  # 终态说明（空仓停机等，进工件）
    # 收益评价拆分（终态存在时）：整体收益会被"停机后现金曲线"稀释，
    # 必须区分运行期与停机期，否则年化数字冒充策略长期表现
    terminal_date: Optional[str] = None                  # 最后有持仓日（None=全程运行）
    active_period_return: Optional[float] = None         # 运行期收益（区间起点→terminal_date）
    idle_trading_days: int = 0                           # 停机期交易日数（terminal_date之后）
    first_liquidation_date: Optional[str] = None         # 首次硬限清仓日
    n_liquidation_events: int = 0                        # 硬限清仓次数（restart模式可>1）
    restart_mode: bool = False                           # 硬限后自动重启（否则停机）

    def summary_markdown(self) -> str:
        from trading.config import BACKTEST_ONLY, LIVE_ONLY
        lines = ["## 每日重放回测（严格生产语义）\n"]
        lines.append(f"> ⚠️ **口径声明**：本重放使用生产闸门（RiskGate），"
                     f"不含回测专属约束 {BACKTEST_ONLY}。"
                     "与近似口径回测的差异包含约束集合差异，非纯调仓频率差异。\n")
        lines.append(f"- 持仓 Top{self.config.top_n} | 缓冲带 {self.config.band:.0%} | "
                     f"风控闸门 {'开' if self.config.use_risk_gate else '关'}")
        # 终态醒目上报：若后期空仓停机（如硬限清仓），必须明说——净值平坦≠策略在运行
        if self.trades:
            last_trade = self.trades[-1]['date']
            last_hold = max((d for d, h in self.daily_holdings.items() if h), default=None)
            last_day = str(self.equity_curve.index[-1])[:10] if self.equity_curve is not None \
                and len(self.equity_curve) else last_trade
            if last_hold and last_hold < last_day:
                n_flat = len([d for d in self.equity_curve.index
                              if str(d)[:10] > last_hold]) if self.equity_curve is not None else 0
                lines.append(f"- **⚠ 终态：{last_hold} 起空仓停机**（最后成交 {last_trade}，"
                             f"此后 {n_flat} 个交易日持币观望——硬限清仓或禁买所致，"
                             f"区间收益为现金收益，不代表策略持续运行）")
                # 收益分相：整体数字会被停机期现金曲线稀释，必须分开呈现
                if self.active_period_return is not None:
                    cash_ret = (1 + self.total_return) / (1 + self.active_period_return) - 1
                    lines.append(f"- **收益分相**：运行期（→{last_hold}）"
                                 f"**{self.active_period_return:+.1%}**｜"
                                 f"停机期（{n_flat} 交易日，现金）{cash_ret:+.1%}｜"
                                 f"整体 {self.total_return:+.1%}——"
                                 f"评价策略用运行期数字，整体数字含现金拖累/缓冲")
        lines.append(f"- 区间交易日: {self.n_dates} | 有成交日: {self.n_trade_days} | "
                     f"止损卖 {self.n_stop_sells} 笔 | 停机 {self.n_halt_days} 天 | "
                     f"禁买 {self.n_no_buy_days} 天")
        lines.append("| 指标 | 值 |")
        lines.append("|------|------|")
        lines.append(f"| 总收益 | {self.total_return:+.2%} |")
        for bname, bret in self.benchmark_suite.items():
            lines.append(f"| 基准：{bname} | {bret:+.2%} | 超额 {self.total_return - bret:+.2%} |")
        lines.append(f"| 年化收益 | {self.annual_return:+.2%} |")
        lines.append(f"| 夏普 | {self.sharpe:.2f} |")
        lines.append(f"| 最大回撤 | {self.max_drawdown:.2%} |")
        lines.append(f"| 累计成本 | {self.total_cost:,.0f} "
                     f"({self.total_cost / self.initial_capital:.2%} 本金) |")
        return "\n".join(lines)


class DailyReplayBacktester:
    """逐日重放生产链路的回测器（组件与生产同源，见模块docstring）"""

    def __init__(self, cache: Dict[str, pd.DataFrame], config: ReplayConfig = None):
        self.cfg = config or ReplayConfig()
        close = {}          # 不复权（撮合/涨跌停/成交价——与真实市场一致）
        adjclose = {}       # 前复权（净值/收益——跨除权连续）
        volume = {}
        amount = {}
        self._bench_df = None
        for sym, df in cache.items():
            if df is None or df.empty:
                continue
            if sym == self.cfg.benchmark_symbol:
                self._bench_df = df.copy()
            idx = pd.DatetimeIndex(pd.to_datetime(df['date'])).as_unit('ns')
            close[sym] = pd.Series(pd.to_numeric(df['close'], errors='coerce').values, index=idx)
            if 'adjclose' in df.columns:
                adjclose[sym] = pd.Series(pd.to_numeric(df['adjclose'], errors='coerce').values, index=idx)
            else:
                adjclose[sym] = close[sym]
            if 'volume' in df.columns:
                volume[sym] = pd.Series(pd.to_numeric(df['volume'], errors='coerce').values, index=idx)
            if 'amount' in df.columns:
                amount[sym] = pd.Series(pd.to_numeric(df['amount'], errors='coerce').values, index=idx)
        self.close_mx = pd.DataFrame(adjclose).sort_index()     # 收益/净值（复权）
        self.prev_close_mx = self.close_mx.shift(1)
        self.raw_close_mx = pd.DataFrame(close).sort_index()    # 撮合/涨跌停（不复权）
        self._check_bench_continuity()
        # 成交额（元）：经由 semantics.turnover_matrix 统一口径
        from data.semantics import turnover_matrix
        vol_mx = pd.DataFrame(volume).reindex(self.close_mx.index).sort_index()
        amount_mx = pd.DataFrame(amount).reindex(self.close_mx.index).sort_index()
        turnover_src = turnover_matrix(self.close_mx, vol_mx, amount_mx)
        self.avg_turnover_20d = turnover_src.rolling(20).mean()
        self.data_age = self.close_mx.notna().cumsum()
        # ST 状态矩阵（PIT）——注意用 self.cfg（config=None 时默认初始化，公共接口安全）
        self.st_matrix = self.cfg.st_matrix
        if self.st_matrix is not None:
            self.st_matrix = self.st_matrix.copy()
            self.st_matrix.index = pd.DatetimeIndex(
                pd.to_datetime(self.st_matrix.index)).as_unit('ns')   # 单位与close_mx对齐
            self.st_matrix = self.st_matrix.reindex(
                columns=self.st_matrix.columns.intersection(self.close_mx.columns))

    def _check_bench_continuity(self):
        """基准连续性防线（同 PortfolioBacktester）：|单日涨跌|>20%即多源拼接断层。"""
        b = self.close_mx.get(self.cfg.benchmark_symbol)
        if b is None or len(b) < 3:
            return
        chg = b.pct_change().abs()
        jumps = chg[chg > 0.20]
        if len(jumps):
            dates = [str(d)[:10] for d in jumps.index]
            logger.error(f"[Bench] 基准{self.cfg.benchmark_symbol}存在{len(jumps)}处"
                         f">20%假跳变（{dates[:3]}...）——数据断层，重放/闸门结果不可信！"
                         f"运行 scripts/rebuild_index_cache.py 重建")

    # ---------------- 主流程 ----------------

    def run(self, score_mx: pd.DataFrame,
            start_date: str = None, end_date: str = None,
            debug_from: str = None) -> ReplayResult:
        """debug_from: 'YYYY-MM-DD' 起打印每日订单/闸门/成交细节（诊断断链用）"""
        cfg = self.cfg
        tmpdir = tempfile.mkdtemp(prefix='daily_replay_')
        ledger = Ledger(os.path.join(tmpdir, 'ledger.json'))
        ledger.initialize(cfg.initial_capital)
        ledger.save = lambda: None   # 性能：重放期间不落盘JSON（O(n²)），结果读内存
        gate = RiskGate(self.cfg.gate_config,
                        state_path=os.path.join(tmpdir, 'risk_state.json')) \
            if cfg.use_risk_gate else None
        broker = PaperBroker()

        dates = self.close_mx.index
        if start_date:
            dates = dates[dates >= pd.Timestamp(start_date)]
        if end_date:
            dates = dates[dates <= pd.Timestamp(end_date)]
        sig_dates = score_mx.index.sort_values()
        n_halt = n_no_buy = 0
        n_liquidations = 0
        first_liq_date = None

        for i, T in enumerate(dates):
            Td = T.date() if hasattr(T, 'date') else T
            prices = {s: float(v) for s, v in self.raw_close_mx.loc[T].dropna().items()}
            prev_prices = {s: float(v) for s, v in self.raw_close_mx.shift(1).loc[T].dropna().items()}

            # 信号 = T-1（前一交易日）；分数行 = ≤T-1 的最新因子行（公告日粒度，
            # 与 generate_target 的 valid_dates[-1] 语义一致，无前视）
            if i > 0:
                sig_t = dates[i - 1]
                spos = sig_dates.searchsorted(sig_t, side='right') - 1
                if spos >= 0:
                    row_date = sig_dates[spos]
                    targets = self._targets_at(score_mx, row_date, sig_t)

                    # 风控前净值快照：用当日价格先行 mark-to-market，确保闸门读到
                    # 当日实际净值（而非上一交易日的滞后值）——否则回撤/单日熔断
                    # 会晚一天触发。交易完成后再做最终 mark-to-market。
                    if ledger.positions:
                        ledger.mark_to_market(prices, Td)

                    nav = ledger.last_nav() or ledger.cash
                    cur_w = ledger.current_weights(prices) if ledger.positions else {}
                    orders = build_orders(targets, cur_w, nav, top_n=cfg.top_n, band=cfg.band)

                    if gate is not None:
                        bench_slice = None
                        if self._bench_df is not None:
                            bench_slice = self._bench_df[self._bench_df['date'] <= T]
                        st_ok = self._st_data_ok(T)
                        decision = gate.review(orders, ledger, prices, prev_prices, targets,
                                               bench_df=bench_slice, as_of=Td,
                                               st_data_ok=st_ok)
                        exec_orders = decision.orders
                        if decision.action == 'halt':
                            n_halt += 1
                        elif decision.action == 'no_buy':
                            n_no_buy += 1
                        # 硬限清仓事件统计（有持仓时的 liquidate 才是真实清仓触发）
                        if decision.action == 'liquidate' and ledger.positions:
                            n_liquidations += 1
                            if first_liq_date is None:
                                first_liq_date = str(Td)
                    else:
                        exec_orders = orders

                    fills = broker.execute(exec_orders, prices, Td, prev_prices=prev_prices,
                                           available_cash=ledger.cash,
                                           st_symbols=self._st_set_at(T),
                                           positions=dict(ledger.positions))
                    # positions 必传（与 run_daily/run_live 同口径）：卖出数量经
                    # round_sell_qty 钳制到实际持仓。漏传时金额制全仓卖单会因
                    # 滑点+费用膨胀出 1~4 股超额 → 账本异常队列 → L2.6 永久禁买，
                    # 重放从首次调仓起就锁死在只卖不买（2026-10-15 十年回测教训）。
                    ledger.apply_fills(fills)
                    if debug_from and str(Td) >= debug_from:
                        st = {}
                        for f_ in fills:
                            k = f_.status if f_.quantity > 0 else f_.status[:12]
                            st[k] = st.get(k, 0) + 1
                        logger.info(
                            f"[replay-dbg {Td}] targets={len(targets)} "
                            f"orders={len(orders)} exec={len(exec_orders)} fills={st} "
                            f"cash={ledger.cash:,.0f} nav={nav:,.0f} "
                            f"gate={getattr(decision, 'action', 'n/a') if gate else 'off'} "
                            f"suppressed={len(getattr(decision, 'suppressed', [])) if gate else 0}")

            # 逐日净值（生产同路径：mark_to_market）
            ledger.mark_to_market(prices, Td)

        return self._summarize(ledger, dates, n_halt, n_no_buy,
                               n_liquidations, first_liq_date)

    # ---------------- 生产过滤（口径=generate_target） ----------------

    def _st_data_ok(self, T) -> bool:
        """ST 数据可用性：矩阵存在且 T 日有覆盖（矩阵缺行=该日按非ST处理=安全风险）"""
        if not self.cfg.st_data_required:
            return True
        if self.st_matrix is None:
            return False
        return T in self.st_matrix.index

    def _st_set_at(self, T) -> set:
        if self.st_matrix is None or T not in self.st_matrix.index:
            return set()
        row = self.st_matrix.loc[T]
        return set(row[row].index)

    def _targets_at(self, score_mx, row_date, filt_t) -> List[str]:
        """分数取 row_date（≤信号日的最新因子行），过滤在信号日 filt_t 口径评估。
        选股池自动约束在有行情的列（K线未覆盖的标的不可交易）。"""
        cfg = self.cfg
        scores = score_mx.loc[row_date].dropna()
        scores = scores[scores.index.intersection(self.close_mx.columns)]   # 防御：仅池内
        tradable = self.close_mx.loc[filt_t].dropna().index
        scores = scores[scores.index.intersection(tradable)]
        # ST 过滤（PIT，与生产 generate_target 同口径）
        st_syms = self._st_set_at(filt_t)
        if st_syms:
            scores = scores[~scores.index.isin(st_syms)]
        if cfg.min_listing_days > 0:
            age = self.data_age.loc[filt_t]
            scores = scores[scores.index.intersection(
                age[age >= cfg.min_listing_days].index)]
        if cfg.min_daily_turnover > 0 and not self.avg_turnover_20d.empty:
            try:
                avg = self.avg_turnover_20d.loc[filt_t]
                illiquid = avg[avg < cfg.min_daily_turnover].index  # NaN(数据不足)不排除
                scores = scores[~scores.index.isin(illiquid)]
            except KeyError:
                pass
        return scores.sort_values(ascending=False).head(cfg.top_n).index.tolist()

    # ---------------- 汇总 ----------------

    def _summarize(self, ledger, dates, n_halt, n_no_buy,
                   n_liquidations=0, first_liq_date=None) -> ReplayResult:
        cfg = self.cfg
        trades = ledger._d['trades']
        total_cost = sum(t['commission'] + t['taxes'] for t in trades)
        n_stop = sum(1 for t in trades if '[L4]' in (t.get('note') or ''))

        hist = ledger.nav_series()
        equity = pd.Series({pd.Timestamp(h['date']): h['nav'] for h in hist}).sort_index()

        # 换手统计（单边/有成交日）
        day_buys, day_sells = {}, {}
        for t in trades:
            v = t['price'] * t['quantity']
            d = day_buys if t['side'] == 'BUY' else day_sells
            d[t['date']] = d.get(t['date'], 0.0) + v
        trade_dates = sorted(set(day_buys) | set(day_sells))
        nav_map = {str(ts.date()): v for ts, v in equity.items()}
        turnover_list = [((day_buys.get(d, 0) + day_sells.get(d, 0)) / 2)
                         / max(nav_map.get(d, cfg.initial_capital), 1e-9)
                         for d in trade_dates]
        avg_turnover = float(np.mean(turnover_list)) if turnover_list else 0.0

        # 审计明细（与 PortfolioBacktester 工件同构）：逐日重放成交推导持仓/贡献
        trade_log = [[t['date'], t['symbol'], t['side'],
                      round(float(t['price']), 3), int(t['quantity'])] for t in trades]
        pos_qty, daily_holdings, daily_contrib = {}, {}, {}
        prev_close_mx = self.close_mx.shift(1)
        for T in dates:
            ts = str(T.date())
            for t in (x for x in trades if x['date'] == ts):
                if t['side'] == 'BUY':
                    pos_qty[t['symbol']] = pos_qty.get(t['symbol'], 0) + t['quantity']
                else:
                    pos_qty[t['symbol']] = pos_qty.get(t['symbol'], 0) - t['quantity']
                    if pos_qty[t['symbol']] <= 0:
                        pos_qty.pop(t['symbol'], None)
            row = self.close_mx.loc[T]
            prow = prev_close_mx.loc[T]
            hold, contrib = [], []
            for s2, q2 in pos_qty.items():
                p2 = row.get(s2, np.nan)
                if pd.isna(p2):
                    continue
                hold.append([s2, round(q2 * float(p2), 0)])
                pp = prow.get(s2, np.nan)
                if not pd.isna(pp):
                    contrib.append([s2, round(q2 * (float(p2) - float(pp)), 0)])
            daily_holdings[ts] = sorted(hold, key=lambda x: -x[1])
            daily_contrib[ts] = contrib

        total_return = float(equity.iloc[-1] / cfg.initial_capital - 1) if len(equity) else 0.0
        n_days = max(len(equity), 1)
        annual = (1 + total_return) ** (252 / n_days) - 1 if total_return > -1 else -1.0
        rets = equity.pct_change().dropna()
        sharpe = float(rets.mean() * 252 / (rets.std() * np.sqrt(252))) \
            if len(rets) > 1 and rets.std() > 0 else 0.0
        mdd = float((equity / equity.cummax() - 1).min()) if len(equity) else 0.0

        suite = {}
        bench_idx = {'沪深300': '000300.SH', '中证500': '000905.SH', '中证1000': '000852.SH'}
        for name, code in bench_idx.items():
            if code in self.close_mx.columns:
                b = self.close_mx[code].loc[dates[0]:dates[-1]].dropna()
                if len(b) >= 2 and b.iloc[0] > 0:
                    suite[name] = float(b.iloc[-1] / b.iloc[0] - 1)
        pool_cols = [c for c in self.close_mx.columns if c not in bench_idx.values()]
        if pool_cols:
            # 统计参照（不可交易口径：无过滤/成本/再平衡；严格对照用 tradable_benchmark）
            pool_ret = self.close_mx[pool_cols].loc[dates[0]:dates[-1]].pct_change().mean(axis=1)
            ew = (1 + pool_ret.fillna(0)).cumprod()
            if len(ew) >= 2:
                suite['等权统计参照'] = float(ew.iloc[-1] - 1)

        # 终态说明（空仓停机检测：最后有持仓日远早于区间末 → 硬限清仓/长期禁买）
        terminal_note = None
        terminal_date = None
        active_return = None
        idle_days = 0
        if trades:
            last_hold = max((day for day, h in daily_holdings.items() if h), default=None)
            last_day = str(dates[-1].date()) if len(dates) else None
            if last_hold and last_day and last_hold < last_day:
                terminal_date = last_hold
                idle_days = int(sum(1 for d in dates if str(d.date()) > last_hold))
                terminal_note = (f"终态：{last_hold} 起空仓停机（最后成交 "
                                 f"{trades[-1]['date']}），此后持币——区间尾部收益为"
                                 f"现金收益，不代表策略持续运行")
                # 运行期收益：起点净值 → terminal_date 净值
                eq_map = {str(ts.date()): float(v) for ts, v in equity.items()}
                if len(equity):
                    start_nav = float(equity.iloc[0])
                    end_nav = eq_map.get(last_hold, start_nav)
                    if start_nav > 0:
                        active_return = end_nav / start_nav - 1

        return ReplayResult(
            config=cfg, n_dates=len(dates), initial_capital=cfg.initial_capital,
            final_capital=float(equity.iloc[-1]) if len(equity) else cfg.initial_capital,
            total_return=total_return, annual_return=float(annual), sharpe=sharpe,
            max_drawdown=mdd,
            n_rebalances=len(trade_dates),
            n_trade_days=len(trade_dates),
            total_cost=float(total_cost), avg_turnover=avg_turnover,
            n_stop_sells=n_stop,
            n_halt_days=n_halt, n_no_buy_days=n_no_buy,
            benchmark_return=suite.get('沪深300'),
            equity_curve=equity, trades=trades, benchmark_suite=suite,
            daily_holdings=daily_holdings, daily_contrib=daily_contrib,
            trade_log=trade_log, terminal_note=terminal_note,
            terminal_date=terminal_date, active_period_return=active_return,
            idle_trading_days=idle_days,
            first_liquidation_date=first_liq_date,
            n_liquidation_events=n_liquidations,
            restart_mode=bool(cfg.gate_config and getattr(
                cfg.gate_config, 'restart_after_liquidation', False)))
