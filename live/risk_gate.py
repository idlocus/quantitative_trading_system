#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
执行风控闸门（Execution Risk Gate）—— 生产链路的强制风控层

位于信号生成（signals.build_orders）与券商执行（broker.execute）之间，
对指令流做最后一道审查：拦截、缩量、注入强制卖出。Paper/QMT 同路径，
即"日志里写了的风险控制，真实交易一定执行"。

风险分层口径（journal 展示与本文一致）：
┌──────┬──────────────────┬──────────────────────────────────────────┐
│ 层级 │ 类别             │ 动作                                     │
├──────┼──────────────────┼──────────────────────────────────────────┤
│ L0   │ 研究指标         │ 无（因子/估值，仅用于选股）              │
│ L1   │ 预警指标         │ 无（RSI/均线/量比，仅日志提示，不交易）  │
│ L2   │ 数据停机规则     │ 拦截全部指令（HALT）                     │
│ L3   │ 组合级熔断规则   │ 缩量/禁买/清仓（SCALE/NO_BUY/LIQUIDATE） │
│ L4   │ 实际卖出规则     │ 注入强制 SELL（个股止损，T+1 感知）      │
└──────┴──────────────────┴──────────────────────────────────────────┘

优先级：L2（停机）> L3（熔断）> L4（止损）> 原始调仓指令。
状态持久化：live/risk_state.json（禁买冷却期等跨日状态）。
"""

import json
import os
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Dict, List, Optional, Tuple

import pandas as pd

from live.broker import Order
from trading import config as chain_cfg
from utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_STATE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'runtime', 'risk_state.json')   # 运行时状态独立于源码（runtime/，gitignored）：
# 避免旧回撤锚点/禁买冷却随代码复制到新环境

# 动作语义：
#   trade     正常放行
#   scale     缩量放行（回撤软限：买入金额×仓位上限，超配部分削减）
#   no_buy    禁止买入（暴跌熔断冷却期：只卖不买）
#   halt      数据停机（全部指令拦截，含强制卖出——数据不可信时不动作）
#   liquidate 熔断清仓（回撤硬限：全部持仓卖出，原调仓指令作废）
ACTIONS = ('trade', 'scale', 'no_buy', 'halt', 'liquidate')


@dataclass
class GateConfig:
    """闸门阈值（生产口径：回撤软限8%/硬限15%，与回测RiskLimits对齐并更严）"""
    # L2 数据停机
    bench_symbol: str = '000300.SH'
    max_bench_staleness: int = 5       # 基准最新K线距 as_of 最大容忍自然日
    min_price_coverage: float = 0.90   # 持仓∪目标 的价格覆盖率下限
    anomaly_move: float = 0.21         # 单日|涨跌|>此值视为数据异常（超最大涨跌停+容差）
    max_anomaly_rate: float = 0.05     # 异常标的占比上限，超过则整体停机
    # L3 组合熔断（阈值单一来源=trading/config，与回测 RiskLimits 真同源）
    drawdown_soft: float = chain_cfg.DRAWDOWN_SOFT        # 回撤软限：开始线性降仓
    drawdown_hard: float = chain_cfg.DRAWDOWN_HARD        # 回撤硬限：全部清仓
    portfolio_daily_crash: float = chain_cfg.PORTFOLIO_DAILY_CRASH   # 组合单日跌幅熔断
    market_daily_crash: float = chain_cfg.MARKET_DAILY_CRASH     # 基准单日跌幅熔断
    cooldown_days: int = 1             # 暴跌熔断后的禁买冷却天数
    # L4 强制卖出
    stop_loss_pct: float = chain_cfg.STOP_LOSS_PCT    # 个股硬止损：成本下方5%
    max_stop_sells_per_day: int = 20   # 单日止损卖出上限（防踩踏，其余顺延）
    min_order_amount: float = 2000.0   # 缩量后低于此金额的买入放弃

    # L3 附加约束（与回测 RiskLimits 同源；此前仅回测有、实盘无）
    vol_target: float = chain_cfg.VOL_TARGET           # 年化波动率目标
    max_turnover: float = chain_cfg.MAX_TURNOVER       # 单次调仓最大换手
    min_cash_ratio: float = chain_cfg.MIN_CASH_RATIO   # 最低现金比例

    # 数据安全（生产默认）：ST状态数据不可用时禁止新买入（只卖不买），
    # 避免 ST 股按普通涨跌停撮合并提交实盘委托；人工确认后恢复。
    st_data_required: bool = True

    # 硬限清仓后是否自动重启（默认停机等待人工决策：硬限=停止是设计意图；
    # True 时清仓完成即重置回撤锚以现金为新起点重启——研究对照模式用）
    restart_after_liquidation: bool = False


@dataclass
class RiskDecision:
    """闸门审查结果（action + 改写后的指令流 + 全部理由，可审计）"""
    action: str = 'trade'
    exposure: float = 1.0              # 允许的目标仓位上限
    orders: List[Order] = field(default_factory=list)      # 放行指令（含强制卖出）
    forced_sells: List[Order] = field(default_factory=list)  # L4 注入的强制卖出
    suppressed: List[Tuple[Order, str]] = field(default_factory=list)  # 被拦截指令
    data_issues: List[str] = field(default_factory=list)   # L2 数据问题
    reasons: List[str] = field(default_factory=list)       # 触发的规则（含层级标签）
    drawdown: float = 0.0
    peak_nav: float = 0.0

    _VERDICT = {'trade': '✅ 放行',
                'scale': '⚠️ 缩量放行',
                'no_buy': '🚫 禁止买入（只卖不买）',
                'halt': '⛔ 数据停机（全部指令拦截）',
                'liquidate': '🔴 熔断清仓'}

    _VERDICT_PLAIN = {'trade': '放行',
                      'scale': '缩量放行',
                      'no_buy': '禁止买入(只卖不买)',
                      'halt': '数据停机(全部指令拦截)',
                      'liquidate': '熔断清仓'}

    @property
    def verdict_cn(self) -> str:
        return self._VERDICT.get(self.action, self.action)

    @property
    def verdict_plain(self) -> str:
        """无emoji版（GBK控制台/print安全）"""
        return self._VERDICT_PLAIN.get(self.action, self.action)


class RiskGate:
    """执行风控闸门：review() 一次调用完成 L2→L3→L4 全层审查"""

    def __init__(self, config: GateConfig = None, state_path: str = DEFAULT_STATE_PATH):
        self.cfg = config or GateConfig()
        self.state_path = state_path
        self._anomalous: set = set()
        self._state = self._load_state()

    # ================= 状态持久化 =================

    def _load_state(self) -> dict:
        if os.path.isfile(self.state_path):
            try:
                with open(self.state_path, encoding='utf-8') as f:
                    return json.load(f)
            except Exception:
                logger.warning("risk_state.json 损坏，重建")
        return {'no_buy_until': None}

    def _save_state(self):
        try:
            os.makedirs(os.path.dirname(os.path.abspath(self.state_path)), exist_ok=True)
            with open(self.state_path, 'w', encoding='utf-8') as f:
                json.dump(self._state, f, ensure_ascii=False, indent=1)
        except Exception as e:
            logger.error(f"风控状态保存失败: {e}")

    # ================= 主入口 =================

    def review(self, orders: List[Order], ledger, prices: Dict[str, float],
               prev_prices: Dict[str, float], targets: List[str],
               bench_df: Optional[pd.DataFrame] = None,
               as_of: Optional[date] = None,
               st_data_ok: bool = True,
               stale_symbols: Optional[set] = None) -> RiskDecision:
        """
        审查一次调仓指令流。

        Args:
            orders:      signals.build_orders 产出的原始指令
            ledger:      Ledger（持仓/成本/净值序列/成交记录）
            prices:      {sym: 最新价}（as_of 收盘口径）
            prev_prices: {sym: 前收盘}（用于异常检测）
            targets:     目标持仓列表（用于覆盖率检查）
            bench_df:    基准日K（date/close 列，用于新鲜度+市场熔断）
            as_of:       信号日期（默认今天）
            st_data_ok:  ST状态数据是否可用（False → 禁止新买入，卖出照常；
                         ST股可能被按普通涨跌停撮合，实盘安全优先）
            stale_symbols: 行情日期明显落后的标的集合（停牌/数据延迟；
                         这些标的的指令将被剔除）
        """
        as_of = as_of or date.today()
        d = RiskDecision()
        self._stale = stale_symbols or set()

        # L2 数据停机（最高优先级：数据不可信 → 一律不动）
        if self._check_data(ledger, prices, prev_prices, targets, bench_df, as_of, d):
            d.action, d.exposure = 'halt', 0.0
            d.suppressed = [(o, '[L2]数据停机，全部拦截') for o in orders]
            self._save_state()
            logger.warning(f"[RiskGate] 停机: {d.data_issues}")
            return d

        # L3 组合熔断
        action, exposure, bench_chg = self._portfolio_breakers(ledger, bench_df, as_of, d)
        d.action, d.exposure = action, exposure

        # L3.1 空仓死锁/停机语义：
        #  a) scale+空仓：预算制下不应发生，作为安全网锚重置（现金为新起点）
        #  b) liquidate+空仓：硬限清仓完成。默认=停机等待人工决策（硬限=停止
        #     是设计意图，静默持币2.5年不是）；restart_after_liquidation=True
        #     时锚重置自动重启（研究对照模式）。两种路径都留痕。
        if hasattr(ledger, 'positions') and not ledger.positions:
            nav_now = ledger.last_nav() or ledger.cash
            if action == 'scale' and exposure < 0.15 and nav_now > 0:
                self._state['dd_anchor'] = nav_now
                d.reasons.append(f'[L3] 空仓死锁解除：回撤锚重置为当前净值 {nav_now:,.0f}'
                                 '（现金为新起点，创新高后自动失效）')
                action, exposure = 'trade', 1.0
                d.action, d.exposure = action, exposure
            elif action == 'liquidate' and nav_now > 0:
                if self.cfg.restart_after_liquidation:
                    self._state['dd_anchor'] = nav_now
                    d.reasons.append(f'[L3] 清仓后自动重启：回撤锚重置为 {nav_now:,.0f}'
                                     '（restart_after_liquidation=True）')
                    action, exposure = 'trade', 1.0
                    d.action, d.exposure = action, exposure
                else:
                    d.reasons.append('[L3] 硬限清仓已完成，策略停机持币——'
                                     '等待人工决策重启（restart_after_liquidation'
                                     '=True 可自动重启）')

        # L2.5 ST数据安全：不可用时禁买（卖出照常，人工确认后恢复）
        if not st_data_ok and self.cfg.st_data_required and d.action != 'halt':
            if d.action in ('trade', 'scale', 'no_buy'):
                d.reasons.append('[L2] ST状态数据不可用：禁止新买入（只卖不买）——'
                                 'ST股可能按普通涨跌停撮合，实盘安全优先；人工确认后恢复')
                action = 'no_buy'          # 局部变量同步（_assemble 以此为准）
                d.action = action

        # L2.6 对账安全闸门：上次对账不一致或存在未处理异常时禁止新交易
        # （此前交易后才对账——错误持仓/现金会继续产生错误订单）
        if hasattr(ledger, 'open_exceptions') and ledger.open_exceptions():
            n_ex = len(ledger.open_exceptions())
            d.reasons.append(f'[L2] 账本存在{n_ex}项未处理异常（超额卖出等）——'
                             '禁止新交易，人工核对后 resolve_exception 解锁')
            if d.action != 'halt':
                action = 'no_buy'
                d.action = action
        last_recon = getattr(ledger, '_d', {}).get('meta', {}).get('last_recon', {})
        if last_recon and not last_recon.get('ok', True):
            d.reasons.append(f'[L2] 上次对账({last_recon.get("date", "?")})不一致'
                             f'（{last_recon.get("n_diffs", "?")}项差异）——'
                             '禁止新交易，人工核对后 --statement --fix 或解锁')
            if d.action != 'halt':
                action = 'no_buy'
                d.action = action
        elif last_recon:
            # 对账新鲜度：上次对账距今超过3天（实盘应每日对账）
            recon_date = last_recon.get('date', '')
            if recon_date and as_of:
                try:
                    from datetime import datetime as _dt
                    rd = _dt.strptime(str(recon_date)[:10], '%Y-%m-%d').date()
                    if (as_of - rd).days > 3:
                        d.reasons.append(f'[L2] 上次对账({recon_date})距今超过3天——'
                                         '禁止新交易，先完成对账（--statement）')
                        if d.action != 'halt':
                            action = 'no_buy'
                            d.action = action
                except (ValueError, TypeError):
                    pass
        # 注：「从未对账」不作为闸门条件——Paper/回测无对账单，会被误拦截。
        # 实盘首次对账由运维流程保障（run_live.py --status 提示"从未对账"）。

        # L4 强制卖出（halt 已排除；liquidate 时由清仓指令覆盖）
        forced = [] if action == 'liquidate' else self._enforce_stops(ledger, prices, as_of, d)

        # 指令重组
        d.orders = self._assemble(orders, forced, action, exposure, ledger, prices, d)
        self._save_state()

        n_forced = len(d.forced_sells)
        logger.info(f"[RiskGate] action={d.action} | 仓位上限{exposure:.0%} | "
                    f"放行{len(d.orders)} 拦截{len(d.suppressed)} 强制卖出{n_forced}"
                    + (f" | 昨基准{bench_chg:+.1%}" if bench_chg is not None else ""))
        return d

    # ================= L2 数据停机 =================

    def _check_data(self, ledger, prices, prev_prices, targets,
                    bench_df, as_of, d: RiskDecision) -> bool:
        """返回 True = 停机。检查：基准新鲜度 / 价格覆盖率 / 异常波动占比"""
        cfg = self.cfg

        # 1) 基准数据新鲜度（无基准=无法判定市场熔断，视为停机）
        bench_ok = False
        if bench_df is not None and len(bench_df) >= 2:
            last_dt = pd.Timestamp(bench_df['date'].iloc[-1])
            if (pd.Timestamp(as_of) - last_dt).days <= cfg.max_bench_staleness:
                bench_ok = True
        if not bench_ok:
            last_dt = str(bench_df['date'].iloc[-1]) if bench_df is not None and len(bench_df) else '无数据'
            d.data_issues.append(
                f"基准({cfg.bench_symbol})数据缺失或滞后（最新:{last_dt}，"
                f"容忍{cfg.max_bench_staleness}天），无法校验市场状态")
            return True

        # 2) 价格覆盖率（持仓∪目标：大量缺价=数据管道故障，停机）
        universe = list(ledger.positions.keys()) + [t for t in (targets or []) if t not in ledger.positions]
        if universe:
            missing = [s for s in universe if prices.get(s, 0) <= 0]
            coverage = 1 - len(missing) / len(universe)
            if coverage < cfg.min_price_coverage:
                d.data_issues.append(
                    f"价格覆盖率{coverage:.0%}<{cfg.min_price_coverage:.0%}"
                    f"（缺失{len(missing)}/{len(universe)}，如{missing[:3]}）")
                return True
            if missing:
                d.data_issues.append(
                    f"{len(missing)}只无最新价（疑似停牌，跳过其指令）：{','.join(missing[:5])}")

        # 3) 单日异常波动（|涨跌|>21% 超出任何板块涨跌停+容差 → 疑似脏数据）
        self._anomalous = set()
        if prev_prices:
            anomalies = [s for s, p in prices.items()
                         if p > 0 and prev_prices.get(s, 0) > 0
                         and abs(p / prev_prices[s] - 1) > cfg.anomaly_move]
            if anomalies:
                rate = len(anomalies) / max(len(prices), 1)
                if rate > cfg.max_anomaly_rate:
                    d.data_issues.append(
                        f"异常价格占比{rate:.0%}>{cfg.max_anomaly_rate:.0%}"
                        f"（单日|涨跌|>{cfg.anomaly_move:.0%}，如{anomalies[:3]}），疑似数据错误")
                    return True
                self._anomalous = set(anomalies)
                d.data_issues.append(
                    f"{len(anomalies)}只单日波动>{cfg.anomaly_move:.0%}（剔除其指令）："
                    f"{','.join(anomalies[:5])}")
        # 过期行情（停牌/数据延迟）标记
        if self._stale:
            d.data_issues.append(
                f"{len(self._stale)}只行情过期/停牌（剔除其指令）："
                f"{','.join(sorted(self._stale)[:5])}")
        return False

    # ================= L3 组合熔断 =================

    def _portfolio_breakers(self, ledger, bench_df, as_of, d: RiskDecision):
        """返回 (action, exposure, bench_chg)。优先级：liquidate > no_buy > scale > trade"""
        cfg = self.cfg
        action, exposure = 'trade', 1.0

        # 1) 回撤（基于账本净值序列的峰值；空仓死锁解除后以锚代替，创新高清锚）
        #    存取款不算市场盈亏：出金20万≠回撤20%——从 cash_log 中提取存取款
        #    并对 nav 序列做调整（加回出金/扣除入金）后再计算回撤。
        nav_hist = ledger.nav_series()
        if nav_hist:
            navs = [h['nav'] for h in nav_hist]
            dates_hist = [h['date'] for h in nav_hist]
            # 提取存取款事件，调整净值序列
            cash_log = getattr(ledger, '_d', {}).get('cash_log', [])
            flow_by_date = {}
            for e in cash_log:
                if e.get('type') in ('deposit', 'withdraw'):
                    d_str = str(e.get('date', ''))[:10]
                    flow_by_date[d_str] = flow_by_date.get(d_str, 0.0) + float(e.get('amount', 0))
            # 从后往前累积流量，调整历史净值使其可比
            if flow_by_date:
                adj_navs = list(navs)
                cum_flow = 0.0
                for i in range(len(navs) - 1, -1, -1):
                    dt = dates_hist[i][:10]
                    cum_flow += flow_by_date.get(dt, 0.0)
                    adj_navs[i] = navs[i] - cum_flow  # 加回出金/扣除入金
                navs = adj_navs

            peak, last = max(navs), navs[-1]
            if last >= peak:
                self._state.pop('dd_anchor', None)   # 创新高：锚失效
            anchor = self._state.get('dd_anchor')
            peak_eff = anchor if (anchor and anchor < peak) else peak
            if peak_eff > 0:
                dd = 1 - last / peak_eff
                d.drawdown, d.peak_nav = dd, peak_eff
                if dd >= cfg.drawdown_hard:
                    d.reasons.append(f'[L3熔断] 回撤{dd:.1%}≥硬限{cfg.drawdown_hard:.0%} → 全部清仓')
                    return 'liquidate', 0.0, None
                if dd >= cfg.drawdown_soft:
                    exposure = 1 - (dd - cfg.drawdown_soft) / (cfg.drawdown_hard - cfg.drawdown_soft)
                    exposure = max(0.0, min(1.0, exposure))
                    action = 'scale'
                    d.reasons.append(f'[L3熔断] 回撤{dd:.1%}≥软限{cfg.drawdown_soft:.0%} → 仓位上限{exposure:.0%}')
                else:
                    d.reasons.append(f'[L3] 回撤{dd:.1%}（软限{cfg.drawdown_soft:.0%}/硬限{cfg.drawdown_hard:.0%}）正常')
            else:
                # peak_eff<=0：回撤保护不可用（不能静默显示"回撤0.0%"）
                if hasattr(ledger, 'positions') and ledger.positions:
                    d.reasons.append('[L3] ⚠️ 有持仓但净值基准异常（peak≤0）——回撤保护不可用')
                else:
                    d.reasons.append('[L3] 新账户（无净值历史）——回撤保护待建立')
            # 2) 组合单日暴跌
            if len(navs) >= 2 and navs[-2] > 0:
                day_ret = navs[-1] / navs[-2] - 1
                if day_ret <= -cfg.portfolio_daily_crash:
                    d.reasons.append(f'[L3熔断] 组合单日{day_ret:.1%}≤-{cfg.portfolio_daily_crash:.0%} → 禁买{cfg.cooldown_days}日')
                    action = 'no_buy'
                    self._state['no_buy_until'] = str(as_of + timedelta(days=cfg.cooldown_days))

        # 3) 市场单日暴跌（基准昨跌幅）
        bench_chg = None
        if bench_df is not None and len(bench_df) >= 2:
            a, b = float(bench_df['close'].iloc[-1]), float(bench_df['close'].iloc[-2])
            if b > 0:
                bench_chg = a / b - 1
                if bench_chg <= -cfg.market_daily_crash:
                    d.reasons.append(f'[L3熔断] 基准单日{bench_chg:.1%}≤-{cfg.market_daily_crash:.0%} → 禁买{cfg.cooldown_days}日')
                    action = 'no_buy'
                    self._state['no_buy_until'] = str(as_of + timedelta(days=cfg.cooldown_days))

        # 4) 冷却期内（持久化状态）
        nbu = self._state.get('no_buy_until')
        if nbu and as_of <= pd.Timestamp(nbu).date() and action in ('trade', 'scale'):
            d.reasons.append(f'[L3熔断] 禁买冷却期内（至{nbu}），只卖不买')
            action = 'no_buy'

        return action, exposure, bench_chg

    # ================= L4 强制卖出 =================

    def _enforce_stops(self, ledger, prices, as_of, d: RiskDecision) -> List[Order]:
        """个股硬止损 → 注入强制 SELL（T+1：当日买入的锁仓顺延；单日上限防踩踏）"""
        cfg = self.cfg
        today_buys = {t['symbol'] for t in ledger._d['trades']
                      if t['side'] == 'BUY' and t['date'] == str(as_of)}
        stopped = []
        _last = getattr(ledger, '_d', {}).get('meta', {}).get('last_prices', {})
        for sym, qty in ledger.positions.items():
            # 异常价格标的冻结止损：数据异常的"低价"不是真实市场价，
            # 用它判断止损会产生假信号（如数据源返回0.01元的错误价格）
            if sym in self._anomalous:
                d.reasons.append(f'[L4] {sym} 价格异常，止损判断冻结（待价格确认）')
                continue
            cost = ledger._d['cost'].get(sym, 0)
            px = prices.get(sym)
            if px is None or px <= 0:
                px = _last.get(sym, 0)     # 缺价沿用最后有效价（与nav同口径）
            if cost <= 0 or px <= 0 or qty <= 0:
                continue
            ret = px / cost - 1
            if ret <= -cfg.stop_loss_pct:
                if sym in today_buys:
                    d.reasons.append(f'[L4] {sym} 已破止损{ret:+.1%}，但当日买入（T+1锁仓），明日强制卖出')
                    continue
                stopped.append((ret, sym, qty, px))

        stopped.sort(key=lambda x: x[0])  # 亏损最深优先
        forced = []
        for ret, sym, qty, px in stopped:
            if len(forced) >= cfg.max_stop_sells_per_day:
                d.reasons.append(f'[L4] 止损达单日上限{cfg.max_stop_sells_per_day}只，'
                                 f'{sym}({ret:+.1%})顺延明日')
                continue
            o = Order(sym, 'SELL', 0.0, quantity=qty, amount=qty * px,
                      note=f'[L4]强制卖出·止损{ret:+.1%}≤-{cfg.stop_loss_pct:.0%}')
            forced.append(o)
        if forced:
            d.reasons.append(f'[L4实际卖出] 止损触发{len(forced)}只'
                             f'（规则：收盘价≤成本×{1 - cfg.stop_loss_pct:.2f}，全仓卖出）')
            d.forced_sells = forced
        return forced

    # ================= 指令重组 =================

    def _assemble(self, orders, forced, action, exposure, ledger, prices, d: RiskDecision) -> List[Order]:
        """合并强制卖出与原始指令，应用熔断动作与仓位上限"""
        cfg = self.cfg
        forced_syms = {o.symbol for o in forced}
        out = list(forced)
        sells_in = set(forced_syms)

        # 缩量预算制参数（在订单循环前预计算；缺价持仓沿用最后有效价，与账本同口径）
        _last_px = getattr(ledger, '_d', {}).get('meta', {}).get('last_prices', {})

        def _val_px(s, default=0.0):
            p = prices.get(s)
            if p is not None and p > 0:
                return p
            return _last_px.get(s, default)

        nav = ledger.cash + sum(q * _val_px(s) for s, q in ledger.positions.items())
        invested = nav - ledger.cash
        buy_budget = max(0.0, exposure * nav - invested) if (exposure < 0.999 and nav > 0) \
            else float('inf')

        if action == 'liquidate':
            for sym, qty in ledger.positions.items():
                if sym in forced_syms or qty <= 0:
                    continue
                px = _val_px(sym)      # 缺价沿用最后有效价（与nav计算同口径）
                out.append(Order(sym, 'SELL', 0.0, quantity=qty, amount=qty * px,
                                 note='[L3]熔断清仓·回撤硬限'))
                sells_in.add(sym)
            d.suppressed += [(o, '[L3]熔断清仓，原调仓指令作废') for o in orders]
            return [o for o in out if o.quantity and o.quantity > 0]

        for o in orders:
            if o.symbol in self._anomalous or o.symbol in self._stale:
                reason = '[L2]数据异常' if o.symbol in self._anomalous else '[L2]行情过期'
                d.suppressed.append((o, f'{reason}，剔除该标的指令'))
                continue
            if o.side == 'SELL':
                if o.symbol in sells_in:  # 已被全量强制卖出/清仓取代
                    continue
                sells_in.add(o.symbol)
                out.append(o)
                continue
            # BUY
            if o.symbol in forced_syms:
                d.suppressed.append((o, '[L4]该股强制止损卖出，撤销买入'))
                continue
            if action == 'no_buy':
                d.suppressed.append((o, '[L3]禁买期（熔断冷却），买入拦截'))
                continue
            if exposure < 0.999:
                # 缩量预算制：总买入预算 = 目标仓位市值−当前持仓市值；每单按原金额
                # 足额占用预算，预算耗尽即止（每票已是 build_orders 的足额单票，
                # 不再×exposure 切碎——碎片<最低申报会整批拒单，即缩量死锁成因）
                alloc = min(o.amount, buy_budget)
                if alloc < cfg.min_order_amount:
                    d.suppressed.append((o, f'[L3]缩量预算耗尽（余{buy_budget:,.0f}），放弃'))
                    continue
                out.append(Order(o.symbol, 'BUY', o.target_weight,
                                 quantity=o.quantity, amount=alloc,
                                 note=f'{o.note}｜[L3]缩量预算制(仓位上限{exposure:.0%})'))
                buy_budget -= alloc
                continue
            out.append(o)

        # 仓位上限削减：当前持仓权重 > exposure → 从最大持仓开始减（nav 已预计算；
        # 全部用 _val_px 回退价——与 nav/current_weights 同口径）
        if nav > 0 and exposure < 0.999:
            inv_w = sum(q * _val_px(s) for s, q in ledger.positions.items()) / nav
            if inv_w > exposure + 0.02:
                remaining = (inv_w - exposure) * nav
                holdings = sorted(((q * _val_px(s), s) for s, q in ledger.positions.items()),
                                  reverse=True)
                for mv, s in holdings:
                    if remaining <= cfg.min_order_amount:
                        break
                    if s in sells_in:
                        continue
                    px = _val_px(s)
                    if px <= 0:
                        continue
                    # 部分减持按申报规则合法化（科创板/北交所1股递增；此前固定>=100整手）
                    from trading.rules import min_order_qty, qty_step
                    raw = int(min(remaining, mv) / px)
                    step = qty_step(s)
                    qty = (raw // step) * step
                    if qty >= min_order_qty(s):
                        out.append(Order(s, 'SELL', 0.0, quantity=qty, amount=qty * px,
                                         note=f'[L3]回撤降仓·仓位{inv_w:.0%}→{exposure:.0%}'))
                        sells_in.add(s)
                        remaining -= qty * px
        return [o for o in out if (o.quantity and o.quantity > 0) or o.amount > 0]


# ================= Journal 渲染 =================

def render_gate_markdown(d: RiskDecision, basic_lines: List[str] = None) -> List[str]:
    """闸门结论渲染为 journal 章节（AM 预览 / PM 执行记录共用）"""
    lines = [
        "> 口径分层：L0研究指标（因子）→仅选股；L1预警（均线/RSI/量比）→仅提示不交易；",
        "> **L2数据停机 / L3组合熔断 / L4强制卖出 → 实际修改今日指令**", "",
        f"- 闸门结论：**{d.verdict_cn}**｜目标仓位上限 **{d.exposure:.0%}**｜"
        f"当前回撤 **{d.drawdown:.1%}**（峰值净值 {d.peak_nav:,.0f}）", "",
    ]
    if d.data_issues:
        lines += ["**L2 数据检查**："]
        lines += [f"- {i}" for i in d.data_issues]
        lines.append("")
    if d.reasons:
        lines += ["**触发的风控规则**："]
        lines += [f"- {r}" for r in d.reasons]
        lines.append("")
    if d.forced_sells:
        lines += ["**L4 强制卖出（今日将执行的止损单）**", "",
                  "| 代码 | 数量 | 金额(约) | 规则 |", "|------|------|----------|------|"]
        for o in d.forced_sells:
            lines.append(f"| {o.symbol} | {o.quantity} | {o.amount:,.0f} | {o.note} |")
        lines.append("")
    if d.suppressed:
        lines += ["**被拦截指令**", "",
                  "| 代码 | 方向 | 金额(约) | 原因 |", "|------|------|----------|------|"]
        for o, why in d.suppressed[:30]:
            lines.append(f"| {o.symbol} | {o.side} | {o.amount:,.0f} | {why} |")
        if len(d.suppressed) > 30:
            lines.append(f"| … | | | 其余 {len(d.suppressed) - 30} 条 |")
        lines.append("")
    if not (d.data_issues or d.reasons or d.forced_sells or d.suppressed):
        lines += ["- 无触发：数据检查通过，回撤在软限内，无止损持仓", ""]
    if basic_lines:
        lines += basic_lines
    return lines
