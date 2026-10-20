#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生产信号生成 —— 5因子合成（ep+bp+sp+size+leverage），PIT 到最新交易日

输出目标持仓与调仓订单（与账本持仓的差额），涨跌停不可买自动顺延。
"""

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from data.layered import LayeredDataSource
from data.fundamentals import WindFundamentalProvider, build_fundamental_matrix
from data.fundamental_pit import PITFinancialProvider
from portfolio.construction import composite_from_matrices
from trading import config as chain
from utils.logger import get_logger

logger = get_logger(__name__)

# 生产配置 —— 单一来源 trading/config.py（回测 PortfolioConfig 默认值同源；
# 旧名字保留 re-export，防止外部引用断裂）
PRODUCTION_FACTORS = chain.PRODUCTION_FACTORS
DEFAULT_TOP_N = chain.TOP_N              # Top100（Top30经回测验证跑输）
WARMUP_DAYS = chain.WARMUP_DAYS
MIN_LISTING_DAYS = chain.MIN_LISTING_DAYS
MIN_DAILY_TURNOVER = chain.MIN_DAILY_TURNOVER  # 20日均成交额下限（与回测同口径）
# 信号质量闸门（数据停机保护：覆盖率崩塌→halt而非正常调仓）
MIN_FACTOR_COVERAGE = 0.30       # 核心因子单日覆盖率下限
MIN_TARGET_COUNT = 20            # 目标持仓下限（<20只视为数据异常）


def generate_target(ds: LayeredDataSource, as_of: date,
                    top_n: int = DEFAULT_TOP_N,
                    st_syms: set = None) -> Tuple[List[str], pd.Series]:
    """
    生成 as_of 收盘后的目标持仓（信号用 ≤as_of 数据，PIT）

    Args:
        st_syms: 预加载的 ST 集合（提供则跳过内部加载，供生产入口复用）

    Returns:
        (目标列表, 综合分Series)
    """
    start = as_of - timedelta(days=WARMUP_DAYS)
    pool = sorted(ds.list_symbols(as_of=as_of))

    # 唯一策略实现：经由 strategy.build（生产/重放/实验共用）
    from trading.strategy import build_strategy_score
    score_mx, cache, mats = build_strategy_score(ds, pool, start, as_of)

    if score_mx.empty:
        return [], pd.Series(dtype=float)
    ts = pd.Timestamp(as_of)
    valid_dates = score_mx.index[score_mx.index <= ts]
    if len(valid_dates) == 0:
        return [], pd.Series(dtype=float)
    # 保留mats供单因子覆盖率检查（来自 strategy.build 的原始因子矩阵）
    # 契约校验已在 build_strategy_score 内完成

    # ---- 信号质量闸门（数据停机保护）----
    # 回退到最后所有因子覆盖率均达标且综合分有效的日期
    # （此前只查综合分覆盖率——单因子缺失被重新加权掩盖）
    from data.semantics import factor_coverage, close_matrix
    close_mx = close_matrix(cache)
    scores = pd.Series(dtype=float)
    sig_date = valid_dates[-1]
    required_factors = set(PRODUCTION_FACTORS.keys())
    for dt in reversed(valid_dates[-10:]):
        # 1. 综合分覆盖率
        coverage = factor_coverage(score_mx, dt)
        n_valid = int(score_mx.loc[dt].notna().sum())
        if n_valid <= 10 or coverage < MIN_FACTOR_COVERAGE or dt not in close_mx.index:
            continue

        # 2. 单因子覆盖率（任一必需因子缺失→该日期不可用）
        factor_ok = True
        for fname, fm in mats.items():
            if fname not in required_factors or fm.empty:
                continue
            if dt not in fm.index:
                factor_ok = False
                logger.warning(f"[signal] 因子'{fname}'在{dt.date()}无数据")
                break
            f_cov = factor_coverage(fm, dt)
            if f_cov < MIN_FACTOR_COVERAGE:
                factor_ok = False
                logger.warning(f"[signal] 因子'{fname}'覆盖率{f_cov:.0%}<{MIN_FACTOR_COVERAGE:.0%}"
                               f"@{dt.date()}——该日期不可用（防止单因子崩塌被重新加权掩盖）")
                break
        if not factor_ok:
            continue

        scores = score_mx.loc[dt].dropna()
        sig_date = dt
        break

    if len(scores) == 0:
        fail_reasons = []
        for fname in required_factors:
            fm = mats.get(fname)
            if fm is not None and not fm.empty:
                last_cov = factor_coverage(fm, fm.index[-1])
                fail_reasons.append(f"{fname}:{last_cov:.0%}")
        logger.error(f"[signal] 信号质量闸门：最近10日无合格日期"
                     f"（综合分或单因子覆盖率不足。各因子最新覆盖率: {', '.join(fail_reasons)}），"
                     f"进入数据停机（禁止交易）")
        return [], pd.Series(dtype=float)
    logger.info(f"[signal] 信号日期: {sig_date.date()}（有效{len(scores)}/"
                f"{score_mx.shape[1]}只，单因子覆盖均已检查）")

    # 排除当日无价格（停牌）——用缓存尾部判断
    # 所有过滤统一使用 sig_date（信号回退后valid_dates[-1]≠sig_date，
    # 此前混用导致因子来自旧日期但过滤来自新日期——前视风险）
    last_prices = close_mx.loc[sig_date] if sig_date in close_mx.index else pd.Series()
    tradable = last_prices.dropna()
    scores = scores[scores.index.intersection(tradable.index)]

    # 上市天数过滤：需≥60个交易日数据（与回测一致）
    data_age = close_mx.notna().cumsum()
    if sig_date in data_age.index:
        age_today = data_age.loc[sig_date]
        eligible = age_today[age_today >= MIN_LISTING_DAYS].index
        scores = scores[scores.index.intersection(eligible)]

    # 流动性过滤：20日均成交额≥500万（与回测同口径；日期也统一用sig_date）
    illiquid = illiquid_set(cache, sig_date, MIN_DAILY_TURNOVER)
    if illiquid:
        scores = scores[~scores.index.isin(illiquid)]

    # ---- v2 质量过滤（ROE>8% + 杠杆<65%，作为筛选而非权重） ----
    # 契约校验：PIT杠杆是百分比(D/E%)，阈值65.0 不是资产/权益倍数3.0
    from data.contracts import get_factor_contract, validate_factor_matrix
    from data.fundamental_pit import PITFinancialProvider
    pit = PITFinancialProvider(ds)   # 2026-10-15 依赖重接线时误删此行，
                                     # paper验证期首跑抓回（合成数据测试路径
                                     # 在覆盖率闸门早退，覆盖不到此分支）
    qf = chain.QUALITY_FILTER

    if qf.get('roe_min') is not None:
        roe = pit.get_factor_matrix('roe', pool, start, as_of)
        if not roe.empty:
            try:
                meta = validate_factor_matrix(roe, get_factor_contract('roe'))
                if not meta['passed']:
                    logger.warning(f"[signal] ROE契约异常: {meta['issues']}")
            except Exception:
                pass
        if not roe.empty and sig_date in roe.index:
            roe_row = roe.loc[sig_date]
            fail_roe = roe_row[roe_row <= qf['roe_min']].index
            if len(fail_roe) > 0:
                scores = scores[~scores.index.isin(fail_roe)]
                logger.info(f"[signal] 质量过滤ROE: 排除{len(fail_roe)}只")

    if qf.get('leverage_max') is not None:
        lev = pit.get_factor_matrix('leverage', pool, start, as_of)
        if not lev.empty:
            try:
                meta = validate_factor_matrix(lev, get_factor_contract('leverage'))
                # 关键校验：如果值域中位数<10，大概率是倍数而非百分比 → 告警
                last_vals = lev.iloc[-1].dropna()
                if len(last_vals) > 10 and last_vals.median() < 10:
                    logger.warning(
                        f"[signal] 杠杆值中位数{last_vals.median():.1f}<10——"
                        f"疑似倍数而非百分比（D/E%），阈值{qf['leverage_max']}可能不适用")
            except Exception:
                pass
        if not lev.empty and sig_date in lev.index:
            lev_row = lev.loc[sig_date]
            fail_lev = lev_row[lev_row >= qf['leverage_max']].index
            if len(fail_lev) > 0:
                scores = scores[~scores.index.isin(fail_lev)]
                logger.info(f"[signal] 质量过滤杠杆: 排除{len(fail_lev)}只")

    # ---- v2 趋势叠加（6个月剔1个月复权收益 × MOMENTUM_WEIGHT） ----
    mw = chain.MOMENTUM_WEIGHT
    if mw > 0 and len(scores) > 20:
        close_dfs = {}
        for sym in scores.index:
            df = cache.get(sym)
            if df is not None and not df.empty:
                idx2 = pd.DatetimeIndex(pd.to_datetime(df['date'])).as_unit('ns')
                adj_col = 'adjclose' if 'adjclose' in df.columns else 'close'
                close_dfs[sym] = pd.Series(
                    pd.to_numeric(df[adj_col], errors='coerce').values, index=idx2)
        if close_dfs:
            close_s = pd.DataFrame(close_dfs).sort_index()
            try:
                px_21 = close_s.shift(21).loc[ts] if ts in close_s.index else None
                px_126 = close_s.shift(126).loc[ts] if ts in close_s.index else None
                if px_21 is not None and px_126 is not None:
                    mom = (px_21 / px_126 - 1).dropna()
                    v_z = (scores - scores.mean()) / (scores.std() + 1e-10)
                    m_z = (mom - mom.mean()) / (mom.std() + 1e-10)
                    common = v_z.index.intersection(m_z.index)
                    scores = ((1 - mw) * v_z + mw * m_z).loc[common].dropna()
            except (KeyError, IndexError):
                pass    # 数据不足时退化为纯价值

    # ST 过滤（PIT，ASHAREST）：主板ST涨跌幅±5%、退市风险高，不入选。
    # 数据不可用时的安全策略由调用方决定：生产入口 generate_production_signal
    # 会携带 st_ok=False → 执行闸门禁买（禁止"降级放行"直接交易）。
    if st_syms is None:
        try:
            from data.st_provider import STProvider
            st_syms = STProvider(ds).st_set(as_of)
        except Exception as e:
            st_syms = set()
            logger.warning(f"[signal] ST过滤未生效（ASHAREST 不可用）: {e}")
    if st_syms:
        n_before = len(scores)
        scores = scores[~scores.index.isin(st_syms)]
        logger.info(f"[signal] ST过滤：剔除 {n_before - len(scores)} 只 @ {as_of}")

    targets = scores.sort_values(ascending=False).head(top_n).index.tolist()

    # 目标数量下限检查（数据停机保护：不能超过top_n本身）
    effective_min = min(MIN_TARGET_COUNT, top_n)
    if len(targets) < effective_min:
        logger.error(f"[signal] 目标持仓仅{len(targets)}只（下限{effective_min}），"
                     f"数据链路异常——返回空目标（闸门将halt，保留原持仓）")
        return [], scores

    return targets, scores


# ================= 执行时序（主链唯一口径，与回测 shift(1) 同构） =================

def prev_trading_day(ds: LayeredDataSource, as_of: date) -> date:
    """最近一个 ≤as_of 的完整交易日（以基准K线最后bar为准）"""
    cache = ds.preload_batch(['000300.SH'], as_of - timedelta(days=20), as_of)
    bench = cache.get('000300.SH')
    if bench is not None and len(bench) >= 1:
        d = bench['date'].iloc[-1]
        return d.date() if hasattr(d, 'date') else d
    return as_of - timedelta(days=1)


def signal_date_for(ds: LayeredDataSource, exec_date: date) -> date:
    """执行日 exec_date 对应的信号日 = 严格早于 exec_date 的最近完整交易日。

    时序口径（回测/纸面/实盘一致，无前视）：
        信号 = T-1 收盘数据（PIT）；执行 = T 日（回测与 Paper 以 T 收盘价+滑点
        撮合；QMT 在 T 日盘中/收盘竞价时段下单）。
        与回测 score_mx.shift(1)（调仓日 t 用 t-1 分数、t 收盘价成交）完全同构。
    """
    d = prev_trading_day(ds, exec_date)
    if d >= exec_date:  # exec_date 当日bar已存在（盘后运行）→ 再往前推一个交易日
        d = prev_trading_day(ds, exec_date - timedelta(days=1))
    return d


@dataclass
class ProductionSignal:
    """生产信号（唯一入口的返回结构：信号+ST数据状态一体传递）

    ST 状态区分两个日期（业务规则）：
      st_symbols      = 信号日(T-1)状态 → 选股过滤
      exec_st_symbols = 执行日(T)状态  → 撮合涨跌停判定（T日戴帽/摘帽以执行日为准）
    st_ok=False 时执行方必须经 RiskGate(st_data_ok=False) 禁买——
    防止 ST 数据不可用被静默降级为按普通涨跌停撮合。
    """
    targets: List[str]
    scores: pd.Series
    signal_date: date
    st_ok: bool = True
    st_symbols: set = None          # 信号日(T-1) ST集合（选股过滤用）
    exec_st_symbols: set = None     # 执行日(T) ST集合（撮合判定用）
    spec: object = None             # StrategySpec（版本治理：本次信号使用的策略版本）

    def __post_init__(self):
        if self.st_symbols is None:
            self.st_symbols = set()
        if self.exec_st_symbols is None:
            self.exec_st_symbols = set()


def generate_production_signal(ds: LayeredDataSource, as_of: date = None,
                               top_n: int = DEFAULT_TOP_N) -> ProductionSignal:
    """
    生产信号唯一入口（run_daily / run_live / journal 一律经由本函数）。

    禁止直接调用 generate_target(ds, date.today()) 后以当日收盘价成交——
    信号用了当日收盘数据却假设能以当日收盘价成交，真实交易不可实现（前视）。
    守护测试：tests/test_chain_consistency.py::TestExecutionTiming。

    Returns:
        ProductionSignal（targets/scores/signal_date/st_ok/st_symbols/exec_st_symbols）
    """
    as_of = as_of or date.today()
    sig_date = signal_date_for(ds, as_of)

    # ST 状态两个日期各加载一次：信号日(T-1)用于选股过滤，执行日(T)用于撮合判定
    # （T日刚戴帽的股票撮合必须按±5%；T日刚摘帽则按±10%）
    st_ok = True
    st_syms_sig, st_syms_exec = set(), set()
    try:
        from data.st_provider import STProvider
        prov = STProvider(ds)
        st_syms_sig = prov.st_set(sig_date)
        st_syms_exec = prov.st_set(as_of)
    except Exception as e:
        st_ok = False
        logger.warning(f"[signal] ST状态数据不可用（执行闸门将禁买）: {e}")

    # 注意语义：st_syms 始终传入（含成功但为空集的情形）——空集=当日无ST（已确认），
    # None=未加载。若空集不传，generate_target 会二次加载，二次失败时选股在
    # st_ok=True 的假象下漏过滤（状态失真）。
    kwargs = {'st_syms': st_syms_sig} if st_ok else {}
    targets, scores = generate_target(ds, sig_date, top_n=top_n, **kwargs)
    from trading.spec import StrategySpec
    return ProductionSignal(targets=targets, scores=scores, signal_date=sig_date,
                            st_ok=st_ok, st_symbols=st_syms_sig,
                            exec_st_symbols=st_syms_exec,
                            spec=StrategySpec.current())


def illiquid_set(cache: Dict, as_of: pd.Timestamp,
                 min_turnover: float = MIN_DAILY_TURNOVER) -> set:
    """20日均成交额低于阈值的标的（与回测 _liquidity_mask 同口径，供选股排除）。
    成交额经 data/turnover 统一口径：amount列优先（元），退化 close×volume（股）。"""
    from data.turnover import avg_daily_turnover
    out = set()
    if min_turnover <= 0:
        return out
    for sym, df in (cache or {}).items():
        if df is None or len(df) < 20:
            continue
        d = df[df['date'] <= as_of]
        if len(d) < 20:
            continue
        amt = avg_daily_turnover(d, window=20)
        if amt is not None and 0 < amt < min_turnover:
            out.add(sym)
    return out


def build_orders(targets: List[str], current_weights: Dict[str, float],
                 total_nav: float, top_n: int = DEFAULT_TOP_N,
                 band: float = chain.BUFFER_BAND) -> List[dict]:
    """
    目标持仓 vs 当前持仓 → 调仓指令（带缓冲带，降低换手）

    规则：目标权重=1/top_n；权重差超过 band×目标权重 才调；
    移出目标的持仓全部卖出。缓冲带默认取主链配置（=回测 PortfolioConfig.buffer_band）。
    """
    from live.broker import Order
    orders = []
    target_w = 1.0 / max(top_n, 1)
    target_set = set(targets)

    # 卖出：不在目标中 或 超配
    sell_proceeds_w = 0.0  # 预计卖出回收的权重
    for s, w in current_weights.items():
        if s not in target_set:
            orders.append(Order(s, 'SELL', 0.0, amount=w * total_nav,
                                note='移出组合'))
            sell_proceeds_w += w
        elif w > target_w * (1 + band):
            excess = (w - target_w) * total_nav
            orders.append(Order(s, 'SELL', target_w, amount=excess, note='超配回收'))
            sell_proceeds_w += (w - target_w)

    # 买入：新目标 或 低配（逐笔扣减可用资金，防止超卖）
    remaining_w = 1.0 - sum(current_weights.values()) + \
        sell_proceeds_w * chain.SELL_PROCEEDS_DISCOUNT  # 卖出回收折扣（费用+滑点）
    for s in targets:
        w = current_weights.get(s, 0.0)
        if w < target_w * (1 - band):
            alloc = min(target_w - w, remaining_w)
            if alloc > 0:
                orders.append(Order(s, 'BUY', target_w, amount=alloc * total_nav,
                                    note='新入/低配'))
                remaining_w -= alloc  # 逐笔扣减，防止多单合计超现金
    return orders


def _make_loader(ds):
    from data.base.data_loader import DataLoader
    loader = DataLoader.__new__(DataLoader)
    loader.data_source_type = 'layered'
    loader.data_source = ds
    loader._cache = None
    return loader
