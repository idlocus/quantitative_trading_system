#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CTA Z10 TOP1 信号：波动率压缩排序与每日跟随持仓的单一实现。

被 scripts/run_cta_daily.py（每日报告）与 webapp/app.py（网页）共用，
消除三入口三语义问题（登记册180 勘误项）。语义与登记版一致：

- 信号：20日收益波动率在250日中的分位 < 20% → 压缩状态；
  方向 = sign(收盘 - 20日均线)，**每日跟随**（压缩段内均线穿越即翻向，
  这是登记册178/180 复现确认的形态；"入场锁方向"变体已被证伪 -0.9%）。
- 强度（TOP-k 排序）：波动率分位升序（压缩越深越强），并列按
  |收盘/均线-1| 降序，再并列按固定品种顺序。
- 执行：信号在收盘后生成，**次一交易日开盘**成交（禁止同bar归因）。
- 成本：单边 2.5bp（往返 5bp）计在每笔交易上。

本模块为纯函数，不做网络请求、不落盘。
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

VOL_WINDOW = 20
VOL_LOOKBACK = 250
VOL_THRESHOLD = 0.20
MA_WINDOW = 20
ONE_WAY_COST = 0.00025  # 单边；往返 = 2 ×
# 形成波动分位所需最少K线：20日波动窗 + 250日分位窗
MIN_BARS = VOL_WINDOW + VOL_LOOKBACK
STALE_WARN_CALENDAR_DAYS = 3  # 决策日距今超过此天数 → 新鲜度警告

# 部署形态（登记册181定案）：k=6 = 等权摊给全部活跃品种
# （target_book/portfolio_daily 的 k ≥ 活跃数 时权重 = 1/活跃数）
# 日报与回测验收共用此常量，避免两处口径漂移。
DEPLOY_K = 6
DEPLOY_LABEL = '等权摊活跃(1/n_active)'

# 固定品种顺序（并列时的最终 tie-break，与 Z10 预注册一致）
SYMBOL_ORDER = ['AU0', 'CU0', 'RB0', 'I0', 'P0', 'HC0']
SYMBOL_NAMES = {'AU0': '黄金', 'CU0': '沪铜', 'RB0': '螺纹钢',
                'I0': '铁矿石', 'P0': '棕榈油', 'HC0': '热卷'}


def signal_frame(close: pd.Series, open_: Optional[pd.Series] = None,
                 vol_window: int = VOL_WINDOW,
                 vol_lookback: int = VOL_LOOKBACK,
                 vol_threshold: float = VOL_THRESHOLD,
                 ma_window: int = MA_WINDOW) -> pd.DataFrame:
    """单品种信号框架。

    返回列：
      close / open  价格（open 可缺，生成成交记录时必需）
      ret           日收益
      vol20         20日波动率（未年化，排序用；年化仅展示）
      vol_pct       波动率250日分位（=压缩深度，越低越强）
      compressed    压缩状态
      direction     sign(收盘-MA20)；NaN 视为 0
    """
    close = close.astype(float).sort_index()
    ret = close.pct_change()
    vol20 = ret.rolling(vol_window).std()
    vol_pct = vol20.rolling(vol_lookback).rank(pct=True)
    ma = close.rolling(ma_window).mean()
    direction = np.sign(close - ma)
    clarity = (close / ma - 1).abs()
    frame = pd.DataFrame({
        'close': close,
        'ret': ret,
        'vol20': vol20,
        'vol_pct': vol_pct,
        'ma': ma,
        'clarity': clarity,
        'compressed': (vol_pct < vol_threshold).fillna(False),
        'direction': pd.Series(direction, index=close.index).fillna(0),
    })
    if open_ is not None:
        frame['open'] = open_.astype(float).sort_index()
    return frame


def rank_active(frames: Dict[str, pd.DataFrame],
                at: Optional[pd.Timestamp] = None) -> List[str]:
    """当日活跃品种按强度排序（压缩且方向非零）。

    排序键：vol_pct 升序 → |close/MA-1| 降序（clarity）→ 固定品种顺序。
    at=None 取每个 frame 的最后一行（各 frame 须已对齐到同一交易日，
    见 coverage_check）。
    """
    rows = []
    for sym, f in frames.items():
        if len(f) == 0 and at is None:
            continue  # 空表品种不参与排序（由覆盖检查提示缺失）
        row = f.iloc[-1] if at is None else f.loc[at]
        if not bool(row['compressed']) or row['direction'] == 0:
            continue
        if not np.isfinite(row['vol_pct']):
            continue
        clarity = float(row['clarity']) if np.isfinite(row['clarity']) else 0.0
        order = SYMBOL_ORDER.index(sym) if sym in SYMBOL_ORDER else len(SYMBOL_ORDER)
        rows.append((float(row['vol_pct']), -clarity, order, sym))
    rows.sort()
    return [r[3] for r in rows]


def target_book(frames: Dict[str, pd.DataFrame], k: int = 1,
                at: Optional[pd.Timestamp] = None) -> Dict[str, float]:
    """TOP-k 目标持仓（带符号权重，|w| = 1/k）。

    k=1 即 Z10 TOP1：全资金给最深压缩品种。
    """
    ranked = rank_active(frames, at=at)
    if not ranked:
        return {}
    kk = min(k, len(ranked))
    book = {}
    for sym in ranked[:kk]:
        row = frames[sym].iloc[-1] if at is None else frames[sym].loc[at]
        book[sym] = float(row['direction']) / kk
    return book


def book_transition(prev: Dict[str, float],
                    curr: Dict[str, float]) -> List[dict]:
    """相邻两日目标持仓 → 人类可读操作清单（次日开盘执行）。"""
    actions: List[dict] = []
    for sym in sorted(set(prev) | set(curr), key=lambda s: s):
        p, c = prev.get(sym, 0.0), curr.get(sym, 0.0)
        name = SYMBOL_NAMES.get(sym, sym)
        if p == 0 and c == 0:
            continue
        if p == 0:
            actions.append({'sym': sym, 'name': name,
                            'action': '开多' if c > 0 else '开空',
                            'weight': c, 'from': 0.0})
        elif c == 0:
            actions.append({'sym': sym, 'name': name, 'action': '平仓',
                            'weight': 0.0, 'from': p})
        elif p * c < 0:
            actions.append({'sym': sym, 'name': name,
                            'action': '翻多' if c > 0 else '翻空',
                            'weight': c, 'from': p})
        elif p != c:
            actions.append({'sym': sym, 'name': name, 'action': '调仓',
                            'weight': c, 'from': p})
        else:
            actions.append({'sym': sym, 'name': name, 'action': '不动',
                            'weight': c, 'from': p})
    return actions


def render_actions(actions: List[dict], lev_held: Optional[float],
                   lev_next: Optional[float],
                   yesterday_known: bool) -> List[str]:
    """操作清单最终文本（模块级以便文本测试）。

    关键语义：
      - 调仓 ≠ 平仓：同向权重变化显示"从 X 调至 Y"；
      - **起点敞口 = 昨日权重 × 今日持有杠杆(lev_held)**——这是当前
        实际持有的敞口；终点 = 今日权重 × 明日杠杆(lev_next)。
        两者必须与"目标持仓"表使用同一对值，否则按清单差额执行会错量；
      - 开/翻/平的目标敞口用 lev_next（T 日收盘决策、T+1 开盘生效）；
      - 昨日未知时起点敞口不可信，只显示目标。
    """
    lines = []
    if lev_next is None or not np.isfinite(lev_next):
        # 审计 2026-11-19 二-5：目标杠杆未知（计算异常）→ 开/翻/调仓等
        # 新增风险指令没有可执行的目标敞口，一律停发；仅保留平仓类指令
        # （降风险不需要目标敞口）。保留信号说明，人工确认后再执行。
        exits = [a for a in actions if a['action'] == '平仓']
        for a in exits:
            lines.append(f'- **{a["name"]}**：平仓，退出全部头寸')
        lines.append('- ⚠️ 杠杆计算异常：开仓/反手/调仓指令已停发（无目标敞口），'
                     '信号变化见上方目标持仓表，请人工确认后操作')
        return lines
    for a in actions:
        if a['action'] == '调仓':
            frm = a['from'] * lev_held \
                if (lev_held is not None and yesterday_known) else None
            to = a['weight'] * lev_next if lev_next is not None else None
            d = '多' if a['weight'] > 0 else '空'
            frm_s = f'{frm:+.2f}倍' if frm is not None else '—'
            to_s = f'{to:+.2f}倍' if to is not None else '—'
            lines.append(f'- **{a["name"]}**：调仓（{d}头）敞口 {frm_s} → {to_s}')
        elif a['action'] == '平仓':
            lines.append(f'- **{a["name"]}**：平仓，退出全部头寸')
        elif a['action'] in ('开多', '开空', '翻多', '翻空'):
            to = a['weight'] * lev_next if lev_next is not None else None
            to_s = f'，目标敞口 {to:+.2f}倍' if to is not None else ''
            lines.append(f'- **{a["name"]}**：{a["action"]}{to_s}')
        # '不动' 不输出
    return lines


def trades_from_frame(frame: pd.DataFrame, sym: str) -> List[dict]:
    """按登记语义生成成交记录：每日跟随 + 次日开盘成交。

    决策在 t 收盘后做出（含翻向），成交在 t+1 开盘。
    样本最后一日仍有仓位 → 不生成交易，返回 pending 标记的占位
    （由调用方展示"待执行"）。
    """
    if 'open' not in frame.columns:
        raise ValueError("trades_from_frame 需要 open 列（次日开盘成交口径）")
    sig = frame['direction'].where(frame['compressed'], 0.0)
    opens = frame['open']
    dates = frame.index

    trades: List[dict] = []
    pos = 0.0
    entry_px = np.nan
    entry_date = None

    for i in range(len(frame)):
        s = float(sig.iloc[i])
        # 次日开盘执行：今日决策，明天成交
        if i + 1 < len(frame):
            exec_px = float(opens.iloc[i + 1])
            exec_date = dates[i + 1]
        else:
            if pos != 0:
                trades.append({'sym': sym, 'direction': 'L' if pos > 0 else 'S',
                               'entryDate': str(entry_date.date()),
                               'entryPrice': float(entry_px),
                               'exitDate': 'pending', 'exitPrice': None,
                               'netRet': None, 'holdDays': None,
                               'pending': True})
            break

        if not np.isfinite(exec_px) or exec_px <= 0 or not np.isfinite(s):
            continue
        if pos == 0 and s != 0:
            pos, entry_px, entry_date = s, exec_px, exec_date
        elif pos != 0 and s != pos:
            # 平仓（信号归零）或翻向（信号反向），成交于次日开盘
            gross = (exec_px / entry_px - 1) * pos
            trades.append({'sym': sym, 'direction': 'L' if pos > 0 else 'S',
                           'entryDate': str(entry_date.date()),
                           'entryPrice': float(entry_px),
                           'exitDate': str(exec_date.date()),
                           'exitPrice': float(exec_px),
                           'netRet': gross - 2 * ONE_WAY_COST,
                           'holdDays': int((exec_date - entry_date).days),
                           'pending': False})
            pos = 0.0
            if s != 0:  # 翻向：同价位反手开仓
                pos, entry_px, entry_date = s, exec_px, exec_date
    return trades


def coverage_check(frames: Dict[str, pd.DataFrame],
                   expected: Optional[List[str]] = None,
                   as_of: Optional[pd.Timestamp] = None) -> dict:
    """数据覆盖核验（决策日口径）。

    ok=True 需同时满足：
      1. 品种齐全（expected 全部在场）；
      2. 各品种最新交易日一致（防残缺对齐）；
      3. 历史足够形成波动分位（≥ MIN_BARS = 270 根）；
      4. **决策日指标有限**：每个品种最后一行的 close 与 vol_pct 均为有限值
         ——否则该品种无法产生信号，"全部空仓"可能是"算不出信号"的假象。
    另返回 stale（决策日距今超过 STALE_WARN_CALENDAR_DAYS 个自然日），
    调用方应提示但不阻断（长假后属正常）。
    """
    expected = expected or SYMBOL_ORDER
    missing = [s for s in expected if s not in frames]
    # 空表（如盘中截断后/上游返回空）len=0 < MIN_BARS 自然落入历史不足，
    # 绝不让 index[-1]/iloc[-1] 越界
    empty = [s for s in expected if s in frames and len(frames[s]) == 0]
    short = [s for s in expected if s in frames and len(frames[s]) < MIN_BARS]
    no_signal = []
    for s in expected:
        if s in frames and len(frames[s]) > 0:
            row = frames[s].iloc[-1]
            if not (np.isfinite(row['close']) and np.isfinite(row['vol_pct'])):
                no_signal.append(s)
    last_dates = {s: frames[s].index[-1] for s in frames if len(frames[s]) > 0}
    dates = sorted(set(last_dates.values()))
    aligned = len(dates) <= 1
    stale = False
    last_date = dates[-1] if dates else None
    ref = pd.Timestamp(as_of) if as_of is not None else pd.Timestamp.today().normalize()
    if last_date is not None:
        stale = (ref - pd.Timestamp(last_date)).days > STALE_WARN_CALENDAR_DAYS
    return {'ok': not missing and aligned and not short and not no_signal,
            'missing': missing, 'empty': empty,
            'insufficient_history': sorted(set(short)),
            'no_signal_on_decision_day': no_signal,
            'last_dates': last_dates, 'aligned': aligned,
            'last_date': last_date, 'stale': stale}


def previous_book_date(frames: Dict[str, pd.DataFrame],
                       decision_date,
                       calendar=None) -> tuple:
    """决策日前一个已收盘交易日的判定（防跳日）。

    prev 候选 = 交易日历（calendar 显式提供时，只含交易日）中决策日之前
    最近的一天；未提供时退化为全品种日期并集（任一品种有 bar 的日期）。

    审计 2026-11-19 二-2（六品种共同缺日）：此前 prev = 各品种各自最近
    bar 的最大值——六品种同缺周二时返回"周一、全部齐全"，若周二发生过
    换仓，周三清单会重复开仓/漏平仓。并集口径修掉"部分品种缺日"；
    "全部品种同缺一个工作日"在无日历时无法与法定休市区分 → 昨日按
    未知处理（all_have=False，compose_actions 停发差额指令人工比对）。
    显式 calendar 可把休市日解释掉 → 法定休市不误报。

    返回 (prev_date or None, all_have)。
    """
    dec = pd.Timestamp(decision_date)
    if calendar is not None:
        cal = sorted({pd.Timestamp(d) for d in calendar})
    else:
        cal = sorted({d for f in frames.values() for d in f.index})
    before = [d for d in cal if d < dec]
    if not before:
        return None, False
    prev = before[-1]
    if calendar is None:
        # prev 与决策日之间存在工作日且并集无任何 bar：可能是六品种共同
        # 数据缺失（周二换过仓 → 周一不是昨日），也可能是法定休市——
        # 无日历不可区分，昨日按未知（宁停发差额指令，不猜昨日）。
        for d in pd.date_range(prev + pd.Timedelta(days=1),
                               dec - pd.Timedelta(days=1), freq='D'):
            if d.weekday() < 5:
                return prev, False
    all_have = all(prev in f.index for f in frames.values())
    return prev, all_have


def compose_actions(book_yesterday: Dict[str, float],
                    book_today: Dict[str, float],
                    yesterday_known: bool):
    """生成操作清单；昨日状态未知时返回 (None, 说明)。

    昨日未知 ≠ 昨日空仓：按"昨日=空"生成指令会导致重复开仓
    （今日同品种同向时）或漏掉平仓（今日空仓时），因此必须跳过
    操作清单，只展示今日目标并要求人工比对昨日报告。
    """
    if not yesterday_known:
        note = ('上一交易日目标持仓**未知**（数据不足以回推）。'
                '今日只展示目标持仓，不生成操作清单——请人工对比昨日报告：'
                '若昨日有持仓而今日目标不同，需自行执行平仓/调仓。')
        return None, note
    actions = [a for a in book_transition(book_yesterday, book_today)
               if a['action'] != '不动']
    return actions, ''


def _desired_signed_book(frames: Dict[str, pd.DataFrame], i: int,
                         k: int) -> Dict[str, float]:
    """第 i 行的 TOP-k 目标带符号权重（决策于当日收盘）。"""
    active = []
    for j, sym in enumerate(sorted(frames)):
        f = frames[sym]
        row = f.iloc[i]
        if not bool(row['compressed']) or row['direction'] == 0:
            continue
        if not np.isfinite(row['vol_pct']):
            continue
        clarity = float(row['clarity']) if np.isfinite(row['clarity']) else 0.0
        order = SYMBOL_ORDER.index(sym) if sym in SYMBOL_ORDER else len(SYMBOL_ORDER)
        active.append((float(row['vol_pct']), -clarity, order, sym,
                       float(row['direction'])))
    if not active:
        return {}
    active.sort()
    kk = min(k, len(active)) if k > 0 else len(active)
    w = 1.0 / len(SYMBOL_ORDER) if k == 0 else 1.0 / kk
    return {sym: d * w for _, _, _, sym, d in active[:kk]}


def portfolio_daily(frames: Dict[str, pd.DataFrame], k: int = 1,
                    execution: str = 'next_open',
                    charge_cost: bool = True,
                    leverage: Optional[pd.Series] = None) -> pd.Series:
    """Mark signed units to market and rebalance at executable prices.

    Next-open execution assigns overnight P&L to the previous units. The
    close comparator rebalances at the prior close. Costs use traded notional.
    """
    if not frames:
        return pd.Series(dtype=float)
    if execution not in ('next_open', 'close'):
        raise ValueError(f'unknown execution: {execution}')
    idx = None
    for f in frames.values():
        idx = f.index if idx is None else idx.union(f.index)
    idx = pd.DatetimeIndex(sorted(idx))
    n = len(idx)

    # 对齐到并集日历（品种日历不一致时以 NaN 行占位，决策时自然跳过）
    aligned = {s: f.reindex(idx) for s, f in frames.items()}
    closes = {s: f['close'] for s, f in aligned.items()}
    opens = {s: (f['open'] if 'open' in f.columns else None)
             for s, f in aligned.items()}
    if execution == 'next_open' and any(o is None for o in opens.values()):
        raise ValueError("next_open 口径需要 open 列")
    if leverage is not None:
        leverage = leverage.reindex(idx)

    returns = np.full(n, np.nan)
    units, marks, target = {}, {}, {}
    nav = 1.0
    unexecuted = []

    for i, day in enumerate(idx):
        previous_nav = nav
        lev = float(leverage.iloc[i]) if leverage is not None else 1.0
        ready = np.isfinite(lev)
        symbols = sorted(set(units) | set(target))
        execution_prices = {}
        for sym in symbols:
            px = (opens[sym].iloc[i] if execution == 'next_open'
                  else closes[sym].iloc[i - 1])
            if np.isfinite(px) and px > 0:
                execution_prices[sym] = px
                if sym in units:
                    nav += units[sym] * (px - marks[sym])
                marks[sym] = px
        opening_nav = nav
        for sym in symbols:
            old = units.get(sym, 0.)
            if ready and i > 0:
                weight = target.get(sym, 0.) * lev
                if sym in execution_prices:
                    px = execution_prices[sym]
                    new = weight * opening_nav / px
                    if charge_cost:
                        nav -= abs(new - old) * px * ONE_WAY_COST
                    if new:
                        units[sym] = new
                    else:
                        units.pop(sym, None)
                elif not np.isclose(old * marks.get(sym, 0.), weight * opening_nav):
                    unexecuted.append(dict(date=str(day), symbol=sym,
                                           from_exposure=old * marks.get(sym, 0.) / opening_nav,
                                           to_exposure=weight, reason='missing_' + ('open' if execution == 'next_open' else 'close')))
            close = closes[sym].iloc[i]
            if sym in units and np.isfinite(close) and close > 0:
                nav += units[sym] * (close - marks[sym])
                marks[sym] = close
        if nav <= 0 or not np.isfinite(nav):
            raise ValueError('Portfolio NAV exhausted')
        if ready or units:
            returns[i] = nav / previous_nav - 1
        target = _desired_signed_book(aligned, i, k)
    out = pd.Series(returns, index=idx)
    out.attrs['unexecuted'] = unexecuted
    if unexecuted:
        logger.warning(f"[portfolio_daily] {len(unexecuted)} 笔目标因缺执行价未成交"
                       "（敞口冻结，详见返回值 attrs['unexecuted']）")
    return out


def max_drawdown_from_returns(r: pd.Series) -> float:
    """净值最大回撤（含初始本金 1.0）。

    审计 2026-11-19 一-7：旧算法 nav/nav.cummax() 的首个峰值是首日净值——
    首日亏损 30% 时回撤算成 0%，25% 回撤门槛被绕过。初始本金必须进入
    峰值序列。报告与验收共用本实现，避免口径漂移。"""
    r = r.dropna()
    if len(r) == 0:
        return 0.0
    nav = (1 + r).cumprod()
    peak = nav.cummax().clip(lower=1.0)
    return float((nav / peak - 1).min())


def criteria_verdict(net_deploy: pd.Series, net_base: pd.Series,
                     split: str, dd_cap: float = 0.25) -> dict:
    """预注册三判据（登记册180定义）：①夏普胜基线 ②两期同号为正
    ③回撤幅度 ≤ dd_cap。部署验收共用，避免口径漂移。"""
    def _stats(r):
        r = r.dropna()
        yrs = len(r) / 252
        ann = (1 + r).prod() ** (1 / yrs) - 1 if len(r) > 20 else 0.0
        sh = r.mean() / r.std() * np.sqrt(252) if (len(r) > 20 and r.std() > 0) else 0.0
        mdd = max_drawdown_from_returns(r) if len(r) > 20 else 0.0
        return ann, sh, mdd

    net_deploy = net_deploy.replace([np.inf, -np.inf], np.nan).dropna()
    net_base = net_base.reindex(net_deploy.index).replace([np.inf, -np.inf], np.nan)
    _, sh_d, mdd = _stats(net_deploy)
    _, sh_b, _ = _stats(net_base)
    p1 = net_deploy[net_deploy.index <= split].dropna()
    p2 = net_deploy[net_deploy.index > split].dropna()
    ann1 = (1 + p1).prod() ** (252 / len(p1)) - 1 if len(p1) > 20 else 0.0
    ann2 = (1 + p2).prod() ** (252 / len(p2)) - 1 if len(p2) > 20 else 0.0
    # 二轮复审#35：基线缺数据/样本不足时夏普记 0 → 判据①空过。基线
    # 无有效统计量 → 整体判不可验收（pass_all=False 并注明原因）。
    base_ok = len(net_base) > 20 and net_base.notna().all()
    verdict = {
        'sharpe': sh_d, 'sharpe_base': sh_b,
        'c1_sharpe_beats_base': bool(base_ok and sh_d >= sh_b),
        'ann_p1': ann1, 'ann_p2': ann2,
        'c2_both_periods_positive': bool(ann1 > 0 and ann2 > 0),
        'max_dd': mdd,
        'c3_dd_within_cap': bool(mdd >= -dd_cap),
        'base_verifiable': bool(base_ok),
        'pass_all': bool(base_ok and sh_d >= sh_b and ann1 > 0 and ann2 > 0
                         and mdd >= -dd_cap),
    }
    if not base_ok:
        verdict['not_verdictable_reason'] = '基线收益样本不足（≤20 个有效观测）'
    return verdict


def vol_target_leverage(rets: pd.Series, target: float = 0.15,
                        window: int = 60, cap: float = 2.0) -> pd.Series:
    """波动率目标杠杆（回测口径）：标签日 T 的杠杆只用 T-1 及以前的收益。

    用于回测逐日归因（无前视）。**日报在 T 日收盘后取明日订单杠杆时，
    必须用 next_leverage()（含 T 日收益、T+1 生效），不能取本序列末值
    ——本序列末值只含到 T-1 的信息，会滞后一天。**
    """
    vol = rets.rolling(window).std() * np.sqrt(252)
    lev = (target / vol).clip(upper=cap).shift(1)
    return lev


def next_leverage(rets: pd.Series, target: float = 0.15,
                  window: int = 60, cap: float = 2.0) -> Optional[float]:
    """T 日收盘后生成、T+1 开盘生效的目标杠杆（含 T 日收益）。

    等价于回测 vol_target_leverage 序列在 T+1 行的值（shift(1) 前的
    未滞后末值）。波动突变日（T 日大波动）会即时反映到明日订单。
    """
    vol = rets.rolling(window).std() * np.sqrt(252)
    lev = (target / vol).clip(upper=cap)
    v = float(lev.iloc[-1]) if len(lev) else np.nan
    return v if np.isfinite(v) else None


def held_leverage(rets: pd.Series, target: float = 0.15,
                  window: int = 60, cap: float = 2.0) -> Optional[float]:
    """T 日实际持有的杠杆（T-1 收盘决策、T 日生效）——即滞后序列末值。"""
    lev = vol_target_leverage(rets, target, window, cap)
    v = float(lev.iloc[-1]) if len(lev) else np.nan
    return v if np.isfinite(v) else None


def truncate_to_closed(frames: Dict[str, pd.DataFrame],
                       last_closed) -> Dict[str, pd.DataFrame]:
    """截取各品种到最近已收盘交易日（含）。

    盘中运行时数据源可能已返回当日未定稿日线；信号、排名与杠杆必须
    只用已收盘数据。last_closed 为 date 或 Timestamp（含当日则当日为
    已收盘）。返回新 dict（原 frames 不被修改）。
    """
    cut = pd.Timestamp(last_closed)
    return {s: f[f.index <= cut] for s, f in frames.items()}
