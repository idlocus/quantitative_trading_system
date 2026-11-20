#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实盘账本 —— 持仓/现金/成交/净值 的持久化与对账（JSON 单文件，简单可审计）

对账支持（v2）：
- 成交去重：dedup_key（券商trade_id优先，否则 日期|代码|方向|数量|价格），
  同日重复执行/重复回报只入账一次
- 部分成交：quantity>0 即入账（partial 状态按实际成交量记账，撤单/废单跳过）
- 公司行动：apply_dividend / apply_split（事件幂等，重复执行不重复入账）
- 资金流水：cash_log 记录每笔现金变动（trade/dividend/deposit/withdraw/调整）
- 账实调整：adjust_cash / adjust_position（对账修正，全部留痕）
"""

import copy
import json
from numbers import Real
from utils.state_file import exclusive_file, replace_json
import math
import os
from datetime import date, datetime
from typing import Dict, List, Optional, Tuple

from .broker import Fill
from utils.logger import get_logger

logger = get_logger(__name__)

TRADE_TYPES = ('trade', 'dividend', 'deposit', 'withdraw', 'cash_adj', 'fee_adj')


class Ledger:
    """
    持仓账本
    文件结构:
    { cash, positions: {sym: qty}, cost: {sym: avg_cost},
      trades: [...], nav_history: [{date, nav, cash, market_value}],
      cash_log: [{date, type, amount, ref, note}],
      events: [{key, type, date, ...}],
      meta }
    """

    def __init__(self, path: str | None):
        self.path = path
        if path is not None and os.path.isfile(path):
            with open(path, encoding='utf-8') as f:
                self._d = json.load(f)
        else:
            self._d = {'cash': 0.0, 'positions': {}, 'cost': {},
                       'trades': [], 'nav_history': [], 'meta': {'created': str(date.today())}}
        # 旧版本文件兼容：新字段缺省补齐
        self._d.setdefault('cash_log', [])
        self._d.setdefault('events', [])
        self._d.setdefault('exceptions', [])   # 待人工核对异常（如超额卖出）
        self._d.setdefault('meta', {})
        # 状态文件版本：语义变更时显式迁移（不靠字段缺省静默兼容——
        # 2026-10-15 实盘变更管理缺口#3 补齐；v1=当前格式，历史上仅靠
        # setdefault 扛过所有变更）
        if 'schema_version' not in self._d['meta']:
            self._d['meta']['schema_version'] = 1
        self._d.setdefault('cash', 0.0)
        self._d.setdefault('positions', {})
        self._d.setdefault('cost', {})
        self._d.setdefault('trades', [])
        self._d.setdefault('nav_history', [])
        # 版本校验基准（审计 2026-11-19 三-7）：本实例加载时的磁盘版本，
        # save() 前据此检测其他进程/实例的并发写入
        try:
            self._base_rev = int(self._d.get('meta', {}).get('rev', 0))
        except (TypeError, ValueError):
            self._base_rev = 0

        self._committed = copy.deepcopy(self._d) if path is not None else None
        self._existed = path is not None and os.path.isfile(path)

    def latest_activity(self, include_initial=True):
        dates = [str(row['date'])[:10] for key in ('trades', 'events') for row in self._d[key]]
        dates.extend(str(row['date'])[:10] for row in self._d['cash_log'][0 if include_initial else 1:])
        return max(dates, default='')

    # ---- 基础 ----

    @property
    def cash(self) -> float:
        return self._d['cash']

    @property
    def positions(self) -> Dict[str, int]:
        return self._d['positions']

    @staticmethod
    def _require_finite(value, name: str, minimum=None):
        """二轮复审#10：入金/调整/估值写入口统一数值校验——NaN 入金后
        现金为 NaN，与正常账单核对竟返回一致（净值/许可/告警全失效）。"""
        try:
            v = float(value)
        except (TypeError, ValueError):
            raise ValueError(f'{name} 非数值: {value!r}') from None
        if not math.isfinite(v):
            raise ValueError(f'{name} 必须为有限数（拒绝 NaN/Infinity）')
        if minimum is not None and v < minimum:
            raise ValueError(f'{name} 不得低于 {minimum}')
        return v

    def initialize(self, capital: float, as_of: date = None):
        """初始入金（仅一次）——同时建立净值基准（回撤计算的起点）"""
        capital = self._require_finite(capital, '初始入金', minimum=0)
        if capital <= 0:
            raise ValueError('初始入金必须为正')
        if self._d['cash'] != 0 or self._d['trades'] or self._d['positions']:
            raise RuntimeError('账本已有资金或交易，不能重新入金')
        as_of = as_of or date.today()
        self._d['cash'] = float(capital)
        self._log_cash(str(as_of), 'deposit', float(capital), '初始入金')
        # 建立净值基准：peak_nav 从初始资金开始，而非从0（0会导致回撤保护静默跳过）
        self._d['nav_history'].append({
            'date': str(as_of), 'nav': round(float(capital), 2),
            'cash': round(float(capital), 2), 'market_value': 0})
        self.save()

    # ---- 成交入账（去重 + 部分成交） ----

    @staticmethod
    def trade_key(trade_id: str, d: str, symbol: str, side: str,
                  quantity: int, price: float) -> str:
        """成交唯一键：券商回报ID优先；无ID时用要素指纹（同日重复执行可识别）。

        数量/价格非法（NaN 等）时指纹退化为 repr——合法成交的指纹不受影响，
        非法成交本就会被无效成交防线拦截，指纹只进异常记录。"""
        if trade_id:
            return f'id:{trade_id}'
        try:
            qf = float(quantity)
            qs = str(int(qf)) if (math.isfinite(qf) and qf.is_integer()) else repr(quantity)
        except (TypeError, ValueError):
            qs = repr(quantity)
        px = f'{price:.3f}' if isinstance(price, Real) else repr(price)
        return '|'.join(map(str, (str(d)[:10], symbol, side, qs, px)))

    def _fill_key(self, f: Fill) -> str:
        return self.trade_key(getattr(f, 'trade_id', '') or '',
                              str(f.trade_date)[:10], f.order.symbol, f.order.side,
                              f.quantity, f.price)

    @staticmethod
    def _invalid_fill_reason(f: Fill) -> str:
        """成交要素合法性校验（写账前），返回原因；空串 = 合法。

        审计 2026-11-19 三-2：未知方向曾落入卖出分支凭空加现金
        （1万→1.1万），NaN 价格把现金永久污染成 NaN，负费用/非整数
        数量破坏对账指纹——一律在入账前拒绝。
        """
        try:
            date.fromisoformat(str(f.trade_date)[:10])
        except ValueError:
            return 'Invalid trade date'
        if not isinstance(f.quantity, Real) or not isinstance(f.order.symbol, str) or not f.order.symbol.strip():
            return 'Invalid quantity or symbol'
        side = getattr(f.order, 'side', None)
        if side not in ('BUY', 'SELL'):
            return f'方向非法 {side!r}'
        price = f.price
        if not (isinstance(price, (int, float)) and math.isfinite(price)
                and price > 0):
            return f'价格非法 {price!r}'
        try:
            qf = float(f.quantity)
        except (TypeError, ValueError):
            return f'数量非法 {f.quantity!r}'
        if not (math.isfinite(qf) and qf > 0 and qf.is_integer()):
            return f'数量非法 {f.quantity!r}'
        for name, v in (('佣金', f.commission), ('税费', f.taxes)):
            if not (isinstance(v, (int, float)) and math.isfinite(v) and v >= 0):
                return f'{name}非法 {v!r}'
        return ''

    def apply_fills(self, fills: List[Fill]) -> Tuple[List[Fill], List[Fill], List[Fill]]:
        """
        成交入账。

        - quantity>0 即入账：含 partial（部分成交按实际量）、含 rejected 但有量的异常单
        - dedup_key 已存在 → 跳过（重复回报/同日重复执行不重复入账）
        - 要素非法（方向/价格/数量/费用）→ 阻止入账记异常（现金不得被污染）
        Returns:
            (booked, dupes, blocked) 本次入账的fills / 被去重跳过的fills / 被防线阻止的fills
        """
        existing = {(t.get('dedup_key') or self.trade_key(
            t.get('trade_id', ''), t['date'], t['symbol'], t['side'],
            t['quantity'], t['price'])): t for t in self._d['trades']}
        booked, dupes, blocked = [], [], []
        for f in fills:
            if f.quantity == 0:
                continue  # 撤单/废单/零股 无现金与持仓影响
            key = self._fill_key(f)
            reason = self._invalid_fill_reason(f)
            prior = existing.get(key)
            if prior is not None and not reason:
                same = (prior['date'], prior['symbol'], prior['side'], prior['quantity'],
                        prior['price'], prior['commission'], prior['taxes'])
                if same != (str(f.trade_date)[:10], f.order.symbol, f.order.side, f.quantity,
                            f.price, f.commission, f.taxes):
                    reason = 'Conflicting execution identity'
            if reason:
                self._d['exceptions'].append({
                    'date': str(f.trade_date)[:10], 'type': 'invalid_fill',
                    'symbol': f.order.symbol, 'side': str(f.order.side),
                    'quantity': repr(f.quantity), 'price': repr(f.price),
                    'ref': key, 'note': reason, 'status': 'open',
                })
                logger.error(f"[Ledger] 无效成交已阻止：{f.order.symbol} "
                             f"{reason}（记入异常队列，需人工核对）")
                blocked.append(f)
                continue
            if key in existing:
                dupes.append(f)
                continue
            sym = f.order.symbol

            # 超额卖出防线：卖量 > 持仓 → 阻止入账（现金不得凭空增加），
            # 记入异常队列待人工核对（常见成因：重复卖出/错误对账单/公司行动未入账）
            if f.order.side == 'SELL':
                holding = self._d['positions'].get(sym, 0)
                if f.quantity > holding:
                    self._d['exceptions'].append({
                        'date': str(f.trade_date)[:10], 'type': 'oversell',
                        'symbol': sym, 'quantity': int(f.quantity),
                        'holding': int(holding), 'price': float(f.price),
                        'ref': key, 'note': getattr(f.order, 'note', '') or '',
                        'status': 'open',
                    })
                    logger.error(f"[Ledger] 超额卖出已阻止：{sym} 卖{f.quantity} > "
                                 f"持仓{holding}（记入异常队列，需人工核对）")
                    blocked.append(f)
                    continue

            gross = f.price * f.quantity
            net = gross - (f.commission + f.taxes) if f.order.side == 'SELL' \
                else gross + (f.commission + f.taxes)
            sym = f.order.symbol
            if f.order.side == 'BUY':
                self._d['cash'] -= net
                old_qty = self._d['positions'].get(sym, 0)
                old_cost = self._d['cost'].get(sym, 0.0) * old_qty
                new_qty = old_qty + f.quantity
                self._d['positions'][sym] = new_qty
                self._d['cost'][sym] = (old_cost + gross) / new_qty if new_qty else 0.0
            else:
                self._d['cash'] += net
                self._d['positions'][sym] = self._d['positions'].get(sym, 0) - f.quantity
                if self._d['positions'][sym] <= 0:
                    self._d['positions'].pop(sym, None)
                    self._d['cost'].pop(sym, None)
            ts = str(f.trade_date)  # date 或 datetime
            self._d['trades'].append({
                'date': ts[:10], 'time': ts, 'symbol': sym, 'side': f.order.side,
                'price': f.price, 'quantity': f.quantity,
                'commission': f.commission, 'taxes': f.taxes,
                'target_weight': f.order.target_weight,
                'note': getattr(f.order, 'note', '') or '',
                'status': f.status,
                'trade_id': getattr(f, 'trade_id', '') or '',
                'dedup_key': key,
            })
            existing[key] = self._d['trades'][-1]
            self._log_cash(ts[:10], 'trade', -net if f.order.side == 'BUY' else net, key)
            booked.append(f)
        self.save()
        return booked, dupes, blocked

    # ---- 公司行动（分红/送转，幂等） ----

    def _event_seen(self, key: str) -> bool:
        return any(e.get('key') == key for e in self._d['events'])

    def apply_dividend(self, symbol: str, amount: float, ex_date: str = '',
                       note: str = '') -> bool:
        """现金分红入账（amount=税后到账金额）。同一事件key重复执行不重复入账。"""
        amount = self._require_finite(amount, 'dividend', minimum=0)
        ex_date = date.fromisoformat(str(ex_date or date.today())).isoformat()
        key = f'div|{symbol}|{ex_date or str(date.today())}|{amount:.2f}'
        if self._event_seen(key):
            return False
        self._d['cash'] += float(amount)
        self._d['events'].append({'key': key, 'type': 'dividend', 'date': ex_date or str(date.today()),
                                  'symbol': symbol, 'amount': round(float(amount), 2), 'note': note})
        self._log_cash(ex_date or str(date.today()), 'dividend', float(amount), key)
        self.save()
        return True

    def apply_split(self, symbol: str, ratio: float, ex_date: str = '',
                    note: str = '') -> bool:
        """送股/转增（ratio=送转比例，如10送3 → 1.3）。数量×ratio，成本÷ratio。"""
        ratio = self._require_finite(ratio, 'split ratio')
        if ratio <= 0:
            raise ValueError('ratio 必须为正（10送3 → 1.3）')
        ex_date = date.fromisoformat(str(ex_date or date.today())).isoformat()
        key = f'split|{symbol}|{ex_date or str(date.today())}|{ratio}'
        if self._event_seen(key):
            return False
        qty = self._d['positions'].get(symbol, 0)
        cost = self._d['cost'].get(symbol, 0.0)
        if qty > 0:
            new_qty = int(round(qty * ratio))
            self._d['positions'][symbol] = new_qty
            last = self._d['meta'].get('last_prices', {})
            if symbol in last:
                last[symbol] /= ratio
            if cost > 0:
                self._d['cost'][symbol] = cost / ratio
        self._d['events'].append({'key': key, 'type': 'split', 'date': ex_date or str(date.today()),
                                  'symbol': symbol, 'ratio': ratio,
                                  'qty_before': qty, 'qty_after': self._d['positions'].get(symbol, 0),
                                  'note': note})
        self.save()
        return True

    # ---- 资金存取与流水 ----

    def deposit(self, amount: float, note: str = '') -> float:
        amount = self._require_finite(amount, '入金额')
        if amount <= 0:
            raise ValueError('入金额必须为正')
        self._d['cash'] += float(amount)
        self._log_cash(str(date.today()), 'deposit', float(amount), '', note)
        self.save()
        return self._d['cash']

    def withdraw(self, amount: float, note: str = '') -> float:
        amount = self._require_finite(amount, '出金额')
        if amount <= 0:
            raise ValueError('出金额必须为正')
        if amount > self._d['cash']:
            raise ValueError(f'出金超可用现金（{amount:,.0f} > {self._d["cash"]:,.0f}）')
        self._d['cash'] -= float(amount)
        self._log_cash(str(date.today()), 'withdraw', -float(amount), '', note)
        self.save()
        return self._d['cash']

    def cash_log(self) -> List[dict]:
        return list(self._d['cash_log'])

    # ---- 账实调整（对账修正，留痕） ----

    def adjust_cash(self, target: float, reason: str, as_of=None) -> float:
        """现金修正到券商口径。差异写入cash_log（type=cash_adj）。"""
        target = self._require_finite(target, '现金修正目标')
        diff = float(target) - self._d['cash']
        if abs(diff) < 0.005:
            return 0.0
        day = date.fromisoformat(str(as_of or date.today())).isoformat()
        self._d['cash'] = float(target)
        self._log_cash(day, 'cash_adj', diff, '', reason)
        self.save()
        return diff

    def adjust_position(self, symbol: str, target_qty: int, reason: str, as_of=None) -> int:
        """持仓数量修正到券商口径（成本不动，差异走事件留痕）。"""
        q = self._require_finite(target_qty, 'quantity', minimum=0)
        if not q.is_integer():
            raise ValueError('Fractional position')
        cur = self._d['positions'].get(symbol, 0)
        diff = int(target_qty) - cur
        if diff == 0:
            return 0
        day = date.fromisoformat(str(as_of or date.today())).isoformat()
        key = f'pos_adj|{symbol}|{day}|{cur}->{target_qty}'
        if int(target_qty) <= 0:
            self._d['positions'].pop(symbol, None)
            self._d['cost'].pop(symbol, None)
        else:
            self._d['positions'][symbol] = int(target_qty)
            self._d['cost'].setdefault(symbol, 0.0)
        self._d['events'].append({'key': key, 'type': 'pos_adj', 'date': day,
                                  'symbol': symbol, 'qty_before': cur,
                                  'qty_after': int(target_qty), 'note': reason})
        self.save()
        return diff

    # ---- 异常队列（超额卖出等，人工核对后 resolve） ----

    def exceptions(self) -> List[dict]:
        return list(self._d.get('exceptions', []))

    def open_exceptions(self) -> List[dict]:
        return [e for e in self._d.get('exceptions', []) if e.get('status') == 'open']

    def resolve_exception(self, index: int, note: str = ''):
        exs = self._d.get('exceptions', [])
        if 0 <= index < len(exs):
            exs[index]['status'] = 'resolved'
            exs[index]['resolved_note'] = note
            self.save()

    def record_recon(self, as_of: str, ok: bool, n_diffs: int,
                     source: str = 'manual', fixed: bool = False,
                     snapshot: str = None):
        """统一对账记录入口（snapshot=本次依据的原始快照路径，审计可追溯）。

        二轮复审#9：snapshot=None 时**保留**上次记录的快照路径——修正流程
        不传 snapshot 会把先前保存的依据清成空字符串。"""
        recon = self._d.setdefault('meta', {}).get('last_recon', {})
        if str(recon.get('date', '')) > str(as_of):
            raise ValueError('An older reconciliation cannot replace a later record')
        if snapshot is None:
            snapshot = recon.get('snapshot', '') if recon.get('date') == as_of else ''
        self._d.setdefault('meta', {})['last_recon'] = {
            'date': as_of, 'ok': ok, 'n_diffs': n_diffs,
            'source': source, 'fixed': fixed, 'snapshot': snapshot}
        self.save()

    # ---- 净值 ----

    def mark_to_market(self, prices: Dict[str, float], as_of: date) -> float:
        """缺价持仓沿用最后有效价（首次缺价记 warning）——不用 0 计值：
        否则净值瞬间被低估→回撤误触→错误净值写入历史→风控被污染。

        二轮复审#15：日期合法性在**任何状态变更之前**校验——此前拒绝的
        补记请求已经污染了 last_prices（下一日缺价估值随之错）。且历史
        重标只能针对最新一日（旧日重标会用当前持仓改写历史净值）。"""
        hist = self._d['nav_history']
        latest = max(self.latest_activity(), hist[-1]['date'] if hist else '')
        if str(as_of) < latest:
            raise ValueError(
                f"拒绝重标历史日期 {as_of}（最新记录 {latest}）："
                f"历史净值须按当日余额快照或事件重放重建，不接受当前持仓改写")
        mv = 0.0
        _last = self._d.setdefault('meta', {}).setdefault('last_prices', {})
        for s, qty in self._d['positions'].items():
            px = prices.get(s)
            if px is not None and math.isfinite(px) and px > 0:
                _last[s] = px                            # 更新最后有效价
            else:
                px = _last.get(s, 0.0)                   # 沿用最后有效价
                if px <= 0:
                    logger.warning(f"[Ledger] {s} 无当前价且无历史价，按0计（首次）")
            mv += qty * px
        nav = self._d['cash'] + mv
        entry = {'date': str(as_of), 'nav': round(nav, 2),
                 'cash': round(self._d['cash'], 2), 'market_value': round(mv, 2)}
        if not hist or hist[-1]['date'] < str(as_of):
            hist.append(entry)
        else:
            hist[-1] = entry
        self.save()
        return nav

    def nav_series(self) -> List[dict]:
        return list(self._d['nav_history'])

    def last_nav(self) -> Optional[float]:
        return self._d['nav_history'][-1]['nav'] if self._d['nav_history'] else None

    # ---- 持仓权重 ----

    def current_weights(self, prices: Dict[str, float]) -> Dict[str, float]:
        """缺价持仓沿用最后有效价（与 mark_to_market 同口径）——否则权重被
        低估→买入预算被高估→可能重复买入已有持仓。"""
        _last = self._d.get('meta', {}).get('last_prices', {})
        def _px(s):
            p = prices.get(s)
            if p is not None and math.isfinite(p) and p > 0:
                return p
            return _last.get(s, 0.0)
        mv = {s: qty * _px(s) for s, qty in self._d['positions'].items()}
        total = self._d['cash'] + sum(mv.values())
        if total <= 0:
            return {}
        return {s: v / total for s, v in mv.items() if v > 0}

    # ---- 内部 ----

    def _log_cash(self, d: str, type_: str, amount: float, ref: str = '', note: str = ''):
        self._d['cash_log'].append({'date': str(d)[:10], 'type': type_,
                                    'amount': round(float(amount), 2),
                                    'ref': ref, 'note': note})

    def save(self):
        if self.path is None:
            return
        try:
            with exclusive_file(self.path):
                if os.path.isfile(self.path):
                    with open(self.path, encoding='utf-8') as stream:
                        rev = int(json.load(stream).get('meta', {}).get('rev', 0))
                    if not self._existed or rev != self._base_rev:
                        raise RuntimeError('账本文件已被其他实例修改，请重新加载')
                elif self._existed:
                    raise RuntimeError('Ledger file disappeared')
                self._d['meta']['rev'] = self._base_rev + 1
                replace_json(self.path, self._d)
                self._base_rev += 1
                self._existed = True
                self._committed = copy.deepcopy(self._d)
        except Exception:
            self._d = copy.deepcopy(self._committed)
            raise
