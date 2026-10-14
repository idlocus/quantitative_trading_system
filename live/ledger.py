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

import json
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

    def __init__(self, path: str):
        self.path = path
        if os.path.isfile(path):
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
        self._d.setdefault('cash', 0.0)
        self._d.setdefault('positions', {})
        self._d.setdefault('cost', {})
        self._d.setdefault('trades', [])
        self._d.setdefault('nav_history', [])

    # ---- 基础 ----

    @property
    def cash(self) -> float:
        return self._d['cash']

    @property
    def positions(self) -> Dict[str, int]:
        return self._d['positions']

    def initialize(self, capital: float):
        """初始入金（仅一次）——同时建立净值基准（回撤计算的起点）"""
        if self._d['cash'] != 0 or self._d['trades'] or self._d['positions']:
            raise RuntimeError('账本已有资金或交易，不能重新入金')
        self._d['cash'] = float(capital)
        self._log_cash(str(date.today()), 'deposit', float(capital), '初始入金')
        # 建立净值基准：peak_nav 从初始资金开始，而非从0（0会导致回撤保护静默跳过）
        self._d['nav_history'].append({
            'date': str(date.today()), 'nav': round(float(capital), 2),
            'cash': round(float(capital), 2), 'market_value': 0})
        self.save()

    # ---- 成交入账（去重 + 部分成交） ----

    @staticmethod
    def trade_key(trade_id: str, d: str, symbol: str, side: str,
                  quantity: int, price: float) -> str:
        """成交唯一键：券商回报ID优先；无ID时用要素指纹（同日重复执行可识别）"""
        if trade_id:
            return f'id:{trade_id}'
        return '|'.join([str(d)[:10], symbol, side, str(int(quantity)), f'{float(price):.3f}'])

    def _fill_key(self, f: Fill) -> str:
        return self.trade_key(getattr(f, 'trade_id', '') or '',
                              str(f.trade_date)[:10], f.order.symbol, f.order.side,
                              f.quantity, f.price)

    def apply_fills(self, fills: List[Fill]) -> Tuple[List[Fill], List[Fill]]:
        """
        成交入账。

        - quantity>0 即入账：含 partial（部分成交按实际量）、含 rejected 但有量的异常单
        - dedup_key 已存在 → 跳过（重复回报/同日重复执行不重复入账）
        Returns:
            (booked, dupes, blocked) 本次入账的fills / 被去重跳过的fills / 被超额卖出防线阻止的fills
        """
        existing = {t.get('dedup_key') or self.trade_key(
            t.get('trade_id', ''), t['date'], t['symbol'], t['side'],
            t['quantity'], t['price']) for t in self._d['trades']}
        booked, dupes, blocked = [], [], []
        for f in fills:
            if f.quantity <= 0:
                continue  # 撤单/废单/零股 无现金与持仓影响
            key = self._fill_key(f)
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

            existing.add(key)
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
        if ratio <= 0:
            raise ValueError('ratio 必须为正（10送3 → 1.3）')
        key = f'split|{symbol}|{ex_date or str(date.today())}|{ratio}'
        if self._event_seen(key):
            return False
        qty = self._d['positions'].get(symbol, 0)
        cost = self._d['cost'].get(symbol, 0.0)
        if qty > 0:
            new_qty = int(round(qty * ratio))
            self._d['positions'][symbol] = new_qty
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
        if amount <= 0:
            raise ValueError('入金额必须为正')
        self._d['cash'] += float(amount)
        self._log_cash(str(date.today()), 'deposit', float(amount), '', note)
        self.save()
        return self._d['cash']

    def withdraw(self, amount: float, note: str = '') -> float:
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

    def adjust_cash(self, target: float, reason: str) -> float:
        """现金修正到券商口径。差异写入cash_log（type=cash_adj）。"""
        diff = float(target) - self._d['cash']
        if abs(diff) < 0.005:
            return 0.0
        self._d['cash'] = float(target)
        self._log_cash(str(date.today()), 'cash_adj', diff, '', reason)
        self.save()
        return diff

    def adjust_position(self, symbol: str, target_qty: int, reason: str) -> int:
        """持仓数量修正到券商口径（成本不动，差异走事件留痕）。"""
        cur = self._d['positions'].get(symbol, 0)
        diff = int(target_qty) - cur
        if diff == 0:
            return 0
        key = f'pos_adj|{symbol}|{date.today()}|{cur}->{target_qty}'
        if int(target_qty) <= 0:
            self._d['positions'].pop(symbol, None)
            self._d['cost'].pop(symbol, None)
        else:
            self._d['positions'][symbol] = int(target_qty)
            self._d['cost'].setdefault(symbol, 0.0)
        self._d['events'].append({'key': key, 'type': 'pos_adj', 'date': str(date.today()),
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
                     source: str = 'manual', fixed: bool = False):
        """统一对账记录入口"""
        self._d.setdefault('meta', {})['last_recon'] = {
            'date': as_of, 'ok': ok, 'n_diffs': n_diffs,
            'source': source, 'fixed': fixed}
        self.save()

    # ---- 净值 ----

    def mark_to_market(self, prices: Dict[str, float], as_of: date) -> float:
        """缺价持仓沿用最后有效价（首次缺价记 warning）——不用 0 计值：
        否则净值瞬间被低估→回撤误触→错误净值写入历史→风控被污染。"""
        mv = 0.0
        _last = self._d.setdefault('meta', {}).setdefault('last_prices', {})
        for s, qty in self._d['positions'].items():
            px = prices.get(s)
            if px is not None and px > 0:
                _last[s] = px                            # 更新最后有效价
            else:
                px = _last.get(s, 0.0)                   # 沿用最后有效价
                if px <= 0:
                    logger.warning(f"[Ledger] {s} 无当前价且无历史价，按0计（首次）")
            mv += qty * px
        nav = self._d['cash'] + mv
        hist = self._d['nav_history']
        if not hist or hist[-1]['date'] < str(as_of):
            hist.append({'date': str(as_of), 'nav': round(nav, 2),
                         'cash': round(self._d['cash'], 2), 'market_value': round(mv, 2)})
        else:
            hist[-1] = {'date': str(as_of), 'nav': round(nav, 2),
                        'cash': round(self._d['cash'], 2), 'market_value': round(mv, 2)}
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
            if p is not None and p > 0:
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
        """原子写入：先写临时文件再 rename——进程崩溃不留半写状态的损坏账本"""
        import tempfile
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        fd, tmp = tempfile.mkstemp(
            dir=os.path.dirname(os.path.abspath(self.path)),
            suffix='.tmp', prefix='ledger_')
        try:
            with os.fdopen(fd, 'w', encoding='utf-8') as f:
                json.dump(self._d, f, ensure_ascii=False, indent=1)
            os.replace(tmp, self.path)     # 原子替换（Windows POSIX均支持）
        except Exception:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
