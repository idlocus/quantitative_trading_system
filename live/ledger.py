#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实盘账本 —— 持仓/现金/成交/净值 的持久化与对账（JSON 单文件，简单可审计）
"""

import json
import os
from datetime import date, datetime
from typing import Dict, List, Optional

from .broker import Fill


class Ledger:
    """
    持仓账本

    文件结构:
    { cash, positions: {sym: qty}, cost: {sym: avg_cost},
      trades: [...], nav_history: [{date, nav, cash, market_value}], meta }
    """

    def __init__(self, path: str):
        self.path = path
        if os.path.isfile(path):
            with open(path, encoding='utf-8') as f:
                self._d = json.load(f)
        else:
            self._d = {'cash': 0.0, 'positions': {}, 'cost': {},
                       'trades': [], 'nav_history': [], 'meta': {'created': str(date.today())}}

    # ---- 基础 ----

    @property
    def cash(self) -> float:
        return self._d['cash']

    @property
    def positions(self) -> Dict[str, int]:
        return self._d['positions']

    def initialize(self, capital: float):
        """初始入金（仅一次）"""
        if self._d['cash'] != 0 or self._d['trades'] or self._d['positions']:
            raise RuntimeError('账本已有资金或交易，不能重新入金')
        self._d['cash'] = float(capital)
        self.save()

    # ---- 成交入账 ----

    def apply_fills(self, fills: List[Fill]):
        for f in fills:
            if f.status != 'filled':
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
            self._d['trades'].append({
                'date': str(f.trade_date), 'symbol': sym, 'side': f.order.side,
                'price': f.price, 'quantity': f.quantity,
                'commission': f.commission, 'taxes': f.taxes,
                'target_weight': f.order.target_weight,
            })
        self.save()

    # ---- 净值 ----

    def mark_to_market(self, prices: Dict[str, float], as_of: date) -> float:
        mv = sum(qty * prices.get(s, 0.0) for s, qty in self._d['positions'].items())
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
        mv = {s: qty * prices.get(s, 0.0) for s, qty in self._d['positions'].items()}
        total = self._d['cash'] + sum(mv.values())
        if total <= 0:
            return {}
        return {s: v / total for s, v in mv.items() if v > 0}

    def save(self):
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        with open(self.path, 'w', encoding='utf-8') as f:
            json.dump(self._d, f, ensure_ascii=False, indent=1)
