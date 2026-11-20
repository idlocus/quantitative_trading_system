#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
券商抽象层 —— 统一下单接口，Paper 模拟与 QMT 实盘同接口切换

QMT 实盘实现见 live/qmt_broker.py（唯一 QMTBroker，勿在此另建占位类）。
执行器统一经 live.make_broker(mode) 构造：paper → PaperBroker，qmt → QMTBroker。
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date
from typing import Dict, List, Optional

from trading import config as chain
from trading.rules import (is_limit_up, is_limit_down, round_buy_qty,
                           round_sell_qty, round_tick)
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class Order:
    symbol: str
    side: str            # 'BUY' / 'SELL'
    target_weight: float  # 目标权重（信息性）
    quantity: int = 0     # 股数（0 = 由 broker 按金额计算）
    amount: float = 0.0   # 金额（quantity=0 时使用）
    note: str = ''


@dataclass
class Fill:
    order: Order
    price: float
    quantity: int
    commission: float
    taxes: float
    trade_date: date
    status: str = 'filled'   # filled / partial(n/m) / cancelled / rejected
    trade_id: str = ''       # 券商成交回报ID（对账去重优先键；Paper为空）


class Broker(ABC):
    """券商接口抽象"""

    @abstractmethod
    def execute(self, orders: List[Order], prices: Dict[str, float],
                trade_date: date) -> List[Fill]:
        """按给定价格执行订单（实盘=真实委托；Paper=模拟撮合）"""

    @abstractmethod
    def name(self) -> str:
        ...


class PaperBroker(Broker):
    """
    模拟撮合（验证全链路用）

    成交假设：收盘价 ± 滑点；费用=佣金万3+过户费（买）/ +印花税（卖），
    与回测引擎同口径。涨跌停不可成交（rejected）。
    """

    def __init__(self, slippage: float = chain.SLIPPAGE,
                 commission: float = chain.COMMISSION,
                 stamp_tax: float = chain.STAMP_TAX,
                 transfer_fee: float = chain.TRANSFER_FEE,
                 enforce_price_limits: bool = True,
                 lot_map: Dict[str, int] = None):
        self.slippage = slippage
        self.commission = commission
        self.stamp_tax = stamp_tax
        self.transfer_fee = transfer_fee
        self.enforce_limits = enforce_price_limits
        self.lot_map = lot_map or {}   # {symbol: 最低申报数量}（BoardParamsProvider，缺省走板块规则）

    def _lot(self, symbol: str) -> int:
        from trading.rules import min_order_qty
        return self.lot_map.get(symbol) or min_order_qty(symbol)

    def name(self):
        return 'PaperBroker'

    def execute(self, orders, prices, trade_date, prev_prices: Optional[Dict[str, float]] = None,
                available_cash: Optional[float] = None,
                st_symbols: Optional[set] = None,
                positions: Optional[Dict[str, int]] = None):
        """
        st_symbols: 当日 ST 股集合（主板 ST 涨跌幅 ±5%，见 trading/rules）；
                    不传则按非 ST 处理（数据不可用时的降级行为）。
        positions:  当前持仓 {symbol: 数量}（卖出数量合法化需要：零股一次性规则）；
                    不传则卖出数量按订单原样（回测/闸门全仓卖出场景恒合法）。
        """
        from datetime import datetime as _dt
        if isinstance(trade_date, date) and not isinstance(trade_date, _dt):
            trade_date = _dt.combine(trade_date, _dt.now().time())  # 记录真实执行时刻
        fills = []
        running_cash = available_cash  # 运行时现金追踪（None=不检查）
        st_symbols = st_symbols or set()
        positions = positions or {}
        for o in orders:
            raw = prices.get(o.symbol)
            if raw is None or raw <= 0:
                fills.append(Fill(o, 0, 0, 0, 0, trade_date, status='rejected'))
                continue
            # 涨跌停拒单（主链共享规则：板块比例+ST差异化，回测同实现）
            if self.enforce_limits and prev_prices:
                prev = prev_prices.get(o.symbol)
                if prev and prev > 0:
                    is_st = o.symbol in st_symbols
                    if o.side == 'BUY' and is_limit_up(o.symbol, raw, prev, st=is_st):
                        fills.append(Fill(o, 0, 0, 0, 0, trade_date, status='rejected(涨停)'))
                        continue
                    if o.side == 'SELL' and is_limit_down(o.symbol, raw, prev, st=is_st):
                        fills.append(Fill(o, 0, 0, 0, 0, trade_date, status='rejected(跌停)'))
                        continue
            px = raw * (1 + self.slippage) if o.side == 'BUY' else raw * (1 - self.slippage)
            px = round_tick(o.symbol, px)   # 价格档位 0.01
            qty = o.quantity
            if qty <= 0 and o.amount > 0:
                fee_rate = self.commission + self.transfer_fee + \
                    (self.stamp_tax if o.side == 'SELL' else 0)
                qty = int(o.amount / (px * (1 + fee_rate)))
            if o.side == 'BUY':
                # 申报数量：主板/创业板100整手；科创板≥200且1股递增；北交所≥100且1股递增
                qty = round_buy_qty(o.symbol, qty, lot=self._lot(o.symbol))
            elif positions:
                # 卖出合法化（含零股一次性：全仓申报恒合法）
                qty = round_sell_qty(o.symbol, qty, positions.get(o.symbol, qty),
                                     lot=self._lot(o.symbol))
            if qty <= 0:
                fills.append(Fill(o, px, 0, 0, 0, trade_date, status='rejected(不足最低申报)'))
                continue

            # 资金约束：检查可用现金（买入时逐笔扣减；金额含佣金+过户费）
            value = px * qty
            if o.side == 'BUY' and running_cash is not None:
                buy_fee_rate = self.commission + self.transfer_fee
                total_cost = value * (1 + buy_fee_rate)
                if total_cost > running_cash:
                    # 资金不足：按申报规则缩减（含费用边界，防现金扣负）
                    affordable_qty = round_buy_qty(o.symbol,
                                                   int(running_cash / (px * (1 + buy_fee_rate))),
                                                   lot=self._lot(o.symbol))
                    if affordable_qty <= 0:
                        fills.append(Fill(o, px, 0, 0, 0, trade_date,
                                         status='rejected(资金不足)'))
                        continue
                    qty = affordable_qty
                    value = px * qty
            commission = value * self.commission
            taxes = value * self.transfer_fee + (value * self.stamp_tax if o.side == 'SELL' else 0)

            # 更新运行时现金
            if running_cash is not None:
                if o.side == 'BUY':
                    running_cash -= value + commission + taxes
                else:
                    running_cash += value - commission - taxes

            fills.append(Fill(o, px, qty, commission, taxes, trade_date))
        return fills
