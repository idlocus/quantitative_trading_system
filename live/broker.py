#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
券商抽象层 —— 统一下单接口，Paper 模拟与 QMT 实盘同接口切换

实盘切换：QMTBroker 需券商开通量化权限后安装 miniQMT(xtquant)，
在 _submit_qmt 中实现真实委托（代码已留位，无需改动上层）。
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date
from typing import Dict, List, Optional

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
    status: str = 'filled'   # filled / rejected


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

    def __init__(self, slippage: float = 0.001, commission: float = 0.0003,
                 stamp_tax: float = 0.0005, transfer_fee: float = 0.00001,
                 enforce_price_limits: bool = True):
        self.slippage = slippage
        self.commission = commission
        self.stamp_tax = stamp_tax
        self.transfer_fee = transfer_fee
        self.enforce_limits = enforce_price_limits

    def name(self):
        return 'PaperBroker'

    def execute(self, orders, prices, trade_date, prev_prices: Optional[Dict[str, float]] = None):
        from datetime import datetime as _dt
        if isinstance(trade_date, date) and not isinstance(trade_date, _dt):
            trade_date = _dt.combine(trade_date, _dt.now().time())  # 记录真实执行时刻
        fills = []
        for o in orders:
            raw = prices.get(o.symbol)
            if raw is None or raw <= 0:
                fills.append(Fill(o, 0, 0, 0, 0, trade_date, status='rejected'))
                continue
            # 涨跌停拒单（近似：|涨跌幅|≥9.8%且方向不利）
            if self.enforce_limits and prev_prices:
                prev = prev_prices.get(o.symbol)
                if prev and prev > 0:
                    chg = raw / prev - 1
                    if o.side == 'BUY' and chg > 0.098:
                        fills.append(Fill(o, 0, 0, 0, 0, trade_date, status='rejected(涨停)'))
                        continue
                    if o.side == 'SELL' and chg < -0.098:
                        fills.append(Fill(o, 0, 0, 0, 0, trade_date, status='rejected(跌停)'))
                        continue
            px = raw * (1 + self.slippage) if o.side == 'BUY' else raw * (1 - self.slippage)
            qty = o.quantity
            if qty <= 0 and o.amount > 0:
                fee_rate = self.commission + self.transfer_fee + \
                    (self.stamp_tax if o.side == 'SELL' else 0)
                qty = int(o.amount / (px * (1 + fee_rate)))
            if qty <= 0:
                fills.append(Fill(o, px, 0, 0, 0, trade_date, status='rejected(零股)'))
                continue
            value = px * qty
            commission = value * self.commission
            taxes = value * self.transfer_fee + (value * self.stamp_tax if o.side == 'SELL' else 0)
            fills.append(Fill(o, px, qty, commission, taxes, trade_date))
        return fills


class QMTBroker(Broker):
    """
    miniQMT 实盘通道（预留）

    前置条件（用户侧）：
    1. 券商开通量化交易权限（资金门槛以券商为准）
    2. 安装 miniQMT 客户端并登录，Python 环境可 import xtquant
    3. 完成程序化交易报备（券商合规流程）
    """

    def __init__(self, account_id: str = ''):
        self.account_id = account_id
        self._xt = None

    def name(self):
        return 'QMTBroker'

    def _connect(self):
        if self._xt is None:
            from xtquant.xttrader import XtQuantTrader  # noqa: 券商环境就位后启用
            raise NotImplementedError(
                "QMT 实盘通道待券商环境就位后启用：安装 miniQMT 并开通权限，"
                "然后在 _submit_qmt 中实现委托调用")

    def execute(self, orders, prices, trade_date, **kw):
        self._connect()
        raise NotImplementedError
