#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实盘模块 V1：信号 → 订单 → 账本 → 净值（券商通道就位前用 Paper 模式验证全链路）

- broker.py: 券商抽象（Broker ABC + PaperBroker 模拟撮合 + QMTBroker 预留）
- ledger.py: 持仓账本（JSON 持久化、成交记录、净值序列）
- signals.py: 生产合成信号（5因子，复用研究管道，PIT 到最新）
合规：正式接入券商后需完成程序化交易报备（券商通道办理）。
"""

from .broker import Order, Broker, PaperBroker
from .qmt_broker import QMTBroker as QMTLiveBroker
from .ledger import Ledger

__all__ = ['Order', 'Broker', 'PaperBroker', 'QMTBroker', 'Ledger']
