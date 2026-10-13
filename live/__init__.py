#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实盘模块：信号 → 风控闸门 → 订单 → 执行器 → 账本 → 净值（Paper/QMT 同路径）

- broker.py:     券商抽象（Broker ABC + PaperBroker 模拟撮合）
- qmt_broker.py: QMT 实盘通道（唯一 QMTBroker 实现）
- ledger.py:     持仓账本（JSON 持久化、成交去重、公司行动、资金流水）
- signals.py:    生产信号（5因子，T-1收盘→T日执行，generate_production_signal 唯一入口）
- risk_gate.py:  执行风控闸门（L2数据停机/L3组合熔断/L4强制卖出，指令改写层）
- reconcile.py:  账实核对（账本 vs 券商对账单）

执行器经 make_broker(mode) 构造——paper 即用，qmt 先做环境检查，
不满足时抛出带修复指引的 RuntimeError（绝不静默跳过）。
合规：正式接入券商后需完成程序化交易报备（券商通道办理）。
"""

from .broker import Order, Broker, PaperBroker, Fill
from .qmt_broker import QMTBroker
from .ledger import Ledger
from .risk_gate import RiskGate, GateConfig, RiskDecision
from .reconcile import Reconciler, BrokerStatement, ReconReport


def make_broker(mode: str, lot_map: dict = None, enforce_window: bool = True):
    """
    执行器工厂（run_daily / run_live 统一入口）。

    Args:
        mode: 'paper' 模拟撮合 | 'qmt' 实盘委托
        lot_map: {symbol: 最低申报数量}（BoardParamsProvider.lot_map；
                 缺省走板块规则：科创200/其余100）
        enforce_window: QMT 非收盘竞价窗口拒单（默认True；CLI --outside-window 关闭）

    Raises:
        ValueError: 未知模式
        RuntimeError: qmt 环境未就绪（消息含逐项修复指引，不静默跳过）
    """
    if mode == 'paper':
        return PaperBroker(lot_map=lot_map)
    if mode == 'qmt':
        env = QMTBroker.check_env()
        if not env['ready']:
            raise RuntimeError(
                "QMT 环境未就绪，本次不执行任何指令。逐项修复：\n  - "
                + "\n  - ".join(env['issues']))
        b = QMTBroker(enforce_window=enforce_window)
        b.lot_map = lot_map or {}
        return b
    raise ValueError(f"未知执行模式: {mode}（可选 paper / qmt）")


def load_lot_map(ds) -> dict:
    """每手数据（ASHAREWINDCUSTOMCODE.S_INFO_LOT_SIZE）；不可用时降级空表=板块规则"""
    try:
        from data.board_params import BoardParamsProvider
        return BoardParamsProvider(ds).lot_map
    except Exception as e:
        from utils.logger import get_logger
        get_logger(__name__).warning(f"[BoardParams] 每手数据不可用，按板块规则处理: {e}")
        return {}


__all__ = ['Order', 'Fill', 'Broker', 'PaperBroker', 'QMTBroker', 'Ledger',
           'RiskGate', 'GateConfig', 'RiskDecision',
           'Reconciler', 'BrokerStatement', 'ReconReport',
           'make_broker', 'load_lot_map']
