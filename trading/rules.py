#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
交易规则 —— 板块涨跌停/申报数量/整手/费用（回测与实盘共用的唯一实现）

涨跌停比例（含 ST 差异化，A股规则）：
- 主板（60/00）非ST ±10%；主板 ST/*ST/PT ±5%（需 ST 状态输入，见 data/st_provider）
- 创业板（300/301）/科创板（688）±20%（注册制后 ST 无差异化，仍 20%）
- 北交所（4/8/92）±30%（无 ST 差异化）
- 港股不限制

申报数量规则（交易所规则；最低申报量可被 ASHAREWINDCUSTOMCODE.S_INFO_LOT_SIZE 覆盖，
实测：主板/创业板/北交所=100，科创板=200）：
- 主板/创业板：买入 100 股整数倍
- 科创板：≥200 股，1 股递增（可买 237 股）
- 北交所：≥100 股，1 股递增
- 卖出：余额不足最低申报数量时必须一次性全仓申报（零股规则）

价格档位：A股最小价差 0.01（round_tick）。

ST 状态由调用方传入（`st: bool` / `st_symbols: set`）——数据来自 ASHAREST（PIT）。
无 ST 数据时按非 ST 处理（与旧行为一致），生产端由 STProvider 降级警告兜底。
"""

from typing import Optional

import pandas as pd

from .config import (COMMISSION, STAMP_TAX, TRANSFER_FEE, LIMIT_TOLERANCE)


def price_limit_ratio(symbol: str, st: bool = False) -> Optional[float]:
    """按板块（+ST状态）的涨跌停幅度；无法判定返回 None（不限制）"""
    if not isinstance(symbol, str) or '.' not in symbol:
        return None
    code, suffix = symbol.split('.')
    if suffix.upper() == 'HK':
        return None
    if code.startswith(('688', '300', '301')):      # 科创板/创业板（含ST仍20%）
        return 0.20
    if code.startswith(('4', '8', '92')):           # 北交所
        return 0.30
    if st and code.startswith(('60', '00')):        # 主板 ST/*ST/PT
        return 0.05
    return 0.10                                       # 主板


def is_limit_up(symbol: str, cur: float, prev: float,
                tol: float = LIMIT_TOLERANCE, st: bool = False) -> bool:
    """当日封涨停（不可买）。cur/prev=当日与前收盘价。"""
    limit = price_limit_ratio(symbol, st=st)
    if limit is None or not prev or prev <= 0 or cur is None:
        return False
    return cur >= prev * (1 + limit) - prev * tol


def is_limit_down(symbol: str, cur: float, prev: float,
                  tol: float = LIMIT_TOLERANCE, st: bool = False) -> bool:
    """当日封跌停（不可卖）。"""
    limit = price_limit_ratio(symbol, st=st)
    if limit is None or not prev or prev <= 0 or cur is None:
        return False
    return cur <= prev * (1 - limit) + prev * tol


def limit_ratio_series(symbols, st_symbols=None) -> pd.Series:
    """向量化版本（回测矩阵用）：{symbol: ratio or NaN}；st_symbols 内的主板按5%"""
    st_symbols = st_symbols or set()
    return pd.Series({s: price_limit_ratio(s, st=(s in st_symbols)) for s in symbols})


# ---- 申报数量（按板块；最低申报量可被券商/数据表覆盖） ----

def min_order_qty(symbol: str, lot: int = None) -> int:
    """最低申报数量：科创板 200（LOT_SIZE 表实测），其余 A股 100；lot 显式覆盖优先"""
    if lot and lot > 0:
        return int(lot)
    if not isinstance(symbol, str) or '.' not in symbol:
        return 100
    code, suffix = symbol.split('.')
    if suffix.upper() == 'HK':
        return 1
    if code.startswith('688'):
        return 200
    return 100


def qty_step(symbol: str) -> int:
    """申报递增单位：科创板/北交所 1 股递增（100/200 起），主板/创业板 100 整手"""
    if not isinstance(symbol, str) or '.' not in symbol:
        return 100
    code, suffix = symbol.split('.')
    if suffix.upper() == 'HK':
        return 1
    if code.startswith(('688', '4', '8', '92')):
        return 1
    return 100


def round_buy_qty(symbol: str, raw_qty: int, lot: int = None) -> int:
    """买入数量向下合法化：不足最低申报→0；step=100→整手；step=1→取整即可"""
    raw_qty = int(raw_qty)
    if raw_qty <= 0:
        return 0
    mn, step = min_order_qty(symbol, lot), qty_step(symbol)
    if raw_qty < mn:
        return 0
    return (raw_qty // step) * step if step > 1 else raw_qty


def sell_qty_valid(symbol: str, qty: int, holding: int, lot: int = None) -> bool:
    """卖出数量合法性：全仓申报恒合法（含零股一次性规则）；否则需≥最低申报且符合步长"""
    qty, holding = int(qty), int(holding)
    if qty <= 0 or qty > holding:
        return False
    if qty == holding:
        return True                      # 余额不足最低申报时一次性卖出
    mn, step = min_order_qty(symbol, lot), qty_step(symbol)
    return qty >= mn and qty % step == 0


def round_sell_qty(symbol: str, qty: int, holding: int, lot: int = None) -> int:
    """卖出数量向下合法化（不超持仓；不足最低申报的非全仓→0 放弃本次）"""
    qty = min(int(qty), int(holding))
    if qty <= 0:
        return 0
    if sell_qty_valid(symbol, qty, holding, lot):
        return qty
    step = qty_step(symbol)
    fixed = (qty // step) * step if step > 1 else qty
    if fixed > 0 and sell_qty_valid(symbol, fixed, holding, lot):
        return fixed
    return 0


# ---- 价格档位 ----

def round_tick(symbol: str, price: float) -> float:
    """A股最小价差 0.01（AShareWINDCUSTOMCODE.S_INFO_MIN_PRICE_CHG_UNIT 实测统一）"""
    if price is None or price <= 0:
        return price
    return round(round(price / 0.01) * 0.01, 2)


# ---- 费用 ----

def buy_fee_rate() -> float:
    """买入费用率（佣金+过户费）"""
    return COMMISSION + TRANSFER_FEE


def sell_fee_rate() -> float:
    """卖出费用率（佣金+过户费+印花税）"""
    return COMMISSION + TRANSFER_FEE + STAMP_TAX


def estimate_fees(side: str, value: float):
    """预估费用（佣金, 税费）——费用拆分公式的唯一实现。

    写路径口径收敛（评审 §2.2）：PaperBroker / QMTBroker / 日报曾各自内联
    同一拆分，改任一处其余不动=账实分叉。实际费用以券商回报为准，
    账实核对修正差异。"""
    commission = value * COMMISSION
    taxes = value * TRANSFER_FEE + (value * STAMP_TAX if side == 'SELL' else 0)
    return commission, taxes
