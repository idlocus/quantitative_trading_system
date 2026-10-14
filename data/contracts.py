#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据契约层 —— 显式声明数据的单位、基准和覆盖范围

每个进入计算管道的数据结构都必须经由本模块构造，构造时即校验：
- unit: 单位（股/手、元/千元、百分比/倍数）
- basis: 价格基准（raw=不复权 / adjusted=前复权）
- coverage: 有效数据覆盖率

违反契约的构造直接抛错（而非静默通过）。

此前审计中的重大 bug 全部源于契约缺失：
- 4688%盈亏 = raw价格 vs adjusted价格无声明
- 100倍成交额差 = 手 vs 股无单位标注
- 100%杠杆排除 = 百分比 vs 倍数无scale声明
"""

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd
import numpy as np


class ContractViolation(Exception):
    """数据契约违反（单位/基准/覆盖率不符合声明）"""
    pass


# ================= 价格序列契约 =================

@dataclass
class PriceContract:
    """价格数据契约：基准与单位必须显式声明"""
    basis: str = 'raw'        # 'raw'=不复权（实盘成交/成本/止损/涨跌停）
                               # 'adjusted'=前复权（回测净值/收益/因子前瞻收益）
    unit: str = 'yuan'        # 人民币元（A股唯一合法单位）


def validate_price_series(df: pd.DataFrame, contract: PriceContract,
                          column: str = 'close') -> pd.DataFrame:
    """
    校验价格列是否符合契约。

    检查：
    - 列存在且非全空
    - 值 > 0（价格不可能为负或零）
    - 无极端跳变（单日|涨跌|>50%且非新上市 → 可能是基准混合）
    """
    if column not in df.columns:
        raise ContractViolation(f"缺少价格列 '{column}'（可用列: {list(df.columns)}）")

    px = pd.to_numeric(df[column], errors='coerce')
    if px.notna().sum() == 0:
        raise ContractViolation(f"价格列 '{column}' 全为空")

    if (px.dropna() <= 0).any():
        raise ContractViolation(f"价格列 '{column}' 含非正值（min={px.min()}）")

    # 极端跳变检测（同一DataFrame内raw和adjusted混合的典型症状）
    if len(px.dropna()) > 2:
        rets = px.pct_change().dropna().abs()
        extreme = rets[rets > 0.50]  # 单日|涨跌|>50%且非新股
        if len(extreme) > len(rets) * 0.01:  # >1%的交易日有极端跳变
            raise ContractViolation(
                f"价格列 '{column}' 有{len(extreme)}个>50%跳变日"
                f"（疑似raw/adjusted基准混合）")

    return df


# ================= 成交量契约 =================

VOLUME_UNITS = {'shares', 'lots'}       # 股 / 手
AMOUNT_UNITS = {'yuan', 'thousand_yuan'} # 元 / 千元


@dataclass
class VolumeContract:
    """成交量/成交额契约"""
    volume_unit: str = 'shares'         # 'shares'=股（DB原始=手，需×100）
    amount_unit: str = 'yuan'           # 'yuan'=元（DB原始=千元，需×1000）


def normalize_volume(df: pd.DataFrame, source: str = 'db') -> pd.DataFrame:
    """
    成交量归一化为股。

    Args:
        source: 'db'（Oracle原始=手，×100）
                'client'（Wind客户端=股，不转换）
                'unknown'（自动检测：值域判断）
    """
    if 'volume' not in df.columns:
        return df

    df = df.copy()
    if source == 'db':
        df['volume'] = pd.to_numeric(df['volume'], errors='coerce') * 100
    elif source == 'client':
        pass  # 已是股
    elif source == 'unknown':
        # 自动检测：A股个股日成交量>100万手的极少，中位数<5000大概率是手
        vol = pd.to_numeric(df['volume'], errors='coerce').dropna()
        if len(vol) > 0 and vol.median() < 5000:
            df['volume'] = vol * 100  # 疑似手单位
    return df


def normalize_amount(df: pd.DataFrame, source: str = 'db') -> pd.DataFrame:
    """
    成交额归一化为元。

    Args:
        source: 'db'（Oracle原始=千元，×1000）
                'client'（Wind客户端=元，不转换）
    """
    if 'amount' not in df.columns:
        return df

    df = df.copy()
    if source == 'db':
        df['amount'] = pd.to_numeric(df['amount'], errors='coerce') * 1000
    return df


# ================= 因子矩阵契约 =================

@dataclass
class FactorContract:
    """因子矩阵数据契约"""
    factor_name: str = ''
    unit: str = ''              # 'ratio'=比率(如ROE=0.08表示8%)
                                # 'pct'=百分比(如D/E=50.5表示50.5%)
                                # 'score'=标准化分数(无量纲)
    expected_range: tuple = None  # (min, max) 合理值域
    min_coverage: float = 0.30   # 最低覆盖率


def validate_factor_matrix(mx: pd.DataFrame, contract: FactorContract) -> dict:
    """
    校验因子矩阵并返回质量元数据。

    Returns:
        {'date': str, 'coverage': float, 'n_valid': int, 'n_total': int,
         'unit': str, 'passed': bool, 'issues': [str]}
    """
    meta = {
        'factor': contract.factor_name,
        'coverage': 0.0, 'n_valid': 0, 'n_total': mx.shape[1] if not mx.empty else 0,
        'unit': contract.unit, 'passed': False, 'issues': []
    }
    if mx.empty:
        meta['issues'].append('矩阵为空')
        return meta

    last_row = mx.iloc[-1]
    n_valid = int(last_row.notna().sum())
    n_total = int(mx.shape[1])
    meta['n_valid'] = n_valid
    meta['coverage'] = n_valid / max(n_total, 1)
    meta['date'] = str(mx.index[-1])

    if meta['coverage'] < contract.min_coverage:
        meta['issues'].append(
            f"覆盖率{meta['coverage']:.0%}<{contract.min_coverage:.0%}")

    if contract.expected_range:
        lo, hi = contract.expected_range
        vals = last_row.dropna()
        if len(vals) > 0:
            out_of_range = vals[(vals < lo) | (vals > hi)]
            if len(out_of_range) > len(vals) * 0.05:  # >5%超值域
                meta['issues'].append(
                    f"{len(out_of_range)}个值超预期值域[{lo},{hi}]"
                    f"（unit='{contract.unit}'可能不正确）")

    meta['passed'] = len(meta['issues']) == 0
    return meta


# ================= 已知的因子契约注册表 =================
# 每个因子的单位、值域、来源——新增因子必须在此登记

FACTOR_CONTRACTS = {
    'ep': FactorContract('ep', 'ratio', (-0.5, 1.0), 0.30),
    'bp': FactorContract('bp', 'ratio', (-0.5, 3.0), 0.30),
    'sp': FactorContract('sp', 'ratio', (0.0, 10.0), 0.30),
    'size': FactorContract('size', 'yuan', (1e7, 1e12), 0.30),  # 市值(元)
    'leverage': FactorContract('leverage', 'pct', (0, 500), 0.20),  # D/E百分比
    'roe': FactorContract('roe', 'ratio', (-0.5, 1.0), 0.20),   # ROE比率
    'momentum': FactorContract('momentum', 'ratio', (-1.0, 3.0), 0.30),  # N月收益
}


def get_factor_contract(name: str) -> FactorContract:
    """获取已注册的因子契约（未注册的因子拒绝使用）"""
    if name not in FACTOR_CONTRACTS:
        raise ContractViolation(
            f"因子 '{name}' 未注册契约。新增因子必须在 FACTOR_CONTRACTS 中登记"
            f"（声明 unit + expected_range + min_coverage）。"
            f"已注册: {list(FACTOR_CONTRACTS.keys())}")
    return FACTOR_CONTRACTS[name]
