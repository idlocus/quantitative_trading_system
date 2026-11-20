#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
个股推荐逻辑生成器 —— 把 5 因子信号翻译成人能读懂的推荐理由

输出结构：结论 / 核心逻辑（估值·质量·规模，含全市场分位）/ 组合定位 / 风险提示。
所有数字来自当日 PIT 因子矩阵（本地缓存，零配额）。
"""

import os
from datetime import date, timedelta

import numpy as np
import pandas as pd

from utils.logger import get_logger

logger = get_logger(__name__)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_RATIONALE_CACHE = {}  # (symbol, day) -> rationale dict


def _load_matrices(ds, day: date):
    """当日生产因子原始矩阵（按日缓存，全市场零配额）"""
    key = str(day)
    if key in _MATS_CACHE:
        return _MATS_CACHE[key]
    with _LOAD_LOCK:  # 并发请求只允许一次加载，其余等待复用
        if key in _MATS_CACHE:
            return _MATS_CACHE[key]
        _MATS_CACHE[key] = _load_matrices_uncached(ds, day)
        return _MATS_CACHE[key]


def _load_matrices_uncached(ds, day):
    prov, pit = _providers(ds)
    start = day - timedelta(days=30)
    pool = _pool(ds, day)
    m = {}
    for f in ['pe_ttm', 'pb_lf', 'ps_ttm', 'mkt_cap_ard']:
        fm = prov.get_field_matrix(f, pool, start, day)
        if not fm.empty:
            m[f] = fm
    lev = pit.get_factor_matrix('leverage', pool, start, day)
    if not lev.empty:
        m['leverage'] = lev
    roe = pit.get_factor_matrix('roe', pool, start, day)
    if not roe.empty:
        m['roe'] = roe
    return m


_POOL_CACHE = {}
_PROVIDERS = {}
_MATS_CACHE = {}
_LOAD_LOCK = __import__('threading').Lock()


def _providers(ds):
    if 'p' not in _PROVIDERS:
        from data.fundamentals import WindFundamentalProvider
        from data.fundamental_pit import PITFinancialProvider
        _PROVIDERS['p'] = (WindFundamentalProvider(ds), PITFinancialProvider(ds))
    return _PROVIDERS['p']


def _pool(ds, day):
    key = str(day)
    if key not in _POOL_CACHE:
        try:
            _POOL_CACHE[key] = sorted(ds.list_symbols(as_of=day))
        except Exception:
            _POOL_CACHE[key] = []
    return _POOL_CACHE[key]


def _at(mx, day, symbol):
    """矩阵在 ≤day 的最后有效值与全市场百分位（高=值大）"""
    if mx is None or mx.empty or symbol not in mx.columns:
        return None, None, 0
    sub = mx.loc[:pd.Timestamp(day), symbol].dropna()
    if sub.empty:
        return None, None, 0
    v = float(sub.iloc[-1])
    row = mx.loc[:pd.Timestamp(day)].iloc[-1].dropna()
    pct = float((row < v).mean()) if len(row) > 50 else None  # 分位：值超过市场中多少股票
    return v, pct, len(row)


def build_rationale(ds, symbol: str, day: date) -> dict:
    """生成推荐逻辑（按 symbol+day 缓存）"""
    key = (symbol, str(day))
    if key in _RATIONALE_CACHE:
        return _RATIONALE_CACHE[key]

    m = _load_matrices(ds, day)
    pe, pe_pct, n1 = _at(m.get('pe_ttm'), day, symbol)
    pb, pb_pct, n2 = _at(m.get('pb_lf'), day, symbol)
    mv, mv_pct, n3 = _at(m.get('mkt_cap_ard'), day, symbol)
    lev, lev_pct, n4 = _at(m.get('leverage'), day, symbol)
    roe, roe_pct, n5 = _at(m.get('roe'), day, symbol)

    def desc(pct, better='low'):
        """分位转人话：better='low' 表示值越低越好"""
        if pct is None:
            return None
        q = pct if better == 'high' else 1 - pct  # 换算成"好过多少股票"
        return q

    points = []

    # 1) 估值
    if pe is not None and pe > 0:
        q = desc(pe_pct, 'low')
        s = f"**估值便宜**：市盈率(PE-TTM) {pe:.1f} 倍"
        s += f"，便宜过全市场 {min(q, 0.99):.0%} 的股票" if q is not None else ""
        if pb is not None and pb > 0:
            s += f"；市净率(PB) {pb:.1f} 倍"
        points.append(s)

    # 2) 质量
    if lev is not None:
        q = desc(lev_pct, 'low')
        q_txt = f"，杠杆低于全市场 {min(q, 0.99):.0%} 的公司" if q is not None else ""
        if roe is not None and roe >= 0:
            points.append(f"**财务稳健**：资产负债率 {lev:.1f}%{q_txt}；ROE {roe:.1f}%")
        else:
            points.append(f"**低杠杆**：资产负债率 {lev:.1f}%{q_txt}"
                          + ("；注意当前盈利偏弱（ROE %.1f%%），入选主因是估值与规模而非盈利质量" % roe
                             if roe is not None else ""))

    # 3) 规模
    if mv is not None and mv > 0:
        q = desc(mv_pct, 'low')
        s = f"**小市值属性**：总市值 {mv / 10000:.0f} 亿元"
        s += f"，规模小于全市场 {min(q, 0.99):.0%} 的股票（A股小市值长期存在溢价，也是本组合的主要收益与风险来源）" if q is not None else ""
        points.append(s)

    # 4) 综合评分与排名
    total = max(n1, n2, n3, n4, 0)
    rank = None
    score = None
    try:
        # 复合分 = 各 directed rank 的均值（与生产一致）
        ranks = []
        if pe is not None and pe_pct is not None: ranks.append(1 - pe_pct)     # ep 越高越好
        if pb is not None and pb_pct is not None: ranks.append(1 - pb_pct)
        if mv_pct is not None: ranks.append(1 - mv_pct)                        # 小市值加分
        if lev_pct is not None: ranks.append(1 - lev_pct)                      # 低杠杆加分
        if ranks:
            score = float(np.mean(ranks))
    except Exception:
        pass

    rationale = {
        'symbol': symbol, 'day': str(day), 'points': points,
        'score': round(score, 3) if score is not None else None,
        'market_total': total,
        'risks': [
            "本组合本质是「价值+小盘」风格暴露：小盘风格逆转（如2017-2018、2024年初）时该股可能随组合大幅回撤",
            "因子基于季频财务与日频估值，财报公告或估值剧烈变化时会触发调仓换手",
            "等权3.3%仓位，单票风险被组合分散，但无法规避系统性风险",
        ],
    }
    _RATIONALE_CACHE[key] = rationale
    return rationale
