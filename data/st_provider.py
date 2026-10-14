#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ST 状态提供器 —— Wind ASHAREST 表的 PIT 判定

ASHAREST：{S_INFO_WINDCODE, S_TYPE_ST, ENTRY_DT(进入日), REMOVE_DT(撤销日,空=仍在)}。
S_TYPE_ST 取值（Wind 编码）：S=特别处理 X=*ST Y=ST P=PT（主板±5%）；
L=退市整理 T=退市 Z=摘牌（非 ST 涨跌幅口径，不纳入）。

规则（trading/rules.price_limit_ratio 同口径）：
- 主板 ST（60/00 开头）→ 涨跌幅 ±5%
- 创业板/科创板 ST → 仍 ±20%（注册制后无 ST 差异化）
- 北交所 → ±30%（无 ST 差异化）

DB 不可用时抛 RuntimeError，调用方自行降级（生产 signals 降级为警告并跳过过滤；
回测 runner 降级为无 ST 矩阵）。
"""

from datetime import date
from typing import Iterable, List, Optional, Set, Tuple

import pandas as pd

from utils.logger import get_logger

logger = get_logger(__name__)

# 视为"ST类"（主板适用±5%）的状态码。R是伴随状态，不独立代表涨跌幅规则。
ST_TYPES = {'S', 'X', 'Y', 'P'}


class STProvider:
    """基于 ASHAREST 区间的 PIT ST 判定（全表仅约3千行，一次加载内存判定）"""

    def __init__(self, ds=None):
        self._rows = self._load(ds) if ds is not None else []
        # [(sym, entry: date|None, remove: date|None)]，仅 ST_TYPES
        self._spans: List[Tuple[str, Optional[date], Optional[date]]] = [
            (sym, _to_date(e), _to_date(r)) for sym, _t, e, r in self._rows]

    @classmethod
    def from_rows(cls, rows: List[Tuple[str, str, str, str]]):
        """直接注入 (sym, s_type_st, entry_dt, remove_dt)，供测试/离线"""
        p = cls.__new__(cls)
        p._rows = list(rows)
        p._spans = [(sym, _to_date(e), _to_date(r))
                    for sym, t, e, r in p._rows if t in ST_TYPES]
        return p

    # 历史基线：ASHAREST 全表约 2765 行（963 只股票的 ST 记录）。
    # 查询成功但行数异常低（权限/同步/字段过滤故障）时必须进入安全模式（抛错→
    # 调用方 st_ok=False → 闸门禁买），而非当作"市场无ST"放行。
    MIN_SANITY_ROWS = 50

    def _load(self, ds) -> List[Tuple[str, str, str, str]]:
        db = ds._db_src()
        if not db:
            raise RuntimeError("DB 不可用，无法加载 ASHAREST")
        cur = db._cursor()
        try:
            cur.execute(
                "SELECT S_INFO_WINDCODE, S_TYPE_ST, ENTRY_DT, REMOVE_DT FROM ASHAREST")
            rows = [(r[0], (r[1] or '').strip(), r[2], r[3]) for r in cur.fetchall()]
            if len(rows) < self.MIN_SANITY_ROWS:
                raise RuntimeError(
                    f"ASHAREST 仅返回 {len(rows)} 行（历史基线≈2765，下限"
                    f"{self.MIN_SANITY_ROWS}）——疑似权限/同步/过滤异常，"
                    "按数据故障进入安全模式（st_ok=False → 禁买）")
            active = sum(1 for _, t, _, rm in rows
                         if t in ST_TYPES and (rm is None or str(rm).strip() == ''))
            if active < 50:
                raise RuntimeError(
                    f"ASHAREST 活跃ST记录仅{active}条（正常应>300）——疑似同步停滞")
            logger.info(f"[STProvider] ASHAREST 加载 {len(rows)} 行（活跃ST {active}）")
            return rows
        finally:
            cur.close()

    # ---------------- 判定 ----------------

    def is_st(self, symbol: str, as_of) -> bool:
        d = _to_date(as_of)
        if d is None:
            return False
        for sym, entry, remove in self._spans:
            if sym != symbol:
                continue
            if entry is not None and entry <= d and (remove is None or remove >= d):
                return True
        return False

    def st_set(self, as_of) -> Set[str]:
        """as_of 日处于 ST 状态的全部股票"""
        d = _to_date(as_of)
        if d is None:
            return set()
        out = set()
        for sym, entry, remove in self._spans:
            if entry is not None and entry <= d and (remove is None or remove >= d):
                out.add(sym)
        return out

    def st_matrix(self, index: pd.DatetimeIndex, columns: Iterable[str]) -> pd.DataFrame:
        """dates×symbols 的 ST 布尔矩阵（供回测 PIT 判定）"""
        cols = list(columns)
        st_cols = {s for s in cols if any(sym == s for sym, _, _ in self._spans)}
        df = pd.DataFrame(False, index=index, columns=cols)
        dates = index.date if hasattr(index, 'date') else [d.date() for d in index]
        for sym in st_cols:
            spans = [(e, r) for s, e, r in self._spans if s == sym]
            df[sym] = [
                any(e is not None and e <= d and (r is None or r >= d)
                    for e, r in spans)
                for d in dates]
        return df


def _to_date(x):
    if x is None or x == '':
        return None
    if isinstance(x, date):
        return x
    s = str(x)[:10].replace('-', '').replace('/', '')
    if len(s) != 8 or not s.isdigit():
        return None
    return date(int(s[:4]), int(s[4:6]), int(s[6:8]))
