#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
申报参数提供器 —— ASHAREWINDCUSTOMCODE 的每手/最低申报数量

实测（2026-10-12）：主板/创业板/北交所 S_INFO_LOT_SIZE=100（5499只），
科创板=200（622只）；S_INFO_MIN_PRICE_CHG_UNIT 统一 0.01。
板块规则兜底（trading/rules.min_order_qty），表数据作为覆盖/校验——
表缺失（退市/未更新行）或 DB 不可用时自动退回板块规则。
"""

from typing import Dict

from utils.logger import get_logger

logger = get_logger(__name__)


class BoardParamsProvider:
    """每手/最低申报数量（当前快照，非 PIT——申报单位极少变化）"""

    def __init__(self, ds=None):
        self.lot_map: Dict[str, int] = {}
        if ds is not None:
            self._load(ds)

    @classmethod
    def from_rows(cls, rows):
        """直接注入 [(symbol, lot_size)]，供测试/离线"""
        p = cls.__new__(cls)
        p.lot_map = {sym: int(lot) for sym, lot in rows if lot and lot > 0}
        return p

    def _load(self, ds):
        db = ds._db_src()
        if not db:
            raise RuntimeError("DB 不可用，无法加载 ASHAREWINDCUSTOMCODE")
        cur = db._cursor()
        try:
            cur.execute(
                "SELECT S_INFO_WINDCODE, S_INFO_LOT_SIZE FROM ASHAREWINDCUSTOMCODE "
                "WHERE S_INFO_LOT_SIZE IS NOT NULL")
            self.lot_map = {r[0]: int(r[1]) for r in cur.fetchall() if r[1]}
            logger.info(f"[BoardParams] 每手数据加载 {len(self.lot_map)} 只")
        finally:
            cur.close()

    def lot(self, symbol: str) -> int:
        """最低申报数量：表优先，缺失退回板块规则"""
        from trading.rules import min_order_qty
        return self.lot_map.get(symbol) or min_order_qty(symbol)
