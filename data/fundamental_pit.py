#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
报告期财务因子（公告日 PIT）—— ROE / 利润增速 / 营收增速 / 毛利率

PIT 规则（防前视）：
- 每条记录在 ANN_DT（公告日）后才可见
- 同一报告期多次公告（更正）→ 取 ANN_DT ≤ t 的最新一次
- 同日多报告期 → 取报告期最新
- 输出为日频矩阵（dates × symbols），报告期之间前向填充

已知边界（诚实声明）：若更正公告晚于新报告期公告（罕见），按公告序覆盖近似处理。
"""

from datetime import date
from typing import Dict, List

import numpy as np
import pandas as pd

from utils.logger import get_logger

logger = get_logger(__name__)

# 财务因子注册表：名字 → (DB字段, 方向, 说明)
PIT_FINANCIAL_FACTORS = {
    'roe':       ('S_FA_ROE', +1.0, '净资产收益率ROE（累计，报告期%）'),
    'np_growth': ('S_FA_YOYNETPROFIT', +1.0, '归母净利润同比%'),
    'rev_growth': ('S_FA_YOY_OR', +1.0, '营业收入同比%'),
    'gmargin':   ('S_FA_GROSSPROFITMARGIN', +1.0, '销售毛利率%'),
    # ---- 深度质量因子（2026-10 扩展：跨年代残差alpha筛选）----
    'salecash':  ('S_FA_SALESCASHINTOOR', +1.0, '销售收现比（盈利含金量）'),
    'asset_g':   ('S_FA_YOYASSETS', -1.0, '总资产同比%（asset growth 异象：低增速好）'),
    'ocf_to_rev': ('S_FA_OCFTOOR', +1.0, '经营现金流/营业收入'),
    'leverage':  ('S_FA_DEBTTOASSETS', -1.0, '资产负债率（低杠杆质量）'),
}


def fetch_announcements(db_cursor, field: str, symbols: List[str],
                        start: date, end: date) -> pd.DataFrame:
    """
    拉取财务指标公告记录（长表：symbol, ann_dt, report_period, value）

    取 start 前 2 年起的所有公告，保证 start 日已有"最近一期"可填。
    """
    lookback = date(start.year - 2, 1, 1)
    frames = []
    for i in range(0, len(symbols), 500):
        chunk = symbols[i:i + 500]
        ph = ",".join(f":s{j}" for j in range(len(chunk)))
        binds = {f"s{j}": s for j, s in enumerate(chunk)}
        sql = f"""
            SELECT S_INFO_WINDCODE, ANN_DT, REPORT_PERIOD, CAST({field} AS FLOAT)
            FROM ASHAREFINANCIALINDICATOR
            WHERE S_INFO_WINDCODE IN ({ph})
              AND ANN_DT IS NOT NULL AND REPORT_PERIOD IS NOT NULL
              AND {field} IS NOT NULL
              AND ANN_DT >= :dt1 AND ANN_DT <= :dt2
        """
        binds.update({'dt1': lookback.strftime('%Y%m%d'), 'dt2': end.strftime('%Y%m%d')})
        db_cursor.execute(sql, binds)
        frames.append(pd.DataFrame(
            db_cursor.fetchall(), columns=['symbol', 'ann_dt', 'report_period', 'value']))
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    return df.sort_values(['symbol', 'ann_dt', 'report_period'])


def build_pit_matrix(ann: pd.DataFrame, index: pd.DatetimeIndex,
                     columns: List[str]) -> pd.DataFrame:
    """
    公告长表 → 日频 PIT 矩阵（dates × symbols）

    规则：按公告序逐步覆盖（后公告覆盖先公告）；同日取最新报告期；
    未公告期为 NaN，公告后前向填充。
    """
    if ann.empty:
        return pd.DataFrame(np.nan, index=index, columns=columns)

    # 同 (symbol, ann_dt) 取最新报告期；同 (symbol, ann_dt, report_period) 取最后一条
    ann = ann.drop_duplicates(subset=['symbol', 'ann_dt', 'report_period'], keep='last')
    ann = ann.sort_values(['symbol', 'ann_dt', 'report_period'])
    ann = ann.drop_duplicates(subset=['symbol', 'ann_dt'], keep='last')

    ann['ts'] = pd.to_datetime(ann['ann_dt'], format='%Y%m%d')
    # 统一时间分辨率（pandas 3 中 s/ns 混用会使 reindex 失败）
    index = pd.DatetimeIndex(index).as_unit('ns')
    ann['ts'] = pd.DatetimeIndex(ann['ts']).as_unit('ns')

    out = pd.DataFrame(np.nan, index=index, columns=columns)
    start_ts = index[0]
    for sym, g in ann.groupby('symbol'):
        if sym not in columns:
            continue
        g = g[g['ts'] >= start_ts - pd.Timedelta(days=800)]  # 保留2年回看
        g = g.sort_values('ts')
        # 公告日序列 → 步进函数（reindex 到日历，前向填充，公告当日生效）
        s = g.set_index('ts')['value']
        s = s[~s.index.duplicated(keep='last')]
        aligned = s.reindex(index, method='ffill')
        # 公告日早于区间起点且之后无新公告 → reindex ffill 已覆盖；
        # 区间起点前的最后值也需带入：
        prior = s[s.index < start_ts]
        if len(prior) > 0 and pd.isna(aligned.iloc[0]):
            aligned.iloc[0] = prior.iloc[-1]
            aligned = aligned.ffill()
        out[sym] = aligned
    return out


class PITFinancialProvider:
    """报告期财务因子（公告日PIT）矩阵提供者（一次性拉取，内存复用）"""

    def __init__(self, layered_source):
        self._layered = layered_source
        self._cache: Dict[str, pd.DataFrame] = {}
        self._ann_cache: Dict[str, pd.DataFrame] = {}

    def get_factor_matrix(self, factor_name: str, symbols: List[str],
                          start: date, end: date) -> pd.DataFrame:
        """日频 PIT 因子矩阵（dates × symbols）"""
        if factor_name not in PIT_FINANCIAL_FACTORS:
            raise ValueError(f"未知财务因子 '{factor_name}'. 可用: {list(PIT_FINANCIAL_FACTORS)}")
        field, _, _ = PIT_FINANCIAL_FACTORS[factor_name]

        key = f"{field}_{start}_{end}_{len(symbols)}"
        if key not in self._ann_cache:
            db = self._layered._db_src()
            if not db:
                logger.warning("[PIT财务] DB 不可用")
                return pd.DataFrame()
            cur = db._cursor()
            try:
                self._ann_cache[key] = fetch_announcements(cur, field, symbols, start, end)
            finally:
                cur.close()
            logger.info(f"[PIT财务] {factor_name}: {len(self._ann_cache[key])} 条公告")

        ann = self._ann_cache[key]
        if ann.empty:
            return pd.DataFrame()
        calendar = self._layered.get_kline_calendar(symbols[:1], start, end)
        if calendar is None or len(calendar) < 10:
            # 回退：按工作日构造
            calendar = pd.bdate_range(start, end)
        return build_pit_matrix(ann, pd.DatetimeIndex(calendar), symbols)
