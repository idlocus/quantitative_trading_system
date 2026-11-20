#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析师一致预期因子（PIT）—— EPS 预期修正（estimate revision）

构造：每日 t 取"下一个未公布的年度报告期"的一致预期 EPS_AVG，
修正 = 当前值 / 20个交易日前值 − 1。预期上调为正信号（经典修正动量）。

PIT：EST_DT 为预期快照发布日；目标报告期在 t 时点即可确定（下一个年度）。
覆盖说明：一致预期仅覆盖有券商跟踪的股票（约半数，偏大市值）——
覆盖外为 NaN，不参与横截面。
"""

from datetime import date
from typing import List

import numpy as np
import pandas as pd

from utils.logger import get_logger

logger = get_logger(__name__)


def fetch_consensus(db_cursor, symbols: List[str], start: date, end: date) -> pd.DataFrame:
    """拉取年度一致预期快照（EPS_AVG）：symbol, est_dt, report_dt, eps_avg"""
    lookback = date(start.year - 2, 1, 1)
    frames = []
    for i in range(0, len(symbols), 500):
        chunk = symbols[i:i + 500]
        ph = ",".join(f":s{j}" for j in range(len(chunk)))
        binds = {f"s{j}": s for j, s in enumerate(chunk)}
        sql = f"""
            SELECT S_INFO_WINDCODE, EST_DT, EST_REPORT_DT, CAST(EPS_AVG AS FLOAT)
            FROM ASHARECONSENSUSDATA
            WHERE S_INFO_WINDCODE IN ({ph})
              AND EST_DT IS NOT NULL AND EST_REPORT_DT IS NOT NULL
              AND EPS_AVG IS NOT NULL
              AND SUBSTR(EST_REPORT_DT, 5, 2) = '12'
              AND EST_DT >= :dt1 AND EST_DT <= :dt2
        """
        binds.update({'dt1': lookback.strftime('%Y%m%d'), 'dt2': end.strftime('%Y%m%d')})
        db_cursor.execute(sql, binds)
        frames.append(pd.DataFrame(
            db_cursor.fetchall(), columns=['symbol', 'est_dt', 'report_dt', 'eps']))
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    return df.sort_values(['symbol', 'est_dt', 'report_dt'])


def build_revision_matrix(cons: pd.DataFrame, index: pd.DatetimeIndex,
                          columns: List[str],
                          revision_window: int = 20) -> pd.DataFrame:
    """
    预期修正矩阵（dates × symbols）

    规则：每日取"下一个年度报告期"的 EPS 步进序列值；
    修正 = 当日值 / 20日前值 − 1（EPS≤0 的期 NaN，避免符号翻转噪声）。
    """
    if cons.empty:
        return pd.DataFrame(np.nan, index=index, columns=columns)
    index = pd.DatetimeIndex(index).as_unit('ns')
    cons = cons.drop_duplicates(subset=['symbol', 'est_dt', 'report_dt'], keep='last')
    cons['ts'] = pd.DatetimeIndex(pd.to_datetime(cons['est_dt'], format='%Y%m%d')).as_unit('ns')
    cons['rdt'] = pd.to_datetime(cons['report_dt'], format='%Y%m%d')

    out = pd.DataFrame(np.nan, index=index, columns=columns)
    for sym, g in cons.groupby('symbol'):
        if sym not in columns:
            continue
        # 每个报告期的 EPS 步进序列
        series_by_period = {}
        for rp, gg in g.groupby('rdt'):
            gg = gg.sort_values('ts').drop_duplicates(subset='ts', keep='last')
            s = gg.set_index('ts')['eps']
            s = s[(s.index >= index[0] - pd.Timedelta(days=200)) & (s.index <= index[-1])]
            if len(s) < 2:
                continue
            series_by_period[rp] = s

        if not series_by_period:
            continue
        # 每日取"下一个年度报告期"的值
        daily = pd.Series(np.nan, index=index)
        for t in index:
            future = [rp for rp in series_by_period if rp > t]
            if not future:
                continue
            rp = min(future)
            s = series_by_period[rp]
            prior = s[s.index <= t]
            if len(prior) == 0:
                continue
            v = prior.iloc[-1]
            if v > 0:  # 亏损期 EPS 修正无稳定符号意义
                daily.loc[t] = v
        # 修正 = 20日变化率
        rev = daily / daily.shift(revision_window) - 1.0
        out[sym] = rev
    return out


class ConsensusProvider:
    """一致预期因子提供者（一次性拉取，内存复用）"""

    def __init__(self, layered_source):
        self._layered = layered_source
        self._cache = None

    def get_revision_matrix(self, symbols: List[str], start: date, end: date) -> pd.DataFrame:
        if self._cache is None:
            db = self._layered._db_src()
            if not db:
                logger.warning("[一致预期] DB 不可用")
                return pd.DataFrame()
            cur = db._cursor()
            try:
                self._cache = fetch_consensus(cur, symbols, start, end)
            finally:
                cur.close()
            logger.info(f"[一致预期] {len(self._cache)} 条快照")
        calendar = self._layered.get_kline_calendar(symbols[:1], start, end)
        if calendar is None or len(calendar) < 10:
            calendar = pd.bdate_range(start, end)
        return build_revision_matrix(self._cache, pd.DatetimeIndex(calendar), symbols)
