#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
公司行动数据层（修复 2026-10-26 勘误#1）：A 股现金分红与送转

来源：Wind ASHAREDIVIDEND（进度=3 已实施）
口径（R2）：
  cash_ps = CASH_DVD_PER_SH_AFTER_TAX（税后每股现金，元/股）
  stk_ps  = STK_DVD_PER_SH（每股送转股数，1.0=10送10 即股数×2）
  ex_date = EX_DT（除权除息日——价格于该日调整，持仓于 T-1 收盘者享有）
缓存：data_cache/corporate_actions.parquet（原子写）
"""
import os
import hashlib
import json
from datetime import date
from utils.state_file import exclusive_file, replace_json
from pathlib import Path

import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent
CACHE = PROJECT / 'data_cache' / 'corporate_actions.parquet'


def load_corporate_actions(ds, symbols=None, start='20160101', end=None,
                           use_cache=True) -> pd.DataFrame:
    """返回 [sym, ex_date, cash_ps, stk_ps]。symbols=None 时全市场。"""
    end = min(end or date.today().strftime('%Y%m%d'), date.today().strftime('%Y%m%d'))
    coverage_path = CACHE.with_suffix('.coverage.json')
    if use_cache:
        with exclusive_file(CACHE):
            coverage = json.loads(coverage_path.read_text(encoding='utf8')) if coverage_path.exists() else {}
            if (CACHE.is_file() and coverage.get('start', '99999999') <= start
                    and coverage.get('end', '') >= end):
                with CACHE.open('rb') as stream:
                    if hashlib.file_digest(stream, 'sha256').hexdigest() == coverage.get('sha256'):
                        stream.seek(0)
                        df = pd.read_parquet(stream)
                        days = pd.to_datetime(df['ex_date'])
                        df = df.loc[days.between(pd.Timestamp(start), pd.Timestamp(end))]
                        if symbols is not None:
                            df = df[df['sym'].isin(symbols)]
                        return df.reset_index(drop=True)

    cur = ds._db_src()._cursor()
    cur.execute(
        "SELECT S_INFO_WINDCODE, EX_DT, CASH_DVD_PER_SH_AFTER_TAX, STK_DVD_PER_SH "
        "FROM ASHAREDIVIDEND WHERE S_DIV_PROGRESS='3' AND EX_DT>=:s AND EX_DT<=:e "
        "AND (CASH_DVD_PER_SH_AFTER_TAX>0 OR STK_DVD_PER_SH>0)",
        {'s': start, 'e': end})
    rows = cur.fetchall()
    if hasattr(cur, 'close'):
        cur.close()
    df = pd.DataFrame(rows, columns=['sym', 'ex_date', 'cash_ps', 'stk_ps'])
    df['ex_date'] = pd.to_datetime(df['ex_date'], format='%Y%m%d')
    for c in ('cash_ps', 'stk_ps'):
        df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0.0)
    full = df.copy()
    with exclusive_file(CACHE):
        tmp = str(CACHE) + '.tmp'
        full.to_parquet(tmp)
        digest = hashlib.sha256(Path(tmp).read_bytes()).hexdigest()
        os.replace(tmp, CACHE)
        replace_json(coverage_path, dict(start=start, end=end, sha256=digest))
    if symbols is not None:
        df = df[df['sym'].isin(set(symbols))]
    return df.reset_index(drop=True)


def by_date_index(ca: pd.DataFrame) -> dict:
    """{date: {sym: (cash_ps, stk_ps)}}，供逐日重放查询。"""
    out = {}
    for _, r in ca.iterrows():
        d = r['ex_date'].date()
        events = out.setdefault(d, {})
        cash, stock = events.get(r['sym'], (0., 0.))
        events[r['sym']] = cash + float(r['cash_ps']), stock + float(r['stk_ps'])
    return out
