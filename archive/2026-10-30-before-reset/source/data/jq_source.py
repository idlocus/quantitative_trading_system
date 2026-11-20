#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JQData（聚宽）分钟数据源 —— 按需增量取数 + 本地持久化 + manifest 账本

设计（2026-10-16，用户要求：按需取数、全量本地存储、组织清晰、便于复用）：
  - 本地布局 data_cache/jq/：
      manifest.json               # 覆盖范围账本：{sym: {freq: {start,end,rows,updated}}}
      {symbol}_{freq}.parquet     # 单文件全历史，index=time，
                                  # cols=open,high,low,close,volume,money
  - 增量：仅拉本地缺失的区间（头/尾缺口），配额只消耗差额——与
    layered 源"配额消耗仅限差额区间"同一纪律
  - 原子写：tmp + os.replace（R4 教训：写一半崩溃不得截断旧数据）
  - 口径（R2 声明）：volume=股，money=元；get_price 分钟线为不复权价
    （收益计算需复权处理——用日频 factor 或短窗口近似，研究层负责）
  - 凭据：.env 的 JQ_USER/JQ_PASSWORD（不入库不入日志）
"""
import json
import logging
import os
from datetime import datetime
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

PROJECT = Path(__file__).resolve().parent.parent
CACHE_DIR = PROJECT / 'data_cache' / 'jq'
MANIFEST = CACHE_DIR / 'manifest.json'
FIELDS = ['open', 'high', 'low', 'close', 'volume', 'money']

_authed = False


def _auth():
    global _authed
    if _authed:
        return
    from dotenv import load_dotenv
    load_dotenv(PROJECT / '.env')
    from jqdatasdk import auth
    auth(os.getenv('JQ_USER'), os.getenv('JQ_PASSWORD'))
    _authed = True
    logger.info("[jq] 认证成功")


def _load_manifest() -> dict:
    if MANIFEST.is_file():
        try:
            return json.loads(MANIFEST.read_text(encoding='utf-8'))
        except Exception:
            logger.warning("[jq] manifest 损坏，重建")
    return {}


def _save_manifest(m: dict):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = MANIFEST.with_suffix('.tmp')
    tmp.write_text(json.dumps(m, ensure_ascii=False, indent=1), encoding='utf-8')
    os.replace(tmp, MANIFEST)


def _path(symbol: str, freq: str) -> Path:
    return CACHE_DIR / f"{symbol.replace('.', '_')}_{freq}.parquet"


def _fetch_range(symbol: str, freq: str, start: str, end: str) -> pd.DataFrame:
    """区间拉取（JQ 单次行数有限制，按月分块）"""
    from jqdatasdk import get_price
    frames = []
    cur = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    while cur <= end_ts:
        seg_end = min(cur + pd.offsets.MonthEnd(1), end_ts)
        df = get_price(symbol, start_date=str(cur.date()),
                       end_date=str(seg_end.date()), frequency=freq,
                       fields=FIELDS, skip_paused=False, fq=None)
        if df is not None and len(df):
            frames.append(df)
        cur = seg_end + pd.offsets.Day(1)
    return pd.concat(frames) if frames else pd.DataFrame(columns=FIELDS)


def ensure(symbol: str, freq: str, start: str, end: str) -> pd.DataFrame:
    """确保本地覆盖 [start, end]（增量补缺口），返回该区间数据。

    Args:
        symbol: JQ 代码（如 '510300.XSHG'）
        freq: '5m' / '1m' / '15m' …
        start/end: 'YYYY-MM-DD'
    """
    _auth()
    m = _load_manifest()
    rec = m.get(symbol, {}).get(freq, {})
    path = _path(symbol, freq)
    have = None
    if path.is_file() and rec:
        try:
            have = pd.read_parquet(path)
        except Exception:
            logger.warning(f"[jq] {path.name} 损坏，重拉")
            have = None
    want_s, want_e = pd.Timestamp(start), pd.Timestamp(end)

    new_frames = []
    if have is None or not len(have):
        new_frames.append(_fetch_range(symbol, freq, start, end))
        merged = new_frames[0]
    else:
        have_s, have_e = have.index[0], have.index[-1]
        if want_s < have_s:                    # 头缺口
            new_frames.append(_fetch_range(symbol, freq, start, str(have_s.date())))
        if want_e > have_e:                    # 尾缺口（增量主力）
            new_frames.append(_fetch_range(symbol, freq,
                                           str((have_e + pd.offsets.Day(1)).date()),
                                           end))
        if new_frames:
            merged = pd.concat([*new_frames, have])
            merged = merged[~merged.index.duplicated(keep='last')].sort_index()
        else:
            merged = have

    if new_frames or have is None or not path.is_file():
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix('.tmp')
        merged.to_parquet(tmp)
        os.replace(tmp, path)
        m.setdefault(symbol, {})[freq] = {
            'start': str(merged.index[0]), 'end': str(merged.index[-1]),
            'rows': int(len(merged)),
            'updated': datetime.now().isoformat()[:19], 'source': 'jqdata'}
        _save_manifest(m)
        logger.info(f"[jq] {symbol} {freq}: 本地覆盖 {merged.index[0]} ~ "
                    f"{merged.index[-1]}（{len(merged)} 行）")
    return merged.loc[want_s: want_e + pd.offsets.Day(1)]


def ensure_batch(symbols, freq: str, start: str, end: str) -> dict:
    """批量确保覆盖（逐只增量）。"""
    out = {}
    for s in symbols:
        out[s] = ensure(s, freq, start, end)
    return out
