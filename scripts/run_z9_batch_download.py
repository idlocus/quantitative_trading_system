#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Z9 分钟数据分块下载（审计 2026-11-19 四-d 重写）。

旧版：任一季度失败仍写出最终 pkl（甚至全失败写出空字典），下次运行只看
文件存在就退出——跨品种/分期比较缺整段样本且永不自愈。
新版：按 品种×季度 分块落盘（各自完整性验收），全部块齐备才组装最终
文件；失败块下次运行自动重试。
"""
import sys, os
if __name__ == '__main__':
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from pathlib import Path
import time
import hashlib
import json
from functools import lru_cache
import pickle

import pandas as pd

DEST = Path('ai-created/research/z9_minute_signal')
CHUNK_DIR = DEST / 'min5_chunks'
FINAL = DEST / 'min5_2yr_6com.pkl'

CORE = {
    'CU.SHF': 'Copper', 'RB.SHF': 'Rebar', 'AP.CZC': 'Apple',
    'BU.SHF': 'Asphalt', 'TA.CZC': 'PTA', 'M.DCE': 'SoyMeal',
}

CHUNKS = [
    ('2025-01-01', '2025-03-31'),
    ('2025-04-01', '2025-06-30'),
    ('2025-07-01', '2025-09-30'),
    ('2025-10-01', '2025-12-31'),
    ('2026-01-01', '2026-03-31'),
    ('2026-04-01', '2026-06-30'),
    ('2026-07-01', '2026-09-30'),
]


def download_chunk(code, start, end):
    from WindPy import w
    d = w.wsi(code, 'close,volume', start + ' 09:00:00', end + ' 15:00:00', 'barsize=5')
    if d.ErrorCode != 0 or len(d.Times) < 100 or len(d.Data) != 2:
        return None
    n = len(d.Times)
    return pd.DataFrame({'close': d.Data[0], 'volume': d.Data[1] if len(d.Data) > 1 else [0] * n},
                        index=pd.DatetimeIndex(d.Times))


def chunk_path(code, start):
    return CHUNK_DIR / f"{code.replace('.', '_')}_{start[:7]}.pkl"


def chunk_valid(path: Path, start: str, end: str) -> bool:
    """块完整性验收：可读、≥100 根 bar、时间范围落在块区间内且末bar
    接近块末（末段缺失=块被截断，不得计为完成）。"""
    try:
        df = pickle.load(open(path, 'rb'))
    except Exception:
        return False
    return frame_valid(df, start, end)


def frame_valid(df, start, end):
    if (not isinstance(df, pd.DataFrame) or len(df) < 100
            or not {'volume', 'close'}.issubset(df) or not isinstance(df.index, pd.DatetimeIndex)):
        return False
    import numpy as np
    values = df[['close', 'volume']].apply(pd.to_numeric, errors='coerce')
    idx = df.index
    if (not idx.is_monotonic_increasing or not idx.is_unique or idx.hasnans
            or not np.isfinite(values.to_numpy()).all() or (values.close <= 0).any()
            or (values.volume < 0).any() or (idx.minute % 5 != 0).any()):
        return False
    lo, hi = pd.Timestamp(start), pd.Timestamp(end)
    days = idx.normalize().unique()
    gaps = days.to_series().diff().dt.days
    sessions = expected_sessions(start, end)
    if sessions is None or not len(sessions):
        return False
    daytime = idx[(idx.hour >= 9) & (idx.hour < 16)]
    offsets = [pd.Timedelta(minutes=m) for first, last in [(545, 615), (635, 690), (815, 900)]
               for m in range(first, last + 1, 5)]
    required = pd.DatetimeIndex([day + offset for day in sessions for offset in offsets])
    if len(required.difference(daytime)):
        return False
    return (days.min() >= lo and days.max() <= hi and days.min() <= lo + pd.Timedelta(days=10)
            and days.max() >= hi - pd.Timedelta(days=10) and not gaps.gt(12).any()
            and len(days) >= len(pd.bdate_range(start, end)) * .65)


@lru_cache(maxsize=32)
def expected_sessions(start, end):
    path = Path('data_cache/preparation/wind/calendar_SSE.parquet')
    if not path.exists() or not path.with_suffix('.json').exists():
        return None
    from data.wind_preparation import cached_response
    request = json.loads(path.with_suffix('.json').read_text(encoding='utf8'))['request']
    if not cached_response(path, request) or request['start'] > start or request['end'] < end:
        return None
    days = pd.DatetimeIndex(pd.read_parquet(path, columns=['date'])['date'])
    return days[(days >= pd.Timestamp(start)) & (days <= pd.Timestamp(end))]


def load_verified_product(path):
    with path.open('rb') as stream:
        manifest = path.with_suffix('.json')
        if manifest.exists():
            digest = json.loads(manifest.read_text(encoding='utf8')).get('sha256')
            if digest != hashlib.file_digest(stream, 'sha256').hexdigest():
                raise ValueError('Minute product digest mismatch')
            stream.seek(0)
        data = pickle.load(stream)
    if not isinstance(data, dict) or set(data) != set(CORE):
        raise ValueError('Incomplete minute product universe')
    for code, df in data.items():
        if not isinstance(df, pd.DataFrame) or not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError(f'Invalid minute product for {code}')
        for start, end in CHUNKS:
            if not frame_valid(df.loc[start:end], start, end):
                raise ValueError(f'Invalid minute data for {code} {start} through {end}')
    return data


def final_valid():
    try:
        load_verified_product(FINAL)
        return True
    except (IndexError, KeyError, AttributeError, OSError, EOFError, TypeError, ValueError, pickle.UnpicklingError):
        return False


def main():
    if final_valid():
        print('Validated existing minute data')
        return
    CHUNK_DIR.mkdir(parents=True, exist_ok=True)
    from WindPy import w
    w.start()
    failures = []
    for code, name in CORE.items():
        print(f'\n{name}:', flush=True)
        for start, end in CHUNKS:
            p = chunk_path(code, start)
            if p.exists() and chunk_valid(p, start, end):
                print(f'  {start[:7]} cached', flush=True)
                continue
            if p.exists():
                p.unlink()          # 无效/截断块删除重下
            print(f'  {start[:7]}...', end='', flush=True)
            df = download_chunk(code, start, end)
            if frame_valid(df, start, end):
                tmp = p.with_suffix('.tmp')
                pickle.dump(df, open(tmp, 'wb'))
                os.replace(tmp, p)
                print(f' {len(df)}', flush=True)
            else:
                failures.append((code, start))
                print(' FAIL', flush=True)
            time.sleep(0.5)
    w.stop()

    missing = [(c, s) for c in CORE for s, _e in CHUNKS
               if not (chunk_path(c, s).exists()
                       and chunk_valid(chunk_path(c, s), s,
                                       dict(CHUNKS)[s]))]
    if missing:
        print(f'\n{len(missing)} 个品种×季度块缺失或未通过验收——最终文件不写出，'
              f'下次运行自动重试缺失块')
        for code, start in missing:
            print(f'  missing: {code} {start[:7]}')
        sys.exit(1)

    all_data = {}
    for code, name in CORE.items():
        frames = [pickle.load(open(chunk_path(code, s), 'rb')) for s, _e in CHUNKS]
        full = pd.concat(frames)
        full = full[~full.index.duplicated(keep='last')].sort_index()
        all_data[code] = full
        print(f'{name}: {len(full)} bars total', flush=True)
    tmp = FINAL.with_suffix('.tmp')
    pickle.dump(all_data, open(tmp, 'wb'))
    os.replace(tmp, FINAL)
    from data.preparation import atomic_json
    atomic_json(FINAL.with_suffix('.json'), {
        'sha256': hashlib.sha256(FINAL.read_bytes()).hexdigest(),
        'chunks': {f'{c}/{start}': hashlib.sha256(chunk_path(c, start).read_bytes()).hexdigest()
                   for c in CORE for start, _ in CHUNKS}})
    print(f'\nSaved {len(all_data)} commodities（全部块完整性验收通过）')


if __name__ == '__main__':
    main()
