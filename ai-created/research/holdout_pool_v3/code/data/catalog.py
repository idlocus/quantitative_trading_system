"""Read-only local cache discovery and access, with no network fallback."""
from dataclasses import dataclass
from datetime import date
from pathlib import Path
import re

import pandas as pd


@dataclass(frozen=True)
class Dataset:
    name: str
    relative_path: str
    size_bytes: int


class LocalDataCatalog:
    def __init__(self, root: Path | str):
        self.root = Path(root).resolve()

    def datasets(self, category: str | None = None) -> tuple[Dataset, ...]:
        folder = self.root if category is None else (self.root / category).resolve()
        if not folder.is_relative_to(self.root):
            raise ValueError('Cache category must stay inside the cache directory')
        if not folder.is_dir():
            return ()
        return tuple(Dataset(path.stem, path.relative_to(self.root).as_posix(), path.stat().st_size)
                     for path in sorted(folder.rglob('*.parquet'))
                     if path.resolve().is_relative_to(self.root))

    def summary(self) -> dict:
        datasets = self.datasets()
        categories = {}
        for item in datasets:
            category = Path(item.relative_path).parts[0] if '/' in item.relative_path else 'root'
            group = categories.setdefault(category, {'files': 0, 'bytes': 0})
            group['files'] += 1
            group['bytes'] += item.size_bytes
        return {'cache_exists': self.root.is_dir(), 'files': len(datasets),
                'bytes': sum(item.size_bytes for item in datasets), 'categories': categories}

    def kline(self, symbol: str, start: date | None = None, end: date | None = None) -> pd.DataFrame:
        if not re.fullmatch(r'\d{5,6}\.(SH|SZ|BJ|HK)', symbol):
            raise ValueError('Expected an exchange-qualified symbol such as 600000.SH')
        path = (self.root / 'kline' / f'{symbol.replace(".", "_")}.parquet').resolve()
        if not path.is_relative_to(self.root):
            raise ValueError('Cache path must stay inside the cache directory')
        if not path.is_file():
            raise FileNotFoundError(f'No local K-line cache for {symbol}; no download was attempted')
        frame = pd.read_parquet(path)
        if 'date' not in frame or 'close' not in frame:
            raise ValueError(f'Missing date/close columns in {symbol}')
        frame = frame.copy()
        frame['date'] = pd.to_datetime(frame['date'].astype('string'), format='mixed', errors='coerce')
        if frame['date'].isna().any() or frame['date'].duplicated().any():
            raise ValueError(f'Invalid or duplicate dates in {symbol}')
        if start is not None and end is not None and start > end:
            raise ValueError('Start date must not be after end date')
        if start is not None:
            frame = frame.loc[frame['date'] >= pd.Timestamp(start)]
        if end is not None:
            frame = frame.loc[frame['date'] <= pd.Timestamp(end)]
        return frame.sort_values('date').reset_index(drop=True)
