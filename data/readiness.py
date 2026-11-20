"""Read-only metadata audit of the retained local research datasets.

Filename categories are inventory hints, never point-in-time eligibility.
No provider is instantiated, no credentials are loaded and no cache is written.
"""
from collections import Counter
from datetime import date, datetime
from pathlib import Path
import re

import pyarrow.parquet as parquet


def classify_kline_name(name: str) -> str:
    match = re.fullmatch(r'(\d{5,6})_(SH|SZ|BJ|HK)', name)
    if not match:
        return 'unclassified'
    code, exchange = match.groups()
    if exchange == 'HK':
        return 'hk_unverified_eligibility'
    if exchange == 'BJ':
        return 'a_stock_filename'
    if (exchange == 'SH' and code.startswith(('60', '68')) or
            exchange == 'SZ' and code.startswith(('00', '30'))):
        return 'a_stock_filename'
    return 'non_stock_or_unclassified'


def _iso_day(value) -> str | None:
    if isinstance(value, (datetime, date)):
        return value.isoformat()[:10]
    if isinstance(value, bytes):
        value = value.decode('utf-8', errors='strict')
    if isinstance(value, str):
        if re.fullmatch(r'\d{8}', value):
            return date.fromisoformat(f'{value[:4]}-{value[4:6]}-{value[6:]}').isoformat()
        return date.fromisoformat(value[:10]).isoformat()
    return None


def parquet_summary(path: Path) -> dict:
    """Inspect schema, row counts and stored date statistics only."""
    with parquet.ParquetFile(path) as reader:
        names = reader.schema_arrow.names
        date_column = next((name for name in ('date', 'ex_date', 'dt', '__index_level_0__')
                            if name in names), None)
        minimum, maximum = [], []
        missing_statistics = 0
        if date_column is not None:
            for group_index in range(reader.metadata.num_row_groups):
                group = reader.metadata.row_group(group_index)
                column = next((group.column(i) for i in range(group.num_columns)
                               if group.column(i).path_in_schema == date_column), None)
                statistics = column.statistics if column is not None else None
                if statistics is None or not statistics.has_min_max:
                    missing_statistics += 1
                    continue
                lower, upper = _iso_day(statistics.min), _iso_day(statistics.max)
                if lower is None or upper is None:
                    missing_statistics += 1
                else:
                    minimum.append(lower)
                    maximum.append(upper)
        return {'rows': reader.metadata.num_rows, 'columns': tuple(names),
                'date_column': date_column, 'date_statistics_complete': (
                    date_column is not None and missing_statistics == 0 and bool(minimum)),
                'first_date': min(minimum) if minimum else None,
                'last_date': max(maximum) if maximum else None}


def audit_local_readiness(cache_root: Path | str) -> dict:
    """Aggregate inventory evidence without printing symbols or raw records."""
    root = Path(cache_root).resolve()
    groups = {}
    errors = []
    required = {'date', 'open', 'high', 'low', 'close', 'volume'}
    paths = sorted((root / 'kline').glob('*.parquet'))
    for path in paths:
        if not path.resolve().is_relative_to(root):
            errors.append({'category': 'kline', 'error': 'Cache path escapes root'})
            continue
        kind = classify_kline_name(path.stem)
        group = groups.setdefault(kind, {'files': 0, 'rows': 0, 'schema_incomplete': 0,
            'with_adjclose_column': 0, 'with_amount_column': 0,
            'date_statistics_incomplete': 0, 'first_date': None, 'last_date': None,
            'latest_date_distribution': Counter()})
        group['files'] += 1
        try:
            summary = parquet_summary(path)
            columns = set(summary['columns'])
            group['rows'] += summary['rows']
            group['schema_incomplete'] += int(not required.issubset(columns))
            group['with_adjclose_column'] += int('adjclose' in columns)
            group['with_amount_column'] += int('amount' in columns)
            group['date_statistics_incomplete'] += int(not summary['date_statistics_complete'])
            if summary['first_date']:
                group['first_date'] = min(filter(None, [group['first_date'], summary['first_date']]))
            if summary['last_date']:
                group['last_date'] = max(filter(None, [group['last_date'], summary['last_date']]))
                group['latest_date_distribution'][summary['last_date']] += 1
        except (OSError, ValueError, TypeError, UnicodeError) as exc:
            errors.append({'category': kind, 'error': type(exc).__name__})
    for group in groups.values():
        group['latest_date_distribution'] = dict(sorted(group['latest_date_distribution'].items()))
    supporting = []
    support_paths = list((root / 'fundamental').glob('*.parquet'))
    support_paths += [root / 'corporate_actions.parquet']
    for path in sorted(support_paths):
        if not path.is_file():
            continue
        if not path.resolve().is_relative_to(root):
            errors.append({'category': 'supporting', 'error': 'Cache path escapes root'})
            continue
        try:
            summary = parquet_summary(path)
            supporting.append({'dataset': path.relative_to(root).as_posix(),
                'rows': summary['rows'], 'column_count': len(summary['columns']),
                'first_date': summary['first_date'], 'last_date': summary['last_date'],
                'date_statistics_complete': summary['date_statistics_complete']})
        except (OSError, ValueError, TypeError, UnicodeError) as exc:
            errors.append({'category': 'supporting', 'error': type(exc).__name__})
    return {'audit_date': date.today().isoformat(), 'mode': 'offline_metadata_only',
            'cache_exists': root.is_dir(), 'primary_kline_files': len(paths),
            'filename_groups': groups, 'supporting_datasets': supporting, 'errors': errors,
            'limitations': [
                'Filename classification does not establish instrument identity or historical eligibility.',
                'Column presence does not verify price basis, units, currency or absence of bad rows.',
                'Parquet date extrema do not prove trading-calendar continuity or delisted-stock coverage.',
                'No remote database, Stock Connect membership, FX or calendar provider was queried.',
                'This report is inventory evidence, not acceptance of a strategy or market execution model.']}
