"""Plan and fetch bounded raw-price patches; original caches remain immutable."""
from collections import defaultdict
from datetime import date
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from .annual_audit import compress_sessions
from .preparation import atomic_json, validate_bars
from .readiness import classify_kline_name
from .wind_preparation import cached_response, save_frame, save_response


FIELDS = ('open', 'high', 'low', 'close', 'volume', 'amt', 'adjfactor')
OPTIONS = 'PriceAdj=U;Days=Trading'


def field_response(response, codes: list[str]):
    """Normalize Wind's single-day cross section and multi-day time series layouts."""
    if response.ErrorCode != 0 or list(response.Codes) != codes:
        raise ValueError('Provider error or mismatched response codes')
    times = pd.DatetimeIndex(pd.to_datetime(response.Times))
    if times.empty or times.has_duplicates or times.hasnans:
        raise ValueError('Provider returned invalid trading dates')
    values = list(response.Data)
    if len(times) == 1 and len(values) == 1 and len(values[0]) == len(codes):
        values = [[value] for value in values[0]]
    if len(values) != len(codes) or any(len(column) != len(times) for column in values):
        raise ValueError('Provider returned an unsupported field layout')
    return SimpleNamespace(ErrorCode=0, Fields=codes, Data=values, Times=list(times))


def lifecycle_dates(master: pd.DataFrame, code: str) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    """Treat provider zero-date sentinels as unknown, never as actual delist dates."""
    selected = master.loc[master['wind_code'] == code]
    if len(selected) != 1:
        return None, None
    values = []
    for field in ('IPO_DATE', 'DELIST_DATE'):
        value = pd.to_datetime(selected.iloc[0][field], errors='coerce')
        values.append(None if pd.isna(value) or value.year < 1900 else value.normalize())
    return tuple(values)


def defect_dates(frame: pd.DataFrame) -> pd.DatetimeIndex:
    """Select observed days requiring authoritative replacement, without rescaling them."""
    names = ['open', 'high', 'low', 'close', 'volume', 'amount', 'adjclose']
    days = pd.to_datetime(frame['date'], errors='coerce', format='mixed').dt.normalize()
    if not set(names).issubset(frame.columns):
        return pd.DatetimeIndex(days.dropna()).unique()
    values = frame[names].apply(pd.to_numeric, errors='coerce')
    prices = values[['open', 'high', 'low', 'close']]
    invalid = ((~np.isfinite(values.to_numpy(dtype=float))).any(axis=1) |
               (values[['volume', 'amount']] < 0).any(axis=1) |
               (values[['open', 'high', 'low', 'close', 'adjclose']] <= 0).any(axis=1) |
               (prices['high'] < prices.max(axis=1)) | (prices['low'] > prices.min(axis=1)) |
               days.duplicated(keep=False))
    return pd.DatetimeIndex(days[invalid].dropna()).unique()


def build_year_plan(cache: Path, output: Path, year: int, cutoff: date) -> dict:
    """Clip candidate gaps to the lifecycle; suspensions remain explicit candidates."""
    if not 1900 <= year <= cutoff.year:
        raise ValueError('Year must be within the requested historical range')
    lifecycle_manifest = json.loads((output / 'lifecycle' / 'manifest.json').read_text(encoding='utf-8'))
    tables = []
    for name, record in lifecycle_manifest['files'].items():
        path = output / 'lifecycle' / name
        if cached_response(path, record['request']) is None:
            raise ValueError('Lifecycle source is missing or changed')
        tables.append(pd.read_parquet(path))
    master = pd.concat(tables, ignore_index=True)
    calendars = {exchange: pd.DatetimeIndex(pd.read_parquet(output / 'wind' / f'calendar_{provider}.parquet')['date']).normalize().unique().sort_values()
                 for exchange, provider in (('SH', 'SSE'), ('SZ', 'SZSE'))}
    lower, upper = pd.Timestamp(date(year, 1, 1)), pd.Timestamp(min(cutoff, date(year, 12, 31)))
    intervals = []
    unresolved = []
    inactive = 0
    for path in sorted((cache / 'kline').glob('*.parquet')):
        exchange = path.stem.rsplit('_', 1)[-1]
        if classify_kline_name(path.stem) != 'a_stock_filename' or exchange not in calendars:
            continue
        if not path.resolve().is_relative_to(cache.resolve()):
            raise ValueError('Source path escapes cache')
        code = path.stem.replace('_', '.')
        ipo, delist = lifecycle_dates(master, code)
        if ipo is None:
            unresolved.append({'code': code, 'reason': 'missing_or_ambiguous_ipo_date'})
            continue
        first, last = max(lower, ipo), min(upper, delist) if delist is not None else upper
        calendar = calendars[exchange]
        expected = calendar[(calendar >= first) & (calendar <= last)]
        if expected.empty:
            inactive += 1
            continue
        frame = pd.read_parquet(path)
        days = pd.to_datetime(frame['date'], errors='coerce', format='mixed').dt.normalize()
        subset = frame.loc[days.between(first, last)]
        absent = expected.difference(pd.DatetimeIndex(days.dropna()))
        bad = defect_dates(subset).intersection(expected)
        targets = absent.union(bad).sort_values()
        for span in compress_sessions(targets, expected):
            intervals.append({'code': code, 'exchange': exchange, **span,
                              'candidate_absences': len(absent[(absent >= span['start']) & (absent <= span['end'])]),
                              'defective_sessions': len(bad[(bad >= span['start']) & (bad <= span['end'])])})
    result = {'year': year, 'cutoff': cutoff.isoformat(), 'intervals': intervals,
              'symbols': len({item['code'] for item in intervals}),
              'sessions': sum(item['sessions'] for item in intervals),
              'estimated_field_values': sum(item['sessions'] for item in intervals) * len(FIELDS),
              'inactive_files': inactive, 'unresolved_lifecycle': unresolved,
              'scope': 'Cached SH/SZ stock codes only; BJ and missing security-master codes are not covered.',
              'research_ready': False}
    atomic_json(output / 'repair' / str(year) / 'plan.json', result)
    return result


def fetch_year_patches(wind, output: Path, plan: dict, max_values: int = 250_000) -> dict:
    """Fetch exactly the planned ranges, stop on provider errors, preserve partial progress."""
    if plan['estimated_field_values'] > max_values:
        raise ValueError('Planned download exceeds explicit field-value budget')
    groups = defaultdict(list)
    for item in plan['intervals']:
        groups[(item['start'], item['end'])].append(item['code'])
    root = output / 'repair' / str(plan['year'])
    records = {}
    for (start, end), codes in sorted(groups.items()):
        for index in range(0, len(codes), 50):
            chunk = sorted(codes[index:index + 50])
            columns = {}
            sources = []
            for field in FIELDS:
                request = {'method': 'wsd', 'codes': chunk, 'fields': field, 'start': start, 'end': end, 'options': OPTIONS}
                identity = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()[:20]
                destination = root / 'responses' / f'{identity}.parquet'
                record = cached_response(destination, request)
                if record is None:
                    response = wind.wsd(','.join(chunk), field, start, end, OPTIONS)
                    if response.ErrorCode or list(response.Codes) != chunk:
                        atomic_json(root / 'failure.json', {'request': request, 'error_code': response.ErrorCode,
                                                           'error': 'provider_error_or_code_mismatch'})
                        return {'status': 'blocked', 'completed_patches': len(records), 'error_code': response.ErrorCode}
                    try:
                        proxy = field_response(response, chunk)
                    except (ValueError, TypeError) as exc:
                        atomic_json(root / 'failure.json', {'request': request, 'error_type': type(exc).__name__,
                                                           'error': 'unsupported_field_response'})
                        return {'status': 'blocked', 'completed_patches': len(records), 'error': 'unsupported_field_response'}
                    record = save_response(proxy, destination, request)
                    if record['status'] == 'error':
                        return {'status': 'blocked', 'completed_patches': len(records), 'error': record.get('error')}
                frame = pd.read_parquet(destination).set_index('date')
                columns[field] = frame
                sources.append({'path': str(destination.relative_to(root)), 'sha256': record['sha256']})
            for code in chunk:
                patch = pd.DataFrame({field: values[code] for field, values in columns.items()})
                patch = patch.rename(columns={'amt': 'amount'}).rename_axis('date').reset_index()
                compatibility = patch.assign(adjclose=patch['close'] * patch['adjfactor'])
                checks = validate_bars(compatibility)
                expected_counts = {item['sessions'] for item in plan['intervals']
                                   if item['code'] == code and item['start'] == start and item['end'] == end
                                   and 'sessions' in item}
                dates = pd.to_datetime(patch['date'])
                if (expected_counts and expected_counts != {len(patch)}) or not dates.between(pd.Timestamp(start), pd.Timestamp(end)).all():
                    checks['status'] = 'invalid'
                    checks['response_coverage_mismatch'] = True
                if not np.isfinite(patch['adjfactor'].to_numpy(dtype=float)).all() or (patch['adjfactor'] <= 0).any():
                    checks['status'] = 'invalid'
                destination = root / 'patches' / f'{code.replace(".", "_")}_{start}_{end}.parquet'
                metadata = save_frame(patch, destination, {'source_responses': sources, 'code': code,
                    'start': start, 'end': end, 'basis': 'raw', 'adjustment': 'Separate cumulative adjfactor; no forward-adjusted prices fabricated.',
                    'volume_unit': 'shares', 'amount_unit': 'CNY', 'validation': checks})
                records[destination.name] = metadata
            atomic_json(root / 'patch_manifest.json', {'files': records, 'research_ready': False})
    return {'status': 'downloaded', 'patch_files': len(records),
            'invalid_patches': sum(item['request']['validation']['status'] != 'validated' for item in records.values()),
            'research_ready': False}
