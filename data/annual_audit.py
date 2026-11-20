"""Yearly cache diagnostics; calendar gaps remain candidates until status is known."""
from collections import Counter
from datetime import date
from pathlib import Path

import pandas as pd

from .preparation import atomic_json, validate_bars
from .readiness import classify_kline_name


def annual_diagnostics(frame: pd.DataFrame, calendar: pd.DatetimeIndex,
                       start: date, end: date) -> dict:
    """Compare observed bars to sessions without inventing listing or suspension dates."""
    dates = pd.to_datetime(frame['date'], errors='coerce', format='mixed').dt.normalize()
    sessions = pd.DatetimeIndex(calendar).normalize().unique().sort_values()
    result = {}
    for year in range(start.year, end.year + 1):
        lower = pd.Timestamp(max(start, date(year, 1, 1)))
        upper = pd.Timestamp(min(end, date(year, 12, 31)))
        expected = sessions[(sessions >= lower) & (sessions <= upper)]
        mask = dates.between(lower, upper)
        observed = pd.DatetimeIndex(dates[mask].dropna()).unique().sort_values()
        checks = validate_bars(frame.loc[mask]) if mask.any() else {'status': 'absent', 'rows': 0}
        absent = expected.difference(observed)
        entry = {**checks, 'exchange_sessions': len(expected), 'observed_sessions': len(observed),
                 'unexplained_absent_sessions': len(absent),
                 'off_calendar_sessions': len(observed.difference(expected)),
                 'absent_session_ranges': compress_sessions(absent, expected),
                 'classification': 'needs_security_lifecycle_and_suspension_review' if len(absent) else 'calendar_dates_present'}
        result[str(year)] = entry
    return result


def compress_sessions(absent: pd.DatetimeIndex, calendar: pd.DatetimeIndex) -> list[dict]:
    """Compress absent consecutive exchange sessions, including intervening holidays."""
    ranges = []
    positions = calendar.get_indexer(absent)
    for index, day in enumerate(absent):
        if index and positions[index] == positions[index - 1] + 1:
            ranges[-1]['end'] = day.date().isoformat()
            ranges[-1]['sessions'] += 1
        else:
            ranges.append({'start': day.date().isoformat(), 'end': day.date().isoformat(), 'sessions': 1})
    return ranges


def audit_years(cache: Path, output: Path, start: date, end: date) -> dict:
    """Write per-year evidence separately; never overwrite retained market data."""
    cache, output = cache.resolve(), output.resolve()
    if start > end:
        raise ValueError('Start must not follow end')
    if output.is_relative_to(cache / 'kline'):
        raise ValueError('Audit output must be outside original daily bars')
    calendars = {}
    for exchange, provider in (('SH', 'SSE'), ('SZ', 'SZSE')):
        frame = pd.read_parquet(output / 'wind' / f'calendar_{provider}.parquet')
        calendars[exchange] = pd.DatetimeIndex(pd.to_datetime(frame['date']))
        if calendars[exchange].hasnans or calendars[exchange].has_duplicates:
            raise ValueError('Invalid reference calendar')
        if calendars[exchange].empty or calendars[exchange].min().date() > start or calendars[exchange].max().date() < end:
            # Start/end may be holidays, but large truncation must not hide absent history.
            if calendars[exchange].empty or (calendars[exchange].min().date() - start).days > 10 or (end - calendars[exchange].max().date()).days > 10:
                raise ValueError('Reference calendar does not cover requested audit interval')
    reports = {str(year): {} for year in range(start.year, end.year + 1)}
    errors = []
    skipped = Counter()
    for path in sorted((cache / 'kline').glob('*.parquet')):
        exchange = path.stem.rsplit('_', 1)[-1]
        if classify_kline_name(path.stem) != 'a_stock_filename' or exchange not in calendars:
            skipped[exchange] += 1
            continue
        if not path.resolve().is_relative_to(cache):
            errors.append({'file': path.name, 'error': 'path_outside_cache'})
            continue
        try:
            source = pd.read_parquet(path)
            entries = annual_diagnostics(source, calendars[exchange], start, end)
            for year, entry in entries.items():
                reports[year][path.name] = entry
        except (OSError, ValueError, TypeError, KeyError) as exc:
            errors.append({'file': path.name, 'error_type': type(exc).__name__})
    summaries = {}
    for year, entries in reports.items():
        summary = {'files': len(entries), 'rows': sum(item['rows'] for item in entries.values()),
                   'status_counts': dict(Counter(item['status'] for item in entries.values())),
                   'issue_counts': dict(sum((Counter(item.get('issues', {})) for item in entries.values()), Counter())),
                   'files_with_unexplained_absences': sum(item['unexplained_absent_sessions'] > 0 for item in entries.values())}
        atomic_json(output / 'annual' / f'{year}.json', {'year': int(year), 'summary': summary, 'files': entries,
                    'research_ready': False, 'note': 'Absent sessions are not confirmed missing prices; lifecycle and suspension evidence is required.'})
        summaries[year] = summary
    result = {'start': start.isoformat(), 'end': end.isoformat(), 'years': summaries,
              'skipped_by_exchange': dict(skipped), 'errors': errors, 'research_ready': False}
    atomic_json(output / 'annual' / 'manifest.json', result)
    return result
