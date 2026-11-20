"""Explicit Wind reference downloads, isolated from legacy provider side effects."""
from datetime import date, datetime, timezone
import hashlib
import json
from pathlib import Path

import pandas as pd

from .preparation import atomic_json


def save_response(response, destination: Path, request: dict, *, include_codes: bool = False) -> dict:
    """Persist successful tabular responses and their exact request provenance."""
    if response.ErrorCode != 0:
        return {'status': 'error', 'error_code': response.ErrorCode, 'request': request}
    fields = list(response.Fields)
    data = list(response.Data)
    if (not fields or len(set(fields)) != len(fields) or len(fields) != len(data)
            or len({len(column) for column in data}) != 1):
        return {'status': 'error', 'error': 'invalid_response_shape', 'request': request}
    frame = pd.DataFrame(dict(zip(fields, data)))
    if frame.empty:
        return {'status': 'error', 'error': 'empty_response', 'request': request}
    if include_codes:
        if len(response.Codes) != len(frame):
            return {'status': 'error', 'error': 'invalid_code_count', 'request': request}
        frame.insert(0, 'wind_code', list(response.Codes))
    if len(response.Times) == len(frame) and 'date' not in frame:
        frame.insert(0, 'date', pd.to_datetime(response.Times))
    return save_frame(frame, destination, request)


def save_frame(frame: pd.DataFrame, destination: Path, request: dict, *, provider: str = 'WindPy') -> dict:
    """Save a provider frame with request provenance and a content digest."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix('.pending.parquet')
    try:
        frame.to_parquet(temporary, index=False)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    with destination.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    record = {'status': 'downloaded_unverified', 'rows': len(frame), 'columns': list(frame.columns),
              'sha256': digest, 'request': request,
              'retrieved_at': datetime.now(timezone.utc).isoformat(),
              'path': destination.name, 'provider': provider}
    atomic_json(destination.with_suffix('.json'), record)
    return record


def cached_response(destination: Path, request: dict) -> dict | None:
    """Reuse only a matching request with verified content, never an existence check."""
    manifest = destination.with_suffix('.json')
    if not destination.exists() or not manifest.exists():
        return None
    try:
        record = json.loads(manifest.read_text(encoding='utf-8'))
        with destination.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        return record if record.get('request') == request and record.get('sha256') == digest else None
    except (OSError, ValueError):
        return None


def prepare_lifecycle(wind, cache: Path, output: Path, chunk_size: int = 200) -> dict:
    """Request listing and delisting dates for both cached and current A-share codes."""
    from .readiness import classify_kline_name

    if not 1 <= chunk_size <= 500:
        raise ValueError('Lifecycle chunk size must be between 1 and 500')
    codes = {path.stem.replace('_', '.') for path in (cache / 'kline').glob('*.parquet')
             if classify_kline_name(path.stem) == 'a_stock_filename'}
    snapshot = output / 'wind' / 'a_share_current_snapshot.parquet'
    if snapshot.exists():
        codes.update(pd.read_parquet(snapshot)['wind_code'].dropna().tolist())
    codes = sorted(codes)
    records = {}
    for index in range(0, len(codes), chunk_size):
        chunk = codes[index:index + chunk_size]
        request = {'method': 'wss', 'codes': chunk, 'fields': 'ipo_date,delist_date', 'options': ''}
        destination = output / 'lifecycle' / f'chunk_{index:05d}.parquet'
        record = cached_response(destination, request)
        if record is None:
            response = wind.wss(','.join(chunk), request['fields'], request['options'])
            record = save_response(response, destination, request, include_codes=True)
        records[destination.name] = record
        atomic_json(output / 'lifecycle' / 'manifest.json', {'files': records, 'requested_codes': len(codes)})
        if record['status'] == 'error':
            break
    return {'requested_codes': len(codes), 'downloaded_rows': sum(item.get('rows', 0) for item in records.values()),
            'errors': sum(item['status'] == 'error' for item in records.values()),
            'note': 'Security lifecycle only; not daily tradability or historical connect eligibility.'}


def prepare_market_status(wind, output: Path, year: int) -> dict:
    """Collect explicit provider statuses for unresolved quote intervals."""
    reconciliation = json.loads((output / 'repair' / str(year) / 'reconciliation.json').read_text(encoding='utf-8'))
    records = []
    for item in reconciliation['records']:
        if item['classification'] != 'requires_market_status_review':
            continue
        request = {'method': 'wsd', 'codes': item['code'], 'fields': 'trade_status',
                   'start': item['start'], 'end': item['end'], 'options': 'Days=Trading'}
        destination = output / 'market_status' / str(year) / f'{item["code"].replace(".", "_")}_{item["start"]}_{item["end"]}.parquet'
        record = cached_response(destination, request)
        if record is None:
            response = wind.wsd(request['codes'], request['fields'], request['start'], request['end'], request['options'])
            record = save_response(response, destination, request)
        records.append(record)
        atomic_json(output / 'market_status' / str(year) / 'manifest.json', {'requests': records, 'research_ready': False})
        if record['status'] == 'error':
            break
    return {'year': year, 'requests': len(records), 'errors': sum(record['status'] == 'error' for record in records),
            'research_ready': False}


def prepare_reference(wind, output: Path, start: date, end: date) -> dict:
    """Fetch calendars and a current A-share snapshot; never claim historical membership."""
    if start > end:
        raise ValueError('Start must not follow end')
    requests = {}
    for market in ('SSE', 'SZSE', 'HKEX'):
        options = f'TradingCalendar={market}'
        response = wind.tdays(start.isoformat(), end.isoformat(), options)
        requests[f'calendar_{market}'] = save_response(response, output / f'calendar_{market}.parquet',
            {'method': 'tdays', 'start': start.isoformat(), 'end': end.isoformat(), 'options': options})
    options = f'date={end.isoformat()};sectorid=a001010100000000'
    response = wind.wset('sectorconstituent', options)
    requests['a_share_current_snapshot'] = save_response(response, output / 'a_share_current_snapshot.parquet',
        {'method': 'wset', 'name': 'sectorconstituent', 'options': options,
         'limitation': 'Snapshot only; not a historical security master'})
    result = {'requests': requests, 'research_ready': False}
    atomic_json(output / 'reference_manifest.json', result)
    return result
