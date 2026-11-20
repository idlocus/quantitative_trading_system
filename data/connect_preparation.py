"""Collect explicit dated provider membership; never infer changes between snapshots."""
from datetime import date
import json
from pathlib import Path

import pandas as pd

from .preparation import atomic_json
from .wind_preparation import cached_response, save_response


ROUTES = {'sh_southbound': ('\u6caa\u5e02\u6e2f\u80a1\u901a', 'HKS'),
          'sz_southbound': ('\u6df1\u5e02\u6e2f\u80a1\u901a', 'HKSZ')}


def download_connect_daily(wind, output: Path, start: date, end: date) -> dict:
    if start > end:
        raise ValueError('Start must not follow end')
    dictionary_path = output / 'oracle/supporting/ASHAREINDUSTRIESCODE.parquet'
    dictionary_record = json.loads(dictionary_path.with_suffix('.json').read_text(encoding='utf-8'))
    if cached_response(dictionary_path, dictionary_record['request']) is None:
        raise ValueError('Sector dictionary failed provenance verification')
    dictionary = pd.read_parquet(dictionary_path)
    calendar_path = output / 'oracle/reference/HKEXCALENDAR.parquet'
    calendar_record = json.loads(calendar_path.with_suffix('.json').read_text(encoding='utf-8'))
    if cached_response(calendar_path, calendar_record['request']) is None:
        raise ValueError('Connect calendar failed provenance verification')
    calendar = pd.read_parquet(calendar_path)
    calendar['day'] = pd.to_datetime(calendar['TRADE_DAYS'], format='%Y%m%d', errors='raise')
    root = output / 'connect_daily'
    manifest = root / 'manifest.json'
    records = json.loads(manifest.read_text(encoding='utf-8')).get('files', {}) if manifest.exists() else {}
    completed = 0
    for route, (label, market) in ROUTES.items():
        selected = dictionary.loc[dictionary['INDUSTRIESNAME'].eq(label)
            & dictionary['INDUSTRIESCODE'].astype('string').str.startswith('a'), 'INDUSTRIESCODE'].unique()
        if len(selected) != 1:
            raise ValueError('Southbound sector identifier is missing or ambiguous')
        days = calendar.loc[calendar['S_INFO_EXCHMARKET'].eq(market)
            & calendar['day'].between(pd.Timestamp(start), pd.Timestamp(end)), 'day'].drop_duplicates().sort_values()
        for day in days:
            options = f'date={day.date().isoformat()};sectorid={selected[0]}'
            request = {'method': 'wset', 'name': 'sectorconstituent', 'options': options,
                       'route': route, 'as_of': day.date().isoformat(),
                       'dictionary_sha256': dictionary_record['sha256'],
                       'calendar_sha256': calendar_record['sha256']}
            destination = root / route / f'{day.date().isoformat()}.parquet'
            record = cached_response(destination, request)
            if record is None:
                record = save_response(wind.wset('sectorconstituent', options), destination, request)
            error = record['status'] == 'error'
            if not error:
                frame = pd.read_parquet(destination)
                error = not {'date', 'wind_code'}.issubset(frame.columns)
                if not error:
                    observed = pd.to_datetime(frame['date'], errors='coerce').dt.normalize()
                    error = (not observed.eq(day).all() or frame['wind_code'].duplicated().any()
                             or not frame['wind_code'].astype('string').str.endswith('.HK').all())
            if error:
                atomic_json(root / 'failure.json', {'request': request, 'provider_result': record,
                    'error': 'provider_error_or_unverified_snapshot_date'})
                atomic_json(manifest, {'files': records, 'research_ready': False, 'last_attempt_status': 'blocked'})
                return {'status': 'blocked', 'completed_this_run': completed,
                        'provider_error_code': record.get('error_code'), 'research_ready': False}
            records[f'{route}/{day.date().isoformat()}'] = record
            completed += 1
            if completed % 20 == 0:
                atomic_json(manifest, {'files': records, 'research_ready': False, 'last_attempt_status': 'running'})
    atomic_json(manifest, {'files': records, 'research_ready': False, 'last_attempt_status': 'completed',
        'scope': 'Dated provider buy-universe snapshots; sell-only qualification and official cross-check remain separate.'})
    return {'status': 'downloaded', 'snapshots': len(records), 'completed_this_run': completed,
            'rows': sum(record['rows'] for record in records.values()), 'research_ready': False}
