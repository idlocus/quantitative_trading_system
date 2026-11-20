"""Explicit static-current HK research universe with lifecycle boundaries."""
from datetime import date
import json
from pathlib import Path

import pandas as pd

from .preparation import atomic_json
from .wind_preparation import cached_response, save_frame


def research_symbols(frame: pd.DataFrame, as_of: date) -> tuple[str, ...]:
    """Apply lifecycle constraints; do not infer tradability or historical connect status."""
    if not {'symbol', 'list_date', 'delist_date'}.issubset(frame.columns):
        raise ValueError('Universe lifecycle columns are missing')
    if frame['symbol'].isna().any() or frame['symbol'].duplicated().any() or frame['list_date'].isna().any():
        raise ValueError('Universe symbols or listing dates are ambiguous')
    day = pd.Timestamp(as_of)
    active = (frame['list_date'] <= day) & (frame['delist_date'].isna() | (day < frame['delist_date']))
    return tuple(sorted(frame.loc[active, 'symbol']))


def build_static_universe(output: Path, policy_path: Path) -> dict:
    """Freeze an explicitly authorized approximation separately from verified facts."""
    policy = json.loads(policy_path.read_text(encoding='utf-8'))
    if (policy.get('hk_universe_mode') != 'current_connect_snapshot' or
            policy.get('historical_connect_membership_required') is not False or
            policy.get('scope') != 'research_only'):
        raise ValueError('Static current membership requires an explicit research policy')
    master_path = output / 'oracle/reference/HKSHAREDESCRIPTION.parquet'
    master_record = json.loads(master_path.with_suffix('.json').read_text(encoding='utf-8'))
    if cached_response(master_path, master_record['request']) is None:
        raise ValueError('Master snapshot failed provenance verification')
    frozen_path = output / 'hk_research_universe/manifest.json'
    if frozen_path.exists():
        frozen = json.loads(frozen_path.read_text(encoding='utf-8'))
        if frozen.get('source_master_sha256') != master_record['sha256']:
            raise ValueError('Frozen research universe cannot switch master snapshots implicitly')
    prices = json.loads((output / 'oracle/hk_current_candidates/manifest.json').read_text(encoding='utf-8'))
    if not prices.get('files'):
        raise ValueError('HK candidate price manifest is empty')
    if any(item['request'].get('candidate_snapshot_sha256') != master_record['sha256']
           for item in prices['files'].values()):
        raise ValueError('Candidate prices and universe use different snapshots')
    master = pd.read_parquet(master_path)
    selected = master['SECURITYTYPE'].eq('ORD') & pd.to_numeric(master['IS_HKSC'], errors='coerce').eq(1)
    source = master.loc[selected]
    frame = pd.DataFrame({'symbol': source['S_INFO_WINDCODE'],
        'list_date': pd.to_datetime(source['S_INFO_LISTDATE'], format='%Y%m%d', errors='coerce'),
        'delist_date': pd.to_datetime(source['S_INFO_DELISTDATE'], format='%Y%m%d', errors='coerce')})
    frame = frame.sort_values('symbol').reset_index(drop=True)
    if frame.empty or len(frame) != prices.get('candidate_count'):
        raise ValueError('Static universe differs from the downloaded candidate count')
    if not frame['symbol'].astype('string').str.endswith('.HK').all():
        raise ValueError('Static HK universe contains non-HK securities')
    research_symbols(frame, date.fromisoformat(policy['user_confirmed_on']))
    if ((frame['delist_date'].notna()) & (frame['delist_date'] <= frame['list_date'])).any():
        raise ValueError('Invalid listing and delisting interval')
    destination = output / 'hk_research_universe' / 'current_snapshot.parquet'
    record = save_frame(frame, destination, {'policy': policy, 'source_master_sha256': master_record['sha256'],
        'source_snapshot_retrieved_at': master_record['retrieved_at'],
        'membership_basis': 'assumed_static_current', 'lifecycle_interval': '[list_date, delist_date)'}, provider='DerivedResearchAssumption')
    result = {'mode': 'current_connect_snapshot', 'scope': 'research_only',
              'assumption_accepted': True, 'historical_eligibility_verified': False,
              'historical_eligibility_blocking': False, 'symbols': len(frame),
              'source_master_sha256': master_record['sha256'], 'policy': policy,
              'universe_file': record, 'research_ready': False}
    atomic_json(destination.parent / 'manifest.json', result)
    return {key: value for key, value in result.items() if key not in ('universe_file', 'policy', 'source_master_sha256')}
