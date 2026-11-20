"""Normalize verified Oracle A-share extracts without altering original source rows."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .preparation import atomic_json
from .readiness import classify_kline_name
from .wind_preparation import cached_response, save_frame


COLUMNS = {'S_INFO_WINDCODE': 'symbol', 'TRADE_DT': 'date', 'CRNCY_CODE': 'currency',
    'S_DQ_PRECLOSE': 'preclose', 'S_DQ_OPEN': 'open', 'S_DQ_HIGH': 'high',
    'S_DQ_LOW': 'low', 'S_DQ_CLOSE': 'close', 'S_DQ_VOLUME': 'volume',
    'S_DQ_AMOUNT': 'amount', 'S_DQ_ADJCLOSE': 'adjclose', 'S_DQ_ADJFACTOR': 'adjfactor',
    'S_DQ_TRADESTATUS': 'source_status', 'S_DQ_TRADESTATUSCODE': 'source_status_code',
    'S_DQ_LIMIT': 'source_upper_limit', 'S_DQ_STOPPING': 'source_lower_limit'}


def normalize_a_frame(source: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    if not set(COLUMNS).issubset(source.columns):
        raise ValueError('Required A-share source columns are missing')
    frame = source[list(COLUMNS)].rename(columns=COLUMNS).copy()
    if not frame['currency'].eq('CNY').all():
        raise ValueError('Unexpected A-share quotation currency')
    frame['date'] = pd.to_datetime(frame['date'], format='%Y%m%d', errors='coerce')
    if frame['date'].isna().any() or frame.duplicated(['symbol', 'date']).any():
        raise ValueError('Invalid or duplicate source security dates')
    numbers = ['open', 'high', 'low', 'close', 'volume', 'amount', 'adjclose', 'adjfactor',
               'preclose', 'source_upper_limit', 'source_lower_limit']
    frame[numbers] = frame[numbers].apply(pd.to_numeric, errors='coerce')
    frame['volume'] = frame['volume'] * 100
    frame['amount'] = frame['amount'] * 1000
    prices = frame[['open', 'high', 'low', 'close']]
    finite = pd.Series(np.isfinite(frame[['open', 'high', 'low', 'close', 'volume', 'amount', 'adjclose', 'adjfactor']].to_numpy(dtype=float)).all(axis=1), index=frame.index)
    positive = (frame[['open', 'high', 'low', 'close', 'adjclose', 'adjfactor']] > 0).all(axis=1)
    nonnegative = (frame[['volume', 'amount']] >= 0).all(axis=1)
    ranges = (prices['high'] >= prices.max(axis=1)) & (prices['low'] <= prices.min(axis=1))
    frame['numeric_bar_valid'] = finite & positive & nonnegative & ranges
    frame['source_suspended'] = frame['source_status'].eq('\u505c\u724c')
    unique_codes = frame['symbol'].dropna().unique()
    hints = {code: classify_kline_name(code.replace('.', '_')) for code in unique_codes}
    frame['instrument_type_hint'] = frame['symbol'].map(hints)
    unsuspended = ~frame['source_suspended']
    issues = {'nonfinite_rows': int((~finite & unsuspended).sum()),
              'nonpositive_price_or_factor_rows': int((~positive & unsuspended).sum()),
              'negative_volume_or_amount_rows': int((~nonnegative & unsuspended).sum()),
              'price_range_rows': int((~ranges & unsuspended).sum())}
    quality = {'rows': len(frame), 'symbols': len(unique_codes),
               'first_date': None if frame.empty else frame['date'].min().date().isoformat(),
               'last_date': None if frame.empty else frame['date'].max().date().isoformat(),
               'source_status_counts': {str(key): int(value) for key, value in frame['source_status'].value_counts(dropna=False).items()},
               'source_suspended_rows': int(frame['source_suspended'].sum()),
               'unsuspended_numeric_invalid_rows': int((~frame['numeric_bar_valid'] & unsuspended).sum()),
               'unsuspended_issue_counts': issues,
               'note': 'Numeric validity and filename hints do not imply historical eligibility or executable liquidity.'}
    return frame.sort_values(['symbol', 'date']).reset_index(drop=True), quality


def normalize_downloads(output: Path) -> dict:
    root = output / 'oracle' / 'ashare_prices'
    source_manifest = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
    records = {}
    summaries = {}
    for name, source in sorted(source_manifest['files'].items()):
        path = root / (name + '.parquet')
        if not path.resolve().is_relative_to(root.resolve()) or cached_response(path, source['request']) is None:
            raise ValueError('Raw source failed content or provenance verification')
        request = {'provider': 'WindOracle', 'schema_version': 1, 'source_sha256': source['sha256'],
                   'source_path': name + '.parquet', 'price_basis': 'raw',
                   'adjustment': 'Source cumulative adjfactor and adjclose are separate; OHLC is never rescaled.',
                   'volume_unit': 'shares', 'amount_unit': 'CNY'}
        destination = output / 'oracle' / 'normalized_a' / (name + '.parquet')
        record = cached_response(destination, request)
        quality_path = destination.with_suffix('.quality.json')
        if record is None or not quality_path.exists():
            frame, quality = normalize_a_frame(pd.read_parquet(path))
            record = save_frame(frame, destination, request, provider='WindOracle')
            atomic_json(quality_path, quality)
        else:
            quality = json.loads(quality_path.read_text(encoding='utf-8'))
        records[name] = record
        year = name.split('/')[0]
        summary = summaries.setdefault(year, {'months': 0, 'rows': 0, 'unsuspended_numeric_invalid_rows': 0,
                                              'source_suspended_rows': 0, 'first_date': None, 'last_date': None})
        for field in ('rows', 'source_suspended_rows', 'unsuspended_numeric_invalid_rows'):
            summary[field] += quality[field]
        summary['months'] += 1
        if quality['first_date'] is not None:
            summary['first_date'] = min(filter(None, [summary['first_date'], quality['first_date']]))
            summary['last_date'] = max(filter(None, [summary['last_date'], quality['last_date']]))
        atomic_json(output / 'oracle' / 'normalized_a' / 'manifest.json',
                    {'files': records, 'years': summaries, 'research_ready': False})
    return {'months': len(records), 'years': summaries, 'research_ready': False}
