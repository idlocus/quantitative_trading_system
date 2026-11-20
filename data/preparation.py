"""Resumable, read-only validation of cached daily bars."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import os
import tempfile

import numpy as np
import pandas as pd

from .readiness import classify_kline_name


VALIDATOR_VERSION = 3


def atomic_json(path: Path, value: dict) -> None:
    """Publish a complete checkpoint without exposing partially written JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, suffix='.tmp')
    try:
        with os.fdopen(descriptor, 'w', encoding='utf-8') as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def validate_bars(frame: pd.DataFrame) -> dict:
    """Report observable defects without inferring exchange calendars or eligibility."""
    required = {'date', 'open', 'high', 'low', 'close', 'volume', 'amount', 'adjclose'}
    missing = sorted(required.difference(frame.columns))
    if missing:
        return {'status': 'invalid', 'rows': len(frame), 'missing_columns': missing}
    dates = pd.to_datetime(frame['date'].astype('string'), errors='coerce', format='mixed')
    numbers = frame[list(required - {'date'})].apply(pd.to_numeric, errors='coerce')
    prices = numbers[['open', 'high', 'low', 'close']]
    issues = {
        'invalid_dates': int(dates.isna().sum()),
        'duplicate_dates': int(dates.duplicated().sum()),
        'nonfinite_values': int((~np.isfinite(numbers.to_numpy(dtype=float))).sum()),
        'nonpositive_prices': int((prices <= 0).any(axis=1).sum()),
        'nonpositive_adjusted_close': int((numbers['adjclose'] <= 0).sum()),
        'negative_volume_or_amount': int((numbers[['volume', 'amount']] < 0).any(axis=1).sum()),
        'invalid_price_range': int(((prices['high'] < prices.max(axis=1)) |
                                    (prices['low'] > prices.min(axis=1))).sum()),
        'date_order_errors': int((dates.diff().dropna() < pd.Timedelta(0)).sum()),
    }
    return {'status': 'invalid' if any(issues.values()) or frame.empty else 'validated',
            'rows': len(frame), 'first_date': None if dates.dropna().empty else dates.min().date().isoformat(),
            'last_date': None if dates.dropna().empty else dates.max().date().isoformat(),
            'issues': issues}


def prepare_local(cache: Path, output: Path) -> dict:
    """Hash every source, reuse matching validation, and never modify source caches."""
    cache, output = cache.resolve(), output.resolve()
    if output.is_relative_to(cache / 'kline'):
        raise ValueError('Preparation output must be outside original daily bars')
    checkpoint = output / 'local_validation.json'
    previous = json.loads(checkpoint.read_text(encoding='utf-8')) if checkpoint.exists() else {}
    records = previous.get('files', {}) if previous.get('validator_version') == VALIDATOR_VERSION else {}
    current = {}
    for index, path in enumerate(sorted((cache / 'kline').glob('*.parquet'))):
        if not path.resolve().is_relative_to(cache):
            current[path.name] = {'status': 'invalid', 'error': 'path_outside_cache'}
            continue
        try:
            before = path.stat()
            with path.open('rb') as stream:
                digest = hashlib.file_digest(stream, 'sha256').hexdigest()
            old = records.get(path.name, {})
            if old.get('sha256') == digest and old.get('status') in ('validated', 'invalid'):
                record = dict(old)
            else:
                record = validate_bars(pd.read_parquet(path))
                record['sha256'] = digest
            after = path.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                record = {'status': 'retry', 'error': 'source_changed_during_validation'}
            record['category'] = classify_kline_name(path.stem)
            current[path.name] = record
        except (OSError, ValueError, TypeError) as exc:
            current[path.name] = {'status': 'error', 'error_type': type(exc).__name__}
        if (index + 1) % 100 == 0:
            atomic_json(checkpoint, {'validator_version': VALIDATOR_VERSION, 'files': {**records, **current}})
    result = {'validator_version': VALIDATOR_VERSION,
              'checked_at': datetime.now(timezone.utc).isoformat(), 'files': current,
              'summary': dict(Counter(item['status'] for item in current.values())),
              'research_ready': False,
              'limitations': ['Calendar completeness, historical membership, units and provider provenance remain unverified.']}
    atomic_json(checkpoint, result)
    return {key: value for key, value in result.items() if key != 'files'}
