"""Build independently versioned yearly staging files from verified raw patches."""
from collections import Counter, defaultdict
from datetime import date
import hashlib
import json
from pathlib import Path

import pandas as pd

from .preparation import atomic_json, validate_bars
from .readiness import classify_kline_name
from .wind_preparation import cached_response, save_frame


def reconcile_empty_patches(output: Path, year: int) -> dict:
    """Separate observed empty delisting-date quotes from unresolved provider values."""
    lifecycle = output / 'lifecycle'
    tables = []
    source_manifest = json.loads((lifecycle / 'manifest.json').read_text(encoding='utf-8'))
    for name, record in source_manifest['files'].items():
        path = lifecycle / name
        if not path.resolve().is_relative_to(lifecycle.resolve()) or cached_response(path, record['request']) is None:
            raise ValueError('Lifecycle source failed provenance verification')
        tables.append(pd.read_parquet(path))
    master = pd.concat(tables, ignore_index=True)
    root = output / 'repair' / str(year)
    patches = json.loads((root / 'patch_manifest.json').read_text(encoding='utf-8'))
    records = []
    for name, record in patches['files'].items():
        request = record['request']
        if request['validation']['status'] == 'validated':
            continue
        path = root / 'patches' / name
        if not path.resolve().is_relative_to((root / 'patches').resolve()) or cached_response(path, request) is None:
            raise ValueError('Patch source failed provenance verification')
        frame = pd.read_parquet(path)
        selected = master.loc[master['wind_code'] == request['code']]
        delist = pd.to_datetime(selected.iloc[0]['DELIST_DATE'], errors='coerce') if len(selected) == 1 else pd.NaT
        trading_fields = ['open', 'high', 'low', 'volume', 'amount']
        missing_trading_bar = set(trading_fields).issubset(frame.columns) and frame[trading_fields].isna().all().all()
        on_delist = not pd.isna(delist) and request['start'] == request['end'] == delist.date().isoformat()
        reason = 'no_full_bar_on_delist_effective_date' if missing_trading_bar and on_delist else 'requires_market_status_review'
        records.append({'file': name, 'code': request['code'], 'start': request['start'],
                        'end': request['end'], 'classification': reason})
    result = {'year': year, 'counts': dict(Counter(item['classification'] for item in records)),
              'records': records, 'research_ready': False,
              'note': 'Observed source absence and lifecycle coincidence; not a general inference that all empty prices are valid.'}
    atomic_json(root / 'reconciliation.json', result)
    return {key: value for key, value in result.items() if key != 'records'}


def stage_year(cache: Path, output: Path, year: int, cutoff: date) -> dict:
    """Stage replacements only when hashes and numeric validation both succeed."""
    manifest_path = output / 'repair' / str(year) / 'patch_manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    root = manifest_path.parent
    by_code = defaultdict(list)
    rejected = []
    for name, record in manifest['files'].items():
        path = root / 'patches' / name
        if not path.resolve().is_relative_to((root / 'patches').resolve()):
            raise ValueError('Patch escapes staging root')
        request = record['request']
        if cached_response(path, request) is None:
            raise ValueError('Patch source hash or provenance mismatch')
        # 审计 2026-11-19 四-e：三方比对——实际文件哈希必须与**上层清单**
        # 记录一致（此前只核单文件旁的 sidecar：补丁与 sidecar 一同更新、
        # 上层清单未同步时，输出会引用旧哈希）。
        with path.open('rb') as stream:
            actual = hashlib.file_digest(stream, 'sha256').hexdigest()
        if actual != record.get('sha256'):
            raise ValueError(
                f'Patch {name}: 文件哈希与 staging 清单不一致'
                f'（sidecar 与上层清单需三方同步）')
        if request['validation']['status'] != 'validated':
            rejected.append(name)
            continue
        patch = pd.read_parquet(path)
        patch['adjclose'] = patch['close'] * patch['adjfactor']
        if validate_bars(patch)['status'] != 'validated':
            rejected.append(name)
            continue
        patch['row_source'] = 'wind_raw_patch'
        by_code[request['code']].append((patch, {'path': name, 'sha256': record['sha256']}))
    files = {}
    staged_codes = set()
    seen_cache_codes = set()   # 二轮复审#38：原缓存文件存在的证券（即使年度切片为空）
    upper = min(cutoff, date(year, 12, 31))
    target = output / 'yearly' / str(year)
    for path in sorted((cache / 'kline').glob('*.parquet')):
        if classify_kline_name(path.stem) != 'a_stock_filename' or not path.stem.endswith(('_SH', '_SZ')):
            continue
        if not path.resolve().is_relative_to(cache.resolve()):
            raise ValueError('Source path escapes cache')
        with path.open('rb') as stream:
            original_hash = hashlib.file_digest(stream, 'sha256').hexdigest()
        original = pd.read_parquet(path)
        original['date'] = pd.to_datetime(original['date'], errors='coerce', format='mixed').dt.normalize()
        invalid_source_dates = int(original['date'].isna().sum())
        frame = original.loc[original['date'].between(pd.Timestamp(date(year, 1, 1)), pd.Timestamp(upper))].copy()
        frame['row_source'] = 'retained_cache_unverified_basis'
        code = path.stem.replace('_', '.')
        seen_cache_codes.add(code)
        sources = []
        for patch, source in by_code[code]:
            if not patch['date'].between(pd.Timestamp(date(year, 1, 1)), pd.Timestamp(upper)).all():
                raise ValueError('Patch dates outside requested year or cutoff')
            frame = pd.concat([frame.loc[~frame['date'].isin(patch['date'])], patch], ignore_index=True)
            sources.append(source)
        if frame.empty:
            continue
        frame = frame.sort_values('date').reset_index(drop=True)
        checks = validate_bars(frame)
        record = save_frame(frame, target / path.name, {'source_cache': path.name, 'source_sha256': original_hash,
            'patches': sources, 'year': year, 'cutoff': cutoff.isoformat(), 'validation': checks,
            'invalid_source_dates': invalid_source_dates,
            'adjustment': 'Patch adjclose is raw close times cumulative adjfactor; retained-cache basis remains unverified.',
            'scope': 'Staging only; not approved for research or execution.'})
        files[path.name] = record
        staged_codes.add(code)
    # 审计 2026-11-19 四-e：只有有效补丁、没有原缓存文件的证券此前被静默
    # 跳过——现在显式列出（未纳入+原因）。
    # 二轮复审#38：原文件存在且**本年度无任何补丁**的证券不得列为孤儿
    # （patch_files 为空的误报会让维护者寻找不存在的补丁）。
    orphans = [{'code': code,
                'patch_files': [source['path'] for _, source in by_code[code]],
                'reason': 'no_original_cache_file'}
               for code in sorted(by_code)
               if code not in staged_codes
               and code not in seen_cache_codes and by_code.get(code)]
    result = {'year': year, 'cutoff': cutoff.isoformat(), 'files': files,
              'patches_without_cache_file': orphans,
              'rejected_patches': rejected,
              'status_counts': dict(Counter(item['request']['validation']['status'] for item in files.values())),
              'research_ready': False,
              'remaining': ['Full raw-price provenance, lifecycle/calendar completeness, suspensions, corporate actions, BJ and southbound coverage.']}
    atomic_json(target / 'manifest.json', result)
    return {key: value for key, value in result.items() if key != 'files'}
