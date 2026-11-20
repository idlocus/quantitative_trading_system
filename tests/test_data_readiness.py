"""Readiness audits must remain offline, aggregate-only and safe on bad files."""
import json

import pandas as pd
import pytest

from application.cli import main
from application.workspace import Workspace
from data.readiness import audit_local_readiness, classify_kline_name, parquet_summary


@pytest.mark.parametrize('name,expected', [
    ('600000_SH', 'a_stock_filename'), ('000001_SZ', 'a_stock_filename'),
    ('920001_BJ', 'a_stock_filename'), ('000300_SH', 'non_stock_or_unclassified'),
    ('510300_SH', 'non_stock_or_unclassified'), ('00700_HK', 'hk_unverified_eligibility'),
    ('../../.env', 'unclassified')])
def test_inventory_names_never_imply_stock_connect_eligibility(name, expected):
    assert classify_kline_name(name) == expected


def test_audit_uses_metadata_without_loading_or_modifying_price_rows(tmp_path, monkeypatch):
    folder = tmp_path / 'kline'
    folder.mkdir()
    frame = pd.DataFrame({'date': pd.to_datetime(['2025-01-02', '2025-01-03']),
        'open': [10., 11.], 'high': [10., 11.], 'low': [10., 11.],
        'close': [10., 11.], 'volume': [100., 200.], 'adjclose': [20., 22.]})
    path = folder / '600000_SH.parquet'
    frame.to_parquet(path, index=False)
    before = (path.read_bytes(), path.stat().st_mtime_ns)
    monkeypatch.setattr(pd, 'read_parquet', lambda *args, **kwargs: pytest.fail('Full data load attempted'))
    report = audit_local_readiness(tmp_path)
    group = report['filename_groups']['a_stock_filename']
    assert group['files'] == 1 and group['rows'] == 2
    assert group['with_adjclose_column'] == 1 and group['with_amount_column'] == 0
    assert group['first_date'] == '2025-01-02' and group['last_date'] == '2025-01-03'
    assert not report['errors']
    assert (path.read_bytes(), path.stat().st_mtime_ns) == before
    assert '600000' not in json.dumps(report)


def test_missing_statistics_are_not_reported_as_verified_coverage(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    path = tmp_path / 'frame.parquet'
    frame = pd.DataFrame({'date': pd.to_datetime(['2025-01-02']), 'close': [10.]})
    pq.write_table(pa.Table.from_pandas(frame), path, write_statistics=False)
    result = parquet_summary(path)
    assert not result['date_statistics_complete']
    assert result['first_date'] is None and result['last_date'] is None


def test_corrupt_cache_is_reported_instead_of_crashing(tmp_path):
    folder = tmp_path / 'kline'
    folder.mkdir()
    (folder / '600000_SH.parquet').write_bytes(b'not-a-parquet-file')
    report = audit_local_readiness(tmp_path)
    assert report['errors'] and report['primary_kline_files'] == 1


def test_cli_audit_handles_an_empty_cache_without_creating_directories(tmp_path, capsys):
    assert main(['data', 'audit'], Workspace(tmp_path)) == 0
    report = json.loads(capsys.readouterr().out)
    assert report['mode'] == 'offline_metadata_only'
    assert report['primary_kline_files'] == 0
    assert not list(tmp_path.iterdir())
