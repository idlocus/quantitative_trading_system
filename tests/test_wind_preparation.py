from datetime import date
from types import SimpleNamespace

import pandas as pd

from data.wind_preparation import cached_response, prepare_lifecycle, prepare_market_status, prepare_reference, save_response
from data.preparation import atomic_json


def response(code=0):
    return SimpleNamespace(ErrorCode=code, Fields=['value'], Data=[[1]], Times=[date(2020, 1, 2)])


def test_save_success(tmp_path):
    target = tmp_path / 'sample.parquet'
    result = save_response(response(), target, {'method': 'test'})
    assert result['rows'] == 1
    assert result['status'] == 'downloaded_unverified'
    assert len(result['sha256']) == 64
    assert 'date' in pd.read_parquet(target)


def test_errors_never_persist(tmp_path):
    target = tmp_path / 'sample.parquet'
    assert save_response(response(-1), target, {})['status'] == 'error'
    malformed = response()
    malformed.Data = []
    assert save_response(malformed, target, {})['status'] == 'error'
    assert not target.exists()


def test_reference_requests_are_explicit(tmp_path):
    calls = []
    class Wind:
        def tdays(self, *args):
            calls.append(args)
            return response()

        def wset(self, *args):
            calls.append(args)
            return response()
    result = prepare_reference(Wind(), tmp_path, date(2020, 1, 1), date(2020, 1, 3))
    assert len(calls) == 4
    assert len(result['requests']) == 4
    assert result['research_ready'] is False


def test_static_codes_and_cache_identity(tmp_path):
    item = response()
    item.Codes = ['600000.SH']
    target = tmp_path / 'metadata.parquet'
    request = {'method': 'wss'}
    save_response(item, target, request, include_codes=True)
    assert pd.read_parquet(target)['wind_code'].tolist() == ['600000.SH']
    assert cached_response(target, request) is not None
    assert cached_response(target, {'method': 'changed'}) is None
    target.write_bytes(b'changed')
    assert cached_response(target, request) is None


def test_lifecycle_includes_retired_cache_and_reuses(tmp_path):
    (tmp_path / 'kline').mkdir()
    (tmp_path / 'kline' / '600000_SH.parquet').touch()
    (tmp_path / 'kline' / '000001_SH.parquet').touch()
    class Wind:
        calls = 0

        def wss(self, codes, fields, options):
            self.calls += 1
            item = response()
            item.Codes = codes.split(',')
            return item
    wind = Wind()
    output = tmp_path / 'prepared'
    result = prepare_lifecycle(wind, tmp_path, output)
    assert result['requested_codes'] == result['downloaded_rows'] == 1
    prepare_lifecycle(wind, tmp_path, output)
    assert wind.calls == 1


def test_market_status_requests_only_unresolved_intervals(tmp_path):
    atomic_json(tmp_path / 'repair' / '2020' / 'reconciliation.json', {'records': [
        {'code': '600000.SH', 'start': '2020-01-02', 'end': '2020-01-02', 'classification': 'requires_market_status_review'},
        {'classification': 'no_full_bar_on_delist_effective_date'}]})
    class Wind:
        calls = 0

        def wsd(self, code, fields, start, end, options):
            self.calls += 1
            assert fields == 'trade_status'
            return response()
    wind = Wind()
    assert prepare_market_status(wind, tmp_path, 2020)['requests'] == 1
    assert prepare_market_status(wind, tmp_path, 2020)['errors'] == 0
    assert wind.calls == 1
