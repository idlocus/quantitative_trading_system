from datetime import date
from types import SimpleNamespace

import pandas as pd
import pytest

from data.year_repair import build_year_plan, defect_dates, fetch_year_patches, field_response, lifecycle_dates
from data.wind_preparation import save_frame
from data.preparation import atomic_json


def sample():
    return pd.DataFrame({'date': pd.to_datetime(['2020-01-02', '2020-01-03']),
        'open': [10., 100.], 'high': [11., 110.], 'low': [9., 90.], 'close': [10., 10.],
        'volume': [100., 100.], 'amount': [1000., 1000.], 'adjclose': [10., 100.]})


def test_defect_dates_and_lifecycle_sentinels():
    assert defect_dates(sample()).tolist() == [pd.Timestamp('2020-01-03')]
    master = pd.DataFrame({'wind_code': ['600000.SH'], 'IPO_DATE': [date(2020, 1, 2)], 'DELIST_DATE': [date(1899, 12, 30)]})
    assert lifecycle_dates(master, '600000.SH') == (pd.Timestamp('2020-01-02'), None)
    assert lifecycle_dates(master, 'missing') == (None, None)


def test_plan_clips_ipo_and_delist_and_preserves_cache(tmp_path):
    cache, output = tmp_path / 'cache', tmp_path / 'prepared'
    (cache / 'kline').mkdir(parents=True)
    (output / 'wind').mkdir(parents=True)
    for market in ('SSE', 'SZSE'):
        pd.DataFrame({'date': pd.to_datetime(['2020-01-01', '2020-01-02', '2020-01-03', '2020-01-06'])}).to_parquet(output / 'wind' / f'calendar_{market}.parquet')
    path = cache / 'kline' / '600000_SH.parquet'
    sample().to_parquet(path)
    original = path.read_bytes()
    master = pd.DataFrame({'wind_code': ['600000.SH'], 'IPO_DATE': [date(2020, 1, 2)], 'DELIST_DATE': [date(2020, 1, 3)]})
    record = save_frame(master, output / 'lifecycle' / 'chunk.parquet', {'method': 'fake'})
    atomic_json(output / 'lifecycle' / 'manifest.json', {'files': {'chunk.parquet': record}})
    plan = build_year_plan(cache, output, 2020, date(2020, 1, 6))
    assert plan['sessions'] == 1
    assert plan['intervals'][0]['start'] == plan['intervals'][0]['end'] == '2020-01-03'
    assert path.read_bytes() == original


def plan():
    return {'year': 2020, 'estimated_field_values': 7, 'intervals': [
        {'code': '600000.SH', 'start': '2020-01-03', 'end': '2020-01-03'}]}


def test_batch_download_raw_fields_and_resume(tmp_path):
    class Wind:
        calls = 0

        def wsd(self, codes, field, start, end, options):
            self.calls += 1
            values = {'open': 10., 'high': 11., 'low': 9., 'close': 10., 'volume': 100., 'amt': 1000., 'adjfactor': 2.}
            return SimpleNamespace(ErrorCode=0, Codes=codes.split(','), Data=[[values[field]]], Times=[date(2020, 1, 3)])
    wind = Wind()
    result = fetch_year_patches(wind, tmp_path, plan())
    assert result['status'] == 'downloaded'
    assert result['invalid_patches'] == 0
    fetch_year_patches(wind, tmp_path, plan())
    assert wind.calls == 7
    patch = pd.read_parquet(next((tmp_path / 'repair' / '2020' / 'patches').glob('*.parquet')))
    assert patch.close.iloc[0] == 10.
    assert patch.adjfactor.iloc[0] == 2.
    assert 'adjclose' not in patch.columns


def test_budget_and_provider_failure(tmp_path):
    class Wind:
        def wsd(self, *args):
            return SimpleNamespace(ErrorCode=-1, Codes=[])
    with pytest.raises(ValueError, match='budget'):
        fetch_year_patches(Wind(), tmp_path, plan(), max_values=1)
    result = fetch_year_patches(Wind(), tmp_path, plan())
    assert result['status'] == 'blocked'
    assert not (tmp_path / 'repair' / '2020' / 'patches').exists()


def test_wind_single_day_cross_section_and_multiple_day_layouts():
    codes = ['600000.SH', '600001.SH']
    response = SimpleNamespace(ErrorCode=0, Codes=codes, Data=[[10., 20.]], Times=[date(2020, 1, 3)])
    assert field_response(response, codes).Data == [[10.], [20.]]
    response.Times = [date(2020, 1, 2), date(2020, 1, 3)]
    response.Data = [[10., 11.], [20., 21.]]
    assert field_response(response, codes).Data == response.Data
    response.Data = [[10.]]
    with pytest.raises(ValueError, match='layout'):
        field_response(response, codes)
