import json

import pandas as pd
import pytest

from data.preparation import prepare_local, validate_bars


def bars():
    return pd.DataFrame({'date': ['2020-01-02', '2020-01-03'], 'open': [10., 11.],
                         'high': [12., 12.], 'low': [9., 10.], 'close': [11., 11.],
                         'volume': [100., 0.], 'amount': [1000., 0.], 'adjclose': [11., 11.]})


def test_valid_bars():
    assert validate_bars(bars())['status'] == 'validated'


@pytest.mark.parametrize('column,value,issue', [('high', 1., 'invalid_price_range'),
    ('volume', -1., 'negative_volume_or_amount'), ('close', float('inf'), 'nonfinite_values'),
    ('date', 'bad', 'invalid_dates'), ('adjclose', 0., 'nonpositive_adjusted_close')])
def test_invalid_bars(column, value, issue):
    frame = bars()
    frame.loc[0, column] = value
    result = validate_bars(frame)
    assert result['status'] == 'invalid'
    assert result['issues'][issue] > 0


def test_resume_and_source_preservation(tmp_path, monkeypatch):
    cache = tmp_path / 'cache'
    (cache / 'kline').mkdir(parents=True)
    path = cache / 'kline' / '600000_SH.parquet'
    bars().to_parquet(path)
    original = path.read_bytes()
    output = tmp_path / 'prepared'
    assert prepare_local(cache, output)['summary'] == {'validated': 1}
    monkeypatch.setattr(pd, 'read_parquet', lambda *args: pytest.fail('Unchanged source reloaded'))
    assert prepare_local(cache, output)['summary'] == {'validated': 1}
    assert path.read_bytes() == original
    assert not json.loads((output / 'local_validation.json').read_text())['research_ready']


def test_missing_columns_and_empty():
    assert validate_bars(pd.DataFrame())['status'] == 'invalid'
    assert validate_bars(bars().iloc[:0])['status'] == 'invalid'
