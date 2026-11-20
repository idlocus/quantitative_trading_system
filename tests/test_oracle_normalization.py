import pandas as pd
import pytest

from data.oracle_normalization import COLUMNS, normalize_a_frame, normalize_downloads
from data.preparation import atomic_json
from data.wind_preparation import save_frame


def sample():
    values = {column: [1.] for column in COLUMNS}
    values.update({'S_INFO_WINDCODE': ['600000.SH'], 'TRADE_DT': ['20200102'], 'CRNCY_CODE': ['CNY'],
                   'S_DQ_OPEN': [10.], 'S_DQ_HIGH': [11.], 'S_DQ_LOW': [9.], 'S_DQ_CLOSE': [10.],
                   'S_DQ_ADJCLOSE': [20.], 'S_DQ_ADJFACTOR': [2.], 'S_DQ_TRADESTATUS': ['trading']})
    return pd.DataFrame(values)


def test_raw_basis_and_units_are_explicit():
    source = sample()
    frame, quality = normalize_a_frame(source)
    assert frame.open.iloc[0] == 10.
    assert frame.adjclose.iloc[0] == 20.
    assert frame.volume.iloc[0] == 100.
    assert frame.amount.iloc[0] == 1000.
    assert source.S_DQ_VOLUME.iloc[0] == 1.
    assert quality['unsuspended_numeric_invalid_rows'] == 0


def test_source_suspension_does_not_fabricate_prices():
    source = sample()
    source['S_DQ_OPEN'] = float('nan')
    source['S_DQ_TRADESTATUS'] = '\u505c\u724c'
    frame, quality = normalize_a_frame(source)
    assert pd.isna(frame.open.iloc[0])
    assert not frame.numeric_bar_valid.iloc[0]
    assert quality['source_suspended_rows'] == 1
    assert quality['unsuspended_numeric_invalid_rows'] == 0


def test_currency_duplicates_and_price_range_fail_closed():
    source = sample()
    source['CRNCY_CODE'] = 'USD'
    with pytest.raises(ValueError, match='currency'):
        normalize_a_frame(source)
    with pytest.raises(ValueError, match='duplicate'):
        normalize_a_frame(pd.concat([sample(), sample()]))
    source = sample()
    source['S_DQ_HIGH'] = 1.
    assert normalize_a_frame(source)[1]['unsuspended_issue_counts']['price_range_rows'] == 1


def test_normalized_bundle_verifies_raw_source_and_reuses(tmp_path, monkeypatch):
    path = tmp_path / 'oracle/ashare_prices/2020/01.parquet'
    record = save_frame(sample(), path, {'source': 'test'}, provider='WindOracle')
    atomic_json(tmp_path / 'oracle/ashare_prices/manifest.json', {'files': {'2020/01': record}})
    original = path.read_bytes()
    assert normalize_downloads(tmp_path)['years']['2020']['rows'] == 1
    import data.oracle_normalization as normalization
    monkeypatch.setattr(normalization, 'normalize_a_frame', lambda *args: pytest.fail('Unchanged data reprocessed'))
    assert normalize_downloads(tmp_path)['months'] == 1
    assert path.read_bytes() == original
    path.write_bytes(b'changed')
    with pytest.raises(ValueError, match='verification'):
        normalize_downloads(tmp_path)
