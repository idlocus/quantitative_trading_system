import numpy as np
import pandas as pd

from research.signal_screen import daily_statistics, forward_labels, hac_summary, load_panel, mean_or_none, signal_values
from data.preparation import atomic_json
from data.wind_preparation import save_frame


def test_labels_use_next_close_and_fixed_market_horizon_without_filling():
    prices = pd.DataFrame({'stock': [10., 20., 30., 40., 50.]})
    labels = forward_labels(prices, 2)
    assert labels.iloc[0, 0] == 1.
    prices.iloc[3, 0] = np.nan
    assert np.isnan(forward_labels(prices, 2).iloc[0, 0])


def test_signals_do_not_change_when_future_is_perturbed():
    prices = pd.DataFrame({'stock': np.exp(np.arange(80) / 100)})
    baseline = signal_values(prices)
    prices.iloc[50:] *= 100
    altered = signal_values(prices)
    for name in baseline:
        pd.testing.assert_frame_equal(baseline[name].iloc[:50], altered[name].iloc[:50])


def test_cross_section_uses_only_eligible_pairs_and_counts_missing_labels():
    score = pd.DataFrame([[1., 2., 100., 4.]])
    labels = pd.DataFrame([[1., 2., -100., np.nan]])
    eligible = pd.DataFrame([[True, True, False, True]])
    result = daily_statistics(score, labels, eligible, 2)
    assert np.isclose(result.ic.iloc[0], 1.)
    assert result.paired_stocks.iloc[0] == 2
    assert result.missing_forward_labels.iloc[0] == 1


def test_hac_uncertainty_accounts_for_positive_serial_dependence():
    values = pd.Series(np.repeat([-.1, .2, -.1, .2], 40))
    independent = hac_summary(values, 0)
    corrected = hac_summary(values, 5)
    assert corrected['mean_ic'] == independent['mean_ic']
    assert abs(corrected['hac_t_stat']) < abs(independent['hac_t_stat'])
    assert hac_summary(pd.Series(dtype=float), 5)['mean_ic'] is None


def test_empty_statistics_are_explicit_not_nonfinite_json():
    assert mean_or_none(pd.Series([np.nan])) is None
    assert mean_or_none(pd.Series([1., 2.])) == 1.5


def test_loader_lifecycle_st_and_holdout_isolation(tmp_path):
    reference = tmp_path / 'oracle/reference'
    save_frame(pd.DataFrame({'S_INFO_WINDCODE': ['600000.SH'], 'S_INFO_LISTDATE': ['20160106'],
        'S_INFO_DELISTDATE': [None]}), reference / 'ASHAREDESCRIPTION.parquet', {})
    days = pd.bdate_range('2016-01-04', periods=10)
    save_frame(pd.DataFrame({'TRADE_DAYS': days.strftime('%Y%m%d'), 'S_INFO_EXCHMARKET': 'SSE'}),
               reference / 'ASHARECALENDAR.parquet', {})
    save_frame(pd.DataFrame({'S_INFO_WINDCODE': ['600000.SH'], 'S_TYPE_ST': ['S'],
        'ENTRY_DT': ['20160107'], 'REMOVE_DT': ['20160108']}), reference / 'ASHAREST.parquet', {})
    source = pd.DataFrame({'date': days, 'symbol': '600000.SH', 'adjclose': 10.,
        'amount': 30000000., 'volume': 100000., 'numeric_bar_valid': True, 'source_suspended': False})
    base = tmp_path / 'oracle/normalized_a'
    record = save_frame(source, base / '2016/01.parquet', {})
    atomic_json(base / 'manifest.json', {'files': {'2016/01': record, '2024/01': {'sha256': 'not-read'}}})
    panel, evidence = load_panel(tmp_path, 'a', {'start': '2016-01-01', 'validation_end': '2023-12-31', 'exclude_a_st_types': ['S']})
    assert not panel['usable'].loc['2016-01-04', '600000.SH']
    assert np.isnan(panel['price'].loc['2016-01-04', '600000.SH'])
    assert not panel['usable'].loc['2016-01-07', '600000.SH']
    assert panel['usable'].loc['2016-01-08', '600000.SH']
    assert '2024/01' not in evidence['source_files']
    assert evidence['outside_lifecycle_price_rows_removed'] == 2
