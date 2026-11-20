from datetime import date

import pandas as pd

from data.annual_audit import annual_diagnostics, audit_years


def frame():
    return pd.DataFrame({'date': ['2020-01-02', '2020-01-06'], 'open': [10., 10.],
                         'high': [11., 11.], 'low': [9., 9.], 'close': [10., 12.],
                         'volume': [100., 100.], 'amount': [1000., 1000.], 'adjclose': [10., 12.]})


def test_gaps_are_unexplained_and_years_are_separate():
    calendar = pd.to_datetime(['2020-01-02', '2020-01-03', '2020-01-06'])
    result = annual_diagnostics(frame(), calendar, date(2019, 12, 31), date(2020, 1, 6))
    assert result['2019']['status'] == 'absent'
    entry = result['2020']
    assert entry['issues']['invalid_price_range'] == 1
    assert entry['unexplained_absent_sessions'] == 1
    assert entry['absent_session_ranges'] == [{'start': '2020-01-03', 'end': '2020-01-03', 'sessions': 1}]
    assert 'lifecycle' in entry['classification']


def test_audit_keeps_originals_and_skips_unknown_calendar(tmp_path):
    cache = tmp_path / 'cache'
    output = cache / 'preparation'
    (cache / 'kline').mkdir(parents=True)
    (output / 'wind').mkdir(parents=True)
    for market in ('SSE', 'SZSE'):
        pd.DataFrame({'date': pd.to_datetime(['2020-01-02', '2020-01-03', '2020-01-06'])}).to_parquet(output / 'wind' / f'calendar_{market}.parquet')
    path = cache / 'kline' / '600000_SH.parquet'
    frame().to_parquet(path)
    original = path.read_bytes()
    frame().to_parquet(cache / 'kline' / '430001_BJ.parquet')
    result = audit_years(cache, output, date(2020, 1, 1), date(2020, 1, 6))
    assert result['years']['2020']['status_counts'] == {'invalid': 1}
    assert result['skipped_by_exchange'] == {'BJ': 1}
    assert path.read_bytes() == original
    assert (output / 'annual' / '2020.json').exists()
