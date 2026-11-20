from datetime import date
from types import SimpleNamespace

import pandas as pd

from data.connect_preparation import ROUTES, download_connect_daily
from data.wind_preparation import save_frame


def setup_sources(output):
    dictionary = pd.DataFrame({'INDUSTRIESNAME': [value[0] for value in ROUTES.values()],
                              'INDUSTRIESCODE': ['a_test_sh', 'a_test_sz']})
    calendar = pd.DataFrame({'TRADE_DAYS': ['20200102', '20200102'], 'S_INFO_EXCHMARKET': ['HKS', 'HKSZ']})
    save_frame(dictionary, output / 'oracle/supporting/ASHAREINDUSTRIESCODE.parquet', {})
    save_frame(calendar, output / 'oracle/reference/HKEXCALENDAR.parquet', {})


def test_exact_date_snapshots_and_resume(tmp_path):
    setup_sources(tmp_path)
    class Wind:
        calls = 0

        def wset(self, name, options):
            self.calls += 1
            assert 'date=2020-01-02' in options
            return SimpleNamespace(ErrorCode=0, Fields=['date', 'wind_code'],
                Data=[[date(2020, 1, 2)], ['00001.HK']], Times=[])
    wind = Wind()
    result = download_connect_daily(wind, tmp_path, date(2020, 1, 2), date(2020, 1, 2))
    assert result['snapshots'] == 2
    download_connect_daily(wind, tmp_path, date(2020, 1, 2), date(2020, 1, 2))
    assert wind.calls == 2


def test_wrong_asof_date_is_not_accepted(tmp_path):
    setup_sources(tmp_path)
    class Wind:
        def wset(self, *args):
            return SimpleNamespace(ErrorCode=0, Fields=['date', 'wind_code'],
                Data=[[date(2020, 1, 3)], ['00001.HK']], Times=[])
    assert download_connect_daily(Wind(), tmp_path, date(2020, 1, 2), date(2020, 1, 2))['status'] == 'blocked'
