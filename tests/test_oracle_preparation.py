from types import SimpleNamespace
from datetime import date

import pandas as pd

import pytest

from data.oracle_preparation import HK_PRICE_COLUMNS, PRICE_COLUMNS, OracleReader, connect_readonly, connection_settings, download_a_prices, download_hk_candidates, inspect_coverage, month_windows, prepare_supporting, safe_error
from data.wind_preparation import save_frame


def test_settings_do_not_load_other_secrets_or_mutate_environment(tmp_path):
    path = tmp_path / '.env'
    path.write_text('ORACLE_DSN=database\nORACLE_USER=reader\nORACLE_PASSWORD=test-secret\nOTHER_SECRET=unrelated\n')
    environment = {'ORACLE_USER': 'override'}
    settings = connection_settings(path, environment)
    assert settings == {'dsn': 'database', 'user': 'override', 'password': 'test-secret'}
    assert environment == {'ORACLE_USER': 'override'}
    assert 'unrelated' not in str(settings)


def test_incomplete_settings_and_safe_errors(tmp_path):
    with pytest.raises(ValueError):
        connection_settings(tmp_path / '.env', {})
    exc = RuntimeError(SimpleNamespace(code=12541, message='private details'))
    assert safe_error(exc) == {'error_type': 'RuntimeError', 'oracle_code': 12541}
    assert 'private' not in str(safe_error(exc))


class Cursor:
    def __init__(self, statements):
        self.statements = statements
        self.description = [('TRADE_DT',)]
        self.fetched = False

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def execute(self, sql, parameters=None):
        self.statements.append((sql, parameters))

    def fetchall(self):
        return [('test',)]

    def fetchmany(self, size):
        if self.fetched:
            return []
        self.fetched = True
        return [('20200102',)]


class Connection:
    def __init__(self):
        self.statements = []
        self.closed = False

    def cursor(self):
        return Cursor(self.statements)

    def close(self):
        self.closed = True


@pytest.mark.parametrize('sql', ['DELETE FROM T', 'UPDATE T SET A=1', 'SELECT * FROM T FOR UPDATE', 'SELECT 1 FROM DUAL; DELETE FROM T'])
def test_reader_rejects_writes(sql):
    connection = Connection()
    reader = OracleReader(connection)
    with pytest.raises(ValueError):
        reader.rows(sql)
    assert len(connection.statements) == 1
    assert connection.statements[0][0] == 'SET TRANSACTION READ ONLY'


def test_connection_closes_after_failure():
    connection = Connection()
    driver = SimpleNamespace(is_thin_mode=lambda: False, connect=lambda **kwargs: connection)
    with pytest.raises(RuntimeError):
        with connect_readonly({}, driver) as reader:
            assert reader.columns('ASHAREEODPRICES') == ['TRADE_DT']
            raise RuntimeError('test')
    assert connection.closed


def test_bounded_frame_and_empty_limit():
    reader = OracleReader(Connection())
    assert reader.frame('SELECT TRADE_DT FROM T')['TRADE_DT'].tolist() == ['20200102']
    with pytest.raises(ValueError, match='row bound'):
        reader.frame('SELECT TRADE_DT FROM T', max_rows=0)


def test_month_windows_are_exclusive_and_cut_off():
    assert month_windows([2020], date(2020, 1, 3)) == [(date(2020, 1, 1), date(2020, 1, 4))]
    assert month_windows([2020, 2020], date(2020, 12, 31))[-1] == (date(2020, 12, 1), date(2021, 1, 1))


def test_monthly_extract_provenance_and_resume(tmp_path):
    class Reader:
        calls = 0

        def columns(self, table):
            return PRICE_COLUMNS

        def frame(self, sql, parameters, max_rows):
            self.calls += 1
            assert ':start_dt' in sql
            values = {column: [1.] for column in PRICE_COLUMNS}
            values['S_INFO_WINDCODE'] = ['600000.SH']
            values['TRADE_DT'] = ['20200102']
            return pd.DataFrame(values)
    reader = Reader()
    result = download_a_prices(reader, tmp_path, [2020], date(2020, 1, 3))
    assert result['rows'] == 1
    download_a_prices(reader, tmp_path, [2020], date(2020, 1, 3))
    assert reader.calls == 1
    import json
    record = json.loads((tmp_path / 'oracle/ashare_prices/2020/01.json').read_text())
    assert record['provider'] == 'WindOracle'
    assert record['request']['parameters'] == {'start_dt': '20200101', 'end_dt': '20200104'}


def test_coverage_is_explicitly_sample_scoped(tmp_path):
    class Reader:
        def rows(self, sql, parameters=None):
            if 'ALL_TAB_COLUMNS' in sql or 'ALL_COL_COMMENTS' in sql:
                return []
            if 'HKSHAREDESCRIPTION' in sql:
                return [('00001.HK',)]
            assert parameters is not None and 'code' in parameters
            return [('20200102',)]
    result = inspect_coverage(Reader(), tmp_path)
    assert len(result['sample_coverage']) == 2
    assert all('sample' in value['scope'] for value in result['sample_coverage'].values())


@pytest.mark.parametrize('flags', [['1', '1', '0'], [1, 1, 0]])
def test_hk_download_is_bound_to_saved_current_ordinary_share_snapshot(tmp_path, flags):
    master = pd.DataFrame({'S_INFO_WINDCODE': ['00001.HK', '00002.HK', '00003.HK'],
                          'SECURITYTYPE': ['ORD', 'WAR', 'ORD'], 'IS_HKSC': flags})
    save_frame(master, tmp_path / 'oracle/reference/HKSHAREDESCRIPTION.parquet', {'source': 'test'})
    class Reader:
        calls = 0

        def columns(self, table):
            return HK_PRICE_COLUMNS

        def frame(self, sql, parameters, max_rows):
            self.calls += 1
            assert parameters['code_0'] == '00001.HK'
            assert 'code_1' not in parameters
            values = {column: [1.] for column in HK_PRICE_COLUMNS}
            values['S_INFO_WINDCODE'] = ['00001.HK']
            values['TRADE_DT'] = ['20200102']
            return pd.DataFrame(values)
    reader = Reader()
    result = download_hk_candidates(reader, tmp_path, [2020], date(2020, 1, 3))
    assert result['candidate_count'] == result['rows'] == 1
    assert result['historical_eligibility_verified'] is False
    download_hk_candidates(reader, tmp_path, [2020], date(2020, 1, 3))
    assert reader.calls == 1


def test_supporting_dates_and_carry_in_are_separate(tmp_path):
    class Reader:
        calls = []

        def frame(self, sql, parameters=None, max_rows=500000):
            self.calls.append((sql, parameters))
            return pd.DataFrame({'S_INFO_WINDCODE': ['600000.SH']})
    reader = Reader()
    result = prepare_supporting(reader, tmp_path, 2020, date(2020, 1, 3))
    assert result['rows']['suspensions/carry_in'] == 1
    assert any('S_DQ_RESUMPDATE IS NULL' in sql for sql, params in reader.calls)
    assert (tmp_path / 'oracle/supporting/suspensions/2020.parquet').exists()
