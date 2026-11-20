"""Read-only Oracle preparation with credential-safe diagnostics."""
import hashlib
import json
from contextlib import contextmanager
from datetime import date, datetime, timedelta
from pathlib import Path
import re

from dotenv import dotenv_values
import pandas as pd

from .preparation import atomic_json
from .wind_preparation import cached_response, save_frame


def connection_settings(path: Path, environment: dict) -> dict:
    """Use only authorized Oracle keys; never mutate the process environment."""
    keys = ('ORACLE_DSN', 'ORACLE_USER', 'ORACLE_PASSWORD')
    local = dotenv_values(path, interpolate=False) if path.exists() else {}
    values = {key: environment.get(key) or local.get(key) for key in keys}
    if not all(values.values()):
        raise ValueError('Oracle connection settings are incomplete')
    return {'dsn': values['ORACLE_DSN'], 'user': values['ORACLE_USER'], 'password': values['ORACLE_PASSWORD']}


def safe_error(exc: Exception) -> dict:
    """Return error classes and numeric Oracle codes, never raw exception messages."""
    detail = exc.args[0] if exc.args else None
    code = getattr(detail, 'code', None)
    result = {'error_type': type(exc).__name__, 'oracle_code': code if isinstance(code, int) else None}
    known = {'Extract exceeded its row bound', 'Current HK candidate snapshot is empty',
             'HK master snapshot failed provenance verification', 'Required HK native price fields are absent',
             'Candidate universe exceeds this bounded Oracle IN-list adapter',
             'HK source returned dates outside the requested interval',
             'HK source returned unexpected or duplicate securities', 'Oracle connection settings are incomplete'}
    if isinstance(exc, ValueError) and str(exc) in known:
        result['reason'] = str(exc)
    return result


class OracleReader:
    """Restrict application queries to SELECT within a read-only transaction."""
    def __init__(self, connection):
        self.connection = connection
        self.connection.call_timeout = 30000
        with self.connection.cursor() as cursor:
            cursor.execute('SET TRANSACTION READ ONLY')

    @contextmanager
    def query(self, sql: str, parameters: dict | None = None):
        if not re.match(r'^\s*SELECT\b', sql, re.IGNORECASE) or ';' in sql or re.search(r'\bFOR\s+UPDATE\b', sql, re.IGNORECASE):
            raise ValueError('Only standalone read-only SELECT statements are allowed')
        with self.connection.cursor() as cursor:
            cursor.arraysize = 2000
            cursor.execute(sql, parameters or {})
            yield cursor

    def rows(self, sql: str, parameters: dict | None = None) -> list:
        with self.query(sql, parameters) as cursor:
            return cursor.fetchall()

    def columns(self, table: str) -> list[str]:
        if not re.fullmatch(r'[A-Z][A-Z0-9_]*', table):
            raise ValueError('Unsupported table identifier')
        with self.query(f'SELECT * FROM {table} WHERE 1=0') as cursor:
            return [column[0] for column in cursor.description]

    def frame(self, sql: str, parameters: dict | None = None, max_rows: int = 500000) -> pd.DataFrame:
        with self.query(sql, parameters) as cursor:
            columns = [item[0] for item in cursor.description]
            batches = []
            count = 0
            while batch := cursor.fetchmany(20000):
                count += len(batch)
                if count > max_rows:
                    raise ValueError('Extract exceeded its row bound')
                batches.append(pd.DataFrame.from_records(batch, columns=columns))
            return pd.concat(batches, ignore_index=True) if batches else pd.DataFrame(columns=columns)


@contextmanager
def connect_readonly(settings: dict, driver):
    if driver.is_thin_mode():
        driver.init_oracle_client()
    connection = driver.connect(**settings, tcp_connect_timeout=10)
    try:
        yield OracleReader(connection)
    finally:
        connection.close()


def probe_database(reader: OracleReader, output: Path) -> dict:
    """Discover accessible dataset names and schemas without persisting login details."""
    tables = reader.rows("""SELECT DISTINCT TABLE_NAME FROM ALL_TABLES
        WHERE TABLE_NAME LIKE 'ASHARE%' OR TABLE_NAME LIKE 'HKSHARE%'
        OR TABLE_NAME LIKE 'HSHARE%' OR TABLE_NAME LIKE '%CONNECT%'
        OR TABLE_NAME LIKE '%CALENDAR%' OR TABLE_NAME LIKE '%EXCHANGERATE%'
        ORDER BY TABLE_NAME""")
    candidates = [row[0] for row in tables]
    schemas, errors = {}, {}
    for table in ('ASHAREEODPRICES', 'ASHAREDESCRIPTION', 'HKSHAREEODPRICES', 'HKSHAREDESCRIPTION'):
        try:
            schemas[table] = reader.columns(table)
        except Exception as exc:
            errors[table] = safe_error(exc)
    result = {'status': 'connected', 'read_only': True, 'candidate_tables': candidates,
              'schemas': schemas, 'table_errors': errors, 'research_ready': False}
    atomic_json(output / 'oracle' / 'probe.json', result)
    return result


def prepare_references(reader: OracleReader, output: Path) -> dict:
    """Download bounded reference tables and inspect temporal dataset schemas."""
    names = reader.rows("""SELECT DISTINCT TABLE_NAME FROM ALL_TABLES
        WHERE TABLE_NAME LIKE '%SHSC%' OR TABLE_NAME LIKE '%HKSC%'
        OR TABLE_NAME LIKE '%CONNECT%' OR TABLE_NAME LIKE '%EXCHANGERATE%'
        OR TABLE_NAME LIKE '%HSHAREDIV%' OR TABLE_NAME LIKE '%HKDIV%'
        ORDER BY TABLE_NAME""")
    discovered = [row[0] for row in names if re.fullmatch(r'[A-Z][A-Z0-9_]*', row[0])]
    inspected = {}
    targets = ('ASHARECALENDAR', 'HKEXCALENDAR', 'ASHAREST', 'ASHAREDIVIDEND',
               'ASHARETRADINGSUSPENSION', 'ASHAREINDUSTRIESCLASS', 'HKSHAREPARVALUELOTSIZE')
    for table in sorted(set(targets).union(discovered)):
        try:
            inspected[table] = reader.columns(table)
        except Exception as exc:
            inspected[table] = safe_error(exc)
    indexes = reader.rows("""SELECT INDEX_NAME, COLUMN_NAME, COLUMN_POSITION
        FROM ALL_IND_COLUMNS WHERE TABLE_NAME='ASHAREEODPRICES'
        ORDER BY INDEX_NAME, COLUMN_POSITION""")
    grouped = {}
    for name, column, position in indexes:
        grouped.setdefault(name, []).append(column)
    records = {}
    for table in ('ASHAREDESCRIPTION', 'HKSHAREDESCRIPTION', 'ASHARECALENDAR', 'HKEXCALENDAR',
                  'ASHAREST', 'ASHAREINDUSTRIESCLASS', 'HKSHAREPARVALUELOTSIZE',
                  'SHSCMEMBERS', 'SHSCSELLMEMBERS', 'ASHARETRADINGSUSPENSION', 'ASHAREDIVIDEND'):
        request = {'provider': 'WindOracle', 'table': table, 'selection': 'all_rows',
                   'limitation': 'Current source snapshot; availability time and historical use require validation.'}
        path = output / 'oracle' / 'reference' / f'{table}.parquet'
        try:
            record = cached_response(path, request)
            if record is None:
                frame = reader.frame(f'SELECT * FROM {table}')
                record = save_frame(frame, path, request, provider='WindOracle')
            records[table] = record
        except Exception as exc:
            records[table] = {'status': 'error', **safe_error(exc)}
    result = {'schemas': inspected, 'price_index_columns': list(grouped.values()),
              'reference_files': records, 'research_ready': False}
    atomic_json(output / 'oracle' / 'reference_manifest.json', result)
    return {'reference_rows': {table: record.get('rows') for table, record in records.items()},
            'errors': {table: record for table, record in records.items() if record['status'] == 'error'},
            'inspected_tables': len(inspected)}


PRICE_COLUMNS = ('S_INFO_WINDCODE', 'TRADE_DT', 'CRNCY_CODE', 'S_DQ_PRECLOSE',
    'S_DQ_OPEN', 'S_DQ_HIGH', 'S_DQ_LOW', 'S_DQ_CLOSE', 'S_DQ_VOLUME', 'S_DQ_AMOUNT',
    'S_DQ_ADJCLOSE', 'S_DQ_ADJFACTOR', 'S_DQ_TRADESTATUS', 'S_DQ_TRADESTATUSCODE',
    'S_DQ_LIMIT', 'S_DQ_STOPPING')


def month_windows(years: list[int], cutoff: date) -> list[tuple[date, date]]:
    windows = []
    for year in dict.fromkeys(years):
        if year < 1900 or year > cutoff.year:
            raise ValueError('Unsupported historical year')
        for month in range(1, 13):
            start = date(year, month, 1)
            end = date(year + 1, 1, 1) if month == 12 else date(year, month + 1, 1)
            if start <= cutoff:
                windows.append((start, min(end, cutoff + timedelta(days=1))))
    return windows


def download_a_prices(reader: OracleReader, output: Path, years: list[int], cutoff: date) -> dict:
    """Extract bounded monthly native rows using the verified TRADE_DT index."""
    import json
    available = set(reader.columns('ASHAREEODPRICES'))
    if not set(PRICE_COLUMNS).issubset(available):
        raise ValueError('Required raw A-share price fields are absent')
    root = output / 'oracle' / 'ashare_prices'
    manifest = root / 'manifest.json'
    previous = json.loads(manifest.read_text(encoding='utf-8')) if manifest.exists() else {}
    records = previous.get('files', {})
    sql = f'SELECT {", ".join(PRICE_COLUMNS)} FROM ASHAREEODPRICES WHERE TRADE_DT >= :start_dt AND TRADE_DT < :end_dt'
    for start, end in month_windows(years, cutoff):
        parameters = {'start_dt': start.strftime('%Y%m%d'), 'end_dt': end.strftime('%Y%m%d')}
        request = {'provider': 'WindOracle', 'table': 'ASHAREEODPRICES', 'columns': list(PRICE_COLUMNS),
                   'parameters': parameters, 'price_basis': 'Native unadjusted OHLC; adjusted close and cumulative factor remain separate.',
                   'volume_unit': 'lots_of_100_shares', 'amount_unit': 'thousand_CNY',
                   'scope': 'Native source rows, not an approved tradable universe.'}
        path = root / str(start.year) / f'{start.month:02d}.parquet'
        record = cached_response(path, request)
        if record is None:
            frame = reader.frame(sql, parameters, max_rows=300000)
            dates = pd.to_datetime(frame['TRADE_DT'], format='%Y%m%d', errors='coerce')
            if dates.isna().any() or not dates.between(pd.Timestamp(start), pd.Timestamp(end) - pd.Timedelta(days=1)).all():
                raise ValueError('Source returned dates outside the requested interval')
            if frame.duplicated(['S_INFO_WINDCODE', 'TRADE_DT']).any():
                raise ValueError('Source returned duplicate security dates')
            record = save_frame(frame, path, request, provider='WindOracle')
        records[f'{start.year}/{start.month:02d}'] = record
        atomic_json(manifest, {'files': records, 'research_ready': False, 'read_only': True})
    return {'months': len(records), 'rows': sum(record['rows'] for record in records.values()),
            'years': sorted({int(key.split('/')[0]) for key in records}), 'research_ready': False}


def inspect_coverage(reader: OracleReader, output: Path) -> dict:
    """Inspect sample coverage and source comments without assuming market direction."""
    candidates = reader.rows("""SELECT DISTINCT TABLE_NAME FROM ALL_TAB_COLUMNS
        WHERE COLUMN_NAME IN ('ENTRY_DT', 'REMOVE_DT', 'SEC_IND_CODE') ORDER BY TABLE_NAME""")
    names = [row[0] for row in candidates if re.fullmatch(r'[A-Z][A-Z0-9_]*', row[0])]
    relevant = [name for name in names if any(token in name for token in ('HK', 'SHSC', 'SZSC', 'CONSTIT', 'SECTOR', 'INDEXMEMBER'))]
    schemas = {}
    for table in relevant:
        try:
            schemas[table] = reader.columns(table)
        except Exception as exc:
            schemas[table] = safe_error(exc)
    comments = reader.rows("""SELECT TABLE_NAME, COLUMN_NAME, COMMENTS FROM ALL_COL_COMMENTS
        WHERE TABLE_NAME IN ('HKSHAREEODPRICES','SHSCMEMBERS','SHSCSELLMEMBERS')
        AND COMMENTS IS NOT NULL""")
    coverage = {}
    hk_codes = reader.rows("SELECT S_INFO_WINDCODE FROM HKSHAREDESCRIPTION WHERE SECURITYTYPE='ORD' AND IS_HKSC='1' AND ROWNUM<=1")
    requests = [('ASHAREEODPRICES', '000001.SZ')]
    if hk_codes:
        requests.append(('HKSHAREEODPRICES', hk_codes[0][0]))
    for table, code in requests:
        endpoints = []
        for direction in ('ASC', 'DESC'):
            rows = reader.rows(f'SELECT TRADE_DT FROM (SELECT TRADE_DT FROM {table} WHERE S_INFO_WINDCODE=:code ORDER BY TRADE_DT {direction}) WHERE ROWNUM<=1', {'code': code})
            endpoints.append(rows[0][0] if rows else None)
        coverage[table] = {'first_date': endpoints[0], 'last_date': endpoints[1], 'scope': 'One sample security only; not full dataset coverage.'}
    result = {'interval_schemas': schemas, 'source_comments': comments, 'sample_coverage': coverage,
              'research_ready': False}
    atomic_json(output / 'oracle' / 'coverage.json', result)
    return {'inspected_interval_tables': len(schemas), 'sample_coverage': coverage, 'source_comment_rows': len(comments)}


def prepare_supporting(reader: OracleReader, output: Path, start_year: int, cutoff: date) -> dict:
    """Preserve bounded history snapshots; never infer eligibility from concepts."""
    records = {}
    for table in ('ASHAREINDUSTRIESCODE', 'HKSHARECONSEPTION', 'HKSHARESWINDUSTRIESCLASS', 'HKSHAREST'):
        request = {'provider': 'WindOracle', 'table': table, 'selection': 'all_rows',
                   'limitation': 'Raw reference only; concepts are not approved connect membership.'}
        destination = output / 'oracle' / 'supporting' / f'{table}.parquet'
        try:
            record = cached_response(destination, request)
            if record is None:
                record = save_frame(reader.frame(f'SELECT * FROM {table}', max_rows=500000),
                                    destination, request, provider='WindOracle')
            records[table] = record
        except Exception as exc:
            records[table] = {'status': 'error', **safe_error(exc)}
    for year in range(start_year, cutoff.year + 1):
        start, end = date(year, 1, 1), min(date(year + 1, 1, 1), cutoff + timedelta(days=1))
        parameters = {'start_dt': start.strftime('%Y%m%d'), 'end_dt': end.strftime('%Y%m%d')}
        request = {'provider': 'WindOracle', 'table': 'ASHARETRADINGSUSPENSION', 'parameters': parameters,
                   'selection': 'Suspension entry dates within the window; intervals remain raw.'}
        destination = output / 'oracle' / 'supporting' / 'suspensions' / f'{year}.parquet'
        record = cached_response(destination, request)
        if record is None:
            frame = reader.frame('SELECT * FROM ASHARETRADINGSUSPENSION WHERE S_DQ_SUSPENDDATE >= :start_dt AND S_DQ_SUSPENDDATE < :end_dt', parameters, max_rows=300000)
            record = save_frame(frame, destination, request, provider='WindOracle')
        records[f'suspensions/{year}'] = record
        atomic_json(output / 'oracle' / 'supporting' / 'manifest.json', {'files': records, 'research_ready': False})
    carry_parameters = {'start_dt': date(start_year, 1, 1).strftime('%Y%m%d')}
    carry_request = {'provider': 'WindOracle', 'table': 'ASHARETRADINGSUSPENSION',
                     'parameters': carry_parameters, 'selection': 'Explicit resumption on or after the window start; unknown dates do not imply ongoing suspension.'}
    carry_path = output / 'oracle' / 'supporting' / 'suspensions' / 'carry_in.parquet'
    record = cached_response(carry_path, carry_request)
    if record is None:
        frame = reader.frame('SELECT * FROM ASHARETRADINGSUSPENSION WHERE S_DQ_SUSPENDDATE < :start_dt AND S_DQ_RESUMPDATE >= :start_dt', carry_parameters, max_rows=300000)
        record = save_frame(frame, carry_path, carry_request, provider='WindOracle')
    records['suspensions/carry_in'] = record
    unknown_request = {'provider': 'WindOracle', 'table': 'ASHARETRADINGSUSPENSION',
                       'parameters': carry_parameters, 'selection': 'Counts by source type of pre-window null resumption dates; no ongoing-interval inference.'}
    unknown_path = output / 'oracle' / 'supporting' / 'suspensions' / 'unknown_resume_counts.parquet'
    record = cached_response(unknown_path, unknown_request)
    if record is None:
        frame = reader.frame('SELECT S_DQ_SUSPENDTYPE, COUNT(*) AS RECORDS FROM ASHARETRADINGSUSPENSION WHERE S_DQ_SUSPENDDATE < :start_dt AND S_DQ_RESUMPDATE IS NULL GROUP BY S_DQ_SUSPENDTYPE', carry_parameters, max_rows=1000)
        record = save_frame(frame, unknown_path, unknown_request, provider='WindOracle')
    records['suspensions/unknown_resume_counts'] = record
    atomic_json(output / 'oracle' / 'supporting' / 'manifest.json', {'files': records, 'research_ready': False})
    return {'rows': {name: record.get('rows') for name, record in records.items()},
            'errors': {name: record for name, record in records.items() if record['status'] == 'error'},
            'research_ready': False}


HK_PRICE_COLUMNS = ('S_INFO_WINDCODE', 'TRADE_DT', 'CRNCY_CODE', 'S_DQ_PRECLOSE',
    'S_DQ_OPEN', 'S_DQ_HIGH', 'S_DQ_LOW', 'S_DQ_CLOSE', 'S_DQ_VOLUME', 'S_DQ_AMOUNT',
    'S_DQ_ADJCLOSE', 'S_DQ_ADJFACTOR')


def download_hk_candidates(reader: OracleReader, output: Path, years: list[int], cutoff: date) -> dict:
    """Collect current eligible ordinary-share history, explicitly excluding historical-universe claims."""
    import json
    if not set(HK_PRICE_COLUMNS).issubset(reader.columns('HKSHAREEODPRICES')):
        raise ValueError('Required HK native price fields are absent')
    root = output / 'oracle' / 'hk_current_candidates'
    manifest = root / 'manifest.json'
    previous = json.loads(manifest.read_text(encoding='utf-8')) if manifest.exists() else {}
    records = previous.get('files', {})
    master_path = output / 'oracle' / 'reference' / 'HKSHAREDESCRIPTION.parquet'
    master_record = json.loads(master_path.with_suffix('.json').read_text(encoding='utf-8'))
    if cached_response(master_path, master_record['request']) is None:
        raise ValueError('HK master snapshot failed provenance verification')
    master = pd.read_parquet(master_path)
    eligible_flag = pd.to_numeric(master['IS_HKSC'], errors='coerce').eq(1)
    codes = sorted(master.loc[master['SECURITYTYPE'].eq('ORD') & eligible_flag, 'S_INFO_WINDCODE'].dropna().unique())
    if not codes:
        raise ValueError('Current HK candidate snapshot is empty')
    # Bind the saved snapshot explicitly rather than re-evaluating a changing server flag.
    placeholders = ','.join(f':code_{index}' for index in range(len(codes)))
    if len(codes) > 900:
        raise ValueError('Candidate universe exceeds this bounded Oracle IN-list adapter')
    code_parameters = {f'code_{index}': code for index, code in enumerate(codes)}
    sql = f'SELECT {", ".join(HK_PRICE_COLUMNS)} FROM HKSHAREEODPRICES WHERE TRADE_DT >= :start_dt AND TRADE_DT < :end_dt AND S_INFO_WINDCODE IN ({placeholders})'
    for start, end in month_windows(years, cutoff):
        bounds = {'start_dt': start.strftime('%Y%m%d'), 'end_dt': end.strftime('%Y%m%d')}
        request = {'provider': 'WindOracle', 'table': 'HKSHAREEODPRICES', 'columns': list(HK_PRICE_COLUMNS),
                   'parameters': bounds, 'candidate_snapshot_sha256': master_record['sha256'],
                   'candidate_count': len(codes), 'price_basis': 'raw', 'volume_unit': 'shares',
                   'amount_unit': 'thousand_quote_currency',
                   'limitation': 'Current snapshot only. Not historical connect eligibility; excludes removed/delisted former members.'}
        path = root / str(start.year) / f'{start.month:02d}.parquet'
        record = cached_response(path, request)
        if record is None:
            frame = reader.frame(sql, {**bounds, **code_parameters}, max_rows=50000)
            dates = pd.to_datetime(frame['TRADE_DT'], format='%Y%m%d', errors='coerce')
            if dates.isna().any() or not dates.between(pd.Timestamp(start), pd.Timestamp(end) - pd.Timedelta(days=1)).all():
                raise ValueError('HK source returned dates outside the requested interval')
            if not frame['S_INFO_WINDCODE'].isin(codes).all() or frame.duplicated(['S_INFO_WINDCODE', 'TRADE_DT']).any():
                raise ValueError('HK source returned unexpected or duplicate securities')
            record = save_frame(frame, path, request, provider='WindOracle')
        records[f'{start.year}/{start.month:02d}'] = record
        atomic_json(manifest, {'files': records, 'candidate_count': len(codes), 'research_ready': False,
                              'historical_eligibility_verified': False})
    return {'months': len(records), 'rows': sum(record['rows'] for record in records.values()),
            'candidate_count': len(codes), 'historical_eligibility_verified': False, 'research_ready': False}


def download_financials(reader, output: Path, cutoff: str) -> dict:
    """拉取 AShareFinancialIndicator 质量字段（公告日 PIT 口径）。
    字段：证券/公告日/报告期/ROE/扣非ROE/资产负债率（2026-11-10 增厚路径B）。"""
    sql = """SELECT S_INFO_WINDCODE, ANN_DT, REPORT_PERIOD,
                    S_FA_ROE, S_FA_ROE_DEDUCTED, S_FA_DEBTTOASSETS
             FROM ASHAREFINANCIALINDICATOR
             WHERE ANN_DT IS NOT NULL AND ANN_DT <= :cutoff"""
    rows = reader.rows(sql, {'cutoff': cutoff.replace('-', '')})
    frame = pd.DataFrame(rows, columns=['symbol', 'ann_date', 'report_period',
                                        'roe', 'roe_deducted', 'debt_to_assets'])
    dest = output / 'oracle' / 'financials'
    dest.mkdir(parents=True, exist_ok=True)
    path = dest / 'ashare_financial_indicator.parquet'
    frame.to_parquet(path, index=False)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = {'files': {'financials/ashare_financial_indicator': {
        'sha256': digest, 'rows': len(frame)}},
        'cutoff': cutoff, 'downloaded_at': datetime.now().isoformat(timespec='seconds')}
    (dest / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding='utf-8')
    return {'rows': len(frame), 'path': str(path),
            'ann_date_range': [str(frame.ann_date.min()), str(frame.ann_date.max())]}
