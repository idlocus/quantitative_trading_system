"""Small preregistered signal screen; never represents an execution backtest."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from data.preparation import atomic_json
from data.readiness import classify_kline_name
from data.wind_preparation import cached_response


def signal_values(price: pd.DataFrame) -> dict[str, pd.DataFrame]:
    returns = price.pct_change(fill_method=None)
    return {'momentum_20': price / price.shift(20) - 1,
            'reversal_5': -(price / price.shift(5) - 1),
            'low_volatility_20': -returns.rolling(20, min_periods=20).std()}


def forward_labels(price: pd.DataFrame, horizon: int) -> pd.DataFrame:
    if horizon < 1:
        raise ValueError('Horizon must be positive')
    labels = price.shift(-(horizon + 1)) / price.shift(-1) - 1
    return labels.where(np.isfinite(labels))


def mean_or_none(values: pd.Series) -> float | None:
    result = values.mean()
    return float(result) if pd.notna(result) and np.isfinite(result) else None


def hac_summary(values: pd.Series, lags: int) -> dict:
    samples = values.dropna().to_numpy(dtype=float)
    if not len(samples):
        return {'days': 0, 'mean_ic': None, 'positive_fraction': None, 'hac_t_stat': None}
    centered = samples - samples.mean()
    variance = float(centered @ centered / len(samples))
    for lag in range(1, min(lags, len(samples) - 1) + 1):
        variance += 2 * (1 - lag / (lags + 1)) * float(centered[lag:] @ centered[:-lag] / len(samples))
    standard_error = np.sqrt(max(0., variance) / len(samples))
    return {'days': len(samples), 'mean_ic': float(samples.mean()),
            'positive_fraction': float((samples > 0).mean()),
            'hac_t_stat': float(samples.mean() / standard_error) if standard_error > 0 else None}


def daily_statistics(score: pd.DataFrame, labels: pd.DataFrame, eligible: pd.DataFrame,
                     minimum_count: int) -> pd.DataFrame:
    paired = eligible & score.notna() & labels.notna()
    x, y = score.where(paired), labels.where(paired)
    signal = score.where(eligible)
    n_group = np.ceil(signal.count(axis=1) / 10)
    top = signal.rank(axis=1, method='first', ascending=False).le(n_group, axis=0)
    bottom = signal.rank(axis=1, method='first').le(n_group, axis=0)
    rank = x.rank(axis=1, pct=True)
    with np.errstate(invalid='ignore', divide='ignore'):
        ic = rank.corrwith(y.rank(axis=1, pct=True), axis=1)
    count = paired.sum(axis=1)
    return pd.DataFrame({'ic': ic.where(count >= minimum_count), 'paired_stocks': count,
        'eligible_stocks': eligible.sum(axis=1),
        'missing_forward_labels': (eligible & score.notna() & labels.isna()).sum(axis=1),
        'top_decile_forward_mean': y.where(top).mean(axis=1).where(count >= minimum_count),
        'bottom_decile_forward_mean': y.where(bottom).mean(axis=1).where(count >= minimum_count)})


def verified_frame(path: Path, record: dict, columns=None) -> pd.DataFrame:
    if cached_response(path, record['request']) is None:
        raise ValueError('Research input failed provenance verification')
    # 二轮复审#13：三方比对——上层清单记录的哈希也必须与实际文件一致。
    # 此前只核"文件 vs 旁文件"：两者一同更新、上层清单未同步时，旧清单
    # 哈希仍被研究证据引用，实际读到了新数据。
    upper_sha = record.get('sha256') if isinstance(record, dict) else None
    if not upper_sha:
        raise ValueError('Missing research input digest')
    with path.open('rb') as stream:
        actual = hashlib.file_digest(stream, 'sha256').hexdigest()
        if actual != upper_sha:
            raise ValueError(f'Research input {path.name}: upper-manifest hash mismatch')
        stream.seek(0)
        return pd.read_parquet(stream, columns=columns)



def load_panel(output: Path, market: str, protocol: dict) -> tuple[dict, dict]:
    start, end = pd.Timestamp(protocol['start']), pd.Timestamp(protocol['validation_end'])
    oracle = output / 'oracle'
    if market == 'a':
        base = oracle / 'normalized_a'
        master_path = oracle / 'reference/ASHAREDESCRIPTION.parquet'
        record = json.loads(master_path.with_suffix('.json').read_text(encoding='utf-8'))
        master = verified_frame(master_path, record)
        reference_hashes = {'master': record['sha256']}
        keep = master['S_INFO_WINDCODE'].map(lambda value: classify_kline_name(value.replace('.', '_')) == 'a_stock_filename')
        master = master.loc[keep].rename(columns={'S_INFO_WINDCODE': 'symbol', 'S_INFO_LISTDATE': 'list_date', 'S_INFO_DELISTDATE': 'delist_date'})
        for column in ('list_date', 'delist_date'):
            master[column] = pd.to_datetime(master[column], format='%Y%m%d', errors='coerce')
        calendar_path = oracle / 'reference/ASHARECALENDAR.parquet'
        calendar_markets = ['SSE', 'SZSE', 'BSE']
    elif market == 'hk':
        base = oracle / 'hk_current_candidates'
        metadata = json.loads((output / 'hk_research_universe/manifest.json').read_text(encoding='utf-8'))
        if metadata.get('assumption_accepted') is not True:
            raise ValueError('HK static-sample assumption was not accepted')
        master = verified_frame(output / 'hk_research_universe/current_snapshot.parquet', metadata['universe_file'])
        reference_hashes = {'frozen_universe': metadata['universe_file']['sha256']}
        calendar_path = oracle / 'reference/HKEXCALENDAR.parquet'
        calendar_markets = ['HKEX']
    else:
        raise ValueError('Unsupported market')
    if master['symbol'].duplicated().any():
        raise ValueError('Ambiguous security-master identity')
    missing_listing = int(master['list_date'].isna().sum())
    master = master.loc[master['list_date'].notna()].sort_values('symbol').set_index('symbol')
    symbols = pd.Index(master.index)
    cal_record = json.loads(calendar_path.with_suffix('.json').read_text(encoding='utf-8'))
    reference_hashes['calendar'] = cal_record['sha256']
    calendar = verified_frame(calendar_path, cal_record)
    dates = pd.DatetimeIndex(pd.to_datetime(calendar.loc[calendar['S_INFO_EXCHMARKET'].isin(calendar_markets), 'TRADE_DAYS'], format='%Y%m%d')).unique().sort_values()
    dates = dates[(dates >= start) & (dates <= end)]
    price_values = np.full((len(dates), len(symbols)), np.nan)
    amount_values = np.full((len(dates), len(symbols)), np.nan)
    usable_values = np.zeros((len(dates), len(symbols)), dtype=bool)
    source_manifest = json.loads((base / 'manifest.json').read_text(encoding='utf-8'))
    sources = {}
    for name, source in sorted(source_manifest['files'].items()):
        year, month = map(int, name.split('/'))
        if pd.Timestamp(year=year, month=month, day=1) > end or year < start.year:
            continue
        path = base / (name + '.parquet')
        if not path.resolve().is_relative_to(base.resolve()):
            raise ValueError('Research data path escapes its root')
        if market == 'a':
            frame = verified_frame(path, source, ['date', 'symbol', 'adjclose', 'amount', 'volume', 'numeric_bar_valid', 'source_suspended'])
            valid = frame['numeric_bar_valid'] & ~frame['source_suspended'] & frame['volume'].gt(0) & frame['amount'].gt(0)
        else:
            frame = verified_frame(path, source, ['TRADE_DT', 'S_INFO_WINDCODE', 'CRNCY_CODE', 'S_DQ_ADJCLOSE', 'S_DQ_AMOUNT', 'S_DQ_VOLUME'])
            frame = frame.loc[frame['CRNCY_CODE'].eq('HKD')].rename(columns={'TRADE_DT': 'date', 'S_INFO_WINDCODE': 'symbol', 'S_DQ_ADJCLOSE': 'adjclose', 'S_DQ_AMOUNT': 'amount', 'S_DQ_VOLUME': 'volume'})
            frame['date'] = pd.to_datetime(frame['date'], format='%Y%m%d')
            frame['amount'] = pd.to_numeric(frame['amount'], errors='coerce') * 1000
            valid = frame['volume'].gt(0) & frame['amount'].gt(0) & frame['adjclose'].gt(0)
        ri, ci = dates.get_indexer(frame['date']), symbols.get_indexer(frame['symbol'])
        selected = (ri >= 0) & (ci >= 0)
        values = pd.to_numeric(frame['adjclose'], errors='coerce').to_numpy(dtype=float, copy=True)
        values[~np.isfinite(values) | (values <= 0)] = np.nan
        price_values[ri[selected], ci[selected]] = values[selected]
        amount_values[ri[selected], ci[selected]] = frame['amount'].to_numpy(dtype=float)[selected]
        usable_values[ri[selected], ci[selected]] = valid.fillna(False).to_numpy(dtype=bool)[selected]
        sources[name] = source['sha256']
    lifetime = ((dates.to_numpy()[:, None] >= master['list_date'].to_numpy()[None, :]) &
                (np.isnat(master['delist_date'].to_numpy())[None, :] | (dates.to_numpy()[:, None] < master['delist_date'].to_numpy()[None, :])))
    outside_lifecycle_rows = int((np.isfinite(price_values) & ~lifetime).sum())
    price_values[~lifetime] = np.nan
    amount_values[~lifetime] = np.nan
    price = pd.DataFrame(price_values, index=dates, columns=symbols)
    amount = pd.DataFrame(amount_values, index=dates, columns=symbols)
    usable = pd.DataFrame(usable_values & lifetime, index=dates, columns=symbols)
    if market == 'a':
        st_path = oracle / 'reference/ASHAREST.parquet'
        st_record = json.loads(st_path.with_suffix('.json').read_text(encoding='utf-8'))
        reference_hashes['st_history'] = st_record['sha256']
        st = verified_frame(st_path, st_record)
        for row in st.loc[st['S_TYPE_ST'].isin(protocol['exclude_a_st_types'])].itertuples(index=False):
            if row.S_INFO_WINDCODE not in symbols:
                continue
            entry = pd.to_datetime(row.ENTRY_DT, format='%Y%m%d', errors='coerce')
            removal = pd.to_datetime(row.REMOVE_DT, format='%Y%m%d', errors='coerce')
            if pd.isna(entry):
                raise ValueError('ST entry date is unknown')
            excluded = (dates >= entry) & (pd.isna(removal) | (dates < removal))
            usable.loc[excluded, row.S_INFO_WINDCODE] = False
    evidence = {'market': market, 'source_files': sources, 'missing_listing_dates_excluded': missing_listing,
        'reference_hashes': reference_hashes, 'outside_lifecycle_price_rows_removed': outside_lifecycle_rows,
        'symbols': len(symbols), 'sessions': len(dates), 'quote_currency': 'CNY' if market == 'a' else 'HKD',
        'hk_static_survivorship_assumption': market == 'hk'}
    return {'price': price, 'amount': amount, 'usable': usable}, evidence


def screen_market(output: Path, destination: Path, market: str, protocol: dict) -> dict:
    panel, evidence = load_panel(output, market, protocol)
    destination.mkdir(parents=True, exist_ok=True)
    evidence_path = destination / f'{market}_inputs.json'
    if evidence_path.exists() and json.loads(evidence_path.read_text(encoding='utf-8')) != evidence:
        raise ValueError('Study inputs changed; use a new registered experiment destination')
    atomic_json(evidence_path, evidence)
    price = panel['price']
    history = price.notna().rolling(protocol['minimum_history_sessions']).sum().eq(protocol['minimum_history_sessions'])
    liquid = panel['amount'].rolling(protocol['liquidity_window'], min_periods=protocol['liquidity_window']).median().ge(protocol['minimum_median_daily_amount_local_currency'])
    eligible = panel['usable'] & history & liquid
    labels = forward_labels(price, protocol['horizon_sessions'])
    label_end = pd.Series(price.index, index=price.index).shift(-(protocol['horizon_sessions'] + 1))
    reports = {}
    for name, signal in signal_values(price).items():
        daily = daily_statistics(signal, labels, eligible, protocol['minimum_cross_section'])
        daily.to_csv(destination / f'{market}_{name}_daily.csv', index_label='signal_date')
        partitions = {}
        for split, lower, upper in [('train', protocol['start'], protocol['train_end']),
                                    ('validation', (pd.Timestamp(protocol['train_end']) + pd.Timedelta(days=1)).date().isoformat(), protocol['validation_end'])]:
            mask = (daily.index >= lower) & (daily.index <= upper) & label_end.le(pd.Timestamp(upper))
            selected = daily.loc[mask]
            partitions[split] = {**hac_summary(selected['ic'], protocol['hac_lags']),
                'mean_paired_stocks': mean_or_none(selected['paired_stocks']),
                'missing_forward_labels': int(selected['missing_forward_labels'].sum()),
                'mean_top_minus_bottom_five_session_label': mean_or_none(selected['top_decile_forward_mean'] - selected['bottom_decile_forward_mean'])}
        reports[name] = partitions
        atomic_json(destination / f'{market}_summary.json', {'market': market, 'signals': reports,
                    'portfolio_backtest': False, 'risk_limits_evaluated': False, 'holdout_evaluated': False})
    return {'market': market, 'signals': reports, 'portfolio_backtest': False, 'holdout_evaluated': False}
