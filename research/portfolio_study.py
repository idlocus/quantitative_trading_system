"""Offline portfolio experiment using fixed prior-close selection and shared day bars."""
from itertools import product
import json
import hashlib
import os
import shutil
from uuid import uuid4
from contextlib import contextmanager
from utils.state_file import exclusive_file
from pathlib import Path

import numpy as np
import pandas as pd

from data.preparation import atomic_json
from .portfolio_simulation import PortfolioSimulation, distributions
from .signal_screen import load_panel, signal_values, verified_frame


def _study_files(root):
    """Enumerate regular artifacts and propagate errors from inaccessible subdirectories."""
    paths = {}
    pending = [root]
    while pending:
        for path in pending.pop().iterdir():
            if path.is_symlink() or path.is_junction():
                raise ValueError(f'Unexpected link in experiment: {path}')
            if path.is_dir():
                pending.append(path)
            elif path.is_file():
                paths[path.relative_to(root).as_posix()] = path
            else:
                raise ValueError(f'Unreadable experiment artifact: {path}')
    return paths


def read_published_verdict(destination, manifest_name):
    """Read the verdict and its release identity from one verified publication."""
    dest = Path(destination)
    with exclusive_file(dest):
        files = _study_files(dest)
        manifest = files.pop(manifest_name).read_bytes()
        expected = json.loads(manifest)
        if files.keys() != expected.keys() or 'verdict.json' not in files:
            raise ValueError(f'Incomplete publication: {dest}')
        for name, path in files.items():
            with path.open('rb') as stream:
                if name == 'verdict.json':
                    verdict = stream.read()
                    digest = hashlib.sha256(verdict)
                else:
                    digest = hashlib.file_digest(stream, 'sha256')
            if digest.hexdigest() != expected[name]:
                raise ValueError(f'Published artifact changed: {name}')
        return json.loads(verdict), hashlib.sha256(manifest).hexdigest()


@contextmanager
def publish_study(destination, manifest_name='portfolio_manifest.json'):
    """Publish complete, immutable experiment files; install their digest manifest last."""
    dest = Path(destination).absolute()
    dest.parent.mkdir(parents=True, exist_ok=True)
    # A 0700 temporary directory on Windows gets a protected, owner-only ACL.
    # Create a sibling with ordinary inherited permissions before the atomic rename.
    stage = dest.parent / f'.study-{uuid4().hex}'
    stage.mkdir(mode=0o755)
    try:
        yield stage
        files = _study_files(stage)
        files.pop(manifest_name, None)
        hashes = {name: hashlib.sha256(files[name].read_bytes()).hexdigest() for name in sorted(files)}
        atomic_json(stage / manifest_name, hashes)
        files[manifest_name] = stage / manifest_name
        with exclusive_file(dest):
            if not dest.exists():
                os.rename(stage, dest)
                return
            existing = _study_files(dest)
            extra = sorted(existing.keys() - files.keys())
            if extra:
                raise ValueError(f'Unexpected experiment files: {extra}; use a new destination')
            for name, target in existing.items():
                if target.read_bytes() != files[name].read_bytes():
                    raise ValueError(f'Experiment file changed: {name}; use a new destination')
            for name, path in files.items():
                target = dest / name
                if not target.exists():
                    target.parent.mkdir(parents=True, exist_ok=True)
                    os.replace(path, target)
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def suspension_days(output: Path, dates: pd.DatetimeIndex, symbols: set[str]) -> dict:
    root = output / 'oracle/supporting'
    manifest = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
    result = {}
    for name, record in manifest['files'].items():
        if not name.startswith('suspensions/') or name.endswith('unknown_resume_counts'):
            continue
        frame = verified_frame(root / (name + '.parquet'), record)
        frame = frame.loc[frame.S_INFO_WINDCODE.isin(symbols)]
        entry = pd.to_datetime(frame.S_DQ_SUSPENDDATE, format='%Y%m%d', errors='coerce')
        resume = pd.to_datetime(frame.S_DQ_RESUMPDATE, format='%Y%m%d', errors='coerce')
        for symbol, first, last in zip(frame.S_INFO_WINDCODE, entry, resume):
            if pd.isna(first) or pd.isna(last) or last <= first:
                continue
            for day in dates[(dates >= first) & (dates < last)]:
                result.setdefault(day, set()).add(symbol)
    return result


def snapshot_code(destination: Path):
    root = Path(__file__).resolve().parents[1]
    import sys
    names = {p.relative_to(root).as_posix() for folder in ('research', 'trading')
             for p in (root / folder).glob('*.py')}
    names.update(('data/preparation.py', 'data/catalog.py'))
    frame = sys._getframe()
    while frame is not None:
        caller = Path(frame.f_code.co_filename).resolve()
        if caller.is_relative_to(root / 'scripts') and caller.is_file():
            names.add(caller.relative_to(root).as_posix())
        frame = frame.f_back
    names = sorted(names)
    hashes = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in names}
    manifest = destination / 'code_manifest.json'
    if manifest.exists() and json.loads(manifest.read_text(encoding='utf-8')) != hashes:
        raise ValueError('Registered research code changed; use a new experiment destination')
    for name in names:
        target = destination / 'code' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest() != hashes[name]:
            raise ValueError('Stored experiment source changed')
        if not target.exists():
            target.write_bytes((root / name).read_bytes())
    atomic_json(manifest, hashes)


def selection_lists(score: pd.DataFrame, eligible: pd.DataFrame, count: int, available=None) -> list[list[str] | None]:
    values = score.where(eligible).to_numpy(dtype=float)
    columns = score.columns.to_numpy()
    result = []
    available = (score.notna().any(axis=1) if available is None else available).reindex(score.index).fillna(False).to_numpy()
    for index, row in enumerate(values):
        if not available[index] and eligible.iloc[index].any():
            result.append(None)
            continue
        valid = np.flatnonzero(np.isfinite(row))
        order = np.argsort(-row[valid], kind='stable')[:count]
        result.append(columns[valid[order]].tolist())
    return result


def run_portfolio_study(output: Path, destination: Path, protocol: dict, signal_protocol: dict) -> dict:
    with publish_study(destination) as stage:
        return _run_portfolio_study(output, stage, protocol, signal_protocol)


def _run_portfolio_study(output, destination, protocol, signal_protocol):
    registered = destination / 'protocol.json'
    if registered.exists() and json.loads(registered.read_text(encoding='utf-8')) != protocol:
        raise ValueError('Portfolio protocol cannot change after registration')
    snapshot_code(destination)
    atomic_json(registered, protocol)
    panel, evidence = load_panel(output, 'a', signal_protocol)
    inputs_path = destination / 'inputs.json'
    if inputs_path.exists() and json.loads(inputs_path.read_text(encoding='utf-8')) != evidence:
        raise ValueError('Portfolio inputs changed; register a new experiment')
    atomic_json(inputs_path, evidence)
    price = panel['price']
    eligible = (panel['usable'] & price.notna().rolling(signal_protocol['minimum_history_sessions']).sum().eq(signal_protocol['minimum_history_sessions'])
        & panel['amount'].rolling(signal_protocol['liquidity_window'], min_periods=signal_protocol['liquidity_window']).median().ge(signal_protocol['minimum_median_daily_amount_local_currency']))
    rankings = {name: selection_lists(score, eligible, protocol['positions'])
                for name, score in signal_values(price).items() if name in protocol['signals']}
    dates, symbols = price.index, set(price.columns)
    suspended = suspension_days(output, dates, symbols)
    del panel, price
    oracle = output / 'oracle'
    master_path = oracle / 'reference/ASHAREDESCRIPTION.parquet'
    master_record = json.loads(master_path.with_suffix('.json').read_text(encoding='utf-8'))
    master = verified_frame(master_path, master_record)
    delist = dict(zip(master.S_INFO_WINDCODE, pd.to_datetime(master.S_INFO_DELISTDATE, format='%Y%m%d', errors='coerce')))
    action_path = oracle / 'reference/ASHAREDIVIDEND.parquet'
    action_record = json.loads(action_path.with_suffix('.json').read_text(encoding='utf-8'))
    actions, action_quality = distributions(verified_frame(action_path, action_record), symbols, dates)
    atomic_json(destination / 'corporate_action_inputs.json', {'source_sha256': action_record['sha256'], 'quality': action_quality})
    simulations = []
    for signal, cost, variant in product(protocol['signals'], protocol['one_way_cost_scenarios'], protocol['risk_variants']):
        name = f'{signal}_{variant}_{round(cost * 10000)}bps'
        simulations.append((signal, PortfolioSimulation(name, protocol, cost, variant == 'budget_guard', actions, dates, delist)))
    base = oracle / 'normalized_a'
    manifest = json.loads((base / 'manifest.json').read_text(encoding='utf-8'))
    steps = 0
    for month, record in sorted(manifest['files'].items()):
        if month not in evidence['source_files']:
            continue
        if record.get('sha256') != evidence['source_files'].get(month):
            raise ValueError('Bar inputs changed between research passes')
        frame = verified_frame(base / (month + '.parquet'), record)
        frame = frame.loc[frame.symbol.isin(symbols) & frame.date.isin(dates)]
        grouped = {day: group for day, group in frame.groupby('date', sort=True)}
        month_dates = dates[dates.strftime('%Y/%m') == month]
        for day in month_dates:
            index = dates.get_loc(day)
            day_frame = grouped.get(day)
            bars = {} if day_frame is None else day_frame.set_index('symbol').to_dict(orient='index')
            for symbol in suspended.get(day, set()):
                if symbol not in bars:
                    bars[symbol] = {'numeric_bar_valid': False, 'source_suspended': True}
            previous_eligible = set(eligible.columns[eligible.iloc[index - 1]]) if index else set()
            for signal, simulation in simulations:
                ranking = rankings[signal][index - 1] if index else []
                simulation.step(index, day, bars, ranking, previous_eligible)
            steps += 1
        if month.endswith('/12'):
            print(json.dumps({'completed_year': int(month[:4]), 'sessions': steps}), flush=True)
    if steps != len(dates):
        raise ValueError('Portfolio replay did not cover every expected market session')
    reports = {}
    for signal, simulation in simulations:
        report = simulation.result(protocol['train_end'])
        reports[simulation.name] = report
        pd.DataFrame(simulation.nav_rows).to_csv(destination / f'{simulation.name}_nav.csv', index=False)
        pd.DataFrame(simulation.trades).to_csv(destination / f'{simulation.name}_trades.csv', index=False)
        pd.DataFrame(simulation.issue_rows, columns=['date', 'symbol', 'issue']).to_csv(destination / f'{simulation.name}_issues.csv', index=False)
    result = {'study': protocol['study'], 'runs': reports, 'run_count': len(reports),
              'holdout_evaluated': False, 'live_or_statutory_cost_validation': False}
    atomic_json(destination / 'summary.json', result)
    return result
