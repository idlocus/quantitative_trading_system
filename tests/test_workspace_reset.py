"""Boundary, offline-data and strategy-free execution acceptance tests."""
import ast
import hashlib
import os
import json
from pathlib import Path
import subprocess
import sys

import pandas as pd
import pytest

from application.cli import main
from application.workspace import Workspace
from data.catalog import LocalDataCatalog
from portfolio.orders import build_orders
from portfolio.replay import ReplayConfig, ReplayEngine
from strategies import StrategyRegistry, registry
from strategies.base import StrategyContext, validate_weights

ROOT = Path(__file__).resolve().parents[1]


def actions():
    return pd.DataFrame(columns=['sym', 'ex_date', 'cash_ps', 'stk_ps'])


def market():
    dates = pd.bdate_range('2026-01-01', periods=5)
    return {symbol: pd.DataFrame({'date': dates, 'close': prices, 'adjclose': prices,
                                   'volume': [1e8] * 5})
            for symbol, prices in {'600000.SH': [10, 10, 10, 10, 10],
                                    '600001.SH': [20, 20, 21, 21, 21]}.items()}


class SyntheticAllocation:
    """A test fixture only; never registered as an application strategy."""
    def generate_targets(self, context):
        return {'600000.SH': .5, '600001.SH': .4}


@pytest.mark.release_gate
def test_empty_registry_never_falls_back_to_a_strategy():
    assert registry.names() == ()
    with pytest.raises(LookupError):
        registry.get('old-production')
    with pytest.raises(ValueError, match='explicit strategy'):
        ReplayEngine().run(None, market(), corporate_actions=actions())


def test_registration_is_explicit_and_duplicate_names_fail():
    local = StrategyRegistry()
    fixture = SyntheticAllocation()
    local.register('test-allocation', fixture)
    assert local.get('test-allocation') is fixture
    with pytest.raises(ValueError):
        local.register('test-allocation', fixture)
    assert registry.names() == ()


@pytest.mark.parametrize('weights', [{'A': float('nan')}, {'A': -1}, {'A': 1.1},
                                      {'A': True}, {'': .5}])
def test_invalid_allocations_are_rejected(weights):
    with pytest.raises(ValueError):
        validate_weights(weights)


def test_orders_use_explicit_weights_and_execute_sells_before_buys():
    orders = build_orders({'B': .4, 'C': .3}, {'A': .5}, 100000)
    assert orders[0].side == 'SELL' and orders[0].symbol == 'A'
    assert {order.symbol for order in orders if order.side == 'BUY'} == {'B', 'C'}
    assert sum(order.amount for order in orders if order.side == 'BUY') <= 97500
    assert build_orders({'A': .5}, {'A': .52}, 100000, band=.1) == []


@pytest.mark.release_gate
def test_replay_is_in_memory_and_never_exposes_current_day_data(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cache = market()
    original = {symbol: frame.copy(deep=True) for symbol, frame in cache.items()}
    seen = []
    class Fixture:
        def generate_targets(self, context):
            seen.append(context.signal_date)
            assert max(context.history['600000.SH']['date']).date() == context.signal_date
            context.history['600000.SH'].loc[:, 'close'] = 999
            return {'600000.SH': .5}
    first = ReplayEngine().run(Fixture(), cache, corporate_actions=actions())
    second = ReplayEngine().run(Fixture(), cache, corporate_actions=actions())
    pd.testing.assert_series_equal(first.equity, second.equity)
    assert len(seen) == 8
    assert first.trades[0]['date'] == '2026-01-02'
    assert first.cash >= 0 and first.positions['600000.SH'] > 0
    assert not list(tmp_path.iterdir())
    for symbol in cache:
        pd.testing.assert_frame_equal(cache[symbol], original[symbol])


def test_empty_targets_are_an_explicit_cash_decision():
    class Cash:
        def generate_targets(self, context):
            return {}
    result = ReplayEngine().run(Cash(), market(), corporate_actions=actions())
    assert not result.trades and not result.positions
    assert result.total_return == 0


def test_replay_rejects_unknown_assets_and_invalid_weights():
    class Unknown:
        def generate_targets(self, context):
            return {'missing': .5}
    with pytest.raises(ValueError, match='outside'):
        ReplayEngine().run(Unknown(), market(), corporate_actions=actions())
    with pytest.raises(ValueError, match='cap'):
        ReplayEngine(ReplayConfig(max_weight=.2)).run(SyntheticAllocation(), market(), corporate_actions=actions())


def test_replay_requires_explicit_corporate_action_schema():
    with pytest.raises(ValueError, match='corporate-action'):
        ReplayEngine().run(SyntheticAllocation(), market(), corporate_actions=pd.DataFrame())


def test_replay_accounts_for_dividends_and_splits():
    cache = market()
    cache['600000.SH']['close'] = [10, 10, 4.5, 4.5, 4.5]
    ca = pd.DataFrame({'sym': ['600000.SH'], 'ex_date': [pd.Timestamp('2026-01-05')],
                        'cash_ps': [1.0], 'stk_ps': [1.0]})
    class Allocation:
        def generate_targets(self, context):
            return {'600000.SH': 1.0}
    result = ReplayEngine().run(Allocation(), cache, corporate_actions=ca)
    assert result.total_return > -.01 and result.max_drawdown > -.01


@pytest.mark.parametrize('column,value', [('cash_ps', -.1), ('stk_ps', float('nan')),
                                          ('ex_date', 'invalid')])
def test_replay_rejects_invalid_corporate_actions(column, value):
    ca = pd.DataFrame({'sym': ['600000.SH'], 'ex_date': ['2026-01-05'],
                        'cash_ps': [1.0], 'stk_ps': [0.0]})
    ca[column] = value
    with pytest.raises(ValueError, match='corporate-action'):
        ReplayEngine().run(SyntheticAllocation(), market(), corporate_actions=ca)


def test_context_allocation_snapshot_is_readonly():
    context = StrategyContext.snapshot(pd.Timestamp('2026-01-02').date(), market(), {'A': .1}, 100, 1000)
    with pytest.raises(TypeError):
        context.current_weights['A'] = .9


@pytest.mark.release_gate
def test_old_commands_refuse_to_start_without_touching_state(tmp_path, capsys):
    workspace = Workspace(tmp_path)
    for command in ('daily', 'am', 'pm', 'scan', 'dashboard', 'report'):
        assert main([command, '--execute', 'qmt'], workspace) == 2
        assert json.loads(capsys.readouterr().out)['status'] == 'disabled'
    assert main(['run', '--strategy', 'old-production'], workspace) == 2
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize('script', ['scripts/paper_day.py', 'scripts/paper_watchdog.py'])
def test_legacy_script_entrypoints_fail_closed(script):
    # Root-level run_*.py stubs removed 2026-11-17 (unified to webapp/app.py);
    # scripts/ stubs retained as defense-in-depth
    result = subprocess.run([sys.executable, str(ROOT / script), '--execute', 'qmt'],
                            cwd=ROOT, capture_output=True, text=True, encoding='utf-8')
    assert result.returncode == 2
    assert json.loads(result.stdout)['status'] == 'disabled'


@pytest.mark.release_gate
def test_live_factory_does_not_reenable_qmt_from_old_deployment_state(monkeypatch):
    import live
    monkeypatch.setenv('QMT_ACCOUNT_ID', 'test-only')
    with pytest.raises(RuntimeError, match='disabled'):
        live.make_broker('qmt')
    assert live.make_broker('paper').name() == 'PaperBroker'


def test_offline_catalog_reads_existing_data_without_rewriting(tmp_path):
    folder = tmp_path / 'kline'
    folder.mkdir()
    path = folder / '600000_SH.parquet'
    market()['600000.SH'].to_parquet(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    before = path.stat().st_mtime_ns
    catalog = LocalDataCatalog(tmp_path)
    frame = catalog.kline('600000.SH', end=pd.Timestamp('2026-01-02').date())
    assert len(frame) == 2 and frame['close'].iloc[-1] == 10
    assert catalog.summary()['files'] == 1
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
    assert path.stat().st_mtime_ns == before
    with pytest.raises(FileNotFoundError, match='no download'):
        catalog.kline('600002.SH')
    with pytest.raises(ValueError):
        catalog.kline('../.env')
    with pytest.raises(ValueError):
        catalog.datasets('../outside')


def test_workspace_commands_work_without_any_cache(tmp_path, capsys):
    workspace = Workspace(tmp_path)
    assert main(['status'], workspace) == 0
    status = json.loads(capsys.readouterr().out)
    assert not status['automatic_trading'] and not status['live_execution']
    assert main(['data', 'summary'], workspace) == 0
    assert json.loads(capsys.readouterr().out)['files'] == 0
    assert main(['data', 'show', '600000.SH'], workspace) == 2
    assert not list(tmp_path.iterdir())


def test_local_data_import_does_not_load_external_provider_sdks():
    code = ('import sys; import data; from data.catalog import LocalDataCatalog; '
            'assert not any(name in sys.modules for name in '
            '("akshare", "oracledb", "WindPy", "jqdatasdk"))')
    result = subprocess.run([sys.executable, '-c', code], cwd=ROOT,
                            capture_output=True, text=True, encoding='utf-8')
    assert result.returncode == 0, result.stderr


@pytest.mark.release_gate
def test_current_source_never_imports_archived_policy_modules():
    forbidden = {'archive', 'backtesting', 'selection', 'strategy', 'market_env', 'sector', 'risk'}
    # 2026-11-04 扩面：覆盖全部活跃目录 + 根目录入口（run_*.py/main.py）；
    # webapp 当前无 .py（源码已归档），保留在列表中防复活
    active = ['application', 'strategies', 'data', 'live', 'trading',
              'portfolio', 'research', 'monitoring', 'webapp', 'indicators',
              'utils', 'scripts', 'tests']
    active += [str(p) for p in ROOT.glob('*.py')]
    failures = []
    for entry in active:
        target = Path(entry) if os.path.isabs(str(entry)) else ROOT / entry
        py_files = target.rglob('*.py') if target.is_dir() else ([target] if target.suffix == '.py' else [])
        for path in py_files:
            for node in ast.walk(ast.parse(path.read_text(encoding='utf-8-sig'))):
                imports = [alias.name for alias in node.names] if isinstance(node, ast.Import) else []
                if isinstance(node, ast.ImportFrom) and node.level == 0:
                    imports.append(node.module or '')
                for name in imports:
                    if name.split('.')[0] in forbidden or name in {'live.signals', 'trading.strategy', 'trading.spec'}:
                        failures.append(f'{path.name}: {name}')
    assert not failures


@pytest.mark.release_gate
def test_backup_and_retained_file_inventory_verify():
    from application.archive import verify_snapshot
    result = verify_snapshot(Workspace.current())
    assert result['source_verified'], result['source_failures']
    assert result['retained_files_unchanged'], result['retained_file_changes']


def test_snapshot_verifier_detects_both_source_and_retained_data_changes(tmp_path):
    from application.archive import verify_snapshot
    workspace = Workspace(tmp_path)
    source = workspace.archive / 'source'
    source.mkdir(parents=True)
    (source / 'example.py').write_text('original', encoding='utf-8')
    retained = tmp_path / 'data_cache' / 'example.bin'
    retained.parent.mkdir()
    retained.write_bytes(b'original-data')
    stat = retained.stat()
    manifest = {'source': [{'path': 'example.py', 'sha256': hashlib.sha256(b'original').hexdigest()}],
                'protected': [{'path': 'data_cache/example.bin', 'size': stat.st_size,
                               'mtime_ns': stat.st_mtime_ns}]}
    (workspace.archive / 'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
    assert verify_snapshot(workspace)['source_verified']
    (source / 'example.py').write_text('changed', encoding='utf-8')
    retained.write_bytes(b'changed-data')
    result = verify_snapshot(workspace)
    assert not result['source_verified'] and not result['retained_files_unchanged']


def test_active_source_has_no_undefined_names():
    targets = ['application', 'strategies', 'data', 'live', 'trading', 'portfolio',
               'research', 'indicators', 'scripts', 'monitoring', 'main.py']
    result = subprocess.run([sys.executable, '-m', 'pyflakes', *targets], cwd=ROOT,
                            capture_output=True, text=True, encoding='utf-8', errors='replace')
    undefined = [line for line in result.stdout.splitlines() if 'undefined name' in line]
    assert not undefined, '\n'.join(undefined)
