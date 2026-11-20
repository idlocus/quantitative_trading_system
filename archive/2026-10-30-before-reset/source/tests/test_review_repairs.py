"""Behavioral regressions for the project review repairs."""
from dataclasses import asdict
from datetime import date
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from data.semantics import flow_adjusted_navs
from live.broker import Fill, Order
from live.ledger import Ledger
from live.qmt_broker import QMTBroker, QMTOrderState
from live.risk_gate import GateConfig, RiskDecision, RiskGate
from trading import config as chain
from trading.strategy import _filter_by_factor, required_signal_factors


@pytest.mark.release_gate
@pytest.mark.parametrize('navs,flows,expected', [
    ([1000000, 800000], [(0, 1000000)], 0.20),
    ([1000000, 1000000, 800000], [(0, 1000000), (2, -200000)], 0.0),
    ([1000000, 1200000], [(0, 1000000), (1, 200000)], 0.0),
    ([1000000, 900000, 1800000], [(0, 1000000), (2, 900000)], 0.10),
])
def test_flow_adjustment_preserves_market_drawdown(tmp_path, navs, flows, expected):
    history = [{'date': f'2026-01-{i + 1:02}', 'nav': value}
               for i, value in enumerate(navs)]
    cash_log = [{'date': history[i]['date'], 'type': 'deposit' if value > 0 else 'withdraw',
                 'amount': value} for i, value in flows]
    ledger = SimpleNamespace(nav_series=lambda: history,
                             _d={'cash_log': cash_log}, positions={'A': 1})
    gate = RiskGate(state_path=str(tmp_path / 'risk.json'))
    decision = RiskDecision()
    action, _, _ = gate._portfolio_breakers(ledger, None, date(2026, 1, len(navs)), decision)
    assert decision.drawdown == pytest.approx(expected)
    assert (action == 'liquidate') == (expected >= chain.DRAWDOWN_HARD)


def test_initial_capital_is_dated_to_replay_start(tmp_path):
    ledger = Ledger(str(tmp_path / 'ledger.json'))
    ledger.initialize(1000, as_of=date(2017, 1, 3))
    assert ledger.cash_log()[0]['date'] == '2017-01-03'
    assert ledger.nav_series()[0]['date'] == '2017-01-03'


def test_cash_flows_do_not_trigger_volatility_derisking(tmp_path):
    history = [{'date': str(dt.date()), 'nav': 100 if i < 10 else 1000}
               for i, dt in enumerate(pd.bdate_range('2026-01-01', periods=25))]
    ledger = SimpleNamespace(nav_series=lambda: history, _d={'cash_log': [
        {'date': history[10]['date'], 'type': 'deposit', 'amount': 900}]})
    gate = RiskGate(state_path=str(tmp_path / 'risk.json'))
    assert gate._recent_portfolio_vol(ledger) == pytest.approx(0)


def test_flow_adjustment_handles_flows_between_observations():
    result = flow_adjusted_navs(
        [{'date': '2026-01-01', 'nav': 100}, {'date': '2026-01-05', 'nav': 120}],
        [{'date': '2026-01-03', 'type': 'deposit', 'amount': 20}])
    assert result == pytest.approx([120, 120])


def test_financial_provider_normalizes_only_roe(monkeypatch):
    from data import fundamental_pit as module
    dates = pd.bdate_range('2026-01-01', periods=12)
    cursor = SimpleNamespace(close=lambda: None)
    ds = SimpleNamespace(_db_src=lambda: SimpleNamespace(_cursor=lambda: cursor),
                         get_kline_calendar=lambda *args: dates)

    def announcements(cur, field, symbols, start, end):
        value = 8.0 if field == 'S_FA_ROE' else 65.0
        return pd.DataFrame({'symbol': symbols, 'ann_dt': ['20251231'] * len(symbols),
                             'report_period': ['20250930'] * len(symbols),
                             'value': [value] * len(symbols)})

    monkeypatch.setattr(module, 'fetch_announcements', announcements)
    provider = module.PITFinancialProvider(ds)
    start, end = dates[0].date(), dates[-1].date()
    assert provider.get_factor_matrix('roe', ['A'], start, end).iloc[-1, 0] == 0.08
    assert provider.get_factor_matrix('leverage', ['A'], start, end).iloc[-1, 0] == 65.0
    assert list(provider.get_factor_matrix('roe', ['B'], start, end).columns) == ['B']
    assert provider.get_factor_matrix('roe', ['B'], start, end).iloc[-1, 0] == 0.08


@pytest.mark.release_gate
@pytest.mark.parametrize('mode,threshold,values,eligible', [
    ('min', .08, [np.nan, .01, .08, .09], ['D']),
    ('max', 65, [np.nan, 70, 65, 64], ['D']),
])
def test_quality_requires_known_strictly_eligible_values(mode, threshold, values, eligible):
    score = pd.DataFrame([[1.0] * 4], columns=list('ABCD'))
    factor = pd.DataFrame([values], columns=list('ABCD'))
    assert list(_filter_by_factor(score, factor, mode, threshold, 'quality').iloc[0].dropna().index) == eligible
    assert _filter_by_factor(score, pd.DataFrame(), mode, threshold, 'quality').isna().all().all()


def test_st_production_and_offline_constructors_match(monkeypatch):
    from data.st_provider import STProvider
    rows = [('A', 'S', '20200101', None), ('B', 'T', '20200101', None),
            ('C', 'R', '20200101', None)]
    monkeypatch.setattr(STProvider, '_load', lambda self, ds: rows)
    assert STProvider(object()).st_set(date(2026, 1, 1)) == {'A'}
    assert STProvider.from_rows(rows).st_set(date(2026, 1, 1)) == {'A'}


@pytest.mark.release_gate
def test_qmt_sync_books_incremental_cost_and_recovers_crash(tmp_path, monkeypatch):
    broker = QMTBroker(state_path=str(tmp_path / 'qmt.json'))
    ledger = Ledger(str(tmp_path / 'ledger.json'))
    ledger.initialize(10000)
    order = Order('600000.SH', 'BUY', 0.0, quantity=200)
    ledger.apply_fills([Fill(order, 10, 100, 0, 0, date.today(), trade_id='ref#100')])
    broker._pending['ref'] = {'symbol': order.symbol, 'side': 'BUY', 'qty': 200,
                               'booked': 100, 'booked_cost': 1000,
                               'submitted_at': str(date.today())}
    monkeypatch.setattr(broker, '_query_order', lambda oid: (QMTOrderState.FILLED, 200, 11.0))
    original = broker._save_state
    monkeypatch.setattr(broker, '_save_state', lambda: (_ for _ in ()).throw(OSError('crash')))
    with pytest.raises(OSError):
        broker.sync_pending(ledger)
    assert ledger.positions[order.symbol] == 200
    assert ledger._d['cost'][order.symbol] == pytest.approx(11.0)
    assert ledger._d['trades'][-1]['price'] == pytest.approx(12.0)
    cash = ledger.cash
    broker._pending['ref'] = {'symbol': order.symbol, 'side': 'BUY', 'qty': 200,
                               'booked': 100, 'booked_cost': 1000,
                               'submitted_at': str(date.today())}
    monkeypatch.setattr(broker, '_save_state', original)
    assert broker.sync_pending(ledger) == []
    assert ledger.cash == cash
    assert not broker._pending


@pytest.mark.release_gate
def test_qmt_stops_batch_when_order_status_cannot_be_confirmed(tmp_path, monkeypatch):
    broker = QMTBroker(state_path=str(tmp_path / 'orders.json'), enforce_window=False)
    submitted = []
    monkeypatch.setattr(broker, '_ensure_connected', lambda: None)
    def execute(order, prices, trade_date, cash, **kwargs):
        submitted.append(order.symbol)
        return Fill(order, 10, 0, 0, 0, trade_date, status='query_failed(status=-1)')
    monkeypatch.setattr(broker, '_execute_single', execute)
    fills = broker.execute([Order('A', 'BUY', .01, amount=100),
                             Order('B', 'BUY', .01, amount=100)],
                            {'A': 10, 'B': 10}, date.today(), available_cash=1000)
    assert submitted == ['A']
    assert broker.query_failed
    assert fills[-1].status.startswith('rejected')


def test_qmt_sync_keeps_unpriced_execution_pending(tmp_path, monkeypatch):
    broker = QMTBroker(state_path=str(tmp_path / 'qmt.json'))
    ledger = Ledger(str(tmp_path / 'ledger.json'))
    ledger.initialize(10000)
    broker._pending['ref'] = {'symbol': '600000.SH', 'side': 'BUY', 'qty': 100,
                               'booked': 0, 'submitted_at': str(date.today())}
    monkeypatch.setattr(broker, '_query_order', lambda oid: (QMTOrderState.FILLED, 100, 0.0))
    assert broker.sync_pending(ledger) == []
    assert broker.query_failed
    assert broker._pending['ref']['booked'] == 0
    assert not ledger.positions


@pytest.mark.parametrize('known_cost', [True, False])
def test_qmt_migrates_legacy_partial_cost_only_with_ledger_evidence(tmp_path, monkeypatch, known_cost):
    broker = QMTBroker(state_path=str(tmp_path / 'qmt.json'))
    ledger = Ledger(str(tmp_path / 'ledger.json'))
    ledger.initialize(10000)
    if known_cost:
        ledger.apply_fills([Fill(Order('600000.SH', 'BUY', 0), 10, 100, 0, 0,
                                 date.today(), trade_id='ref#100')])
    broker._pending['ref'] = {'symbol': '600000.SH', 'side': 'BUY', 'qty': 200,
                               'booked': 100, 'submitted_at': str(date.today())}
    monkeypatch.setattr(broker, '_query_order', lambda oid: (QMTOrderState.FILLED, 200, 11.0))
    fills = broker.sync_pending(ledger)
    if known_cost:
        assert fills[0].price == pytest.approx(12)
        assert ledger._d['cost']['600000.SH'] == pytest.approx(11)
        assert not broker._pending
    else:
        assert fills == [] and broker.query_failed
        assert broker._pending['ref']['booked'] == 100


def test_etf_entrypoint_constrains_full_rotation(tmp_path, monkeypatch):
    import run_etf
    from scripts import trading_switch
    symbol, target = list(chain.ETF_ASSETS)[:2]
    ledger = Ledger(str(tmp_path / 'etf.json'))
    ledger.initialize(10000)
    ledger.apply_fills([Fill(Order(symbol, 'BUY', 1), 10, 1000, 0, 0,
                             date(2026, 9, 30))])
    dates = pd.to_datetime(['2026-09-30', '2026-10-01'])
    cache = {s: pd.DataFrame({'date': dates, 'close': [10.0, 10.0]})
             for s in chain.ETF_ASSETS}
    class FixedDate(date):
        @classmethod
        def today(cls):
            return cls(2026, 10, 1)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(run_etf, 'date', FixedDate)
    monkeypatch.setattr(run_etf, 'Ledger', lambda path: ledger)
    monkeypatch.setattr(run_etf, 'LayeredDataSource', lambda: SimpleNamespace(preload_batch=lambda *args: cache))
    monkeypatch.setattr(run_etf, 'build_etf_signal', lambda *args: ({target: 1.0}, cache, {target: .1}))
    monkeypatch.setattr(trading_switch, 'effective_state', lambda: 'NORMAL')
    monkeypatch.setattr('sys.argv', ['run_etf.py'])
    run_etf.main()
    assert symbol not in ledger.positions
    assert ledger.positions[target] > 0
    assert ledger.cash >= 0
    assert not ledger.open_exceptions()


def test_fundamental_cache_expands_universe_and_revises_values(tmp_path, monkeypatch):
    from data.layered import LayeredDataSource
    ds = LayeredDataSource(str(tmp_path))
    dates = pd.bdate_range('2026-01-01', periods=10)
    pd.DataFrame({'A': [10.0] * 10}, index=dates).to_parquet(ds.fund_dir / 'pe_ttm.parquet')
    fetched = []
    monkeypatch.setattr(ds, '_db_src', lambda: True)
    def fetch(field, symbols, start, end):
        fetched.extend(symbols)
        return pd.DataFrame({s: [20.0] * 10 for s in symbols}, index=dates)
    monkeypatch.setattr(ds, '_db_fundamental_bulk', fetch)
    result = ds.get_field_matrix('pe_ttm', ['A', 'B'], dates[0].date(), dates[-1].date())
    assert fetched == ['A', 'B']
    assert result.iloc[-1].to_dict() == {'A': 20, 'B': 20}
    assert list(ds.get_field_matrix('pe_ttm', ['B'], dates[0].date(), dates[-1].date()).columns) == ['B']


def test_automatic_qmt_reconciliation_accepts_statement_object(tmp_path, monkeypatch):
    from scripts import auto_reconcile
    from live.reconcile import BrokerStatement
    import live
    statement = BrokerStatement('2026-10-30', 100, {}, source='qmt')
    received = []
    broker = SimpleNamespace(connect=lambda: None, disconnect=lambda: received.append('disconnect'),
                              fetch_statement=lambda: statement)
    monkeypatch.setattr(live, 'QMTBroker', lambda: broker)
    monkeypatch.setattr(auto_reconcile, 'run_reconciliation',
                        lambda ledger, data: received.append(data) or {'ok': True, 'raw_snapshot': 'mock'})
    monkeypatch.setattr('sys.argv', ['auto_reconcile.py', '--ledger', str(tmp_path / 'ledger.json')])
    auto_reconcile.main()
    assert received == [asdict(statement), 'disconnect']


def test_ml_training_is_invariant_to_prices_after_cutoff(monkeypatch):
    from research.ml_model import DailyMLModel
    import sklearn.ensemble
    class CapturingModel:
        def __init__(self, **kwargs):
            pass
        def fit(self, features, target):
            self.features, self.target = features.copy(), target.copy()
    monkeypatch.setattr(sklearn.ensemble, 'HistGradientBoostingRegressor', CapturingModel)
    dates = pd.bdate_range('2026-01-01', periods=80)
    rng = np.random.RandomState(7)
    prices = pd.DataFrame(10 + rng.rand(80, 60), index=dates)
    factors = {'feature': pd.DataFrame(rng.rand(80, 60), index=dates)}
    first = DailyMLModel(train_months=2, pred_horizon=5)
    second = DailyMLModel(train_months=2, pred_horizon=5)
    assert first.train(factors, prices, dates[60])
    prices.loc[dates[61]:] *= 100
    assert second.train(factors, prices, dates[60])
    pd.testing.assert_frame_equal(first.model.features, second.model.features)
    pd.testing.assert_series_equal(first.model.target, second.model.target)


def test_spec_tracks_effective_runtime_parameters(monkeypatch):
    from trading.spec import StrategySpec
    baseline = StrategySpec.current()
    monkeypatch.setattr(chain, 'MIN_FACTOR_COVERAGE', .40)
    monkeypatch.setattr(chain, 'REENTRY_MAX_LIQUIDATIONS', 5)
    current = StrategySpec.current(top_n=50)
    assert current.min_factor_coverage == .40
    assert current.reentry_max_liquidations == 5
    assert current.top_n == 50
    assert current.rebalance_frequency == 'daily'
    assert current.hash != baseline.hash
    old_hash = current.hash
    monkeypatch.setitem(chain.QUALITY_FILTER, 'roe_min', .15)
    assert current.hash == old_hash


def test_artifact_spec_preserves_actual_replay_settings(tmp_path, monkeypatch):
    from portfolio.daily_replay import ReplayConfig, ReplayResult
    from research.artifacts import portfolio_payload, save_artifact
    from research import artifacts
    import json
    cfg = ReplayConfig(top_n=50, signal_gates_required=False,
                       use_risk_gate=False, band=.1,
                       gate_config=GateConfig(reentry_max_liquidations=8))
    result = ReplayResult(config=cfg, n_dates=2, initial_capital=100,
                           final_capital=100, total_return=0, annual_return=0,
                           sharpe=0, max_drawdown=0, n_rebalances=0,
                           n_trade_days=0, total_cost=0, avg_turnover=0,
                           n_stop_sells=0, n_halt_days=0, n_no_buy_days=0,
                           equity_curve=pd.Series([100, 100], index=pd.bdate_range('2026-01-01', periods=2)))
    monkeypatch.setattr(artifacts, 'DATA_DIR', str(tmp_path))
    payload = portfolio_payload(result)
    path = save_artifact('portfolio', 'repaired', payload)
    stored = json.loads(open(path, encoding='utf-8').read())['_meta']
    spec = stored['strategy_spec']
    assert spec['top_n'] == 50 and spec['buffer_band'] == .1
    assert spec['rebalance_frequency'] == 'daily'
    assert spec['reentry_max_liquidations'] == 8
    assert not spec['signal_gates_required'] and not spec['use_risk_gate']
    assert stored['strategy_hash'] == payload['_meta']['strategy_hash']
    from portfolio.backtester import PortfolioConfig
    from portfolio.risk_manager import RiskLimits
    result.config = PortfolioConfig(top_n=50, risk_limits=RiskLimits(
        max_drawdown=.25, drawdown_derisk=.10))
    approximate = portfolio_payload(result)['_meta']['strategy_spec']
    assert approximate['drawdown_hard'] == .25
    assert approximate['drawdown_soft'] == .10
    assert approximate['rebalance_frequency'] == 'every_5_trading_days'


def test_formal_artifact_can_be_generated_from_clean_checkout(monkeypatch):
    from research import artifacts
    monkeypatch.setattr(artifacts, '_run_git', lambda args: '' if args[0] == 'status' else 'abc123')
    assert artifacts.build_meta(formal=True)['git_dirty'] is False


@pytest.mark.release_gate
def test_replay_requires_each_raw_factor_even_when_composite_is_full():
    from portfolio.daily_replay import DailyReplayBacktester, ReplayConfig
    dates = pd.bdate_range('2026-01-01', periods=3)
    symbols = [f'S{i}.SH' for i in range(20)]
    cache = {s: pd.DataFrame({'date': dates, 'close': [10.0] * 3, 'volume': [1e8] * 3})
             for s in symbols}
    score = pd.DataFrame(1.0, index=dates, columns=symbols)
    matrices = {name: score.copy() for name in required_signal_factors()}
    cfg = ReplayConfig(top_n=20, min_listing_days=0, min_daily_turnover=0,
                       use_risk_gate=False, signal_gates_required=True,
                       factor_matrices=matrices)
    replay = DailyReplayBacktester(cache, cfg)
    assert len(replay._targets_at(score, dates[0], dates[0])) == 20
    matrices['ep'].iloc[1] = np.nan
    assert replay._targets_at(score, dates[1], dates[1]) is None


@pytest.mark.release_gate
def test_production_replay_cli_wires_actions_and_raw_factor_gates(monkeypatch):
    import run_daily_replay as cli
    import data.st_provider
    import data.corporate_actions
    symbols = [f'S{i}.SH' for i in range(20)]
    dates = pd.bdate_range('2026-01-01', periods=3)
    score = pd.DataFrame(1.0, index=dates, columns=symbols)
    mats = {name: score.copy() for name in required_signal_factors()}
    cache = {s: pd.DataFrame({'date': dates, 'close': [10.0] * 3}) for s in symbols}
    actions = pd.DataFrame({'sym': [symbols[0]], 'ex_date': [dates[-1]],
                             'cash_ps': [1.0], 'stk_ps': [1.0]})
    ds = SimpleNamespace(list_symbols=lambda **kw: symbols,
                         preload_batch=lambda *args: {})
    monkeypatch.setattr(cli, 'LayeredDataSource', lambda: ds)
    monkeypatch.setattr(cli, 'build_score_matrix', lambda *args: (score, cache, mats))
    monkeypatch.setattr(data.st_provider, 'STProvider', lambda ds: SimpleNamespace(
        st_matrix=lambda idx, pool: pd.DataFrame(False, index=idx, columns=pool)))
    monkeypatch.setattr(data.corporate_actions, 'load_corporate_actions', lambda *args, **kw: actions)
    captured = {}
    class Replay:
        def __init__(self, data, config, corporate_actions=None):
            captured.update(config=config, actions=corporate_actions)
        def run(self, values, **kwargs):
            assert values is score
            return SimpleNamespace(summary_markdown=lambda: 'synthetic replay')
    monkeypatch.setattr(cli, 'DailyReplayBacktester', Replay)
    monkeypatch.setattr('sys.argv', ['run_daily_replay.py', '--universe', '20', '--top', '20'])
    cli.main()
    assert captured['actions'] is actions
    assert captured['config'].signal_gates_required
    assert captured['config'].st_data_required
    assert captured['config'].factor_matrices is mats


def test_replay_books_split_and_dividend_without_phantom_loss():
    from portfolio.daily_replay import DailyReplayBacktester, ReplayConfig
    dates = pd.bdate_range('2026-01-01', periods=4)
    cache = {'600000.SH': pd.DataFrame({'date': dates, 'close': [10, 10, 4.5, 4.5],
                                       'volume': [1e8] * 4})}
    score = pd.DataFrame(1.0, index=dates, columns=['600000.SH'])
    ca = pd.DataFrame({'sym': ['600000.SH'], 'ex_date': [dates[2]],
                       'cash_ps': [1.0], 'stk_ps': [1.0]})
    cfg = ReplayConfig(top_n=1, initial_capital=100000, min_listing_days=0,
                       min_daily_turnover=0, use_risk_gate=False)
    result = DailyReplayBacktester(cache, cfg, corporate_actions=ca).run(score)
    assert result.total_return > -.01
    assert result.max_drawdown > -.01

