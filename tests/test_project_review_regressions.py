from datetime import date, datetime
import json
import os
import subprocess
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from live.broker import Fill, Order, PaperBroker
from live.ledger import Ledger
from live.qmt_broker import QMTBroker, QMTOrderState
from live.reconcile import BrokerStatement, Reconciler
from research.portfolio_simulation import PortfolioSimulation
from tests.test_portfolio_baseline import bar, config


def test_blocked_exit_keeps_its_slot_and_missing_scores_keep_the_position():
    days = pd.bdate_range('2020-01-02', periods=3)
    cfg = dict(config(), gross_target=.6, rebalance_sessions=1)
    sim = PortfolioSimulation('synthetic', cfg, .0025, False, [], days, {})
    sim.step(0, days[0], {'A': bar()}, ['A'], {'A'})
    sim.step(1, days[1], {'A': bar(lower=10), 'B': bar()}, ['B'], {'B'})
    assert list(sim.positions) == ['A']
    other = PortfolioSimulation('synthetic', cfg, 0, False, [], days, {})
    other.step(0, days[0], {'B': bar()}, ['B'], {'B'})
    other.step(1, days[1], {'B': bar()}, None, set())
    assert list(other.positions) == ['B']
    other.step(2, days[2], {'B': bar()}, [], set())
    assert not other.positions


def test_partial_weight_map_does_not_invent_allocations():
    days = pd.bdate_range('2020-01-02', periods=2)
    sim = PortfolioSimulation('synthetic', dict(config(), positions=2), 0, False, [], days, {},
                              daily_weights={days[0]: {'B': .3}})
    sim.step(1, days[1], {'A': bar(), 'B': bar()}, ['A', 'B'], {'A', 'B'})
    assert list(sim.positions) == ['B']


def test_invalid_accounting_inputs_and_failed_write_preserve_balances(tmp_path):
    path = tmp_path / 'ledger.json'
    ledger = Ledger(str(path))
    ledger.initialize(1000, date(2020, 1, 2))
    for func, value in [(ledger.apply_dividend, np.nan), (ledger.apply_split, np.inf)]:
        with pytest.raises(ValueError):
            func('A', value)
    order = Order('A', 'BUY', 0.)
    fill = Fill(order, 5., 10, 0., 0., date(2020, 1, 3), trade_id='one')
    assert ledger.apply_fills([fill])[0]
    conflicting = Fill(order, 6., 10, 0., 0., fill.trade_date, trade_id='one')
    bad = Fill(order, None, None, 0., 0., fill.trade_date)
    assert ledger.apply_fills([conflicting, bad])[2] == [conflicting, bad]
    with pytest.raises(ValueError):
        ledger.mark_to_market({}, date(2020, 1, 2))
    ledger.mark_to_market({'A': 5.}, date(2020, 1, 3))
    ledger.apply_split('A', 2., '2020-01-06')
    assert ledger.mark_to_market({}, date(2020, 1, 6)) == 1000
    path.write_bytes(b'{')
    with pytest.raises(ValueError):
        ledger.deposit(50)
    assert path.read_bytes() == b'{' and ledger.cash == 950


def test_historical_reconciliation_and_cross_date_attachment():
    ledger = Ledger(None)
    ledger.initialize(100, date(2020, 1, 2))
    ledger.apply_dividend('A', 3, '2020-01-06')
    statement = BrokerStatement('2020-01-03', 100, {})
    reconciler = Reconciler()
    with pytest.raises(ValueError):
        reconciler.apply_adjustments(ledger, statement, reconciler.reconcile(ledger, statement))
    ledger.record_recon('2020-01-06', True, 0, snapshot='day1')
    ledger.record_recon('2020-01-07', True, 0)
    assert not ledger._d['meta']['last_recon']['snapshot']
    with pytest.raises(ValueError):
        BrokerStatement('2020-01-07', 0, {'A': -2})


def test_qmt_ambiguous_ack_and_stale_writer(tmp_path):
    path = str(tmp_path / 'orders.json')
    first, second = (QMTBroker(state_path=path, enforce_window=False) for _ in range(2))
    calls = []
    stub = SimpleNamespace(order_stock_async=lambda *args: calls.append(args))
    first._xt_trader = second._xt_trader = stub
    first._connected = second._connected = True
    buy = Order('A', 'BUY', .1, quantity=100)
    fills = first.execute([buy, buy], {'A': 10.}, date.today(), available_cash=5000)
    assert len(calls) == 1 and first.query_failed and not any(f.quantity for f in fills)
    with pytest.raises(RuntimeError):
        second._submit_order(buy, 100, 10.)
    ref = next(iter(first._pending))
    first.on_stock_order(SimpleNamespace(order_id=7, stock_code='A', order_remark='foreign'))
    assert first._pending[ref]['oid'] is None and len(calls) == 1


@pytest.mark.parametrize('state,qty,price', [(56, 10, 1.), (55, 101, 1.), (55, 50, 0.), (55, 50, np.nan)])
def test_broker_refuses_inconsistent_execution(tmp_path, state, qty, price):
    broker = QMTBroker(state_path=str(tmp_path / 'orders.json'))
    broker._pending['ref'] = dict(symbol='A', side='BUY', qty=100, booked=0)
    fill = broker._book_order_fill(Order('A', 'BUY', 0.), 'ref', state, qty, price,
                                   1., 100, date.today(), Ledger(None))
    assert broker.query_failed and not fill.quantity and 'ref' in broker._pending


def test_broker_books_partial_executions_using_exchange_ids_and_dates(tmp_path):
    broker = QMTBroker(state_path=str(tmp_path / 'orders.json'))
    ledger = Ledger(None)
    ledger.initialize(2000, date(2020, 1, 2))
    broker._pending['ref'] = dict(oid='7', qty=100, symbol='A', side='BUY', booked=0,
                                  submitted_at='2020-01-03')
    times = [datetime(2020, 1, 6, 10), datetime(2020, 1, 6, 11)]
    rows = [SimpleNamespace(order_id=7, stock_code='A', traded_id=str(i),
                traded_time=t.timestamp(), order_type=23, traded_price=10., traded_volume=50)
            for i, t in enumerate(times, 1)]
    broker._xt_trader = SimpleNamespace(query_stock_trades=lambda account: rows,
        query_stock_orders=lambda account: [SimpleNamespace(order_id=7, order_remark='ref',
                       order_status=QMTOrderState.FILLED, traded_volume=100, traded_price=10.)])
    assert sum(f.quantity for f in broker.sync_pending(ledger)) == 100
    stmt = BrokerStatement('2020-01-06', 1000, {'A': 100},
                           [broker._execution_row(row) for row in rows], source='qmt')
    assert Reconciler().reconcile(ledger, stmt).is_ok and not broker._pending


def test_paper_and_qmt_reject_invalid_orders_before_submitting(tmp_path):
    broker = QMTBroker(state_path=str(tmp_path / 'orders.json'), enforce_window=False)
    broker._connected = True
    orders = [Order('X', 'TYPO', 0, quantity=100), Order('Y', 'BUY', 0, quantity=100),
              Order('Z', 'SELL', 0, amount=1000)]
    for adapter in (PaperBroker(), broker):
        fills = adapter.execute(orders, {'X': 10., 'Y': np.nan, 'Z': 10.},
                                date.today(), available_cash=10000, positions={})
        assert len(fills) == 3 and all(f.quantity == 0 for f in fills)


def test_research_and_data_boundaries(tmp_path):
    from data.semantics import actual_cash_ratio, current_drawdown, extract_prices
    from research.overfitting import TrialRegistry
    from research.portfolio_metrics import amount_turnover_per_month
    from research.robustness import RobustnessReport
    from research.signal_screen import daily_statistics
    dates = pd.to_datetime(['2020-01-31', '2020-02-03'])
    nav = pd.DataFrame({'date': dates, 'nav': [100., 200.]})
    trades = pd.DataFrame({'date': dates, 'price': [10., 10.], 'quantity': [1, 4]})
    assert amount_turnover_per_month(trades, nav) == pytest.approx(.15)
    prices = pd.DataFrame({'close': [2., 5.]}, index=dates)
    assert extract_prices({'A': prices}, ['A'], dates[0]) == {'A': 2.}
    with pytest.raises(ValueError):
        actual_cash_ratio(10, {'A': 2}, {})
    assert actual_cash_ratio(10, {'A': 2}, {}, {'A': 5}) == .5
    assert current_drawdown([{'date': '2020-01-02', 'nav': 100}, {'date': '2020-01-03', 'nav': 50}],
                            [{'date': '2020-01-03', 'type': 'withdraw', 'amount': -50}])[0] == 0
    a, b = (TrialRegistry(str(tmp_path / 'registry.json')) for _ in range(2))
    a.record('one')
    assert b.record('two') == 2
    assert '数据不足' in RobustnessReport().summary_markdown()
    scores = pd.DataFrame([range(10)], dtype=float)
    labels = scores.copy()
    labels.iloc[0, -1] = np.nan
    result = daily_statistics(scores, labels, scores.notna(), 5)
    assert np.isnan(result.top_decile_forward_mean.iloc[0])
    assert result.bottom_decile_forward_mean.iloc[0] == 0


def test_failed_publication_preserves_all_previous_artifacts(tmp_path):
    from research.portfolio_study import publish_study
    dest = tmp_path / 'published'
    with publish_study(dest) as stage:
        (stage / 'inputs.json').write_bytes(b'{}')
    manifest = (dest / 'portfolio_manifest.json').read_bytes()
    with pytest.raises(ValueError, match='new destination'):
        with publish_study(dest) as stage:
            (stage / 'inputs.json').write_bytes(b'{"changed":true}')
            (stage / 'another.csv').write_bytes(b'new')
    assert not (dest / 'another.csv').exists()
    assert (dest / 'inputs.json').read_bytes() == b'{}'
    assert (dest / 'portfolio_manifest.json').read_bytes() == manifest


@pytest.mark.parametrize('extra', ['obsolete.csv', 'nested/.old.lock'])
def test_publication_rejects_unexpected_files_before_backfilling(tmp_path, extra):
    from research.portfolio_study import publish_study
    dest = tmp_path / 'published'
    dest.mkdir()
    stale = dest / extra
    stale.parent.mkdir(exist_ok=True)
    stale.write_bytes(b'old')
    with pytest.raises(ValueError, match='Unexpected experiment files'):
        with publish_study(dest) as stage:
            (stage / 'new.csv').write_bytes(b'new')
    assert stale.read_bytes() == b'old'
    assert not (dest / 'new.csv').exists()
    assert not (dest / 'portfolio_manifest.json').exists()


def test_atomic_publication_inherits_normal_directory_access(tmp_path):
    from research.portfolio_study import publish_study
    dest = tmp_path / 'published'
    for attempt in range(2):
        with publish_study(dest) as stage:
            (stage / 'inputs.json').write_bytes(b'{}')
            if attempt == 0:
                assert not dest.exists()
    assert (dest / 'inputs.json').read_bytes() == b'{}'
    if os.name == 'nt':
        result = subprocess.run(
            ['powershell', '-NoProfile', '-NonInteractive', '-Command',
             '[IO.Directory]::GetAccessControl($env:TEST_RELEASE_PATH).AreAccessRulesProtected'],
            env=dict(os.environ, TEST_RELEASE_PATH=str(dest)), check=True,
            capture_output=True, text=True)
        assert result.stdout.strip() == 'False'
    else:
        reference = tmp_path / 'reference'
        reference.mkdir(mode=0o755)
        assert dest.stat().st_mode == reference.stat().st_mode


def test_registry_refuses_inconsistent_count_and_observed_rollback(tmp_path):
    from research.overfitting import TrialRegistry
    path = tmp_path / 'registry.json'
    registry = TrialRegistry(str(path))
    assert registry.record('first') == 1
    for count, reason in [(2, '不一致'), (0, '低于')]:
        prior = json.dumps({'count': count, 'trials': []})
        path.write_text(prior, encoding='utf-8')
        with pytest.raises(RuntimeError, match=reason):
            registry.record('second')
        assert path.read_text(encoding='utf-8') == prior


def test_wind_retries_incomplete_values_and_isolates_cached_results():
    from data.fundamentals import WindFundamentalProvider
    days = list(pd.bdate_range('2026-04-06', periods=2))
    responses = iter([[1, np.inf], [1, 2]])
    wind = SimpleNamespace(wsd=lambda *args: SimpleNamespace(
        ErrorCode=0, Times=days, Codes=['A'], Data=[next(responses)]))
    provider = WindFundamentalProvider(wind)
    args = ('pe_ttm', ['A'], days[0].date(), days[-1].date())
    assert pd.isna(provider.get_field_matrix(*args).iloc[-1, 0])
    complete = provider.get_field_matrix(*args)
    assert complete.iloc[-1, 0] == 2
    complete.iloc[-1, 0] = 99
    assert provider.get_field_matrix(*args).iloc[-1, 0] == 2


def test_field_cache_preserves_interleaved_fetch_and_does_not_rewrite_hits(tmp_path, monkeypatch):
    from data.layered import LayeredDataSource
    source = LayeredDataSource(str(tmp_path))
    days = pd.bdate_range('2026-04-06', periods=2)
    start, end = days[0].date(), days[-1].date()
    monkeypatch.setattr(source, 'get_kline_calendar', lambda *args: days)
    def fetch(field, symbols, first, last):
        if symbols == ['A']:
            source.get_field_matrix(field, ['B'], first, last)
        return pd.DataFrame(1., index=days, columns=symbols)
    monkeypatch.setattr(source, '_client_src_matrix', fetch)
    source.get_field_matrix('dividendyield2', ['A'], start, end)
    cache = source.fund_dir / 'dividendyield2.parquet'
    assert set(pd.read_parquet(cache)) == {'A', 'B'}
    def reject_write(*args, **kwargs):
        pytest.fail('A complete cache hit must not rewrite the published file')
    monkeypatch.setattr(pd.DataFrame, 'to_parquet', reject_write)
    assert list(source.get_field_matrix('dividendyield2', ['B'], start, end)) == ['B']


def test_daily_report_with_unchanged_book_and_unknown_leverage(tmp_path, monkeypatch):
    from scripts import run_cta_daily as daily
    days = pd.bdate_range('2020-01-02', periods=daily.cs.MIN_BARS + 3)
    frame = pd.DataFrame(dict(open=10., close=10., compressed=True, direction=1.,
                              vol_pct=.1, clarity=.1, vol20=.1, ma=9.), index=days)
    raw = frame.rename(columns={'close': '收盘价', 'open': '开盘价'})
    monkeypatch.setattr(daily, 'OUTPUT_DIR', tmp_path)
    monkeypatch.setattr(daily, 'SYMBOLS', {'AU0': ('Sample', 'AU', 'Exchange')})
    monkeypatch.setattr(daily, 'fetch_frame', lambda symbol: raw)
    monkeypatch.setattr(daily, '_expected_last_trading_day', lambda: days[-1].date())
    monkeypatch.setattr(daily.cs, 'signal_frame', lambda *args, **kw: frame)
    monkeypatch.setattr(daily.cs, 'next_leverage', lambda *args: None)
    monkeypatch.setattr(daily.time, 'sleep', lambda seconds: None)
    daily.main()
    report = next(tmp_path.glob('*.md')).read_text(encoding='utf-8')
    assert '无法判断调仓数量' in report
    assert '今日无品种级操作' not in report


def test_risk_gate_scales_executable_quantity_and_preserves_hard_exit(tmp_path):
    from live.risk_gate import RiskDecision, RiskGate
    from tests.test_risk_gate import AS_OF, bench_df, make_ledger
    gate = RiskGate(state_path=None)
    gate._stale = set()
    ledger = make_ledger(tmp_path, cash=10_000)
    orders = gate._assemble([Order('A', 'BUY', 1, quantity=900, amount=9000)], [],
                            'scale', .3, ledger, {'A': 10.}, RiskDecision(), AS_OF)
    assert len(orders) == 1 and orders[0].quantity == 0 and orders[0].amount == 3000
    ledger = make_ledger(tmp_path, positions={'A': 900}, costs={'A': 10.}, cash=1000,
                         navs=[15000, 10000],
                         trades=[dict(symbol='A', side='BUY', date=str(AS_OF), quantity=100)])
    ledger._d['meta']['last_recon'] = {'date': str(AS_OF), 'ok': False}
    decision = gate.review([], ledger, {'A': 10.}, {'A': 10.}, ['A'], bench_df(AS_OF), AS_OF)
    assert decision.action == 'liquidate'
    assert [(o.symbol, o.quantity) for o in decision.orders] == [('A', 800)]


def test_qmt_cancel_request_requires_confirmed_terminal_state(tmp_path):
    broker = QMTBroker(state_path=str(tmp_path / 'orders.json'))
    broker._connected = True
    active = SimpleNamespace(order_id=10, order_status=QMTOrderState.SUBMITTED)
    broker._xt_trader = SimpleNamespace(query_stock_orders=lambda _: [active],
                                         cancel_order_stock_async=lambda *_: 7)
    assert broker.cancel_all_pending() == 1
    active.order_status = QMTOrderState.CANCELLED
    assert broker.cancel_all_pending() == 0


def test_prepared_date_and_company_action_aggregation():
    from data.corporate_actions import by_date_index
    from data.preparation import validate_bars
    frame = pd.DataFrame({key: [5.] for key in ('open', 'high', 'low', 'close',
                                                'volume', 'amount', 'adjclose')})
    frame['date'] = [20200102]
    assert validate_bars(frame)['first_date'] == '2020-01-02'
    events = pd.DataFrame(dict(sym=['A', 'A'], ex_date=pd.to_datetime(['2020-01-02'] * 2),
                               cash_ps=[.5, .25], stk_ps=[.1, .2]))
    assert by_date_index(events)[date(2020, 1, 2)]['A'] == pytest.approx((.75, .3))


def test_close_comparator_reentry_excludes_flat_period():
    from research.cta_signal import portfolio_daily
    days = pd.bdate_range('2020-01-02', periods=5)
    frame = pd.DataFrame(dict(close=[100., 100., 100., 180., 198.],
                              direction=[1., 0., 0., 1., 0.], compressed=True,
                              vol_pct=.05, clarity=1.), index=days)
    returns = portfolio_daily({'A': frame}, execution='close', charge_cost=False)
    assert returns.tolist() == pytest.approx([0., 0., 0., 0., .1])


def test_corporate_action_cache_cannot_mask_another_year(tmp_path, monkeypatch):
    from data import corporate_actions as ca
    monkeypatch.setattr(ca, 'CACHE', tmp_path / 'events.parquet')
    spans = []
    cursor = SimpleNamespace(execute=lambda _, binds: spans.append(binds),
                             fetchall=lambda: [('A', spans[-1]['s'], .2, 0)])
    source = SimpleNamespace(_db_src=lambda: SimpleNamespace(_cursor=lambda: cursor))
    for year in [2019, 2020, 2020]:
        result = ca.load_corporate_actions(source, ['A'], f'{year}0101', f'{year}1231')
        assert result.ex_date.iloc[0].year == year
    assert len(spans) == 2


def test_stale_risk_gate_and_missing_risk_prices(tmp_path):
    from live.risk_gate import RiskGate
    from tests.test_risk_gate import AS_OF, bench_df, make_ledger
    path = str(tmp_path / 'risk.json')
    first, stale = RiskGate(state_path=path), RiskGate(state_path=path)
    first._state['protocol_halted'] = True
    first._save_state()
    with pytest.raises(RuntimeError):
        stale._save_state()
    ledger = make_ledger(tmp_path)
    decision = stale.review([], ledger, {}, {}, [], bench_df(AS_OF), AS_OF)
    assert decision.action == 'halt'
    gate = RiskGate(state_path=None)
    future = bench_df(AS_OF)
    future['date'] += pd.Timedelta(days=1)
    assert gate.review([], ledger, {}, {}, [], future, AS_OF).action == 'halt'
    assert gate.review([], ledger, {'A': np.nan}, {}, ['A'], bench_df(AS_OF), AS_OF).action == 'halt'


def test_infeasible_style_budget_is_reported():
    from portfolio.optimizer import OptimizerConfig, solve_weights
    symbols = list('ABCDEF')
    alpha = pd.Series(range(6), index=symbols, dtype=float)
    returns = pd.DataFrame(np.random.default_rng(11).normal(size=(20, 6)), columns=symbols)
    style = pd.DataFrame({'size': np.ones(6)}, index=symbols)
    with pytest.raises(ValueError, match='infeasible'):
        solve_weights(alpha, returns, style, {'size': 0.}, config=OptimizerConfig(max_weight=.3))


def test_empty_cta_history_and_nonoverlapping_benchmark():
    from research.cta_signal import criteria_verdict, held_leverage, next_leverage
    from scripts.run_cta_daily import _append_symbol_table
    empty = pd.Series(dtype=float)
    assert next_leverage(empty) is None and held_leverage(empty) is None
    lines = []
    _append_symbol_table(lines, {'A': pd.DataFrame()}, {'A': ('Sample', 'A')})
    series = pd.Series(np.repeat(.01, 60), index=pd.bdate_range('2019-01-02', periods=60))
    shifted = series.set_axis(series.index + pd.DateOffset(years=1))
    assert not criteria_verdict(series, shifted, '2019-02-01')['base_verifiable']
