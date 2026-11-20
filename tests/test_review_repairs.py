"""Retained infrastructure regressions; policy-specific tests are archived."""
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


class TestRiskGateReentryProtocol:
    """审计 2026-11-19 三-5 裁决（用户确认保留现行为）：通用 RiskGate 在
    硬限清仓后"以现金为新锚立即满仓重入 + 365 日重入熔断"（2026-10-15
    十年重放实证），与研究评价口径（回撤锚跨清仓终身延续，见
    docs/RESEARCH_MANDATE.md）有意分离。守护测试锁住现行为与熔断。"""

    @staticmethod
    def _hollow_ledger(tmp_path):
        led = Ledger(str(tmp_path / 'ledger.json'))
        led._d.update({
            'cash': 840_000.0, 'positions': {}, 'cost': {}, 'trades': [],
            'nav_history': [
                {'date': '2026-01-01', 'nav': 1_000_000,
                 'cash': 1_000_000, 'market_value': 0},
                {'date': '2026-01-02', 'nav': 840_000,
                 'cash': 840_000, 'market_value': 0}],
            'cash_log': [{'date': '2026-01-01', 'type': 'deposit',
                          'amount': 1_000_000}],
        })
        led.save()
        return led

    @staticmethod
    def _bench():
        dates = pd.bdate_range('2026-01-01', periods=3)
        return pd.DataFrame({'date': dates, 'close': [4000.0, 3990.0, 4000.0]})

    def test_hard_liquidation_reanchors_cash_and_reenters(self, tmp_path):
        led = self._hollow_ledger(tmp_path)      # -16% ≥ DRAWDOWN_HARD 15%
        gate = RiskGate(state_path=str(tmp_path / 'risk.json'))
        d = gate.review([], led, {}, {}, [], bench_df=self._bench(),
                        as_of=date(2026, 1, 5))
        assert d.action == 'trade' and d.exposure == 1.0
        assert any('立即重入' in r or '新锚' in r for r in d.reasons)
        assert gate._state['dd_anchor'] == pytest.approx(840_000)

    def test_reentry_circuit_breaker_halts_after_max(self, tmp_path):
        led = self._hollow_ledger(tmp_path)
        gate = RiskGate(state_path=str(tmp_path / 'risk.json'))
        gate._state['liq_dates'] = ['2026-01-01', '2026-01-02', '2026-01-03']
        gate._save_state()
        d = gate.review([], led, {}, {}, [], bench_df=self._bench(),
                        as_of=date(2026, 1, 6))
        assert d.action == 'halt'
        assert gate._state.get('protocol_halted') is True


def test_raw_snapshot_unique_per_run_and_validated(tmp_path, monkeypatch):
    """审计 2026-11-19 三-8：同日多次对账不得互相覆盖快照；
    外部 as_of 严格解析，不得直接进文件名/路径。"""
    import os
    from scripts import auto_reconcile
    monkeypatch.setattr(auto_reconcile, 'RAW_DIR', str(tmp_path / 'raw'))
    data = {'as_of': '2026-10-30', 'cash': 100.0, 'positions': {}}
    p1 = auto_reconcile.save_raw_snapshot(data)
    p2 = auto_reconcile.save_raw_snapshot(data)                 # 同日同内容
    p3 = auto_reconcile.save_raw_snapshot({**data, 'cash': 200.0})
    assert p1 != p2, '同日第二次对账不得覆盖第一次快照'
    assert p2 != p3
    for p in (p1, p2, p3):
        assert os.path.basename(p).startswith('2026-10-30_')
        assert os.path.isfile(p)
    with pytest.raises(ValueError, match='非法'):
        auto_reconcile.save_raw_snapshot({'as_of': '../../evil'})
    # 对账记录携带快照标识
    led = Ledger(str(tmp_path / 'ledger.json'))
    led.initialize(10_000)
    led.record_recon('2026-10-30', True, 0, source='manual', snapshot=p1)
    assert led._d['meta']['last_recon']['snapshot'] == p1
