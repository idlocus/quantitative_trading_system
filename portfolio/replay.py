"""Strategy-independent, in-memory T-1 signal / T-close replay.

No strategy is imported or selected here. Callers supply both a strategy and
an explicit corporate-action table (which may be empty for synthetic data).
"""
from dataclasses import dataclass
from math import isfinite
from typing import Mapping

import pandas as pd

from live.broker import PaperBroker
from live.ledger import Ledger
from live.risk_gate import GateConfig, RiskGate
from strategies.base import Strategy, StrategyContext, validate_weights
from trading import config as execution
from .orders import build_orders


@dataclass(frozen=True)
class ReplayConfig:
    initial_capital: float = 1_000_000.0
    band: float = 0.0
    max_weight: float = 1.0
    slippage: float = execution.SLIPPAGE
    commission: float = execution.COMMISSION
    stamp_tax: float = execution.STAMP_TAX
    transfer_fee: float = execution.TRANSFER_FEE
    benchmark_symbol: str = '000300.SH'
    risk_config: GateConfig | None = None

    def __post_init__(self):
        if not isfinite(self.initial_capital) or self.initial_capital <= 0:
            raise ValueError('Initial capital must be finite and positive')
        if not 0 <= self.band < 1 or not 0 < self.max_weight <= 1:
            raise ValueError('Invalid weight cap or buffer band')
        for value in (self.slippage, self.commission, self.stamp_tax, self.transfer_fee):
            if not isfinite(value) or value < 0:
                raise ValueError('Execution costs must be finite and nonnegative')


@dataclass(frozen=True)
class ReplayResult:
    equity: pd.Series
    trades: tuple[dict, ...]
    cash: float
    positions: dict[str, int]
    total_return: float
    max_drawdown: float
    risk_actions: tuple[tuple[str, str], ...]


class ReplayEngine:
    def __init__(self, config: ReplayConfig | None = None):
        self.config = config or ReplayConfig()

    def run(self, strategy: Strategy, market_data: Mapping[str, pd.DataFrame], *,
            corporate_actions: pd.DataFrame,
            st_matrix: pd.DataFrame | None = None) -> ReplayResult:
        if not callable(getattr(strategy, 'generate_targets', None)):
            raise ValueError('An explicit strategy implementing generate_targets is required')
        cache = self._prepare(market_data)
        dates = pd.DatetimeIndex(sorted({dt for frame in cache.values() for dt in frame['date']}))
        if len(dates) < 2:
            raise ValueError('Replay requires at least two market dates')
        from data.corporate_actions import by_date_index
        required = {'sym', 'ex_date', 'cash_ps', 'stk_ps'}
        if not required.issubset(corporate_actions.columns):
            raise ValueError('Supply an explicit corporate-action table, including its schema')
        action_frame = corporate_actions.copy(deep=True)
        action_frame['ex_date'] = pd.to_datetime(action_frame['ex_date'], errors='coerce')
        if action_frame['ex_date'].isna().any():
            raise ValueError('Invalid or duplicate corporate-action dates')
        if not action_frame['sym'].map(lambda value: isinstance(value, str) and bool(value.strip())).all():
            raise ValueError('Invalid corporate-action symbols')
        for column in ('cash_ps', 'stk_ps'):
            values = pd.to_numeric(action_frame[column], errors='coerce')
            if not values.map(lambda value: isfinite(value) and value >= 0).all():
                raise ValueError('Invalid corporate-action amounts')
            action_frame[column] = values
        actions = by_date_index(action_frame)
        cfg = self.config
        ledger = Ledger(None)
        ledger.initialize(cfg.initial_capital, as_of=dates[0].date())
        broker = PaperBroker(slippage=cfg.slippage, commission=cfg.commission,
                             stamp_tax=cfg.stamp_tax, transfer_fee=cfg.transfer_fee)
        gate = RiskGate(cfg.risk_config, state_path=None) if cfg.risk_config is not None else None
        risk_actions = []
        last_prices = {}
        for i, timestamp in enumerate(dates):
            day = timestamp.date()
            prices = self._prices(cache, timestamp)
            previous = dict(last_prices)
            if i:
                context = StrategyContext.snapshot(dates[i - 1].date(), cache,
                    ledger.current_weights(previous), ledger.cash, ledger.last_nav())
                targets = validate_weights(strategy.generate_targets(context))
                if set(targets) - set(context.history):
                    raise ValueError('Strategy returned symbols outside the supplied market data')
                if any(weight > cfg.max_weight + 1e-12 for weight in targets.values()):
                    raise ValueError('Strategy target exceeds the explicit weight cap')
            self._apply_actions(ledger, actions.get(day, {}), previous, day)
            last_prices = {**previous, **prices}
            nav = ledger.mark_to_market(prices, day)
            if not i:
                continue
            orders = build_orders(targets, ledger.current_weights(prices), nav, cfg.band)
            if gate is not None:
                benchmark = cache.get(cfg.benchmark_symbol)
                if benchmark is not None:
                    benchmark = benchmark.loc[benchmark['date'] <= timestamp]
                st_ok = st_matrix is not None and timestamp in st_matrix.index
                if st_ok and gate.cfg.st_data_required:
                    flags = st_matrix.loc[timestamp]
                    unknown = {s for s in targets if s not in flags or pd.isna(flags[s])
                               or flags[s] not in (True, False)}
                    if unknown == set(targets) and unknown:
                        st_ok = False
                    else:
                        orders = [o for o in orders if o.side != 'BUY' or o.symbol not in unknown]
                decision = gate.review(orders, ledger, prices, previous, list(targets),
                                       bench_df=benchmark, as_of=day, st_data_ok=st_ok)
                risk_actions.append((str(day), decision.action))
                orders = decision.orders
            st_symbols = set()
            if st_matrix is not None and timestamp in st_matrix.index:
                row = st_matrix.loc[timestamp]
                st_symbols = set(row.index[row.eq(True).fillna(False)])
            fills = broker.execute(orders, prices, day, prev_prices=previous,
                                   available_cash=ledger.cash, positions=dict(ledger.positions),
                                   st_symbols=st_symbols)
            _, _, blocked = ledger.apply_fills(fills)
            if blocked:
                raise RuntimeError('Replay accounting blocked an invalid fill')
            ledger.mark_to_market(prices, day)
        equity = pd.Series({pd.Timestamp(item['date']): item['nav'] for item in ledger.nav_series()})
        return ReplayResult(equity, tuple(dict(item) for item in ledger._d['trades']),
                            ledger.cash, dict(ledger.positions),
                            float(equity.iloc[-1] / cfg.initial_capital - 1),
                            float((equity / equity.cummax() - 1).min()), tuple(risk_actions))

    @staticmethod
    def _prepare(market_data):
        cache = {}
        for symbol, frame in market_data.items():
            if frame is None or frame.empty:
                continue
            if not {'date', 'close'}.issubset(frame.columns):
                raise ValueError(f'Missing date/close columns: {symbol}')
            frame = frame.copy(deep=True)
            frame['date'] = pd.to_datetime(frame['date'])
            if frame['date'].isna().any() or frame['date'].duplicated().any():
                raise ValueError(f'Invalid or duplicate dates: {symbol}')
            prices = pd.to_numeric(frame['close'], errors='coerce')
            if not prices.map(lambda value: isfinite(value) and value > 0).all():
                raise ValueError(f'Invalid raw prices: {symbol}')
            frame['close'] = prices
            cache[symbol] = frame.sort_values('date').reset_index(drop=True)
        return cache

    @staticmethod
    def _prices(cache, timestamp):
        return {symbol: float(frame.loc[frame['date'] == timestamp, 'close'].iloc[-1])
                for symbol, frame in cache.items() if frame['date'].eq(timestamp).any()}

    @staticmethod
    def _apply_actions(ledger, actions, previous, day):
        for symbol, (cash_ps, stock_ps) in actions.items():
            quantity = ledger.positions.get(symbol, 0)
            if symbol in previous:
                previous[symbol] = (previous[symbol] - cash_ps) / (1 + stock_ps)
            if quantity > 0 and cash_ps > 0:
                ledger.apply_dividend(symbol, quantity * cash_ps, ex_date=str(day))
            if quantity > 0 and stock_ps > 0:
                ledger.apply_split(symbol, 1 + stock_ps, ex_date=str(day))
