"""Contracts separating strategy decisions from data and execution."""
from dataclasses import dataclass
from datetime import date
from math import isfinite
from types import MappingProxyType
from typing import Mapping, Protocol

import pandas as pd


def validate_weights(weights: Mapping[str, float]) -> dict[str, float]:
    """Validate a long-only allocation; an empty allocation means cash."""
    result = {}
    for symbol, value in weights.items():
        if not isinstance(symbol, str) or not symbol.strip():
            raise ValueError('Target symbols must be nonempty strings')
        if isinstance(value, bool):
            raise ValueError('Boolean target weights are invalid')
        number = float(value)
        if not isfinite(number) or number < 0:
            raise ValueError(f'Invalid target weight: {symbol}')
        if number > 0:
            result[symbol] = number
    if sum(result.values()) > 1 + 1e-10:
        raise ValueError('Target weights must sum to at most one')
    return result


@dataclass(frozen=True)
class StrategyContext:
    signal_date: date
    history: Mapping[str, pd.DataFrame]
    current_weights: Mapping[str, float]
    cash: float
    nav: float

    @classmethod
    def snapshot(cls, signal_date, cache, current_weights, cash, nav):
        """Give strategies independent copies bounded by the signal cutoff."""
        cutoff = pd.Timestamp(signal_date)
        history = {symbol: frame.loc[pd.to_datetime(frame['date']) <= cutoff].copy(deep=True)
                   for symbol, frame in cache.items()}
        history = {s: f for s, f in history.items() if not f.empty}
        return cls(signal_date, MappingProxyType(history),
                   MappingProxyType(dict(current_weights)), float(cash), float(nav))


class Strategy(Protocol):
    def generate_targets(self, context: StrategyContext) -> Mapping[str, float]:
        """Return explicit target weights using only the supplied history."""
        ...
