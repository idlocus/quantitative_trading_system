"""Convert explicit target weights to orders without choosing assets or factors."""
from math import isfinite
from typing import Mapping

from live.broker import Order
from strategies.base import validate_weights
from trading.config import SELL_PROCEEDS_DISCOUNT


def build_orders(target_weights: Mapping[str, float], current_weights: Mapping[str, float],
                 total_nav: float, band: float = 0.0) -> list[Order]:
    targets = validate_weights(target_weights)
    current = {symbol: float(weight) for symbol, weight in current_weights.items()}
    if not isfinite(total_nav) or total_nav <= 0 or not 0 <= band < 1:
        raise ValueError('Positive NAV and a buffer band in [0, 1) are required')
    if any(not isfinite(weight) or weight < 0 for weight in current.values()):
        raise ValueError('Current weights must be finite and nonnegative')
    sells, buys = [], []
    proceeds = 0.0
    for symbol, weight in current.items():
        target = targets.get(symbol, 0.0)
        excess = weight - target
        if excess > max(band * target, 1e-12):
            sells.append(Order(symbol, 'SELL', target, amount=excess * total_nav,
                               note='target reduction'))
            proceeds += excess
    budget = max(0.0, 1.0 - sum(current.values()) + proceeds * SELL_PROCEEDS_DISCOUNT)
    for symbol, target in targets.items():
        shortage = target - current.get(symbol, 0.0)
        if shortage > max(band * target, 1e-12):
            allocation = min(shortage, budget)
            if allocation > 0:
                buys.append(Order(symbol, 'BUY', target, amount=allocation * total_nav,
                                  note='target increase'))
                budget -= allocation
    return sells + buys
