"""Explicit registration without discovery, default selection or import side effects."""
import re

from .base import Strategy


class StrategyRegistry:
    def __init__(self):
        self._strategies: dict[str, Strategy] = {}

    def register(self, name: str, strategy: Strategy):
        if not re.fullmatch(r'[a-z][a-z0-9_-]*', name):
            raise ValueError('Use a lowercase strategy identifier')
        if name in self._strategies:
            raise ValueError(f'Strategy already registered: {name}')
        if not callable(getattr(strategy, 'generate_targets', None)):
            raise TypeError('Strategy must implement generate_targets(context)')
        self._strategies[name] = strategy

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._strategies))

    def get(self, name: str) -> Strategy:
        if name not in self._strategies:
            raise LookupError(f'No active strategy named {name!r}; no fallback strategy is configured')
        return self._strategies[name]
