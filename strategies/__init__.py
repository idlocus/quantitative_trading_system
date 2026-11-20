"""Strategy contracts and an empty explicit registry. No built-in strategies."""
from .base import Strategy, StrategyContext, validate_weights
from .registry import StrategyRegistry

registry = StrategyRegistry()

__all__ = ['Strategy', 'StrategyContext', 'StrategyRegistry', 'registry', 'validate_weights']
