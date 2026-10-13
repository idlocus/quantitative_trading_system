# DEPRECATED (2026-10-13): research legacy, not main chain. See docs/README.md

from .config import IndicatorRule, SelectionConfig, ScoredStock
from .scanner import StockScanner, load_strategy_from_yaml

__all__ = [
    "IndicatorRule",
    "SelectionConfig",
    "ScoredStock",
    "StockScanner",
    "load_strategy_from_yaml",
]