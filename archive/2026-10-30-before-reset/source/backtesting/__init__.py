#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# DEPRECATED (2026-10-13): research legacy, not main chain. See docs/README.md

"""
回测模块
"""

from .engine import BacktestEngine, BacktestResult, Trade, BacktestConfig, MarketData
from .reporter import BacktestReporter
from .runner import BacktestRunner

__all__ = [
    "BacktestEngine",
    "BacktestResult",
    "Trade",
    "BacktestConfig",
    "MarketData",
    "BacktestReporter",
    "BacktestRunner",
]
