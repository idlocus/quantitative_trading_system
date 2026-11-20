"""Execution assumptions only; strategy parameters belong to explicit strategies.

These defaults describe the retained simulation infrastructure. No active
strategy, asset allocation, factor selection or portfolio size is configured.
"""
COMMISSION = 0.0003
STAMP_TAX = 0.0005
TRANSFER_FEE = 0.00001
SLIPPAGE = 0.001
LIMIT_TOLERANCE = 0.002
SELL_PROCEEDS_DISCOUNT = 0.95
TRADING_DAYS_PER_YEAR = 252
MAX_WEIGHT = 0.05
DRAWDOWN_SOFT = 0.08
DRAWDOWN_HARD = 0.15
MARKET_DAILY_CRASH = 0.05
PORTFOLIO_DAILY_CRASH = 0.07
STOP_LOSS_PCT = 0.05
VOL_TARGET = 0.25
MAX_TURNOVER = 0.50
MIN_CASH_RATIO = 0.05
REENTRY_MAX_LIQUIDATIONS = 3
REENTRY_CIRCUIT_WINDOW_DAYS = 365


def validate():
    """Reject internally inconsistent execution assumptions."""
    if not 0 < DRAWDOWN_SOFT < DRAWDOWN_HARD < 1:
        raise ValueError('Expected 0 < soft drawdown < hard drawdown < 1')
    if not 0 < MAX_WEIGHT <= 1 or not 0 <= MIN_CASH_RATIO < 1:
        raise ValueError('Invalid position or cash constraint')
    if REENTRY_MAX_LIQUIDATIONS < 1 or REENTRY_CIRCUIT_WINDOW_DAYS < 30:
        raise ValueError('Invalid liquidation circuit-breaker configuration')
    if any(rate < 0 for rate in (COMMISSION, STAMP_TAX, TRANSFER_FEE, SLIPPAGE)):
        raise ValueError('Execution costs must not be negative')
