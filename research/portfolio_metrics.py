"""Daily risk accounting and monthly metrics without resetting lifetime peaks."""
from dataclasses import dataclass
from math import isfinite

import numpy as np
import pandas as pd


@dataclass
class DailyRisk:
    initial: float
    loss_limit: float = .15
    drawdown_limit: float = .15
    adverse_shock: float = .25
    monthly_amount_limit: float | None = None
    peak_amount_limit: float | None = None
    peak: float = 0.
    previous_close: float = 0.
    month_base: float = 0.
    month: tuple | None = None
    halted: bool = False

    def __post_init__(self):
        if not isfinite(self.initial) or self.initial <= 0:
            raise ValueError('Risk baseline must be finite and positive')
        if not 0 < self.loss_limit < 1 or not 0 < self.drawdown_limit < 1 or not 0 < self.adverse_shock <= 1:
            raise ValueError('Invalid risk limits')
        if any(value is not None and (not isfinite(value) or value <= 0) for value in (self.monthly_amount_limit, self.peak_amount_limit)):
            raise ValueError('Invalid absolute risk limits')
        self.peak = self.previous_close = self.month_base = self.initial

    def begin_day(self, day):
        month = (day.year, day.month)
        if month != self.month:
            self.month_base = self.previous_close
            self.month = month

    def observe(self, nav: float, *, closing: bool = False) -> dict:
        if not isfinite(nav) or nav < 0:
            raise ValueError('Invalid account NAV')
        if closing:
            self.peak = max(self.peak, nav)
            self.previous_close = nav
        drawdown = nav / self.peak - 1
        month_return = nav / self.month_base - 1 if self.month_base else -1.
        drawdown_amount = max(0., self.peak - nav)
        month_loss_amount = max(0., self.month_base - nav)
        breached = (drawdown < -self.drawdown_limit - 1e-10 or month_return < -self.loss_limit - 1e-10
            or (self.peak_amount_limit is not None and drawdown_amount > self.peak_amount_limit + 1e-6)
            or (self.monthly_amount_limit is not None and month_loss_amount > self.monthly_amount_limit + 1e-6))
        self.halted |= breached
        return {'drawdown': drawdown, 'month_to_date': month_return, 'breached': breached,
                'drawdown_amount': drawdown_amount, 'month_loss_amount': month_loss_amount}

    def exposure_cap(self, nav: float, gross_target: float) -> float:
        if self.halted or nav <= 0:
            return 0.
        floor = max(self.peak * (1 - self.drawdown_limit), self.month_base * (1 - self.loss_limit))
        if self.peak_amount_limit is not None:
            floor = max(floor, self.peak - self.peak_amount_limit)
        if self.monthly_amount_limit is not None:
            floor = max(floor, self.month_base - self.monthly_amount_limit)
        return min(gross_target, max(0., (nav - floor) / (self.adverse_shock * nav)))


def performance(nav: pd.Series, initial: float) -> dict:
    if nav.empty or not np.isfinite(nav.to_numpy()).all() or (nav <= 0).any():
        raise ValueError('Metrics require positive finite NAV observations')
    monthly_end = nav.groupby(nav.index.to_period('M')).last()
    bases = monthly_end.shift(1)
    bases.iloc[0] = initial
    monthly = monthly_end / bases - 1
    peak = nav.cummax().clip(lower=initial)
    daily_bases = pd.Series(nav.index.to_period('M').map(bases), index=nav.index, dtype=float)
    losing_run = current = 0
    for value in monthly:
        current = current + 1 if value < 0 else 0
        losing_run = max(losing_run, current)
    underwater_run = underwater = 0
    for below_peak in nav.lt(peak - 1e-8):
        underwater = underwater + 1 if below_peak else 0
        underwater_run = max(underwater_run, underwater)
    return {'total_return': float(nav.iloc[-1] / initial - 1),
        'geometric_monthly_return': float((nav.iloc[-1] / initial) ** (1 / len(monthly)) - 1),
        'median_monthly_return': float(monthly.median()), 'winning_month_fraction': float((monthly > 0).mean()),
        'worst_calendar_month': float(monthly.min()),
        'worst_daily_month_to_date': float((nav / daily_bases - 1).min()),
        'max_drawdown': float((nav / peak - 1).min()), 'longest_losing_month_run': losing_run,
        'max_drawdown_amount': float((peak - nav).max()),
        'max_daily_month_loss_amount': float((daily_bases - nav).clip(lower=0).max()),
        'longest_underwater_sessions': underwater_run, 'recovered_by_end': bool(nav.iloc[-1] >= peak.iloc[-1] - 1e-8),
        'months': len(monthly), 'monthly_returns': {str(key): float(value) for key, value in monthly.items()}}


def amount_turnover_per_month(trades, nav) -> float:
    """Average monthly two-sided traded notional divided by each day's closing NAV."""
    if trades is None or len(trades) == 0:
        return 0.0
    closes = nav.set_index(pd.to_datetime(nav['date']))['nav']
    notionals = (trades['price'] * trades['quantity']).groupby(pd.to_datetime(trades['date'])).sum()
    daily_nav = closes.reindex(notionals.index)
    if daily_nav.isna().any() or not np.isfinite(daily_nav).all() or (daily_nav <= 0).any():
        raise ValueError('Missing or invalid trade-day NAV')
    return float((notionals / daily_nav).sum() / closes.index.to_period('M').nunique())
