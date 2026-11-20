#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DataFrame / Series 处理工具
"""

import pandas as pd


def to_scalar(val):
    """Convert pandas Series/DataFrame value to scalar float."""
    if isinstance(val, pd.Series):
        return float(val.iloc[-1]) if len(val) > 0 else 0.0
    elif isinstance(val, pd.DataFrame):
        return float(val.iloc[-1, -1]) if len(val) > 0 else 0.0
    return float(val) if val is not None else 0.0


def validate_ohlcv(data: pd.DataFrame) -> bool:
    """Check if DataFrame has required OHLCV columns."""
    return all(col in data.columns for col in ['open', 'high', 'low', 'close', 'volume'])