#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
日成交额计算——薄委托层（2026-10-30 收敛）

唯一实现在 data/semantics.py（R2：此前两份逐行同构的实现并存，可各自漂移）。
本模块保留原导入路径（live/signals 等历史调用方不变），仅 re-export。
"""

from data.semantics import daily_turnover, avg_daily_turnover  # noqa: F401
