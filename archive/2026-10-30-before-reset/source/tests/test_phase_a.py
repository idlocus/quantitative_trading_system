#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase A 测试：过拟合防御（DSR）+ 公告日 PIT 财务因子"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import pytest

from research.overfitting import deflated_sharpe_ratio, expected_max_sharpe, TrialRegistry
from data.fundamental_pit import build_pit_matrix


class TestDeflatedSharpe:

    def test_noise_fails_at_high_trials(self):
        """纯噪声策略在大量尝试次数下 DSR 应低（不能声称显著）"""
        rng = np.random.default_rng(7)
        rets = rng.normal(0, 0.01, 1000)  # 无真实alpha
        dsr1 = deflated_sharpe_ratio(rets, n_trials=1)
        dsr50 = deflated_sharpe_ratio(rets, n_trials=50)
        assert dsr50.dsr < dsr1.dsr, "尝试次数越多，同一夏普越不显著"
        assert dsr50.dsr < 0.95, "噪声+50次尝试不应通过"

    def test_true_signal_passes(self):
        """真实强信号（日均0.2%）在适度尝试下应通过"""
        rng = np.random.default_rng(3)
        rets = 0.002 + rng.normal(0, 0.01, 1500)
        dsr = deflated_sharpe_ratio(rets, n_trials=10)
        assert dsr.dsr > 0.95, f"真信号应通过, got {dsr.dsr}"

    def test_expected_max_grows_with_trials(self):
        assert expected_max_sharpe(1) == 0.0
        assert expected_max_sharpe(100) > expected_max_sharpe(10) > expected_max_sharpe(2)

    def test_registry_counts(self, tmp_path):
        reg = TrialRegistry(path=str(tmp_path / 'reg.json'))
        assert reg.count == 0
        reg.record('策略A', sharpe=1.2)
        reg.record('策略B', sharpe=0.8)
        assert reg.count == 2
        reg2 = TrialRegistry(path=str(tmp_path / 'reg.json'))
        assert reg2.count == 2, "登记应持久化"


class TestPITFinancialMatrix:

    def test_announcement_timing(self):
        """公告日前不可见：Q1报告4/27公告，4/26的值应为NaN、4/27起为公告值"""
        ann = pd.DataFrame({
            'symbol': ['A'] * 2,
            'ann_dt': ['20240427', '20240809'],
            'report_period': ['20240331', '20240630'],
            'value': [10.0, 20.0],
        })
        idx = pd.bdate_range('2024-04-25', '2024-08-15')
        mx = build_pit_matrix(ann, idx, ['A'])
        assert pd.isna(mx.loc[pd.Timestamp('2024-04-26'), 'A'])
        assert mx.loc[pd.Timestamp('2024-04-29'), 'A'] == 10.0  # 下一交易日（周一）
        # 8/9 公告 20.0 后覆盖
        assert mx.loc[pd.Timestamp('2024-08-12'), 'A'] == 20.0
        assert mx.loc[pd.Timestamp('2024-08-08'), 'A'] == 10.0

    def test_correction_overrides(self):
        """更正公告：同报告期两次公告，第二次（更正值）在其公告日后生效"""
        ann = pd.DataFrame({
            'symbol': ['A', 'A', 'A'],
            'ann_dt': ['20240427', '20240510', '20240809'],
            'report_period': ['20240331', '20240331', '20240630'],
            'value': [10.0, 12.0, 20.0],
        })
        idx = pd.bdate_range('2024-04-25', '2024-08-15')
        mx = build_pit_matrix(ann, idx, ['A'])
        assert mx.loc[pd.Timestamp('2024-05-01'), 'A'] == 10.0   # 更正前用原值
        assert mx.loc[pd.Timestamp('2024-05-13'), 'A'] == 12.0   # 更正后用新值

    def test_prior_period_carried_into_window(self):
        """区间起点前已公告的值应带入区间起点（前视回看）"""
        ann = pd.DataFrame({
            'symbol': ['A'],
            'ann_dt': ['20240427'],
            'report_period': ['20240331'],
            'value': [10.0],
        })
        idx = pd.bdate_range('2024-06-01', '2024-06-30')
        mx = build_pit_matrix(ann, idx, ['A'])
        assert (mx['A'] == 10.0).all()

    def test_symbols_without_announcements_are_nan(self):
        ann = pd.DataFrame({
            'symbol': ['A'], 'ann_dt': ['20240427'],
            'report_period': ['20240331'], 'value': [10.0],
        })
        idx = pd.bdate_range('2024-06-01', '2024-06-30')
        mx = build_pit_matrix(ann, idx, ['A', 'B'])
        assert mx['B'].isna().all()
