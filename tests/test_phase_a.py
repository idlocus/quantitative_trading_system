#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase A 测试：过拟合防御（DSR）+ 公告日 PIT 财务因子"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import pytest
import json

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

    def test_registry_corrupt_file_refuses_to_reset(self, tmp_path):
        """审计 2026-11-19 四-a：损坏的登记册不得静默从零重建（下次
        record() 会覆盖全部历史）。"""
        p = str(tmp_path / 'reg.json')
        reg = TrialRegistry(path=p)
        reg.record('策略A', sharpe=1.0)
        with open(p, 'w', encoding='utf-8') as f:
            f.write('{corrupt')
        with pytest.raises(RuntimeError, match='拒绝写入'):
            TrialRegistry(path=p)

    def test_registry_records_ran_at_and_verdict(self, tmp_path):
        reg = TrialRegistry(path=str(tmp_path / 'reg.json'))
        reg.record('G1终极区间验证', extra={'verdict': 'PASS +0.55%/月'})
        entry = json.loads(open(str(tmp_path / 'reg.json'), encoding='utf-8').read())['trials'][0]
        assert entry['label'] == 'G1终极区间验证'
        assert entry['verdict'] == 'PASS +0.55%/月'
        assert 'ran_at' in entry

    def test_factor_screen2_register_delegates_to_registry(self, tmp_path, monkeypatch):
        """统一登记入口：factor_screen2.register 走 TrialRegistry（同文件）。"""
        from research import factor_screen2 as fs2
        monkeypatch.setattr('research.overfitting.TRIALS_PATH',
                            str(tmp_path / 'trial_registry.json'))
        fs2.register('测试结论', label='测试研究')
        data = json.loads((tmp_path / 'trial_registry.json').read_text(encoding='utf-8'))
        assert data['count'] == 1
        assert data['trials'][0]['label'] == '测试研究'
        assert data['trials'][0]['verdict'] == '测试结论'


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


class TestFundamentalProviderCacheKey:
    """审计 2026-11-19 四-b：缓存键含完整证券集合与日期范围；残缺结果不入缓存。"""

    @staticmethod
    def _provider(recorder, fail_codes=()):
        from data.fundamentals import WindFundamentalProvider
        from types import SimpleNamespace

        class FakeSource:
            def wsd(self, codes, field, start, end, opts):
                recorder.append((codes, field, start, end))
                syms = codes.split(',')
                if any(c in fail_codes for c in syms):
                    raise RuntimeError('client down')
                times = pd.date_range('2024-01-01', periods=3)
                data = [[1.0] * len(times) for _ in syms]
                return SimpleNamespace(ErrorCode=0, Times=list(times),
                                       Codes=syms, Data=data)
        return WindFundamentalProvider(FakeSource())

    def test_middle_symbol_change_not_served_from_cache(self):
        from datetime import date
        calls = []
        p = self._provider(calls)
        a = p.get_field_matrix('pe_ttm', ['S1.SH', 'S2.SH', 'S3.SH'],
                               date(2024, 1, 1), date(2024, 1, 31))
        b = p.get_field_matrix('pe_ttm', ['S1.SH', 'X9.SH', 'S3.SH'],
                               date(2024, 1, 1), date(2024, 1, 31))
        assert list(b.columns) == ['S1.SH', 'X9.SH', 'S3.SH'], \
            '同首尾不同中间证券不得命中旧缓存'
        assert len(calls) == 2

    def test_different_date_range_not_served_from_cache(self):
        from datetime import date
        calls = []
        p = self._provider(calls)
        p.get_field_matrix('pe_ttm', ['S1.SH', 'S2.SH'],
                           date(2024, 1, 1), date(2024, 1, 31))
        p.get_field_matrix('pe_ttm', ['S1.SH', 'S2.SH'],
                           date(2025, 1, 1), date(2025, 1, 31))
        assert len(calls) == 2, '相同证券不同年份不得命中旧缓存'

    def test_failed_chunk_result_not_cached(self):
        from datetime import date
        calls = []
        p = self._provider(calls, fail_codes=('S2.SH',))
        first = p.get_field_matrix('pe_ttm', ['S1.SH', 'S2.SH'],
                                   date(2024, 1, 1), date(2024, 1, 31))
        assert 'S2.SH' not in first.columns, '失败批次证券不得出现在结果里'
        second = p.get_field_matrix('pe_ttm', ['S1.SH', 'S2.SH'],
                                    date(2024, 1, 1), date(2024, 1, 31))
        assert len(calls) == 2, '残缺结果不得入缓存（重试应重新拉取）'


class TestKlineCalendarUnion:
    """审计 2026-11-19 四-c：日历优先指数；否则并集（首只证券的缺失
    不再定义公共日历）。"""

    def _write_kline(self, ds, sym, dates):
        pd.DataFrame({'date': pd.DatetimeIndex(dates)}).to_parquet(
            ds._kline_path(sym))

    def test_stock_union_used_without_index(self, tmp_path):
        from data.layered import LayeredDataSource
        ds = LayeredDataSource(str(tmp_path))
        full = pd.bdate_range('2024-01-01', periods=5)
        partial = full[[0, 2, 4]]           # 首只证券缺 2 天
        self._write_kline(ds, 'A.SH', partial)
        self._write_kline(ds, 'B.SH', full)
        cal = ds.get_kline_calendar(['A.SH', 'B.SH'],
                                    full[0].date(), full[-1].date())
        assert list(cal) == list(full), '并集口径：A 缺的日期不得从日历删除'

    def test_partial_index_calendar_includes_observed_stock_sessions(self, tmp_path):
        from data.layered import LayeredDataSource
        ds = LayeredDataSource(str(tmp_path))
        full = pd.bdate_range('2024-01-01', periods=5)
        idx_days = full[[1, 3]]
        self._write_kline(ds, 'A.SH', full)
        self._write_kline(ds, '000300.SH', idx_days)
        cal = ds.get_kline_calendar(['A.SH'],
                                    full[0].date(), full[-1].date())
        assert list(cal) == list(full), 'Observed stock sessions must survive a truncated index cache'


class TestPitStaleRevisionIgnored:
    """审计 2026-11-19 四-c：旧报告期的迟发更正不得覆盖已公布的新报告期。"""

    def test_late_correction_of_older_period_ignored(self):
        idx = pd.bdate_range('2024-08-01', periods=80)
        ann = pd.DataFrame([
            {'symbol': 'A', 'ann_dt': '20240830', 'report_period': '20240630',
             'value': 20.0},
            # 10-15 迟发更正 Q1（旧报告期）——不得覆盖 Q2
            {'symbol': 'A', 'ann_dt': '20241015', 'report_period': '20240331',
             'value': 99.0},
            # 11-01 对 Q2 的正常修订——应生效
            {'symbol': 'A', 'ann_dt': '20241101', 'report_period': '20240630',
             'value': 25.0},
        ])
        m = build_pit_matrix(ann, idx, ['A'])
        assert m.loc[m.index >= pd.Timestamp('2024-08-30'), 'A'].dropna().iloc[0] == 20.0
        assert m.loc[pd.Timestamp('2024-10-16'), 'A'] == 20.0, \
            '旧报告期的迟发更正不得覆盖新报告期'
        assert m.loc[pd.Timestamp('2024-11-01'), 'A'] == 25.0, \
            '同报告期的正常修订应生效'


def test_yearly_breakdown_uses_prior_year_base():
    """审计 2026-11-19 四-h：年度收益以上年末净值为基数（跨年首个交易日
    的盈亏不得遗漏）。"""
    from research.robustness import yearly_breakdown
    idx_2020 = pd.date_range('2020-01-01', periods=12, freq='D')
    idx_2021 = pd.date_range('2021-01-01', periods=12, freq='D')
    values = list(np.linspace(100.0, 110.0, 12)) + list(np.linspace(110.0, 121.0, 12))
    equity = pd.Series(values, index=idx_2020.append(idx_2021))
    result = yearly_breakdown(equity)
    assert result[2020] == pytest.approx(110.0 / 100.0 - 1)
    assert result[2021] == pytest.approx(121.0 / 110.0 - 1), \
        '跨年首个交易日的盈亏必须计入新年份'
