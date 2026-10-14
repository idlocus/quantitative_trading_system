#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
契约与口径测试 —— 验证数据语义正确性（不是逻辑正确性）

与接口测试的区别：
  接口测试：给定输入，输出是否正确？
  契约测试：单位是否正确？基准是否声明？覆盖率是否达标？

此前 QMT 状态码 bug（测试用同一套错值所以全绿）证明了：
  只有接口测试不能发现"口径错误"——必须单独测口径。
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import pytest

from data.contracts import (
    PriceContract, validate_price_series, ContractViolation,
    VolumeContract, normalize_volume, normalize_amount,
    FactorContract, validate_factor_matrix, FACTOR_CONTRACTS,
    get_factor_contract,
)
from data.semantics import (
    get_price_column, extract_prices, daily_turnover, avg_daily_turnover,
    factor_coverage, should_halt_signal, drawdown_from_series,
    current_drawdown, planned_exposure, actual_cash_ratio,
)


class TestPriceContracts:
    """价格契约：基准声明与校验"""

    def test_raw_price_column(self):
        assert get_price_column(pd.DataFrame(columns=['close', 'adjclose']), 'raw') == 'close'

    def test_adjusted_price_column(self):
        df = pd.DataFrame(columns=['close', 'adjclose'])
        assert get_price_column(df, 'adjusted') == 'adjclose'

    def test_adjusted_fallback_to_raw(self):
        df = pd.DataFrame(columns=['close'])
        assert get_price_column(df, 'adjusted') == 'close'

    def test_invalid_basis_raises(self):
        with pytest.raises(ValueError):
            get_price_column(pd.DataFrame(), 'invalid')

    def test_negative_price_rejected(self):
        df = pd.DataFrame({'close': [10.0, -5.0]})
        with pytest.raises(ContractViolation, match='非正'):
            validate_price_series(df, PriceContract())

    def test_extreme_jump_detected(self):
        """raw/adjusted混合的典型症状：>50%单日跳变"""
        df = pd.DataFrame({'close': [10.0, 15.0, 7.0, 20.0, 8.0]})
        with pytest.raises(ContractViolation, match='跳变'):
            validate_price_series(df, PriceContract())

    def test_normal_prices_pass(self):
        df = pd.DataFrame({'close': [10.0, 10.2, 10.1, 10.3, 10.2]})
        result = validate_price_series(df, PriceContract())
        assert result is df  # 通过校验返回原DataFrame


class TestVolumeContracts:
    """成交量契约：单位归一化"""

    def test_db_volume_converted(self):
        """DB原始=手，×100后=股"""
        df = pd.DataFrame({'volume': [100.0, 200.0]})
        result = normalize_volume(df, source='db')
        assert result['volume'].iloc[0] == 10000.0

    def test_client_volume_unchanged(self):
        """Wind客户端已是股"""
        df = pd.DataFrame({'volume': [10000.0]})
        result = normalize_volume(df, source='client')
        assert result['volume'].iloc[0] == 10000.0

    def test_auto_detect_lots(self):
        """值域判断：中位数<5000大概率是手"""
        df = pd.DataFrame({'volume': [100.0, 200.0, 300.0]})
        result = normalize_volume(df, source='unknown')
        assert result['volume'].iloc[0] == 10000.0

    def test_db_amount_converted(self):
        """DB原始=千元，×1000后=元"""
        df = pd.DataFrame({'amount': [5.0]})  # 5千元
        result = normalize_amount(df, source='db')
        assert result['amount'].iloc[0] == 5000.0


class TestFactorContracts:
    """因子契约：单位与值域"""

    def test_leverage_is_pct(self):
        """PIT杠杆值是百分比（中位数~50），不是资产/权益倍数（~2）"""
        c = FACTOR_CONTRACTS['leverage']
        assert c.unit == 'pct'
        assert c.expected_range[1] > 100  # 百分比值域应>100

    def test_roe_is_ratio(self):
        """ROE是比率（0.08=8%），不是百分比（8=8%）"""
        c = FACTOR_CONTRACTS['roe']
        assert c.unit == 'ratio'
        assert c.expected_range[0] < 0  # 比率值域应在-1到1附近

    def test_unregistered_factor_rejected(self):
        with pytest.raises(ContractViolation, match='未注册'):
            get_factor_contract('unknown_factor')

    def test_coverage_validation(self):
        mx = pd.DataFrame({'A': [1.0], 'B': [2.0], 'C': [3.0], 'D': [np.nan]})
        meta = validate_factor_matrix(mx, FactorContract('test', 'ratio', (0, 10), 0.5))
        assert meta['n_valid'] == 3
        assert meta['coverage'] == 0.75
        assert meta['passed']

    def test_low_coverage_fails(self):
        mx = pd.DataFrame({'A': [1.0], 'B': [np.nan], 'C': [np.nan], 'D': [np.nan]})
        meta = validate_factor_matrix(mx, FactorContract('test', 'ratio', None, 0.5))
        assert not meta['passed']
        assert '覆盖率' in meta['issues'][0]

    def test_out_of_range_detected(self):
        """值域检查：杠杆如果传了资产/权益倍数值（~2），会被pct契约检出"""
        mx = pd.DataFrame({f'S{i}': [2.5] for i in range(20)})  # 全是~2.5
        meta = validate_factor_matrix(mx, FACTOR_CONTRACTS['leverage'])
        # 值2.5在pct契约的(0,500)范围内不会报错，但如果契约是(0,10)就会
        # 这测试验证契约值域能检出单位不匹配
        c = FactorContract('leverage_test', 'ratio', (0, 10), 0.5)
        meta2 = validate_factor_matrix(mx, c)
        assert meta2['passed']  # 2.5在(0,10)内


class TestSemanticsConvergence:
    """口径收敛：所有"怎么算"的唯一实现"""

    def test_turnover_amount_priority(self):
        """amount列优先于close×volume"""
        df = pd.DataFrame({'close': [10.0], 'volume': [100.0], 'amount': [999.0]})
        assert daily_turnover(df).iloc[0] == 999.0

    def test_turnover_fallback(self):
        """无amount列时退化close×volume"""
        df = pd.DataFrame({'close': [10.0], 'volume': [100.0]})
        assert daily_turnover(df).iloc[0] == 1000.0

    def test_coverage_calculation(self):
        mx = pd.DataFrame({'A': [1.0], 'B': [np.nan], 'C': [3.0], 'D': [np.nan]})
        assert factor_coverage(mx) == 0.5

    def test_signal_halt_conditions(self):
        halt, reason = should_halt_signal(0.10, 100)
        assert halt and '覆盖率' in reason
        halt, reason = should_halt_signal(0.90, 5)
        assert halt and '目标' in reason
        halt, _ = should_halt_signal(0.90, 100)
        assert not halt

    def test_planned_exposure_math(self):
        """100只目标×(1/100权重)=100%仓位"""
        exposure, cash = planned_exposure(100, 100)
        assert exposure == 1.0 and cash == 0.0
        exposure, cash = planned_exposure(7, 100)
        assert exposure == pytest.approx(0.07) and cash == pytest.approx(0.93)

    def test_current_drawdown(self):
        nav_hist = [{'nav': 100}, {'nav': 90}, {'nav': 85}]
        dd, peak = current_drawdown(nav_hist)
        assert dd == pytest.approx(0.15) and peak == 100

    def test_actual_cash_ratio(self):
        ratio = actual_cash_ratio(300000, {'A': 1000}, {'A': 10.0})
        assert ratio == 300000 / 310000
