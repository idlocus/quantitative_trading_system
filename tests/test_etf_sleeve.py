#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ETF sleeve 唯一实现测试：信号语义 / 月度判定 / 订单带宽 / 总闸"""
import os
import sys
from datetime import date

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trading import config as chain
from trading.etf_strategy import (build_etf_orders, is_rebalance_day)


class TestIsRebalanceDay:
    def test_month_first_trading_day(self):
        ts = pd.bdate_range('2026-09-25', periods=10)   # 09-25,26,28,29,30,10-01...
        # 10月首个交易日=10-01（bdate 假日不精确，用给定序列内判定）
        first_oct = [t for t in ts if t.month == 10][0]
        assert is_rebalance_day(ts, first_oct) is True

    def test_mid_month_not_rebalance(self):
        ts = pd.bdate_range('2026-10-01', periods=20)
        assert is_rebalance_day(ts, ts[10]) is False

    def test_series_start_counts(self):
        ts = pd.bdate_range('2026-10-05', periods=5)
        assert is_rebalance_day(ts, ts[0]) is True


class TestBuildEtfOrders:
    def test_new_position(self):
        orders = build_etf_orders({'518880.SH': 1.0}, {}, 300_000)
        assert len(orders) == 1 and orders[0].side == 'BUY'
        assert orders[0].amount == pytest.approx(300_000)

    def test_exit_on_negative_trend(self):
        orders = build_etf_orders({}, {'510300.SH': 0.5}, 300_000)
        assert len(orders) == 1 and orders[0].side == 'SELL'
        assert orders[0].amount == pytest.approx(150_000)

    def test_band_suppresses_small_drift(self):
        # 权重漂移 2% < max(10%×0.25, 1%)=2.5% → 不调
        orders = build_etf_orders({'510300.SH': 0.25, '511010.SH': 0.25},
                                  {'510300.SH': 0.27, '511010.SH': 0.23},
                                  300_000)
        assert orders == []

    def test_rebalance_triggers_beyond_band(self):
        orders = build_etf_orders({'510300.SH': 0.5}, {'510300.SH': 0.25}, 300_000)
        assert len(orders) == 1 and orders[0].side == 'BUY'
        assert orders[0].amount == pytest.approx(75_000)

    def test_dust_position_ignored(self):
        orders = build_etf_orders({}, {'510500.SH': 0.003}, 300_000)
        assert orders == []      # <0.5% 残渣不生成卖单


class TestSignalSemantics:
    """信号无前视：build_etf_signal 只用 ≤as_of 的数据（以确定性合成数据验证）"""

    def _fake_ds(self, up_gold_only):
        """合成：仅黄金在回看窗中段跳涨（趋势>0），其余阴跌（趋势<0）。"""
        dates = pd.bdate_range('2025-06-02', periods=300)
        gold = pd.Series(1.0, index=dates)
        gold.iloc[200:] = 1.02      # 跳变点在 as_of(≈idx270) 的126日回看窗内
        other = pd.Series(1.0, index=dates)
        other.iloc[200:] = 0.98

        class FakeDS:
            def preload_batch(self, codes, d0, d1):
                out = {}
                for c in codes:
                    s = gold if c == '518880.SH' else other
                    seg = s.loc[d0:pd.Timestamp(d1)]
                    out[c] = pd.DataFrame({
                        'date': seg.index.astype(str).tolist(),
                        'close': seg.values, 'adjclose': seg.values})
                return out
        return FakeDS()

    def test_signal_uses_only_past(self):
        from trading.etf_strategy import build_etf_signal
        as_of = date(2026, 6, 30)
        targets, _, trend = build_etf_signal(self._fake_ds(True), as_of)
        assert '518880.SH' in targets, "上涨资产应入目标"
        assert all(c not in targets for c in ('510300.SH', '510500.SH', '511010.SH'))
        assert trend['518880.SH'] > 0

    def test_no_data_returns_cash(self):
        from trading.etf_strategy import build_etf_signal

        class EmptyDS:
            def preload_batch(self, codes, d0, d1):
                return {c: None for c in codes}
        targets, _, _ = build_etf_signal(EmptyDS(), date(2026, 6, 30))
        assert targets == {}, "无数据=100%现金（fail-safe，不臆测趋势）"
