#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
主链一致性守护测试 —— 回测与实盘必须共用同一套交易语义

目的：任何一侧擅自改动语义（参数另抄一份、规则另写一份），
本测试立即失败，杜绝"测试通过但生产链路使用了另一套逻辑"。
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import date

import pandas as pd
import pytest

from trading import config as chain
from trading import rules
from portfolio.backtester import PortfolioConfig, PortfolioBacktester
from live.broker import Order, PaperBroker
from live import signals


class TestExecutionTiming:
    """执行时序：T-1收盘信号 → T日成交（与回测 shift(1) 同构，无前视）"""

    @staticmethod
    def _fake_ds(last_bench_dates):
        class FakeDS:
            def preload_batch(self, syms, start, end):
                dates = [d for d in last_bench_dates if d <= end.isoformat()]
                return {'000300.SH': pd.DataFrame({
                    'date': pd.to_datetime(dates),
                    'close': [4000.0] * len(dates)})}
        return FakeDS()

    def test_production_signal_strictly_prior_day(self, monkeypatch):
        """盘后运行（当日bar已存在）时，信号日必须是严格更早的交易日——修复前这里会用到当日收盘数据"""
        import live.signals as sig
        captured = {}

        def fake_gt(ds, as_of, top_n=100):
            captured['as_of'] = as_of
            return ['600527.SH'], pd.Series({'600527.SH': 1.0})

        monkeypatch.setattr(sig, 'generate_target', fake_gt)
        # 基准含 10-09 和 10-12 两根bar：今天=10-12 盘后
        ds = self._fake_ds(['2026-10-09', '2026-10-12'])
        ps = sig.generate_production_signal(ds, as_of=date(2026, 10, 12))
        sig_date, st_ok = ps.signal_date, ps.st_ok
        assert captured['as_of'] == date(2026, 10, 9)   # T-1，不是今天
        assert sig_date == date(2026, 10, 9)
        assert sig_date < date(2026, 10, 12)

    def test_signal_date_intraday_run(self, monkeypatch):
        """盘中/盘前运行（当日bar缺失）：信号日=最近完整交易日"""
        import live.signals as sig
        ds = self._fake_ds(['2026-10-08', '2026-10-09'])   # 10-12 bar 不存在
        assert sig.signal_date_for(ds, date(2026, 10, 12)) == date(2026, 10, 9)

    def test_run_flows_use_production_signal_entry(self):
        """源码守护：执行入口不得绕过 generate_production_signal 直接用当日信号"""
        import inspect
        import run_daily
        import run_live
        assert 'generate_production_signal' in inspect.getsource(run_daily.run_pm)
        assert 'generate_target(ds, date.today' not in inspect.getsource(run_daily.run_pm)
        assert 'generate_production_signal' in inspect.getsource(run_live.main)
        assert 'generate_target(ds, as_of' not in inspect.getsource(run_live.main)

    def test_backtest_signal_lag_locked(self):
        """回测前视修正存在且与主链 SIGNAL_LAG 一致（shift(1) = 滞后1日）"""
        import inspect
        assert 'shift(1)' in inspect.getsource(PortfolioBacktester.run)
        assert chain.SIGNAL_LAG == 1

    def test_am_pm_same_signal(self, monkeypatch):
        """AM 预览与 PM 执行必须出自同一信号（此前 AM 用 T-1、PM 重算 T 当日，单子不一致）"""
        import live.signals as sig
        calls = []

        def fake_gt(ds, as_of, top_n=100):
            calls.append(as_of)
            return ['600527.SH'], pd.Series({'600527.SH': 1.0})

        monkeypatch.setattr(sig, 'generate_target', fake_gt)
        ds = self._fake_ds(['2026-10-09', '2026-10-12'])
        # AM（journal.write_am 内部）与 PM（run_pm）都经由 signal_date_for/generate_production_signal
        assert sig.signal_date_for(ds, date(2026, 10, 12)) == date(2026, 10, 9)
        sig.generate_production_signal(ds, as_of=date(2026, 10, 12))
        assert calls == [date(2026, 10, 9)]


class TestPriceLimitRules:
    """涨跌停规则：板块比例（此前 PaperBroker 用9.8%一刀切，北交所/科创创业全部误判）"""

    def test_ratio_by_board(self):
        assert rules.price_limit_ratio('600527.SH') == 0.10   # 沪主板
        assert rules.price_limit_ratio('002177.SZ') == 0.10   # 深主板
        assert rules.price_limit_ratio('688279.SH') == 0.20   # 科创板
        assert rules.price_limit_ratio('300235.SZ') == 0.20   # 创业板
        assert rules.price_limit_ratio('301139.SZ') == 0.20   # 创业板注册制
        assert rules.price_limit_ratio('920275.BJ') == 0.30   # 北交所
        assert rules.price_limit_ratio('830799.BJ') == 0.30   # 北交所老代码
        assert rules.price_limit_ratio('00700.HK') is None    # 港股不限制

    def test_limit_up_down_boundary(self):
        # 北交所 30%：+25% 未涨停（旧9.8%实现会误拒），+30% 涨停
        assert not rules.is_limit_up('920275.BJ', 12.5, 10.0)
        assert rules.is_limit_up('920275.BJ', 13.0, 10.0)
        # 创业板 20%：+15% 未涨停（旧实现误拒）
        assert not rules.is_limit_up('300235.SZ', 11.5, 10.0)
        assert rules.is_limit_up('300235.SZ', 12.0, 10.0)
        # 主板 10% 跌停
        assert rules.is_limit_down('600527.SH', 9.0, 10.0)
        assert not rules.is_limit_down('600527.SH', 9.1, 10.0)

    def test_backtester_uses_shared_rules(self):
        """回测的板块比例矩阵 = 共享规则（同一实现，不允许再抄一份）"""
        bt = PortfolioBacktester({}, PortfolioConfig())
        syms = ['600527.SH', '688279.SH', '300235.SZ', '920275.BJ', '00700.HK']
        ratios = bt._limit_ratios(syms)
        assert ratios['600527.SH'] == 0.10
        assert ratios['688279.SH'] == 0.20
        assert ratios['920275.BJ'] == 0.30
        assert pd.isna(ratios['00700.HK'])

    def test_paper_broker_uses_shared_rules(self):
        """PaperBroker 拒单行为与共享规则一致（含北交所修复验证）"""
        b = PaperBroker()
        # 北交所 +25%：未涨停 → 买入不拒（旧实现 chg>0.098 会误拒）
        fills = b.execute([Order('920275.BJ', 'BUY', 0.1, amount=10000)],
                          {'920275.BJ': 12.5}, date(2026, 10, 12),
                          prev_prices={'920275.BJ': 10.0})
        assert fills[0].status == 'filled'
        # 北交所 +30%：涨停 → 拒买
        fills = b.execute([Order('920275.BJ', 'BUY', 0.1, amount=10000)],
                          {'920275.BJ': 13.0}, date(2026, 10, 12),
                          prev_prices={'920275.BJ': 10.0})
        assert '涨停' in fills[0].status
        # 主板 -10%：跌停 → 拒卖
        fills = b.execute([Order('600527.SH', 'SELL', 0, quantity=100)],
                          {'600527.SH': 9.0}, date(2026, 10, 12),
                          prev_prices={'600527.SH': 10.0})
        assert '跌停' in fills[0].status
        # 主板 -9%：未跌停 → 卖出成交
        fills = b.execute([Order('600527.SH', 'SELL', 0, quantity=100)],
                          {'600527.SH': 9.1}, date(2026, 10, 12),
                          prev_prices={'600527.SH': 10.0})
        assert fills[0].status == 'filled'


class TestParamsSingleSource:
    """主链参数单一来源：回测默认 = 生产默认 = trading.config"""

    def test_backtest_defaults_from_chain(self):
        cfg = PortfolioConfig()
        assert cfg.buffer_band == chain.BUFFER_BAND          # 此前回测默认0≠生产0.20
        assert cfg.top_n == chain.TOP_N
        assert cfg.min_listing_days == chain.MIN_LISTING_DAYS
        assert cfg.min_daily_turnover == chain.MIN_DAILY_TURNOVER
        assert cfg.commission == chain.COMMISSION
        assert cfg.stamp_tax == chain.STAMP_TAX
        assert cfg.transfer_fee == chain.TRANSFER_FEE
        assert cfg.slippage == chain.SLIPPAGE
        assert cfg.max_participation == chain.MAX_PARTICIPATION

    def test_paper_broker_defaults_from_chain(self):
        b = PaperBroker()
        assert b.commission == chain.COMMISSION
        assert b.stamp_tax == chain.STAMP_TAX
        assert b.transfer_fee == chain.TRANSFER_FEE
        assert b.slippage == chain.SLIPPAGE

    def test_live_signals_from_chain(self):
        assert signals.PRODUCTION_FACTORS is chain.PRODUCTION_FACTORS
        assert signals.DEFAULT_TOP_N == chain.TOP_N
        assert signals.WARMUP_DAYS == chain.WARMUP_DAYS
        assert signals.MIN_DAILY_TURNOVER == chain.MIN_DAILY_TURNOVER

    def test_build_orders_default_band_from_chain(self):
        import inspect
        sig = inspect.signature(signals.build_orders)
        assert sig.parameters['band'].default == chain.BUFFER_BAND

    def test_risk_thresholds_aligned(self):
        """执行风控（live gate）与回测风控（risk_manager）阈值真同源（精确相等，无容差）"""
        from live.risk_gate import GateConfig
        from portfolio.risk_manager import RiskLimits
        g, r = GateConfig(), RiskLimits()
        assert g.drawdown_soft == chain.DRAWDOWN_SOFT
        assert g.drawdown_hard == chain.DRAWDOWN_HARD
        assert g.market_daily_crash == chain.MARKET_DAILY_CRASH
        assert g.portfolio_daily_crash == chain.PORTFOLIO_DAILY_CRASH
        assert r.drawdown_derisk == chain.DRAWDOWN_SOFT          # 此前0.10≠生产0.08
        assert r.max_drawdown == chain.DRAWDOWN_HARD
        assert r.circuit_breaker_market == chain.MARKET_DAILY_CRASH
        assert r.circuit_breaker_portfolio == chain.PORTFOLIO_DAILY_CRASH


class TestLiquidityFilter:
    """流动性过滤：生产选股与回测 _liquidity_mask 同口径（20日均成交额）"""

    @staticmethod
    def _df(turnover, days=25):
        dates = pd.date_range('2026-09-01', periods=days, freq='B')
        return pd.DataFrame({'date': dates, 'close': [10.0] * days,
                             'volume': [turnover / 10.0] * days})

    def test_illiquid_set_threshold(self):
        cache = {
            'A.SH': self._df(600e4),    # 600万 → 可入选
            'B.SZ': self._df(400e4),    # 400万 → 排除
            'C.BJ': self._df(500e4),    # 恰好500万 → 不低于阈值，可入选
        }
        illiquid = signals.illiquid_set(cache, pd.Timestamp('2026-10-08'),
                                        min_turnover=500e4)
        assert illiquid == {'B.SZ'}

    def test_insufficient_history_excluded_from_filter_only(self):
        """数据不足20日的标的不误杀（也无法判定流动性，交给上市天数过滤）"""
        cache = {'D.SH': self._df(400e4, days=10)}
        assert signals.illiquid_set(cache, pd.Timestamp('2026-10-08'), 500e4) == set()

    def test_backtest_liquidity_mask_same_threshold(self):
        """回测的流动性 mask 用同一阈值语义（avg<min → 不可买）"""
        cache = {'A.SH': self._df(400e4)}   # 400万，不足
        bt = PortfolioBacktester(cache, PortfolioConfig())
        t = bt.close_mx.index[-1]
        mask = bt._liquidity_mask(t)
        assert bool(mask.get('A.SH', False))
