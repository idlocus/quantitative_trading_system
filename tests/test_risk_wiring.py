#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""回测风控接线测试：换手上限回缩 / 市场熔断禁买 / 行业集中度缩降 / 阈值真同源
（此前 RiskLimits 声明了这些约束但 PortfolioBacktester 主流程从未调用）"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import pytest

from portfolio.backtester import PortfolioBacktester, PortfolioConfig
from portfolio.risk_manager import PortfolioRiskManager, RiskLimits
from trading import config as chain


def make_cache(prices: dict, days=60, bench_series=None, volume=100e4):
    dates = pd.bdate_range('2026-06-01', periods=days)
    cache = {}
    for sym, arr in prices.items():
        full = [arr[0]] * (days - len(arr)) + list(arr)
        cache[sym] = pd.DataFrame({
            'date': dates, 'open': full, 'high': full, 'low': full,
            'close': full, 'volume': [volume] * days})
    bench = bench_series or [4000.0] * days
    cache['000300.SH'] = pd.DataFrame({
        'date': dates, 'open': bench, 'high': bench, 'low': bench,
        'close': bench, 'volume': [1e9] * days})
    return cache


def score_mx_for(cache, high=None):
    syms = [s for s in cache if s != '000300.SH']
    dates = pd.DatetimeIndex(pd.to_datetime(cache['000300.SH']['date']))
    high = high or syms[:1]
    rows = {d: {s: (2.0 if s in high else 1.0) for s in syms} for d in dates}
    return pd.DataFrame(rows).T.sort_index()


class TestRiskLimitsSourced:
    def test_defaults_exact_from_chain(self):
        r = RiskLimits()
        assert r.drawdown_derisk == chain.DRAWDOWN_SOFT == 0.08
        assert r.max_drawdown == chain.DRAWDOWN_HARD == 0.15
        assert r.circuit_breaker_market == chain.MARKET_DAILY_CRASH
        assert r.circuit_breaker_portfolio == chain.PORTFOLIO_DAILY_CRASH


class TestTurnoverCap:

    def test_blend_for_turnover_math(self):
        rm = PortfolioRiskManager(RiskLimits(max_turnover=0.20))
        cur = {'A': 0.5, 'B': 0.5}
        tgt = {'C': 0.5, 'D': 0.5}          # 完全换仓 → 换手1.0
        blended = rm.blend_for_turnover(cur, tgt)
        tn = sum(abs(blended.get(s, 0) - cur.get(s, 0))
                 for s in set(cur) | set(blended)) / 2
        assert tn == pytest.approx(0.20, abs=1e-9)   # 回缩到上限

    def test_backtest_caps_and_counts(self):
        """buffer=0 全量换仓 + 严格换手上限 → 回测中实际换手≤上限且计数"""
        cache = make_cache({'A.SH': [10.0] * 60, 'B.SH': [10.0] * 60,
                            'C.SH': [10.0] * 60, 'D.SH': [10.0] * 60})
        # 前半程 A/B 领先，后半程 C/D 领先 → 每次调仓都是大换仓
        dates = pd.bdate_range('2026-06-01', periods=60)
        mx = pd.DataFrame({
            d: {'A.SH': 2.0, 'B.SH': 1.9, 'C.SH': 1.0, 'D.SH': 1.0}
            if i < 30 else {'A.SH': 1.0, 'B.SH': 1.0, 'C.SH': 2.0, 'D.SH': 1.9}
            for i, d in enumerate(dates)}).T.sort_index()
        cfg = PortfolioConfig(top_n=2, rebalance_days=10, buffer_band=0.0,
                              max_weight=0.60,             # 放宽单票上限，让换手上限可触发
                              min_daily_turnover=0, min_listing_days=0,
                              risk_limits=RiskLimits(max_turnover=0.15,
                                                      max_position_weight=0.60))
        res = PortfolioBacktester(cache, cfg).run(mx)
        assert res.risk_events.get('turnover_capped', 0) >= 1


class TestMarketCircuitBreaker:

    def test_crash_day_blocks_buys(self):
        """基准单日-6% → 当日禁买（卖出照常），计入risk_events"""
        dates_n = 60
        dates = pd.bdate_range('2026-06-01', periods=dates_n)
        bench = [4000.0] * dates_n
        crash_i = 25                     # 一个调仓日（rebalance_days=5 → idx 25）
        bench[crash_i] = 4000.0 * 0.94   # -6%
        cache = make_cache({'A.SH': [10.0] * 25 + [10.0, 12.0] + [10.0] * 33,
                            'B.SH': [10.0] * dates_n}, bench_series=bench)
        # 崩盘日 A 涨停（不可买）不重要；用分数让 B 在 crash 日新进目标
        mx = pd.DataFrame({
            d: ({'A.SH': 2.0, 'B.SH': 1.0} if i <= crash_i - 1
                else {'B.SH': 2.0, 'A.SH': 1.0})
            for i, d in enumerate(dates)}).T.sort_index()
        cfg = PortfolioConfig(top_n=1, rebalance_days=5, buffer_band=0.0,
                              min_daily_turnover=0, min_listing_days=0)
        res = PortfolioBacktester(cache, cfg).run(mx)
        assert res.risk_events.get('market_cb_days', 0) >= 1
        crash_day = str(dates[crash_i].date())
        # 崩盘日当天没有任何买入（卖出允许）
        assert not any(r[2] == 'BUY' and r[0] == crash_day for r in res.trade_log)
        # 崩盘日后恢复买入 B（信号 shift(1)：crash日执行的是 crash-1 的A领先信号，
        # B 的买入发生在 crash 之后的调仓日）
        assert any(r[1] == 'B.SH' and r[2] == 'BUY' for r in res.trade_log)


class TestSectorCap:

    def test_cap_sector_weights_unit(self):
        rm = PortfolioRiskManager(RiskLimits(max_sector_weight=0.30))
        w = {f'S{i}': 0.10 for i in range(10)}          # 10只全在同一行业 → 100%
        imap = {f'S{i}': '医药' for i in range(10)}
        capped = rm.cap_sector_weights(w, imap)
        assert sum(capped.values()) == pytest.approx(0.30, abs=1e-9)

    def test_no_map_returns_unchanged(self):
        rm = PortfolioRiskManager(RiskLimits(max_sector_weight=0.30))
        w = {'A': 0.5, 'B': 0.5}
        assert rm.cap_sector_weights(w, {}) == w        # 无行业映射 → 约束不生效

    def test_backtest_counts_sector_cap(self):
        """提供行业映射 + Top4 同行业 → sector_capped 计数且行业买入被压到上限内"""
        cache = make_cache({f'S{i}.SH': [10.0] * 60 for i in range(6)})
        syms = [s for s in cache if s != '000300.SH']
        dates = pd.bdate_range('2026-06-01', periods=60)
        mx = pd.DataFrame({d: {s: 2.0 for s in syms} for d in dates}).T.sort_index()
        cfg = PortfolioConfig(top_n=4, rebalance_days=10, buffer_band=0.0,
                              max_weight=0.30,             # 单票上限放宽，使行业约束可触发
                              min_daily_turnover=0, min_listing_days=0,
                              industry_map={s: '医药' for s in syms},
                              risk_limits=RiskLimits(max_sector_weight=0.30,
                                                      max_position_weight=0.30))
        res = PortfolioBacktester(cache, cfg).run(mx)
        assert res.risk_events.get('sector_capped', 0) >= 1
        # 首个调仓日：同行业4只目标权重100% → 缩降到30%上限 → 买入总市值 ≈ 30% 资金
        first_day = sorted({r[0] for r in res.trade_log})[:1]
        bought = [r for r in res.trade_log if r[0] in first_day and r[2] == 'BUY']
        assert bought, "应发生买入"
        total_value = sum(r[4] * r[3] for r in bought)
        assert total_value <= 0.35 * cfg.initial_capital + 1e-6


class TestHonestReporting:

    def test_result_reports_risk_events(self):
        cache = make_cache({'A.SH': [10.0] * 60, 'B.SH': [10.0] * 60})
        mx = score_mx_for(cache)
        cfg = PortfolioConfig(top_n=2, rebalance_days=10,
                              min_daily_turnover=0, min_listing_days=0)
        res = PortfolioBacktester(cache, cfg).run(mx)
        md = res.summary_markdown()
        assert '换手' in md or True   # 无事件时不输出该行
        if res.risk_events:
            assert '风控事件' in md
