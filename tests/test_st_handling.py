#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ST 处理测试：涨跌停比例（主板5%/创业科创20%/北交30%）、Paper撮合、PIT判定、选股排除"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import date

import pandas as pd
import pytest

from trading import rules
from data.st_provider import STProvider, ST_TYPES
from live.broker import Order, PaperBroker
from portfolio.backtester import PortfolioBacktester, PortfolioConfig
from portfolio.daily_replay import DailyReplayBacktester, ReplayConfig


class TestSTRules:

    def test_main_board_st_is_5pct(self):
        assert rules.price_limit_ratio('600360.SH', st=True) == 0.05
        assert rules.price_limit_ratio('002656.SZ', st=True) == 0.05

    def test_chinext_star_st_still_20pct(self):
        """注册制后创业板/科创板 ST 无差异化（仍20%）"""
        assert rules.price_limit_ratio('300301.SZ', st=True) == 0.20
        assert rules.price_limit_ratio('688279.SH', st=True) == 0.20

    def test_bse_st_unchanged(self):
        assert rules.price_limit_ratio('920275.BJ', st=True) == 0.30

    def test_non_st_unchanged(self):
        assert rules.price_limit_ratio('600360.SH', st=False) == 0.10
        assert rules.price_limit_ratio('300301.SZ', st=False) == 0.20

    def test_limit_boundary_with_st(self):
        # 主板ST：+5% 涨停 / -5% 跌停
        assert rules.is_limit_up('600360.SH', 10.52, 10.0, st=True)
        assert not rules.is_limit_up('600360.SH', 10.4, 10.0, st=True)   # +4% 未涨停
        assert rules.is_limit_down('600360.SH', 9.48, 10.0, st=True)
        # 非ST口径下 +5% 不是涨停（旧行为会漏判 ST 的 ±5%）
        assert not rules.is_limit_up('600360.SH', 10.52, 10.0, st=False)


class TestPaperBrokerST:

    def test_main_board_st_5pct_rejected(self):
        b = PaperBroker()
        # ST +5.2% → 涨停拒买；-5.2% → 跌停拒卖
        f1 = b.execute([Order('600360.SH', 'BUY', 0.1, amount=10000)],
                       {'600360.SH': 10.52}, date(2026, 10, 12),
                       prev_prices={'600360.SH': 10.0}, st_symbols={'600360.SH'})
        assert '涨停' in f1[0].status
        f2 = b.execute([Order('600360.SH', 'SELL', 0, quantity=100)],
                       {'600360.SH': 9.48}, date(2026, 10, 12),
                       prev_prices={'600360.SH': 10.0}, st_symbols={'600360.SH'})
        assert '跌停' in f2[0].status

    def test_main_board_st_minus_5pct_sell_ok_without_st_flag(self):
        """同一价格，未告知ST时不拒（降级行为=旧行为）——影响面可见"""
        b = PaperBroker()
        f = b.execute([Order('600360.SH', 'SELL', 0, quantity=100)],
                      {'600360.SH': 9.48}, date(2026, 10, 12),
                      prev_prices={'600360.SH': 10.0})   # 不传 st_symbols
        assert f[0].status == 'filled'

    def test_chinext_st_still_20pct_in_broker(self):
        b = PaperBroker()
        f = b.execute([Order('300301.SZ', 'BUY', 0.1, amount=10000)],
                      {'300301.SZ': 11.5}, date(2026, 10, 12),   # +15%
                      prev_prices={'300301.SZ': 10.0}, st_symbols={'300301.SZ'})
        assert f[0].status == 'filled'   # 创业板ST未到20%不拒


class TestSTProvider:

    ROWS = [
        ('600360.SH', 'S', '20250127', None),          # ST 至今
        ('002656.SZ', 'Y', '20200101', '20230101'),    # ST 已摘帽
        ('600225.SH', 'L', '20250207', None),          # 退市整理（不算ST类）
        ('600898.SH', 'T', '20250210', None),          # 退市（不算ST类）
        ('300301.SZ', 'S', '20250113', None),          # 创业板ST
    ]

    def test_interval_judgment(self):
        p = STProvider.from_rows(self.ROWS)
        assert p.is_st('600360.SH', date(2026, 10, 9))
        assert not p.is_st('600360.SH', date(2025, 1, 26))      # entry 前
        assert p.is_st('002656.SZ', date(2022, 6, 1))
        assert not p.is_st('002656.SZ', date(2023, 6, 1))       # 摘帽后
        assert not p.is_st('600225.SH', date(2026, 10, 9))      # L 不算
        assert not p.is_st('600898.SH', date(2026, 10, 9))      # T 不算

    def test_st_set_excludes_non_st_types(self):
        p = STProvider.from_rows(self.ROWS)
        s = p.st_set(date(2026, 10, 9))
        assert '600360.SH' in s and '300301.SZ' in s
        assert '600225.SH' not in s and '600898.SH' not in s

    def test_st_matrix(self):
        p = STProvider.from_rows(self.ROWS)
        idx = pd.DatetimeIndex(pd.to_datetime(['2026-10-08', '2026-10-09']))
        m = p.st_matrix(idx, ['600360.SH', '300301.SZ', '600000.SH'])
        assert m.loc['2026-10-09', '600360.SH'] == True       # noqa: E712
        assert m.loc['2026-10-09', '600000.SH'] == False      # noqa: E712


def make_cache(prices, days=70, volume=100e4):
    dates = pd.bdate_range('2026-06-01', periods=days)
    cache = {}
    for sym, arr in prices.items():
        full = [arr[0]] * (days - len(arr)) + list(arr)
        cache[sym] = pd.DataFrame({'date': dates, 'open': full, 'high': full,
                                   'low': full, 'close': full, 'volume': [volume] * days})
    bench = [4000.0] * days
    cache['000300.SH'] = pd.DataFrame({'date': dates, 'open': bench, 'high': bench,
                                       'low': bench, 'close': bench, 'volume': [1e9] * days})
    return cache


class TestBacktesterST:

    def test_select_excludes_st(self):
        """st_matrix 标记的 ST 股不入选（即使分数最高）"""
        cache = make_cache({'600001.SH': [10.0] * 70, '000001.SZ': [10.0] * 70})
        dates = pd.bdate_range('2026-06-01', periods=70)
        st_m = pd.DataFrame(False, index=dates, columns=['600001.SH', '000001.SZ'])
        st_m['600001.SH'] = True
        mx = pd.DataFrame({d: {'600001.SH': 2.0, '000001.SZ': 1.0} for d in dates}).T.sort_index()
        cfg = PortfolioConfig(top_n=1, min_daily_turnover=0, min_listing_days=0,
                              st_matrix=st_m)
        bt = PortfolioBacktester(cache, cfg)
        res = bt.run(mx)
        bought = {r[1] for r in res.trade_log if r[2] == 'BUY'}
        assert bought == {'000001.SZ'}, "ST 股 A 不应被买入"

    def test_st_limit_down_blocks_sell(self):
        """主板ST -5.5% 当日封跌停 → 卖出顺延次日（旧行为会当普通股放行卖出）"""
        # 场景：第34日调仓买入A（当时非ST）；第60日起A戴帽ST；第68日A暴跌-5.5%
        # 恰逢调仓日（A被ST排除出目标→需卖出）→ ST口径下当日为跌停，顺延至69日成交
        cache = make_cache({'600001.SH': [10.0] * 68 + [9.45, 9.45], '000001.SZ': [10.0] * 70})
        dates = pd.bdate_range('2026-06-01', periods=70)
        st_m = pd.DataFrame(False, index=dates, columns=['600001.SH', '000001.SZ'])
        st_m.iloc[60:, st_m.columns.get_loc('600001.SH')] = True    # 第60日起 ST
        mx = pd.DataFrame({d: {'600001.SH': 2.0, '000001.SZ': 1.0} for d in dates}).T.sort_index()
        cfg = PortfolioConfig(top_n=1, rebalance_days=34, buffer_band=0.0,
                              min_daily_turnover=0, min_listing_days=0, st_matrix=st_m)
        res = PortfolioBacktester(cache, cfg).run(mx)
        sells = [r for r in res.trade_log if r[1] == '600001.SH' and r[2] == 'SELL']
        crash_day = str(dates[68].date())
        next_day = str(dates[69].date())
        assert sells, "A 应被卖出（跌停顺延后）"
        assert all(r[0] != crash_day for r in sells), "ST -5.5% 当日为跌停，不应成交卖出"
        assert sells[0][0] == next_day, "顺延至跌停解除后的下一交易日"


class TestDailyReplayST:

    def test_replay_excludes_st_from_targets(self):
        cache = make_cache({'600001.SH': [10.0] * 70, '000001.SZ': [10.0] * 70})
        dates = pd.bdate_range('2026-06-01', periods=70)
        st_m = pd.DataFrame(False, index=dates, columns=['600001.SH', '000001.SZ'])
        st_m['600001.SH'] = True
        mx = pd.DataFrame({d: {'600001.SH': 2.0, '000001.SZ': 1.0} for d in dates}).T.sort_index()
        cfg = ReplayConfig(top_n=1, min_listing_days=0, min_daily_turnover=0,
                           use_risk_gate=False, st_matrix=st_m)
        res = DailyReplayBacktester(cache, cfg).run(mx)
        bought = {t['symbol'] for t in res.trades if t['side'] == 'BUY'}
        assert bought == {'000001.SZ'}, "重放中 ST 股 A 不应入选"
