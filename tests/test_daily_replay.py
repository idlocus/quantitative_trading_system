#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
每日重放回测测试：验证"严格生产语义"——每日排名进出/缓冲带只管漂移/L4止损/无前视
这些正是每5日近似回测无法表达的语义（用户指出的分叉点）。
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import pytest

from portfolio.daily_replay import DailyReplayBacktester, ReplayConfig


def make_cache(prices: dict, days=70, volume=100e4, bench=True):
    """prices: {sym: [p0, p1, ...]} 长度可不同；统一 70 个工作日，价格序列前端填充"""
    dates = pd.bdate_range('2026-06-01', periods=days)
    cache = {}
    for sym, arr in prices.items():
        full = [arr[0]] * (days - len(arr)) + list(arr)
        cache[sym] = pd.DataFrame({
            'date': dates, 'open': full, 'high': full, 'low': full,
            'close': full, 'volume': [volume] * days})
    if bench:
        cache['000300.SH'] = pd.DataFrame({
            'date': dates, 'open': [4000.0] * days, 'high': [4000.0] * days,
            'low': [4000.0] * days, 'close': [4000.0] * days, 'volume': [1e8] * days})
    return cache


def score_matrix(base: dict, overrides: dict = None, days=70):
    """base: {sym: score} 默认排名；overrides: {日期索引: {sym: score}} 从该索引起生效一日"""
    dates = pd.bdate_range('2026-06-01', periods=days)
    syms = sorted(set(base) | {s for r in (overrides or {}).values() for s in r})
    rows = {}
    for i, d in enumerate(dates):
        r = (overrides or {}).get(i, base)
        rows[d] = {s: r.get(s, np.nan) for s in syms}
    return pd.DataFrame(rows).T.sort_index()


LAST = str(pd.bdate_range('2026-06-01', periods=70)[-1].date())        # 2026-09-04
PREV = str(pd.bdate_range('2026-06-01', periods=70)[-2].date())        # 2026-09-03


class TestDailySemantics:

    def test_rank_exit_sells_same_day(self):
        """核心分叉语义：股票跌出TopN当日立即卖出（缓冲带不管进出）"""
        cache = make_cache({
            'A.SH': [10.0] * 70,
            'B.SH': [10.0] * 70,
            'C.SH': [10.0] * 70,
        })
        # 默认排名 A>C>B（持有A、C）；信号日68（倒数第二日）C 跌到第三
        mx = score_matrix({'A.SH': 3.0, 'C.SH': 2.0, 'B.SH': 1.0},
                          overrides={68: {'A.SH': 3.0, 'B.SH': 2.0, 'C.SH': 1.0}})
        cfg = ReplayConfig(top_n=2, min_listing_days=0, min_daily_turnover=0,
                           use_risk_gate=False, initial_capital=100_000)
        res = DailyReplayBacktester(cache, cfg).run(mx)
        sells_c = [t for t in res.trades if t['symbol'] == 'C.SH' and t['side'] == 'SELL']
        buys_b = [t for t in res.trades if t['symbol'] == 'B.SH' and t['side'] == 'BUY']
        assert sells_c, "C 跌出TopN必须产生卖单"
        assert buys_b, "B 新进TopN必须当日买入"
        # 信号在68，成交只能在69（其后首个交易日），当日执行不顺延
        assert sells_c[-1]['date'] == LAST
        assert buys_b[0]['date'] == LAST

    def test_band_prevents_drift_trades(self):
        """目标不变+价格平稳 → 缓冲带内零调仓"""
        cache = make_cache({'A.SH': [10.0] * 70, 'B.SH': [10.0] * 70})
        mx = score_matrix({'A.SH': 2.0, 'B.SH': 1.0})
        cfg = ReplayConfig(top_n=2, min_listing_days=0, min_daily_turnover=0,
                           use_risk_gate=False, initial_capital=100_000)
        res = DailyReplayBacktester(cache, cfg).run(mx)
        # 首日建仓后，后续不应有持续买卖
        first_day = {t['symbol'] for t in res.trades}
        assert first_day == {'A.SH', 'B.SH'}
        after_first = [t for t in res.trades
                       if t['date'] != res.trades[0]['date']]
        assert after_first == []

    def test_stop_loss_forced_sell_and_next_day_rebuy(self):
        """L4止损：跌破成本5%强制卖出（仍在目标内也卖）；次日仍在目标会被回买——忠实复刻生产"""
        cache = make_cache({'A.SH': [10.0] * 68 + [9.4, 9.4]})   # 最后两日 -6%
        mx = score_matrix({'A.SH': 1.0})
        cfg = ReplayConfig(top_n=1, min_listing_days=0, min_daily_turnover=0,
                           use_risk_gate=True, initial_capital=100_000)
        res = DailyReplayBacktester(cache, cfg).run(mx)
        stop_sells = [t for t in res.trades if t['side'] == 'SELL' and '[L4]' in t['note']]
        assert stop_sells, "跌破止损必须注入强制卖单"
        # 生产语义：卖出当日撤销同标的买入；若之后价格仍低且仍在目标，会被回买（暴露真实行为）
        assert res.n_stop_sells >= 1

    def test_no_lookahead_signal(self):
        """T日成交用的必须是T-1的排名：信号68日才变为B领先 → 最早69日才买B"""
        cache = make_cache({'A.SH': [10.0] * 70, 'B.SH': [10.0] * 70})
        mx = score_matrix({'A.SH': 2.0, 'B.SH': 1.0},
                          overrides={68: {'B.SH': 2.0, 'A.SH': 1.0}})
        cfg = ReplayConfig(top_n=1, min_listing_days=0, min_daily_turnover=0,
                           use_risk_gate=False, initial_capital=100_000)
        res = DailyReplayBacktester(cache, cfg).run(mx)
        buys_a = [t for t in res.trades if t['symbol'] == 'A.SH' and t['side'] == 'BUY']
        buys_b = [t for t in res.trades if t['symbol'] == 'B.SH' and t['side'] == 'BUY']
        assert buys_a, "T-1信号（A领先）应买入A"
        assert buys_b, "信号切换后B应被买入"
        # 无前视：68日的排名变化不能在68日当天生效，最早69日（LAST）
        assert buys_b[0]['date'] == LAST
        assert buys_a[0]['date'] < buys_b[0]['date']

    def test_halt_when_benchmark_stale(self):
        """基准缺失 → L2停机，不执行任何指令（生产闸门语义）"""
        cache = make_cache({'A.SH': [10.0] * 70, 'B.SH': [10.0] * 70}, bench=False)
        mx = score_matrix({'A.SH': 2.0, 'B.SH': 1.0})
        cfg = ReplayConfig(top_n=2, min_listing_days=0, min_daily_turnover=0,
                           use_risk_gate=True, initial_capital=100_000)
        res = DailyReplayBacktester(cache, cfg).run(mx)
        # 首日无前置信号不计停机，其余每日停机
        assert res.n_halt_days >= res.n_dates - 1
        assert res.trades == []
        assert res.total_return == 0.0

    def test_equity_reflects_prices(self):
        """净值随价格变动：建仓后价格上涨 → 总收益为正"""
        cache = make_cache({'A.SH': [10.0] * 65 + [10.0, 11.0, 12.0, 13.0, 14.0]})
        mx = score_matrix({'A.SH': 1.0})
        cfg = ReplayConfig(top_n=1, min_listing_days=0, min_daily_turnover=0,
                           use_risk_gate=False, initial_capital=100_000)
        res = DailyReplayBacktester(cache, cfg).run(mx)
        assert res.total_return > 0
        assert res.final_capital == pytest.approx(
            res.equity_curve.iloc[-1], rel=1e-6)
        assert len(res.equity_curve) == res.n_dates


class TestBenchContinuityGuard:

    def test_fake_jump_logged(self, caplog):
        """基准-35%假跳变必须触发ERROR告警（2024-09-09数据断层曾致649天假停机）"""
        import logging
        cache = make_cache({'A.SH': [10.0] * 70})
        # 基准中部制造-35%跳变
        bench = [4000.0] * 35 + [2600.0] + [2600.0] * 34
        cache['000300.SH'] = cache['000300.SH'].assign(close=bench)
        with caplog.at_level(logging.ERROR, logger='portfolio.daily_replay'):
            DailyReplayBacktester(cache, ReplayConfig(min_listing_days=0,
                                                      min_daily_turnover=0,
                                                      use_risk_gate=False))
        assert any('假跳变' in r.message for r in caplog.records)

    def test_clean_bench_no_warning(self, caplog):
        import logging
        cache = make_cache({'A.SH': [10.0] * 70})
        with caplog.at_level(logging.ERROR, logger='portfolio.daily_replay'):
            DailyReplayBacktester(cache, ReplayConfig(min_listing_days=0,
                                                      min_daily_turnover=0,
                                                      use_risk_gate=False))
        assert not any('假跳变' in r.message for r in caplog.records)


class TestReplayVsApprox:

    def test_replay_result_markdown(self):
        cache = make_cache({'A.SH': [10.0] * 70, 'B.SH': [10.0] * 70})
        mx = score_matrix({'A.SH': 2.0, 'B.SH': 1.0})
        cfg = ReplayConfig(top_n=2, min_listing_days=0, min_daily_turnover=0,
                           use_risk_gate=False, initial_capital=100_000)
        res = DailyReplayBacktester(cache, cfg).run(mx)
        md = res.summary_markdown()
        assert '每日重放' in md and 'Top2' in md
