#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""research/cta_signal.py 行为测试（登记册180 部署形态 TOP1 的单一信号源）。

覆盖：压缩检测、每日跟随方向、TOP1 深度排序、持仓转换文案、
次日开盘成交（禁止同bar归因）、成本扣除、末日待执行、数据覆盖核验。
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import pytest

from research import cta_signal as cs


def _frame(close, vol_pct, compressed, direction, clarity=0.0, open_=None):
    idx = pd.bdate_range('2026-01-05', periods=len(np.atleast_1d(close)))
    n = len(idx)
    data = {
        'close': np.atleast_1d(close).astype(float),
        'ret': np.atleast_1d(close).astype(float) * 0,
        'vol_pct': np.full(n, vol_pct, dtype=float),
        'clarity': np.full(n, clarity, dtype=float),
        'compressed': np.full(n, compressed, dtype=bool),
        'direction': np.full(n, direction, dtype=float),
    }
    if open_ is not None:
        data['open'] = np.atleast_1d(open_).astype(float)
    else:
        data['open'] = np.atleast_1d(close).astype(float)
    return pd.DataFrame(data, index=idx)


class TestSignalFrame:
    def test_compression_detected_after_low_vol_regime(self):
        """高波动 260 天 → 低波动 30 天：末期应进入压缩状态。"""
        rng = np.random.RandomState(3)
        rets = np.concatenate([rng.randn(280) * 0.02, rng.randn(40) * 0.002])
        close = pd.Series(100 * (1 + pd.Series(rets)).cumprod().values,
                          index=pd.bdate_range('2025-01-02', periods=len(rets)))
        f = cs.signal_frame(close)
        assert not f['compressed'].iloc[100]
        assert f['compressed'].iloc[-1]

    def test_direction_follows_ma_side_daily(self):
        """方向=sign(收盘-MA20)，每日跟随（段内穿越即翻向）。"""
        n = 320
        # 先横盘建立基线，再连续上行、再连续下行
        path = np.concatenate([np.full(300, 100.0), np.linspace(100, 120, 10),
                               np.linspace(120, 95, 10)])
        close = pd.Series(path, index=pd.bdate_range('2025-01-02', periods=n))
        f = cs.signal_frame(close)
        assert f['direction'].iloc[305] == 1
        assert f['direction'].iloc[-1] == -1
        assert 'clarity' in f.columns


class TestRankAndBook:
    def test_top1_picks_deepest_compression(self):
        """vol_pct 最低（压缩最深）的品种排第一。"""
        frames = {
            'CU0': _frame(1, 0.05, True, 1),
            'RB0': _frame(1, 0.12, True, -1),
            'AU0': _frame(1, 0.30, False, 1),  # 分位30%>20% → 未压缩
        }
        assert cs.rank_active(frames) == ['CU0', 'RB0']
        book = cs.target_book(frames, k=1)
        assert book == {'CU0': 1.0}

    def test_tie_break_by_clarity_then_fixed_order(self):
        """vol_pct 并列 → clarity 大者优先；再并列 → 固定品种顺序。"""
        frames = {
            'RB0': _frame(1, 0.10, True, 1, clarity=0.01),
            'CU0': _frame(1, 0.10, True, 1, clarity=0.02),
            'AU0': _frame(1, 0.10, True, 1, clarity=0.02),
        }
        assert cs.rank_active(frames) == ['AU0', 'CU0', 'RB0']

    def test_direction_zero_excluded_and_empty_book(self):
        frames = {
            'CU0': _frame(1, 0.05, True, 0),
            'RB0': _frame(1, 0.05, False, 1),
        }
        assert cs.rank_active(frames) == []
        assert cs.target_book(frames, k=1) == {}


class TestBookTransition:
    def test_open_flat_flip_and_hold(self):
        """开多 → 不动 → 翻空 → 平仓 全链路文案。"""
        a = {'CU0': 0.0}
        assert cs.book_transition({}, {'CU0': 1.0})[0]['action'] == '开多'
        assert cs.book_transition({'CU0': 1.0}, {'CU0': 1.0})[0]['action'] == '不动'
        flip = cs.book_transition({'CU0': 1.0}, {'CU0': -1.0})
        assert flip[0]['action'] == '翻空'
        assert cs.book_transition({'CU0': 1.0}, {})[0]['action'] == '平仓'

    def test_symbol_switch_is_close_plus_open(self):
        """TOP1 换品种 = 平旧 + 开新，两条操作。"""
        acts = cs.book_transition({'CU0': 1.0}, {'RB0': -1.0})
        assert {a['action'] for a in acts} == {'平仓', '开空'}
        assert {a['sym'] for a in acts} == {'CU0', 'RB0'}


class TestTradesNextOpenExecution:
    def test_entry_uses_next_day_open_not_signal_close(self):
        """同bar禁令：t 日收盘出信号，成交必须是 t+1 开盘价。"""
        # 3天：信号日收盘100，次日开盘105；信号随即消失
        f = _frame([100, 105, 104], vol_pct=0.05,
                   compressed=[True, False, False],
                   direction=[1, 1, 1], open_=[99, 105, 103])
        trades = cs.trades_from_frame(f, 'CU0')
        assert len(trades) == 1
        t = trades[0]
        assert t['entryPrice'] == 105.0, "入场应为次日开盘105，而非信号日收盘100"
        assert t['exitPrice'] == 103.0, "离场应为信号消失次日开盘103"
        assert t['direction'] == 'L'
        # 毛收益 (103/105-1)，扣往返5bp
        expect = (103 / 105 - 1) - 2 * cs.ONE_WAY_COST
        assert t['netRet'] == pytest.approx(expect)

    def test_flip_closes_and_reopens_at_same_price(self):
        """段内穿越均线（每日跟随）：旧仓按次日开盘平仓并反手。"""
        # 信号 +1 → -1（t=1 决策翻向，t=2 开盘成交平多开空）→ 归零
        f = _frame([100, 98, 96, 95, 94], vol_pct=0.05,
                   compressed=[True, True, True, False, False],
                   direction=[1, -1, -1, -1, -1],
                   open_=[100, 99, 96, 95, 94])
        trades = cs.trades_from_frame(f, 'CU0')
        assert len(trades) == 2
        assert trades[0]['direction'] == 'L'
        assert trades[0]['exitPrice'] == 96.0 and trades[0]['entryPrice'] == 99.0
        assert trades[1]['direction'] == 'S'
        assert trades[1]['entryPrice'] == 96.0, "翻向同价位反手"

    def test_open_position_on_last_day_is_pending(self):
        """末日仍持仓 → pending，不虚构成交价。"""
        f = _frame([100, 101, 102], vol_pct=0.05,
                   compressed=[True, True, True],
                   direction=[1, 1, 1], open_=[100, 100, 101])
        trades = cs.trades_from_frame(f, 'CU0')
        assert len(trades) == 1 and trades[0]['pending'] is True
        assert trades[0]['exitPrice'] is None

    def test_requires_open_column(self):
        f = _frame([100, 101], 0.05, [True, True], [1, 1]).drop(columns=['open'])
        with pytest.raises(ValueError):
            cs.trades_from_frame(f, 'CU0')


class TestCoverageCheck:
    def test_missing_symbol_blocks_decision(self):
        frames = {'CU0': _frame(1, 0.1, True, 1)}
        cov = cs.coverage_check(frames, expected=['AU0', 'CU0', 'RB0'])
        assert cov['ok'] is False
        assert cov['missing'] == ['AU0', 'RB0']

    def test_misaligned_last_dates_block_decision(self):
        """品种间最新交易日不一致 → 无法判定（防残缺数据出结论）。"""
        long = _frame([1] * 5, 0.1, True, 1)
        short = _frame([1] * 4, 0.1, True, 1)
        frames = {'AU0': long, 'CU0': short}
        cov = cs.coverage_check(frames, expected=['AU0', 'CU0'])
        assert cov['ok'] is False and cov['aligned'] is False

    def test_full_coverage_ok(self):
        a = _frame([1] * 280, 0.1, True, 1)
        frames = {'AU0': a, 'CU0': a.copy()}
        assert cs.coverage_check(frames, expected=['AU0', 'CU0'])['ok'] is True


class TestPortfolioDailyGapAttribution:
    """次日开口径的跳空归属：旧仓吃隔夜gap，新仓吃日内。"""

    @staticmethod
    def _frames(compressed, direction, opens, closes):
        n = len(closes)
        idx = pd.bdate_range('2026-01-05', periods=n)
        return {'CU0': pd.DataFrame({
            'close': np.asarray(closes, dtype=float),
            'open': np.asarray(opens, dtype=float),
            'ret': np.zeros(n), 'vol20': np.zeros(n),
            'vol_pct': np.full(n, 0.05), 'ma': np.asarray(closes, dtype=float),
            'clarity': np.zeros(n),
            'compressed': np.asarray(compressed, dtype=bool),
            'direction': np.asarray(direction, dtype=float),
        }, index=idx)}

    def test_entry_day_gets_intraday_only(self):
        """开仓日只吃 O→C，不得吃昨收→开盘的隔夜跳空。"""
        f = self._frames([True, True, False, False], [1, 1, 0, 0],
                         [100, 104, 103, 99], [100, 106, 105, 98])
        net = cs.portfolio_daily(f, k=1, execution='next_open')
        c = cs.ONE_WAY_COST
        assert net.iloc[0] == 0.0
        assert net.iloc[1] == pytest.approx((106 / 104 - 1) - c), \
            '开仓日损益应为 106/104-1（日内），而非 106/100-1（含跳空）'
        units = 1 / 104
        open_nav = units * 103 - c
        new_units = open_nav / 103
        closing_nav = open_nav + new_units * 2 - abs(new_units - units) * 103 * c
        assert net.iloc[2] == pytest.approx(closing_nav / (units * 106 - c) - 1)
        exit_nav = closing_nav + new_units * (99 - 105) - new_units * 99 * c
        assert net.iloc[3] == pytest.approx(exit_nav / closing_nav - 1)

    def test_flip_splits_gap_and_intraday(self):
        """翻向：旧仓吃gap + 新仓反向吃日内；成本按2单位变动。"""
        f = self._frames([True, True, True, False], [1, -1, -1, 0],
                         [100, 100, 104, 99], [100, 100, 102, 98])
        net = cs.portfolio_daily(f, k=1, execution='next_open')
        c = cs.ONE_WAY_COST
        assert net.iloc[1] == pytest.approx((100 / 100 - 1) - c)
        # day2: 旧多仓吃 gap 104/100-1，新空仓吃 102/104-1 的反向，成本 2×c
        open_nav = 1.04 - c
        close_nav = open_nav * (1 + 2 / 104) - (1.04 + open_nav) * c
        expect2 = close_nav / (1 - c) - 1
        assert net.iloc[2] == pytest.approx(expect2)

    def test_close_execution_gives_new_position_full_day(self):
        """对照口径（旧登记）：新仓吃全日 close-to-close——正是被修正的偏差。"""
        f = self._frames([True, True, False, False], [1, 1, 0, 0],
                         [100, 104, 103, 99], [100, 106, 105, 98])
        net = cs.portfolio_daily(f, k=1, execution='close')
        c = cs.ONE_WAY_COST
        assert net.iloc[1] == pytest.approx((106 / 100 - 1) - c), \
            '对照：信号日收盘执行=新仓吃全日（含跳空）'


class TestCoverageStrictness:
    def test_260_aligned_rows_still_insufficient(self):
        """270根才够形成波动分位——260根对齐也必须拦截。"""
        a = _frame([1] * 260, 0.1, True, 1)
        frames = {'AU0': a, 'CU0': a.copy()}
        cov = cs.coverage_check(frames, expected=['AU0', 'CU0'])
        assert cov['ok'] is False
        assert cov['insufficient_history'] == ['AU0', 'CU0']

    def test_aligned_but_stale_flags(self):
        """品种一致但共同滞后 → stale 警告。"""
        n = 280
        idx = pd.bdate_range('2025-01-06', periods=n) - pd.Timedelta(days=10)
        base = pd.DataFrame({
            'close': np.full(n, 100.0), 'open': np.full(n, 100.0),
            'ret': np.zeros(n), 'vol20': np.zeros(n),
            'vol_pct': np.full(n, 0.1), 'ma': np.full(n, 100.0),
            'clarity': np.zeros(n), 'compressed': np.full(n, True),
            'direction': np.full(n, 1.0)}, index=idx)
        frames = {'AU0': base, 'CU0': base.copy()}
        cov = cs.coverage_check(frames, expected=['AU0', 'CU0'],
                                as_of=pd.Timestamp.today())
        assert cov['ok'] is True
        assert cov['stale'] is True, '共同滞后应触发新鲜度警告'

    def test_nan_decision_day_close_blocks(self):
        """决策日收盘缺失 → 无法产生信号，不得输出"全部空仓"。"""
        a = _frame([1] * 280, 0.1, True, 1)
        a.iloc[-1, a.columns.get_loc('close')] = np.nan
        frames = {'AU0': a, 'CU0': _frame([1] * 280, 0.1, True, 1)}
        cov = cs.coverage_check(frames, expected=['AU0', 'CU0'])
        assert cov['ok'] is False
        assert 'AU0' in cov['no_signal_on_decision_day']


class TestComposeActionsUnknownYesterday:
    def test_unknown_yesterday_suppresses_actions(self):
        """昨日未知 → 不出操作清单（防重复开仓/漏平仓）。"""
        actions, note = cs.compose_actions({}, {'CU0': 1.0}, False)
        assert actions is None
        assert '未知' in note

    def test_known_flat_today_empty_orders_close(self):
        actions, _ = cs.compose_actions({'CU0': 1.0}, {}, True)
        assert [a['action'] for a in actions] == ['平仓']

    def test_known_flat_today_long_orders_open(self):
        actions, _ = cs.compose_actions({}, {'CU0': 1.0}, True)
        assert [a['action'] for a in actions] == ['开多']


class TestDeployKSemantics:
    def test_k_ge_active_gives_equal_weights(self):
        """部署形态 k=6：3个活跃品种各得 1/3（等权摊活跃）。"""
        frames = {
            'CU0': _frame(1, 0.05, True, 1),
            'RB0': _frame(1, 0.12, True, -1),
            'AU0': _frame(1, 0.15, True, 1),
            'I0': _frame(1, 0.30, False, 1),
        }
        book = cs.target_book(frames, k=6)
        assert book == {'CU0': 1 / 3, 'RB0': -1 / 3, 'AU0': 1 / 3}


class TestRenderActions:
    def test_weight_change_is_not_close(self):
        """活跃1→2：调仓文案=从X调至Y，绝不能出现'退出全部头寸'。"""
        acts = cs.book_transition({'CU0': 1.0}, {'CU0': 0.5, 'RB0': 0.5})
        lines = cs.render_actions(acts, lev_held=1.0, lev_next=1.2,
                                  yesterday_known=True)
        text = '\n'.join(lines)
        assert '退出全部头寸' not in text, f'调仓被写成清仓: {text}'
        assert '调仓（多头）敞口 +1.00倍 → +0.60倍' in text
        assert '开多，目标敞口 +0.60倍' in text

    def test_weight_increase_on_shrink(self):
        """活跃2→1：留存品种加仓 0.5→1.0。"""
        acts = cs.book_transition({'CU0': 0.5, 'RB0': 0.5}, {'CU0': 1.0})
        lines = cs.render_actions(acts, lev_held=1.0, lev_next=1.0,
                                  yesterday_known=True)
        text = '\n'.join(lines)
        assert '调仓（多头）敞口 +0.50倍 → +1.00倍' in text
        assert '平仓' in text  # RB0 退出

    def test_flip_and_unknown_yesterday(self):
        """翻向带目标敞口；昨日未知时 from 显示 — 不给虚假历史。"""
        acts = cs.book_transition({'CU0': 1.0}, {'CU0': -1.0})
        lines = cs.render_actions(acts, lev_held=1.0, lev_next=0.8,
                                  yesterday_known=False)
        text = '\n'.join(lines)
        assert '翻空，目标敞口 -0.80倍' in text


class TestLeverageTiming:
    def test_next_leverage_includes_last_day_vol_spike(self):
        """末日波动突增：明日订单杠杆须即时反映，滞后序列末值不反映。"""
        n = 120
        rets = pd.Series(np.r_[np.full(n - 1, 0.001), 0.08],
                         index=pd.bdate_range('2026-01-05', periods=n))
        nxt = cs.next_leverage(rets)
        lagged = cs.vol_target_leverage(rets)
        vol_with_spike = rets.rolling(60).std() * np.sqrt(252)
        expect = float((0.15 / vol_with_spike).clip(upper=2).iloc[-1])
        assert nxt == pytest.approx(expect, rel=1e-9)
        assert abs(nxt - float(lagged.iloc[-1])) > 1e-6, \
            '滞后序列末值不含末日信息，两者必须不同（末日大波动时）'


class TestLeverageTurnoverCost:
    def test_leverage_change_alone_trades_and_costs(self):
        """同品种同向、杠杆1→0.5：开盘减仓，成本按|Δ敞口|计。"""
        n = 4
        idx = pd.bdate_range('2026-01-05', periods=n)
        f = {'CU0': pd.DataFrame({
            'close': [100.0, 100.0, 106.0, 108.0],
            'open': [100.0, 100.0, 104.0, 107.0],
            'ret': np.zeros(n), 'vol20': np.zeros(n),
            'vol_pct': np.full(n, 0.05), 'ma': [100.0] * 4,
            'clarity': np.zeros(n), 'compressed': [True] * 4,
            'direction': [1.0] * 4}, index=idx)}
        # 手工杠杆序列：day0/day1=1.0, day2 起 0.5（day2 开盘减仓）
        lev = pd.Series([1.0, 1.0, 0.5, 0.5], index=idx)
        net = cs.portfolio_daily(f, k=1, execution='next_open',
                                 charge_cost=True, leverage=lev)
        # day0 无持仓
        assert net.iloc[0] == pytest.approx(0.0)
        # day1: book(close0)=+1 × lev1=1.0 新开（只吃日内）扣建仓成本
        assert net.iloc[1] == pytest.approx((100 / 100 - 1) - cs.ONE_WAY_COST)
        # day2: 敞口从 1.0 → 0.5：旧敞口吃 gap 104/100-1，
        #       新敞口吃日内 106/104-1，成本=|0.5-1.0|×c
        c = cs.ONE_WAY_COST
        open_nav = 1.04 - c
        target = open_nav * .5
        close_nav = open_nav + target * 2 / 104 - (1.04 - target) * c
        expect2 = close_nav / (1 - c) - 1
        assert net.iloc[2] == pytest.approx(expect2), \
            '杠杆调仓须按实际敞口分别计隔夜/日内并扣换手成本'


class TestCriteriaVerdict:
    def test_pass_and_fail_paths(self):
        idx = pd.bdate_range('2015-01-05', periods=3000)
        rng = np.random.RandomState(1)
        good = pd.Series(rng.randn(3000) * 0.005 + 0.0008, index=idx)
        weak = pd.Series(rng.randn(3000) * 0.005 - 0.0004, index=idx)
        v = cs.criteria_verdict(good, weak, '2020-12-31')
        assert v['pass_all'] is True
        v2 = cs.criteria_verdict(weak, good, '2020-12-31')
        assert v2['c1_sharpe_beats_base'] is False
        assert v2['pass_all'] is False


class TestRenderFromUsesHeldLeverage:
    def test_rebal_from_exposure_uses_held_lev_not_next(self):
        """复核例子：今日权重100%/持有杠杆1.5，明日权重50%/杠杆1.0
        → 清单必须是 +1.50倍 → +0.50倍（起点用今日持有杠杆）。"""
        acts = cs.book_transition({'AU0': 1.0}, {'AU0': 0.5})
        lines = cs.render_actions(acts, lev_held=1.5, lev_next=1.0,
                                  yesterday_known=True)
        assert '调仓（多头）敞口 +1.50倍 → +0.50倍' in '\n'.join(lines), \
            '起点若误用明日杠杆会显示 +1.00倍 → 按差额执行将少减0.50倍'


class TestTruncateToClosed:
    def test_intraday_bar_excluded(self):
        """周三上午数据源已含周三盘中日线 → 只能用周二收盘。"""
        n = 280
        # 周三 2026-11-18 09:35 盘中运行，周三bar已出现
        idx = pd.bdate_range('2025-09-01', periods=n)
        # 强制末两根为周二(11-17)与周三(11-18)
        idx = idx[:-2].append(pd.DatetimeIndex(['2026-11-17', '2026-11-18']))
        f = pd.DataFrame({
            'close': np.linspace(100, 120, n), 'open': np.linspace(100, 120, n),
            'ret': np.zeros(n), 'vol20': np.zeros(n),
            'vol_pct': np.full(n, 0.1), 'ma': np.linspace(100, 120, n),
            'clarity': np.zeros(n), 'compressed': np.full(n, True),
            'direction': np.full(n, 1.0)}, index=idx)
        out = cs.truncate_to_closed({'AU0': f}, '2026-11-17')
        assert out['AU0'].index[-1] == pd.Timestamp('2026-11-17'), \
            '周三盘中bar必须被截掉'
        assert (out['AU0'].index <= pd.Timestamp('2026-11-17')).all()


class TestPreviousBookDate:
    @staticmethod
    def _frame_idx(idx, vol_pct=0.1):
        n = len(idx)
        return pd.DataFrame({
            'close': np.full(n, 100.0), 'open': np.full(n, 100.0),
            'ret': np.zeros(n), 'vol20': np.zeros(n),
            'vol_pct': np.full(n, vol_pct), 'ma': np.full(n, 100.0),
            'clarity': np.zeros(n), 'compressed': np.full(n, True),
            'direction': np.full(n, 1.0)}, index=pd.DatetimeIndex(idx))

    def test_top_symbol_missing_tuesday_no_skip(self):
        """复核用例：排第一的品种缺周二bar → 昨日=周二但该品种缺席，
        不得把周一当昨日（否则漏掉周二换仓）。"""
        mon, tue, wed = '2026-11-16', '2026-11-17', '2026-11-18'
        base = pd.bdate_range('2026-10-19', periods=20).tolist()
        au = self._frame_idx(
            [d for d in base if d.strftime('%Y-%m-%d') < mon] +
            [pd.Timestamp(mon), pd.Timestamp(wed)], vol_pct=0.05)  # 缺周二
        others = {s: self._frame_idx(
            [d for d in base if d.strftime('%Y-%m-%d') < mon] +
            [pd.Timestamp(mon), pd.Timestamp(tue), pd.Timestamp(wed)],
            vol_pct=0.10) for s in ['CU0', 'RB0']}
        frames = {'AU0': au, **others}
        prev, all_have = cs.previous_book_date(frames, wed)
        assert prev == pd.Timestamp(tue), '昨日必须是周二（其余品种的最近bar）'
        assert all_have is False, 'AU0 缺周二 → 昨日未知，不得回退到周一'

    def test_all_present_known(self):
        idx = pd.bdate_range('2026-10-19', periods=22)
        frames = {'AU0': self._frame_idx(idx), 'CU0': self._frame_idx(idx)}
        prev, all_have = cs.previous_book_date(frames, idx[-1])
        assert prev == idx[-2] and all_have is True

    def test_common_missing_day_unknown(self):
        """审计 2026-11-19 二-2：六品种同缺周二（只有周一+周三）→ 无日历时
        无法区分共同数据缺失与休市，昨日必须判未知——不得返回"周一、
        全部齐全"（若周二换过仓，周三清单会重复开仓/漏平仓）。"""
        mon, wed = '2026-11-16', '2026-11-18'
        base = [d for d in pd.bdate_range('2026-10-19', periods=20).tolist()
                if d.strftime('%Y-%m-%d') < mon]
        frames = {s: self._frame_idx(base + [pd.Timestamp(mon),
                                             pd.Timestamp(wed)])
                  for s in ['AU0', 'CU0', 'RB0']}
        prev, all_have = cs.previous_book_date(frames, wed)
        assert prev == pd.Timestamp(mon)
        assert all_have is False, '共同缺日必须判昨日未知'

    def test_legal_holiday_calendar_vs_fallback(self):
        """法定休市（周一假日，周二决策）：显式交易日历 → 昨日=上周五、
        全部齐全；无日历退化为保守未知（不猜昨日）。"""
        cal = [d for d in pd.bdate_range('2026-10-19', periods=22).tolist()
               if d != pd.Timestamp('2026-11-16')]      # 11-16(周一) 法定休市
        frames = {s: self._frame_idx(cal) for s in ['AU0', 'CU0']}
        prev, all_have = cs.previous_book_date(
            frames, pd.Timestamp('2026-11-17'), calendar=cal)
        assert prev == pd.Timestamp('2026-11-13') and all_have is True
        # 无日历：周一缺口解释不掉 → 保守判未知
        prev2, all_have2 = cs.previous_book_date(
            frames, pd.Timestamp('2026-11-17'))
        assert prev2 == pd.Timestamp('2026-11-13') and all_have2 is False


class TestCoverageEmptyFrame:
    def test_empty_frame_is_insufficient_no_crash(self):
        """截断后空表 → 归入历史不足，不得 IndexError。"""
        good = _frame([1] * 280, 0.1, True, 1)
        empty = good.iloc[0:0]
        frames = {'AU0': empty, 'CU0': good}
        cov = cs.coverage_check(frames, expected=['AU0', 'CU0'])
        assert cov['ok'] is False
        assert 'AU0' in cov['insufficient_history']
        assert cov['empty'] == ['AU0']


class TestMaxDrawdownIncludesInitialCapital:
    """审计 2026-11-19 一-7：首日亏损时旧算法以首日净值为首个峰值 →
    回撤算成 0%、25% 门槛被绕过。初始本金 1.0 必须进入峰值序列。"""

    def test_first_day_loss_counts(self):
        idx = pd.bdate_range('2026-01-01', periods=30)
        r = pd.Series([-.30] + [0.001] * 29, index=idx)
        assert cs.max_drawdown_from_returns(r) == pytest.approx(-0.30)

    def test_all_up_has_zero_drawdown(self):
        idx = pd.bdate_range('2026-01-01', periods=30)
        r = pd.Series([.01] * 30, index=idx)
        assert cs.max_drawdown_from_returns(r) == pytest.approx(0.0)

    def test_mid_series_peak_counts(self):
        idx = pd.bdate_range('2026-01-01', periods=30)
        rets = [.10] * 10 + [-.05] * 20
        r = pd.Series(rets, index=idx)
        nav = (1 + r).cumprod()
        # 峰值高于 1.0 时 clip 不生效：期望=段内 cummax 口径（0.95^20-1）
        expected = float((nav / nav.cummax() - 1).min())
        assert cs.max_drawdown_from_returns(r) == pytest.approx(expected)

    def test_criteria_verdict_dd_cap_uses_initial_capital(self):
        """验收判据：首日 -30% 后逐日小幅上涨 → 必须判 ③ 失败（旧算法 PASS）"""
        idx = pd.bdate_range('2020-01-01', periods=300)
        rets = pd.Series([-.30] + [0.001] * 299, index=idx)
        base = pd.Series([0.0005] * 300, index=idx)
        v = cs.criteria_verdict(rets, base, '2020-12-31')
        assert v['max_dd'] == pytest.approx(-0.30)
        assert v['c3_dd_within_cap'] is False
        assert v['pass_all'] is False


class TestPortfolioDailyMissingOpen:
    """审计 2026-11-19 一-6：执行日缺开盘价时目标仓位不得无条件推进
    （旧口径：无效损益被跳过但持仓照建，开仓日只扣费、次日起凭空吃收益）。
    目标与实际成交持仓分离；缺价=保留旧仓记未成交，行情恢复后从真实持仓执行。"""

    def test_entry_missing_open_does_not_create_position(self):
        f = _frame([100, 100, 100, 115], 0.05,
                   [False, True, True, True], [0, 1, 1, 1],
                   open_=[100, 100, np.nan, 110])
        r = cs.portfolio_daily({'AU0': f}, k=1, execution='next_open',
                               charge_cost=True)
        assert r.iloc[2] == 0.0, '缺执行价当日：无成交无换手，收益与成本都必须为 0'
        recs = r.attrs.get('unexecuted', [])
        assert any(rec['symbol'] == 'AU0' and rec['reason'] == 'missing_open'
                   for rec in recs)
        # 次日开盘恢复：从真实持仓（空仓）执行目标，只吃日内
        assert r.iloc[3] == pytest.approx((115 / 110 - 1) - cs.ONE_WAY_COST)

    def test_exit_missing_open_keeps_position(self):
        f = _frame([100, 100, 105, 100], 0.05,
                   [True, False, False, False], [1, 0, 0, 0],
                   open_=[100, 100, np.nan, 100])
        r = cs.portfolio_daily({'AU0': f}, k=1, execution='next_open',
                               charge_cost=False)
        # 平仓日缺开盘价：持仓延续，吃全程 close-to-close，无平仓成本
        assert r.iloc[2] == pytest.approx(105 / 100 - 1)
        recs = r.attrs.get('unexecuted', [])
        assert any(rec['symbol'] == 'AU0' and rec['reason'] == 'missing_open'
                   for rec in recs)
        # 次日开盘恢复：平仓于开盘
        assert r.iloc[3] == pytest.approx(100 / 105 - 1)

    def test_flip_missing_open_keeps_old_side(self):
        f = _frame([100, 100, 104, 104], 0.05,
                   [True, True, True, True], [1, -1, -1, -1],
                   open_=[100, 100, np.nan, 104])
        r = cs.portfolio_daily({'AU0': f}, k=1, execution='next_open',
                               charge_cost=False)
        # 反手日缺开盘价：旧多头冻结吃全程，无反手成本
        assert r.iloc[2] == pytest.approx(104 / 100 - 1)
        # 次日开盘恢复：旧仓平于开盘 + 新仓开于开盘，敞口变化 2
        assert r.iloc[3] == pytest.approx(0.0)

    def test_valid_opens_unchanged_semantics(self):
        """开盘全有效时与原口径一致：开仓只吃日内、持仓吃全程。"""
        f = _frame([100, 100, 110, 121], 0.05,
                   [False, True, True, True], [0, 1, 1, 1],
                   open_=[100, 100, 105, 120])
        r = cs.portfolio_daily({'AU0': f}, k=1, execution='next_open',
                               charge_cost=False)
        assert r.iloc[2] == pytest.approx(110 / 105 - 1)
        assert r.iloc[3] == pytest.approx(121 / 110 - 1)
        assert r.attrs.get('unexecuted', []) == []


class TestPortfolioDailyExposureContinuity:
    """二轮复审 #1：缺价期间的敞口与盈亏连续性。

    旧实现两个洞：①冻结"仓位"而非敞口——杠杆上调日缺开盘价，次日按
    仓位×新杠杆计收益（1 倍仓位、隔夜 +20% 被记 +40%）；②缺收盘价后
    恢复日的价差被记为 0。"""

    def test_leverage_increase_missing_open_does_not_double_exposure(self):
        f = _frame([100, 100, 100, 120], 0.05,
                   [True, True, True, True], [1, 1, 1, 1],
                   open_=[100, 100, np.nan, 120])
        lev = pd.Series([1.0, 1.0, 2.0, 2.0], index=f.index)
        r = cs.portfolio_daily({'AU0': f}, k=1, execution='next_open',
                               charge_cost=True, leverage=lev)
        assert r.iloc[2] == pytest.approx(0.0), '调仓未成交，全日按 1 倍敞口'
        # 日3 行情恢复：按实际 1 倍敞口吃 +20%（旧实现 +40%）
        assert r.iloc[3] == pytest.approx(0.20 - cs.ONE_WAY_COST)

    def test_missing_close_carries_valuation_to_recovery(self):
        f = _frame([100, 100, np.nan, 126], 0.05,
                   [True, True, True, True], [1, 1, 1, 1],
                   open_=[100, 100, 110, 130])
        r = cs.portfolio_daily({'AU0': f}, k=1, execution='next_open',
                               charge_cost=False)
        assert r.iloc[2] == pytest.approx(110 / 100 - 1)
        # 恢复日从最后有效估值（100）续算：126/100-1，价差不丢
        assert r.iloc[3] == pytest.approx(126 / 110 - 1)
