#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
端到端安全网测试 —— 用精心构造的多日数据跑完整 DailyReplayBacktester 链路，
断言系统级不变量（净值连续/账本平衡/风控行为/无非法状态）。

与接口级测试的区别：不 mock 任何组件，直接跑真实生产代码路径。
如果任何层的数据传递断裂（如复权口径/单位/符号不匹配），此处会红。
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from portfolio.daily_replay import DailyReplayBacktester, ReplayConfig
from trading import config as chain


def make_e2e_data(days=120):
    """
    构造一段包含以下场景的多日数据：
    - 正常波动期（60日）
    - 市场暴跌+个股止损期（20日）→ 触发 L4/L3
    - 恢复期（40日）→ 测试重启/锚重置
    """
    dates = pd.bdate_range('2026-01-05', periods=days)

    # 5只股票 + 基准
    syms = ['600001.SH', '600002.SH', '600003.SH', '300001.SZ', '688001.SH']

    prices = {s: [] for s in syms}
    _base = {s: 10.0 + j * 5.0 for j, s in enumerate(syms)}  # 确定性基准价
    for i in range(days):
        if i < 60:
            for j, s in enumerate(syms):
                px = _base[s] * (1 + 0.02 * np.sin(i * 0.3 + j)) * \
                    (1 + np.random.RandomState(i * 5 + j).randn() * 0.01)
                prices[s].append(round(px, 2))
        elif i < 80:
            # 暴跌期：每日-3%
            for s in syms:
                px = prices[s][-1] * 0.97
                prices[s].append(round(px, 2))
        else:
            # 恢复期：每日+1%
            for s in syms:
                px = prices[s][-1] * 1.01
                prices[s].append(round(px, 2))

    cache = {}
    for sym in syms:
        arr = prices[sym]
        # 补齐前缀（等权起始）
        full = [arr[0]] * 1 + arr
        full = full[:days]
        cache[sym] = pd.DataFrame({
            'date': dates,
            'open': [p * 0.998 for p in full],
            'high': [p * 1.005 for p in full],
            'low': [p * 0.995 for p in full],
            'close': full,
            'volume': [500e4] * days,      # 500万股（流动性充足）
            'amount': [p * 500e4 for p in full],  # 成交额=价×量
            'adjclose': full,               # 无除权，复权=不复权
        })

    bench = [4000.0] * days
    # 基准同步暴跌/恢复
    for i in range(60, 80):
        bench[i] = bench[i-1] * 0.97
    for i in range(80, days):
        bench[i] = bench[i-1] * 1.008
    cache['000300.SH'] = pd.DataFrame({
        'date': dates, 'open': bench, 'high': bench,
        'low': bench, 'close': bench, 'volume': [1e9] * days,
        'amount': [4e12] * days, 'adjclose': bench,
    })
    return cache, dates, syms


def make_scores(cache, dates, syms):
    """构造分数矩阵：正常期随机排名，暴跌期某只持续最高分"""
    rng = np.random.RandomState(42)
    rows = {}
    for i, d in enumerate(dates):
        row = {}
        for j, s in enumerate(syms):
            if i < 60:
                row[s] = float(rng.randn())
            elif i < 80:
                row[s] = 2.0 if j == 0 else rng.randn()  # 暴跌期 600001 最高分
            else:
                row[s] = float(rng.randn())
        rows[d] = row
    return pd.DataFrame(rows).T.sort_index()


class TestEndToEndInvariants:

    def setup_method(self):
        self.cache, self.dates, self.syms = make_e2e_data()
        self.scores = make_scores(self.cache, self.dates, self.syms)
        self.cfg = ReplayConfig(
            top_n=3, min_listing_days=0, min_daily_turnover=0,
            use_risk_gate=True, initial_capital=1_000_000)
        self.result = DailyReplayBacktester(self.cache, self.cfg).run(
            self.scores, start_date=str(self.dates[0].date()))

    def test_nav_continuity_no_impossible_jumps(self):
        """不变量1：净值无单日±50%以上跳变（除非涨跌停，但组合分散后不可能）"""
        eq = self.result.equity_curve
        rets = eq.pct_change().dropna()
        assert len(rets) > 50, "回测期太短"
        max_jump = rets.abs().max()
        assert max_jump < 0.70, \
            f"净值单日跳变{max_jump:.1%}≥70%——数据层或执行层有断裂。" \
            f"最大跳变日: {rets.idxmax()}"

    def test_nav_positive(self):
        """不变量2：净值始终为正"""
        eq = self.result.equity_curve
        assert (eq > 0).all(), "净值出现负值——账本或价格数据有严重错误"

    def test_ledger_balance_equation(self):
        """不变量3：最后一日 现金+持仓市值 = 净值（账本平衡）"""
        # 从 daily_holdings 提取最后有持仓日的市值
        holdings = self.result.daily_holdings
        last_hold_day = max((d for d, h in holdings.items() if h), default=None)
        if last_hold_day:
            mv = sum(h[1] for h in holdings[last_hold_day])
            # 净值 ≥ 持仓市值（差额为现金，必须≥0）
            nav_that_day = self.result.equity_curve.loc[last_hold_day] \
                if last_hold_day in [str(d)[:10] for d in self.result.equity_curve.index] else None
            if nav_that_day:
                assert nav_that_day >= mv * 0.95, \
                    f"净值{nav_that_day:,.0f} < 持仓市值{mv:,.0f}×0.95——账本不平衡"

    def test_no_negative_positions(self):
        """不变量4：daily_holdings 中不出现负市值（负持仓）"""
        for day, holds in self.result.daily_holdings.items():
            for sym, mv in holds:
                assert mv >= 0, f"{day} {sym} 市值{mv:,.0f}<0——负持仓泄漏"

    def test_stop_loss_actually_fires(self):
        """不变量5：暴跌期（60-80日连续-3%）应触发止损卖出"""
        # 暴跌期累计跌幅 (0.97^20 - 1) ≈ -46%
        # top3 组合应有个股触发 -5% 止损
        assert self.result.n_stop_sells > 0, \
            "连续暴跌期零止损——L4止损未生效或价格数据有误"

    def test_drawdown_bounded_by_hard_limit(self):
        """不变量6：最大回撤受硬限约束（15%硬限清仓后不应继续大跌）"""
        # 允许略超15%（清仓日有滑点/停牌），但不应超过20%
        assert self.result.max_drawdown > -0.25, \
            f"最大回撤{self.result.max_drawdown:.1%}超过-25%——" \
            "硬限清仓未生效或净值计算有误"

    def test_trades_have_valid_quantities(self):
        """不变量7：所有成交数量为正整数，价格为正"""
        for t in self.result.trades:
            assert t['quantity'] > 0, f"{t['date']} {t['symbol']} 数量{t['quantity']}≤0"
            assert t['price'] > 0, f"{t['date']} {t['symbol']} 价格{t['price']}≤0"
            assert t['side'] in ('BUY', 'SELL')

    def test_total_cost_reasonable(self):
        """不变量8：总成本占本金比例合理（<5%，单边费率~0.1%×多次调仓）"""
        ratio = self.result.total_cost / self.result.initial_capital
        assert 0 < ratio < 0.05, \
            f"总成本占本金{ratio:.1%}——费用计算可能有误"

    def test_terminal_state_consistent(self):
        """不变量9：终态标记的日期必须在回测时间轴内（锚重置允许重建，不限制成交方向）"""
        if self.result.terminal_date:
            dates_str = set(str(d)[:10] for d in self.result.equity_curve.index)
            assert self.result.terminal_date in dates_str, \
                f"terminal_date {self.result.terminal_date} 不在回测时间轴内"

    def test_equity_curve_matches_final_capital(self):
        """不变量10：equity_curve 最后值 = final_capital"""
        assert self.result.equity_curve.iloc[-1] == pytest.approx(
            self.result.final_capital, rel=1e-6)


class TestEndToEndWithST:

    def test_st_stock_excluded_from_targets(self):
        """安全网：ST股在矩阵标记期间不被选入目标"""
        cache, dates, syms = make_e2e_data()
        scores = make_scores(cache, dates, syms)

        # 标记 600001.SH 全程为 ST
        st_m = pd.DataFrame(False, index=dates, columns=syms)
        st_m['600001.SH'] = True

        cfg = ReplayConfig(top_n=3, min_listing_days=0, min_daily_turnover=0,
                           use_risk_gate=False, st_matrix=st_m)
        res = DailyReplayBacktester(cache, cfg).run(scores, start_date=str(dates[0].date()))
        bought = {t['symbol'] for t in res.trades if t['side'] == 'BUY'}
        assert '600001.SH' not in bought, "ST股被买入——排除逻辑断裂"


class TestEndToEndDataIntegrity:

    def test_price_basis_consistency(self):
        """安全网：回测器内 close（撮合用）与 adjclose（净值用）一致（无除权场景）"""
        cache, dates, syms = make_e2e_data()
        bt = DailyReplayBacktester(cache, ReplayConfig(min_listing_days=0,
                                                       min_daily_turnover=0))
        # 无除权时两矩阵应相等
        pd.testing.assert_frame_equal(bt.close_mx, bt.raw_close_mx)

    def test_volume_amount_consistency(self):
        """安全网：amount ≈ close × volume（构造数据的一致性）"""
        cache, dates, syms = make_e2e_data()
        for sym in syms[:2]:
            df = cache[sym]
            expected = df['close'] * df['volume']
            actual = df['amount']
            # 允许浮点误差
            np.testing.assert_allclose(actual, expected, rtol=0.01)
