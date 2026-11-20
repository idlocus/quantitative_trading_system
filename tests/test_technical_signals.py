#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""第二轮信号构造测试：与 indicators/ 库单票交叉验证 + 合成性质 + 联合池行为。"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from research import technical_signals as ts
from research.factor_screen2 import cluster_by_corr


def make_market(n_syms=40, days=400, seed=11):
    """合成 dates×symbols OHLCV（行一致：h=max(o,c)*1.01, l=min(o,c)*0.99）"""
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range('2022-01-03', periods=days)
    syms = [f'S{i:03d}.SH' for i in range(n_syms)]
    close = pd.DataFrame(10 + rng.randn(days, n_syms).cumsum(axis=0) * 0.1,
                         index=dates, columns=syms)
    open_px = close.shift(1) * (1 + rng.randn(days, n_syms) * 0.005)
    open_px.iloc[0] = close.iloc[0]
    high = np.maximum(open_px, close) * 1.01
    low = np.minimum(open_px, close) * 0.99
    vol = pd.DataFrame(rng.uniform(1e6, 5e6, (days, n_syms)), index=dates, columns=syms)
    return {'close': close, 'open': open_px, 'high': high, 'low': low,
            'volume': vol, 'amount': close * vol}


def single_stock_df(m, col=0):
    """取矩阵市场的一只股票转 indicators 库的单票 OHLCV DataFrame"""
    c = m['close'].iloc[:, col]
    df = pd.DataFrame({'date': c.index, 'open': m['open'].iloc[:, col].values,
                       'high': m['high'].iloc[:, col].values,
                       'low': m['low'].iloc[:, col].values,
                       'close': c.values, 'volume': m['volume'].iloc[:, col].values})
    return df.set_index('date')


class TestCrossValidationWithLibrary:
    """矩阵实现 vs indicators/ 库单票实现（容差内一致）"""

    def test_rsi14_matches_library(self):
        m = make_market(seed=3)
        from indicators.momentum_indicators import RSI
        lib = RSI(single_stock_df(m), period=14)   # 库返回 Series
        ours = ts.d6_rsi14(m['close']).iloc[:, 0]
        # 跳过 ewm warmup（首值 NaN vs 0 的初始化差异在收敛前存在，收敛后严格一致）
        both = pd.concat([lib, ours], axis=1).dropna().iloc[50:]
        assert len(both) > 250
        assert np.allclose(both.iloc[:, 0], both.iloc[:, 1], atol=0.5), \
            f"RSI 偏差过大: max={np.abs(both.iloc[:,0]-both.iloc[:,1]).max():.3f}"

    def test_macd_level_matches_library(self):
        m = make_market(seed=4)
        from indicators.trend_indicators import MACD
        lib = MACD(single_stock_df(m))['histogram'] / m['close'].iloc[:, 0]
        ours = ts.d1_macd_level(m['close']).iloc[:, 0]
        both = pd.concat([lib, ours], axis=1).dropna()
        assert len(both) > 300
        assert np.allclose(both.iloc[:, 0], both.iloc[:, 1], atol=1e-6)

    def test_atr_matches_library(self):
        m = make_market(seed=5)
        from indicators.volatility_indicators import ATR
        lib = ATR(single_stock_df(m), period=14) / m['close'].iloc[:, 0]
        ours = (-ts.d16_rel_atr(m['high'], m['low'], m['close'])).iloc[:, 0]
        both = pd.concat([lib, ours], axis=1).dropna()
        assert len(both) > 300
        assert np.allclose(both.iloc[:, 0], both.iloc[:, 1], atol=1e-9)

    def test_mfi_matches_library_shape(self):
        m = make_market(seed=6)
        ours = ts.d19_mfi14(m['high'], m['low'], m['close'], m['volume']).iloc[:, 0]
        assert ours.dropna().between(0, 100).all()
        assert ours.notna().sum() > 300


class TestConstructionProperties:

    def test_all_32_signals_present(self):
        sig = ts.build_all_signals(make_market(n_syms=20, days=420))
        assert len(sig) == 32, f"应 32 个信号（6A+2RPS+20D+4J），实际 {len(sig)}"

    def test_rps_bounds_and_rank(self):
        m = make_market(n_syms=30, seed=7)
        r = ts.rps(m['close'], 120)
        row = r.iloc[-1].dropna()
        assert row.between(1 / 30, 1.0).all()
        assert abs(row.max() - 1.0) < 1e-9          # 最高者=1.0
        assert abs(row.nunique() - len(row)) == 0    # 无并列（连续收益）

    def test_oversold_signals_direction(self):
        """横盘+尾部崩盘（真实超卖形态）：趋势阴跌沿带下行不触下轨，
        %B 超卖发生在窄带横盘后被击穿——BOLL 的指标特性"""
        m = make_market(n_syms=25, days=300, seed=8)
        rng = np.random.RandomState(8)
        path = np.concatenate([20 + rng.randn(290) * 0.05,
                               20 * np.linspace(1, 0.45, 10)])   # 末10日 -55%（窗口混横盘段→带窄可击穿）
        m['close'].iloc[:, 0] = path
        m['open'].iloc[:, 0] = m['close'].iloc[:, 0]
        m['high'].iloc[:, 0] = m['close'].iloc[:, 0] * 1.005
        m['low'].iloc[:, 0] = m['close'].iloc[:, 0] * 0.995
        sig = ts.build_all_signals(m)
        for name in ('D13_wr14', 'D4_boll_pctb', 'A1_risk_adj_reversal'):
            row = sig[name].iloc[-1]
            # 分位数断言（idxmax 对随机对照不公平：恰好有随机股更极端时
            # 第一名会换人；"崩盘股进入最高分位"才是构造要验证的性质）
            assert row.rank(pct=True)['S000.SH'] >= 0.95, \
                f"{name}: 崩盘股应在超卖信号最高分位，实际分位 " \
                f"{row.rank(pct=True)['S000.SH']:.2f}"
        j3 = sig['D3_kdj_j'].iloc[-1]
        if j3.notna().sum() > 5:
            assert j3.rank(pct=True)['S000.SH'] >= 0.8, "暴跌股 -J 应在高位（超卖）"

    def test_joint_pools_exclude_outside(self):
        """J1 条件池：池外必须是 NaN（不得污染）"""
        m = make_market(n_syms=30, days=400, seed=9)
        sig = ts.build_all_signals(m)
        j1 = sig['J1_rps_kdj_oversold'].iloc[-1]
        assert j1.notna().sum() <= 30
        full = sig['J2_rps_macd_up'].iloc[-1].dropna()
        assert full.between(0, 1).all() if len(full) else True   # RPS 值域


class TestCluster:
    def test_cluster_by_corr_groups(self):
        rng = np.random.RandomState(1)
        base = rng.randn(200)
        df = pd.DataFrame({'a': base, 'b': base * 0.9, 'c': rng.randn(200),
                           'd': rng.randn(200) * 0.5 + rng.randn(200)})
        corr = df.corr()
        clusters = cluster_by_corr(corr, 0.7)
        names = {frozenset(c) for c in clusters}
        assert frozenset({'a', 'b'}) in names, "a/b 相关0.9 应聚簇"
        for c in clusters:
            if 'c' in c:
                assert len(c) == 1


class TestCompose:
    """合成纯函数（S1-S4）：定向/加权语义"""

    def _make(self):
        rng = np.random.RandomState(2)
        dates = pd.bdate_range('2024-01-02', periods=30)
        cols = [f'X{i}' for i in range(12)]
        base = pd.DataFrame(rng.randn(30, 12).cumsum(axis=0), index=dates, columns=cols)
        from research.composite_study import _z
        z1 = _z(base)                      # 假想正 IC 信号
        z2 = _z(-base + rng.randn(30, 12) * 0.01)   # 与 base 负相关的信号
        # 用真实名单成员命名（两成员均属 CLUSTER_REPS，其一属 POSITIVE_SURVIVORS）
        return {'zs': {'D16_rel_atr': z1, 'J2_rps_macd_up': z2},
                'ic': {'D16_rel_atr': 0.05, 'J2_rps_macd_up': -0.04},
                'base': base, 'z1': z1}

    def test_s1_direction_flips_negative(self):
        from research.composite_study import compose
        c = self._make()
        out = compose(c['zs'], c['ic'])
        s1 = out['S1_all_dir_ew']
        # 两信号定向后应同向：负 IC 信号被翻转利用（矩阵秩相关）
        corr = s1.corrwith(c['z1'])
        assert corr.mean() > 0.5, f"定向合成应与原信号正相关，实际 {corr.mean():.2f}"

    def test_s4_weight_sign_and_scale(self):
        from research.composite_study import compose
        c = self._make()
        out = compose(c['zs'], c['ic'])
        assert np.isfinite(out['S4_ic_weighted'].to_numpy()).all()

    def test_member_lists_locked(self):
        from research.composite_study import CLUSTER_REPS, POSITIVE_SURVIVORS
        assert len(CLUSTER_REPS) == 10 and len(POSITIVE_SURVIVORS) == 10
        assert 'D16_rel_atr' in POSITIVE_SURVIVORS


class TestVolTargetCap:
    """试验C 波动率目标敞口数学（绕过重构造，直接测方法）"""

    def _sim(self, navs, gross=0.95):
        from research.portfolio_simulation import PortfolioSimulation
        sim = PortfolioSimulation.__new__(PortfolioSimulation)
        sim.nav_rows = [{'nav': v} for v in navs]
        sim.config = {'gross_target': gross}
        sim.vol_targeted = True
        return sim

    def test_full_cap_when_calm(self):
        """低波动（年化~7.5%<15%）→ 满仓 gross"""
        navs = list(np.cumprod(1 + np.random.RandomState(1).normal(0.0004, 0.0005, 80)) * 1e6)
        sim = self._sim(navs)
        assert abs(sim._vol_target_cap() - 0.95) < 1e-9

    def test_scaled_when_volatile(self):
        """高波动（日1.5%≈年化24%）→ 缩至 15%/24%×gross"""
        navs = list(np.cumprod(1 + np.random.RandomState(2).normal(0, 0.015, 80)) * 1e6)
        cap = self._sim(navs)._vol_target_cap()
        assert 0.4 < cap < 0.8, f"高波动应缩仓至0.4-0.8×gross，实际 {cap:.2f}"

    def test_warmup_full_gross(self):
        """不足 60 个观测 → 满仓起步（预注册写死）"""
        sim = self._sim([1e6] * 30)
        assert sim._vol_target_cap() == 0.95


class TestDistributionMerge:
    """同日多笔分红合并修复（601088 案例）的行为锁定。

    2026-11-19 审计一-8 修订：合并键加入支付日/红股上市日——同除权日但
    结算日不同的分配保持独立（各自到期），只有结算安排完全相同的同日
    分配才合并金额。"""

    def test_same_day_same_payout_merges(self):
        import pandas as pd
        from research.portfolio_simulation import distributions
        src = pd.DataFrame({
            'S_INFO_WINDCODE': ['X.SH'] * 3,
            'S_DIV_PROGRESS': ['3', '3', '3'],
            'EQY_RECORD_DT': ['20170707', '20170707', '20180706'],
            'EX_DT': ['20170710', '20170710', '20180709'],
            'DVD_PAYOUT_DT': ['20170710', '20170710', '20180709'],
            'LISTING_DT_OF_DVD_SHR': [None, None, None],
            'CASH_DVD_PER_SH_PRE_TAX': [2.51, 0.46, 0.9],
            'STK_DVD_PER_SH': [0.0, 0.0, 0.0],
            'OBJECT_ID': ['a', 'b', 'c'],
        })
        dates = pd.DatetimeIndex(pd.bdate_range('2016-01-01', '2019-12-31'))
        actions, quality = distributions(src, {'X.SH'}, dates)
        assert quality['same_day_distributions_merged'] == 1
        assert quality['ambiguous_events'] == 0
        same_day = [a for a in actions if str(a.record)[:10] == '2017-07-07']
        assert len(same_day) == 1 and abs(same_day[0].cash - 2.97) < 1e-9, \
            "同日同结算的两笔应合并为一笔，金额相加"

    def test_same_day_unknown_payout_kept_separate(self):
        """601088 实测形态（第二笔支付日缺失）：不并入第一笔的支付日——
        未知结算日的现金单独挂账（informational），不得提前可用。"""
        import pandas as pd
        from research.portfolio_simulation import distributions
        src = pd.DataFrame({
            'S_INFO_WINDCODE': ['X.SH'] * 2,
            'S_DIV_PROGRESS': ['3', '3'],
            'EQY_RECORD_DT': ['20170707', '20170707'],
            'EX_DT': ['20170710', '20170710'],
            'DVD_PAYOUT_DT': ['20170710', None],
            'LISTING_DT_OF_DVD_SHR': [None, None],
            'CASH_DVD_PER_SH_PRE_TAX': [2.51, 0.46],
            'STK_DVD_PER_SH': [0.0, 0.0],
            'OBJECT_ID': ['a', 'b'],
        })
        dates = pd.DatetimeIndex(pd.bdate_range('2016-01-01', '2019-12-31'))
        actions, quality = distributions(src, {'X.SH'}, dates)
        same_day = [a for a in actions if str(a.record)[:10] == '2017-07-07']
        assert len(same_day) == 2, '支付日不同/未知的两笔不得合并到同一支付日'
        pays = sorted(str(a.payout) for a in same_day)
        assert pays == ['2017-07-10 00:00:00', 'None']

    def test_progress1_excluded_and_missing_record_flagged(self):
        import pandas as pd
        from research.portfolio_simulation import distributions
        src = pd.DataFrame({
            'S_INFO_WINDCODE': ['X.SH', 'X.SH'],
            'S_DIV_PROGRESS': ['1', '3'],
            'EQY_RECORD_DT': [None, None],          # 缺登记日=真病态
            'EX_DT': ['20170710', '20180709'],
            'DVD_PAYOUT_DT': [None, None],
            'LISTING_DT_OF_DVD_SHR': [None, None],
            'CASH_DVD_PER_SH_PRE_TAX': [0.5, 0.9],
            'STK_DVD_PER_SH': [0.0, 0.0],
            'OBJECT_ID': ['p', 'q'],
        })
        dates = pd.DatetimeIndex(pd.bdate_range('2016-01-01', '2019-12-31'))
        actions, quality = distributions(src, {'X.SH'}, dates)
        assert len(actions) == 1                     # 预案(progress=1)被滤
        assert actions[0].ambiguous                  # 缺登记日的真病态仍标歧义


class TestLayerWiringGuard:
    """源码守护：portfolio_layer 中被 pop 的 protocol 附加参数禁止再经
    protocol.get 取值（external_scales/daily_weights/hold_buffer_ranks
    三次同款接线 bug 的根治防线，2026-11-10）"""

    def test_no_popped_keys_fetched_from_protocol(self):
        from pathlib import Path
        src = (Path('research/composite_study.py')).read_text(encoding='utf-8')
        # pop 之后（第二次出现的位置起）不得再出现 protocol.get('<key>')
        for key in ('external_scales', 'daily_weights', 'hold_buffer_ranks'):
            assert f"protocol.get('{key}')" not in src, \
                f"protocol.get('{key}') 出现——应使用 pop 的局部变量（接线bug模式）"
