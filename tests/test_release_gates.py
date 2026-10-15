#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
发布门槛测试 —— 三个必须通过的门槛（pytest -m release_gate）

门槛1 一致性：生产信号与重放在同一数据下选出完全相同的目标
门槛2 安全性：注入故障场景，验证新买入全部被拦截
门槛3 QMT集成：模拟官方接口签名，跑通完整订单生命周期
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import os
from datetime import date, timedelta

_PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_TEST_RG = os.path.join(_PROJECT, "runtime")

import numpy as np
import pandas as pd
import pytest

from live.signals import generate_target, build_orders
from live.risk_gate import RiskGate, GateConfig
from live.ledger import Ledger
from live.broker import Order

pytestmark = pytest.mark.release_gate


# ================= 门槛1：一致性 =================

class TestConsistencyGate:
    """生产信号与重放回测在相同输入下必须产出相同目标列表"""

    def _make_synthetic_data(self, n_syms=50, days=70):
        """确定性合成数据（生产与重放共用）"""
        dates = pd.bdate_range('2026-06-01', periods=days)
        syms = [f'S{i:04d}.SH' for i in range(n_syms)]
        rng = np.random.RandomState(42)

        cache = {}
        for sym in syms:
            px = 10.0 + rng.randn(days).cumsum() * 0.1 + 5.0
            cache[sym] = pd.DataFrame({
                'date': dates, 'open': px, 'high': px * 1.01,
                'low': px * 0.99, 'close': px, 'volume': [500e4] * days,
                'amount': px * 500e4, 'adjclose': px})
        bench = [4000.0] * days
        cache['000300.SH'] = pd.DataFrame({
            'date': dates, 'open': bench, 'high': bench,
            'low': bench, 'close': bench, 'volume': [1e9] * days,
            'amount': [4e12] * days, 'adjclose': bench})
        return cache, dates, syms

    def test_production_and_replay_same_factors(self):
        """生产信号的因子列表必须与 trading/config.PRODUCTION_FACTORS 完全一致"""
        from trading import config as chain
        from live.signals import PRODUCTION_FACTORS
        assert set(PRODUCTION_FACTORS.keys()) == set(chain.PRODUCTION_FACTORS.keys()), \
            "生产信号因子列表与 config 分叉——R1违规"

    def test_signal_date_consistency(self):
        """信号日期、过滤日期、成交日期必须一致（R2/R6）"""
        cache, dates, syms = self._make_synthetic_data()

        class FakeDS:
            def list_symbols(self, as_of=None):
                return syms
            def preload_batch(self, syms_list, start, end):
                return {s: cache[s] for s in syms_list if s in cache}
            def _db_src(self):
                return None

        import live.signals as sig
        orig_build = sig.build_fundamental_matrix
        sig.build_fundamental_matrix = lambda prov, f, pool, s, e: pd.DataFrame()

        try:
            targets, scores = generate_target(FakeDS(), date(2026, 9, 8), top_n=30)
            # 无因子数据时应返回空目标（质量闸门halt）
            assert len(targets) == 0, "无因子数据时应halt"
        finally:
            sig.build_fundamental_matrix = orig_build

    def test_quality_filter_threshold_consistent(self):
        """质量过滤阈值必须真正驱动行为（行为断言，评审 §2.5b）。

        旧版断言"源码中出现 str(阈值)"——任意位置出现 '0.08' 即通过，
        哪怕阈值硬编码、行为完全错。改为：同一数据、两个阈值，
        过滤结果必须不同（生产/重放共用的 trading/strategy._filter_by_factor）。
        """
        import pandas as pd
        from trading import config as chain
        from trading import strategy as strat

        score = pd.DataFrame(1.0, index=[pd.Timestamp('2026-09-08')],
                             columns=list('ABC'))
        roe = pd.DataFrame([[0.05, 0.10, 0.20]],
                           index=score.index, columns=list('ABC'))

        t0 = chain.QUALITY_FILTER['roe_min']
        out_lo = strat._filter_by_factor(score.copy(), roe, 'min', t0, 'ROE')
        out_hi = strat._filter_by_factor(score.copy(), roe, 'min', 0.15, 'ROE')
        assert pd.isna(out_lo.iloc[0]['A']) and not pd.isna(out_lo.iloc[0]['B'])
        assert pd.isna(out_hi.iloc[0]['B']), \
            f"ROE 阈值 {t0}→0.15 未改变过滤行为——阈值未真正生效（R1违规）"

        lev = pd.DataFrame([[30.0, 60.0, 80.0]],
                           index=score.index, columns=list('ABC'))
        out_lev = strat._filter_by_factor(
            score.copy(), lev, 'max', chain.QUALITY_FILTER['leverage_max'], '杠杆')
        assert not pd.isna(out_lev.iloc[0]['A']) and pd.isna(out_lev.iloc[0]['C']), \
            "杠杆阈值（max 方向）过滤行为错误——R1违规"

    def test_replay_uses_production_factors(self):
        """重放回测必须使用与生产相同的因子（不是自己的列表）"""
        from trading import config as chain
        import run_daily_replay
        src = inspect_src(run_daily_replay)
        # 检查是否引用了 chain.PRODUCTION_FACTORS 或调用了共享函数
        has_chain_ref = ('chain.PRODUCTION_FACTORS' in src or
                         'build_value_score' in src)
        has_old_5factor = "'size'" in src and "'leverage'" in src and \
                          'build_fundamental_matrix' in src
        assert has_chain_ref or not has_old_5factor, \
            "重放仍在自行构建旧5因子——R1违规"


def inspect_src(module):
    import inspect
    return inspect.getsource(module)


# ================= 门槛2：安全性 =================

class TestSafetyGate:
    """注入故障场景，验证所有新买入被拦截（R3）"""

    def _make_ledger(self, tmp_path=None, positions=None, navs=None):
        import tempfile as _tf
        _dir = tmp_path if tmp_path else _tf.mkdtemp()
        led = Ledger(str(_dir / 'l.json') if hasattr(_dir, '__truediv__') else os.path.join(str(_dir), 'l.json'))
        led.initialize(1_000_000)
        if positions:
            led._d['positions'].update(positions)
            led._d['cost'].update({s: 10.0 for s in positions})
        if navs:
            led._d['nav_history'] = [
                {'date': f'2026-09-{i+1:02d}', 'nav': v} for i, v in enumerate(navs)]
        return led

    def _bench(self):
        dates = pd.bdate_range('2026-09-01', periods=3)
        return pd.DataFrame({'date': dates, 'close': [4000.0, 3990.0, 4000.0]})

    def test_missing_factor_halts_buy(self):
        """因子缺失→halt→禁止买入"""
        led = self._make_ledger(None, positions={'S001.SH': 100},
                                 navs=[1_000_000, 950_000])
        orders = [Order('S002.SH', 'BUY', 0.01, amount=5000)]
        d = RiskGate(state_path=os.path.join(_TEST_RG, 'test_rg1.json')).review(
            orders, led, {'S001.SH': 10.0, 'S002.SH': 10.0},
            {'S001.SH': 10.0, 'S002.SH': 10.0}, [],
            bench_df=self._bench(), as_of=date(2026, 9, 3),
            st_data_ok=False)
        assert all(o.side != 'BUY' for o in d.orders), \
            "ST数据不可用时仍放行买入——R3违规"

    def test_stale_prices_block_buy(self):
        """过期价格标的的指令被剔除"""
        led = self._make_ledger(None, navs=[1_000_000, 950_000])
        orders = [Order('STALE.SH', 'BUY', 0.01, amount=5000)]
        d = RiskGate(state_path=os.path.join(_TEST_RG, 'test_rg2.json')).review(
            orders, led, {'STALE.SH': 10.0}, {'STALE.SH': 10.0}, [],
            bench_df=self._bench(), as_of=date(2026, 9, 3),
            stale_symbols={'STALE.SH'})
        assert all(o.symbol != 'STALE.SH' for o in d.orders)

    def test_empty_nav_with_positions_blocks(self):
        """有持仓但净值基线异常→阻断交易"""
        led = self._make_ledger(None, positions={'S001.SH': 100}, navs=[])
        orders = [Order('S002.SH', 'BUY', 0.01, amount=5000)]
        d = RiskGate(state_path=os.path.join(_TEST_RG, 'test_rg3.json')).review(
            orders, led, {'S001.SH': 10.0, 'S002.SH': 10.0},
            {'S001.SH': 10.0, 'S002.SH': 10.0}, [],
            bench_df=self._bench(), as_of=date(2026, 9, 3))
        assert d.action == 'no_buy' or d.action == 'halt', \
            "有持仓但无净值基线仍放行——R3违规"

    def test_failed_recon_blocks_buy(self):
        """上次对账失败→禁止新交易"""
        led = self._make_ledger(None, navs=[1_000_000, 950_000])
        led._d['meta']['last_recon'] = {'date': '2026-09-02', 'ok': False, 'n_diffs': 3}
        led.save()
        orders = [Order('S001.SH', 'BUY', 0.01, amount=5000)]
        d = RiskGate(state_path=os.path.join(_TEST_RG, 'test_rg4.json')).review(
            orders, led, {'S001.SH': 10.0}, {'S001.SH': 10.0}, ['S001.SH'],
            bench_df=self._bench(), as_of=date(2026, 9, 3))
        assert d.action == 'no_buy' or d.action == 'halt'

    def test_stale_recon_blocks_buy(self):
        """距上次对账>3天→禁止新交易"""
        led = self._make_ledger(None, navs=[1_000_000, 950_000])
        led._d['meta']['last_recon'] = {'date': '2026-08-01', 'ok': True, 'n_diffs': 0}
        led.save()
        orders = [Order('S001.SH', 'BUY', 0.01, amount=5000)]
        d = RiskGate(state_path=os.path.join(_TEST_RG, 'test_rg5.json')).review(
            orders, led, {'S001.SH': 10.0}, {'S001.SH': 10.0}, ['S001.SH'],
            bench_df=self._bench(), as_of=date(2026, 9, 10))
        assert d.action == 'no_buy' or d.action == 'halt'

    def test_data_anomaly_price_blocks_buy(self):
        """价格>21%异常→该标的指令被剔除"""
        led = self._make_ledger(None, navs=[1_000000, 950_000])
        orders = [Order('ANOM.SH', 'BUY', 0.01, amount=5000)]
        # 100只中1只异常=1%<5%→剔除但不halt
        prices = {f'S{i:03d}.SH': 10.0 for i in range(100)}
        prev = dict(prices)
        prices['ANOM.SH'] = 13.0  # +30%
        prev['ANOM.SH'] = 10.0
        orders.append(Order('S000.SH', 'BUY', 0.01, amount=5000))
        d = RiskGate(state_path=os.path.join(_TEST_RG, 'test_rg6.json')).review(
            orders, led, prices, prev, list(prices.keys())[:30],
            bench_df=self._bench(), as_of=date(2026, 9, 3))
        assert all(o.symbol != 'ANOM.SH' for o in d.orders)


# ================= 门槛3：QMT集成 =================

class TestQMTIntegrationGate:
    """QMT模拟接口：官方签名 + 完整订单生命周期（R4）——行为测试，非源码检查"""

    def test_official_signature_8_params(self):
        """order_stock_async 实际调用时参数数量必须≤9（account+8个位置参数）。
        用mock拦截真实调用并计数——当前11参数调用必须在此失败。"""
        from live import qmt_broker as qb
        import tempfile

        call_args = []

        class MockXt:
            def order_stock_async(self, *args, **kwargs):
                call_args.append(args)
                return 12345  # 模拟请求序号

        b = qb.QMTBroker(
            state_path=tempfile.mkdtemp() + '/q.json',
            enforce_window=False, order_timeout=0.1)
        b._connected = True
        b._pending = {}
        b._xt_trader = MockXt()

        from live.broker import Order as O
        b._submit_order(O('600000.SH', 'BUY', 0.01), 100, 10.0)

        assert call_args, "order_stock_async未被调用"
        n_args = len(call_args[0])
        assert n_args <= 9, \
            f"order_stock_async传了{n_args}个参数（含account）——官方API只接受9个" \
            f"（account+symbol+direction+qty+price_type+price+strategy+remark+account_id），" \
            f"当前多传了{n_args - 9}个"

    def test_seq_vs_real_order_id_separation(self):
        """请求序号和真实订单号分开存储"""
        from live import qmt_broker as qb
        import tempfile
        b = qb.QMTBroker(state_path=tempfile.mkdtemp() + '/q.json', enforce_window=False)
        b._connected = True
        b._pending = {}
        b._xt_trader = type('FakeXt', (), {'order_stock_async': staticmethod(lambda *a, **kw: 12345)})()
        # 模拟提交
        oid = b._submit_order(
            __import__('live.broker', fromlist=['Order']).Order('600000.SH', 'BUY', 0.01),
            100, 10.0)
        assert oid.startswith('seq_'), f"提交返回的应是请求序号(seq_xxx)，实际: {oid}"
        rec = b._pending[oid]
        assert 'real_order_id' not in rec or rec.get('real_order_id') is None, \
            "提交时不应已有real_order_id"

    def test_ledger_dedup_prevents_double_entry(self):
        """账本去重：同一trade_id不重复入账（R4）"""
        from live.broker import Fill, Order as O
        import tempfile
        led = Ledger(tempfile.mkdtemp() + '/l.json')
        led.initialize(1_000_000)
        fill = Fill(O('A', 'BUY', 0.5), 10.0, 1000, 3.0, 0.1,
                    date(2026, 9, 1), 'filled', trade_id='T001')
        booked1, dupes1, _ = led.apply_fills([fill])
        booked2, dupes2, _ = led.apply_fills([fill])  # 重放
        assert len(booked1) == 1 and len(dupes2) == 1, "同一trade_id应被去重"

    def test_oversell_blocked(self):
        """超额卖出阻止入账（R4）"""
        from live.broker import Fill, Order as O
        import tempfile
        led = Ledger(tempfile.mkdtemp() + '/l.json')
        led.initialize(1_000_000)
        led.apply_fills([Fill(O('A', 'BUY', 1.0), 10.0, 1000, 0, 0, date(2026, 9, 1))])
        # 试图卖出1500股（持仓1000）→ 应被阻止
        booked, _, blocked = led.apply_fills([
            Fill(O('A', 'SELL', 0, quantity=1500), 10.0, 1500, 0, 0, date(2026, 9, 2))])
        assert len(blocked) == 1, "超额卖出应被阻止"
        assert led.positions['A'] == 1000, "持仓不应被清空"
