#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""执行风控闸门测试：L2数据停机 / L3组合熔断 / L4强制卖出 / T+1 / 状态持久化"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import date, timedelta

import pandas as pd
import pytest

from live.broker import Order
from live.ledger import Ledger
from live.risk_gate import RiskGate, GateConfig, RiskDecision, render_gate_markdown


def make_ledger(tmp_path, positions=None, costs=None, cash=100000.0, navs=None, trades=None):
    led = Ledger(str(tmp_path / 'l.json'))
    led._d = {
        'cash': cash,
        'positions': positions or {},
        'cost': costs or {},
        'trades': trades or [],
        'nav_history': [{'date': f'2026-09-{i+1:02d}', 'nav': v} for i, v in enumerate(navs)] if navs else [],
        'meta': {},
    }
    return led


def bench_df(as_of, last_close=4000.0, chg=-0.01, stale_days=0):
    """基准两根K线；stale_days>0 模拟数据滞后"""
    d2 = as_of - timedelta(days=stale_days)
    return pd.DataFrame({'date': pd.to_datetime([d2 - timedelta(days=1), d2]),
                         'close': [last_close / (1 + chg), last_close]})


AS_OF = date(2026, 10, 12)


class TestL2DataHalt:

    def test_stale_bench_halts_everything(self, tmp_path):
        led = make_ledger(tmp_path, positions={'A': 1000}, costs={'A': 10.0})
        gate = RiskGate(state_path=str(tmp_path / 'rs.json'))
        orders = [Order('A', 'SELL', 0, quantity=100)]
        d = gate.review(orders, led, {'A': 9.0}, {'A': 9.0}, ['A'],
                        bench_df(AS_OF, stale_days=30), as_of=AS_OF)
        assert d.action == 'halt'
        assert d.orders == []
        assert len(d.suppressed) == 1
        assert any('滞后' in i for i in d.data_issues)

    def test_low_price_coverage_halts(self, tmp_path):
        led = make_ledger(tmp_path)
        targets = [f'S{i}' for i in range(10)]
        prices = {f'S{i}': 10.0 for i in range(5)}   # 仅覆盖50%
        gate = RiskGate(state_path=str(tmp_path / 'rs.json'))
        d = gate.review([], led, prices, prices, targets, bench_df(AS_OF), as_of=AS_OF)
        assert d.action == 'halt'
        assert any('覆盖率' in i for i in d.data_issues)

    def test_single_anomaly_excluded_not_halt(self, tmp_path):
        """个别异常剔除指令；异常占比超限才停机"""
        led = make_ledger(tmp_path)
        syms = [f'S{i}' for i in range(25)]
        prices = {s: 10.0 for s in syms}
        prev = dict(prices)
        prices['S0'] = 13.0          # +30% > 21% 异常
        prev['S0'] = 10.0
        gate = RiskGate(state_path=str(tmp_path / 'rs.json'))
        orders = [Order('S0', 'BUY', 0.01, amount=5000), Order('S1', 'BUY', 0.01, amount=5000)]
        d = gate.review(orders, led, prices, prev, syms, bench_df(AS_OF), as_of=AS_OF)
        assert d.action == 'trade'   # 1/25=4% ≤ 5%，不停机
        assert len(d.suppressed) == 1 and d.suppressed[0][0].symbol == 'S0'
        assert len(d.orders) == 1 and d.orders[0].symbol == 'S1'

    def test_anomaly_rate_over_limit_halts(self, tmp_path):
        led = make_ledger(tmp_path)
        syms = [f'S{i}' for i in range(20)]
        prices = {s: 10.0 for s in syms}
        prev = dict(prices)
        for i in range(3):           # 3/20=15% > 5%
            prices[f'S{i}'] = 13.0
        gate = RiskGate(state_path=str(tmp_path / 'rs.json'))
        d = gate.review([], led, prices, prev, syms, bench_df(AS_OF), as_of=AS_OF)
        assert d.action == 'halt'
        assert any('异常价格占比' in i for i in d.data_issues)


class TestL4ForcedSells:

    def test_stop_loss_injects_forced_sell(self, tmp_path):
        led = make_ledger(tmp_path, positions={'A': 1000, 'B': 1000},
                          costs={'A': 10.0, 'B': 10.0})
        prices = {'A': 9.4, 'B': 10.5, 'C': 10.0}   # A -6% 破止损
        gate = RiskGate(state_path=str(tmp_path / 'rs.json'))
        orders = [Order('A', 'BUY', 0.01, amount=5000), Order('C', 'BUY', 0.01, amount=5000)]
        d = gate.review(orders, led, prices, prices, ['A', 'B', 'C'],
                        bench_df(AS_OF), as_of=AS_OF)
        assert d.action == 'trade'
        assert len(d.forced_sells) == 1
        fs = d.forced_sells[0]
        assert fs.symbol == 'A' and fs.quantity == 1000 and fs.side == 'SELL'
        assert '[L4]' in fs.note
        # A 的买入被撤销（同标的既卖又买矛盾）
        assert any(o.symbol == 'A' and why.startswith('[L4]') for o, why in d.suppressed)
        # 强制卖出在放行指令中
        assert any(o is fs for o in d.orders)

    def test_t1_lock_defers_stop_to_tomorrow(self, tmp_path):
        led = make_ledger(tmp_path, positions={'A': 1000}, costs={'A': 10.0},
                          trades=[{'date': str(AS_OF), 'symbol': 'A', 'side': 'BUY'}])
        prices = {'A': 9.0}
        gate = RiskGate(state_path=str(tmp_path / 'rs.json'))
        d = gate.review([], led, prices, prices, ['A'], bench_df(AS_OF), as_of=AS_OF)
        assert d.forced_sells == []
        assert any('T+1' in r for r in d.reasons)

    def test_stop_daily_cap_defers_excess(self, tmp_path):
        positions = {f'S{i}': 1000 for i in range(25)}
        costs = {f'S{i}': 10.0 for i in range(25)}
        prices = {f'S{i}': 9.0 for i in range(25)}   # 全部 -10%
        led = make_ledger(tmp_path, positions=positions, costs=costs)
        gate = RiskGate(state_path=str(tmp_path / 'rs.json'))
        d = gate.review([], led, prices, prices, list(positions), bench_df(AS_OF), as_of=AS_OF)
        assert len(d.forced_sells) == 20             # 默认单日上限
        assert any('顺延' in r for r in d.reasons)
        # 亏损最深的优先：S0..S19（成本同、价格同为9时按字典序）——至少全部是破止损标的
        assert all(o.side == 'SELL' for o in d.forced_sells)

    def test_healthy_positions_no_forced_sells(self, tmp_path):
        led = make_ledger(tmp_path, positions={'A': 1000}, costs={'A': 10.0})
        gate = RiskGate(state_path=str(tmp_path / 'rs.json'))
        d = gate.review([], led, {'A': 11.0}, {'A': 10.8}, ['A'], bench_df(AS_OF), as_of=AS_OF)
        assert d.forced_sells == [] and d.action == 'trade'


class TestL3PortfolioBreakers:

    def test_drawdown_soft_scales_buys(self, tmp_path):
        # 峰值100 → 现90：回撤10% → exposure = 1 - (0.10-0.08)/0.07 ≈ 0.714
        # （最后两日 95→90 = -5.3%，不触发单日暴跌熔断）
        led = make_ledger(tmp_path, positions={'A': 1000}, costs={'A': 10.0},
                          cash=80000.0, navs=[100000, 95000, 90000])
        gate = RiskGate(state_path=str(tmp_path / 'rs.json'))
        orders = [Order('B', 'BUY', 0.01, amount=10000)]
        prices = {'A': 10.0, 'B': 10.0}
        d = gate.review(orders, led, prices, prices, ['A', 'B'], bench_df(AS_OF), as_of=AS_OF)
        assert d.action == 'scale'
        assert 0.70 < d.exposure < 0.72
        buy = [o for o in d.orders if o.side == 'BUY'][0]
        # 预算制：单票足额（10000 ≤ 预算），不再×exposure 切碎
        assert buy.amount == 10000
        assert '[L3]缩量预算制' in buy.note

    def test_drawdown_soft_trims_overweight_holdings(self, tmp_path):
        # 持仓占比90% > exposure+2% → 从最大持仓削减
        led = make_ledger(tmp_path, positions={'A': 9000}, costs={'A': 10.0},
                          cash=10000.0, navs=[100000, 98000, 90000])
        gate = RiskGate(state_path=str(tmp_path / 'rs.json'))
        prices = {'A': 10.0}
        d = gate.review([], led, prices, prices, ['A'], bench_df(AS_OF), as_of=AS_OF)
        trims = [o for o in d.orders if o.side == 'SELL']
        assert len(trims) == 1 and trims[0].symbol == 'A'
        assert trims[0].quantity >= 100

    def test_drawdown_hard_liquidates_all(self, tmp_path):
        led = make_ledger(tmp_path, positions={'A': 1000, 'B': 2000},
                          costs={'A': 10.0, 'B': 5.0}, navs=[100000, 90000, 84000])  # -16%
        gate = RiskGate(state_path=str(tmp_path / 'rs.json'))
        orders = [Order('C', 'BUY', 0.01, amount=5000)]
        prices = {'A': 10.0, 'B': 5.0, 'C': 10.0}
        d = gate.review(orders, led, prices, prices, ['A', 'B', 'C'], bench_df(AS_OF), as_of=AS_OF)
        assert d.action == 'liquidate'
        assert d.exposure == 0.0
        sells = {o.symbol: o for o in d.orders}
        assert set(sells) == {'A', 'B'}
        assert sells['A'].quantity == 1000 and sells['B'].quantity == 2000
        assert all('熔断清仓' in o.note for o in d.orders)
        assert len(d.suppressed) == 1   # 原BUY作废

    def test_market_crash_no_buy_with_cooldown_persisted(self, tmp_path):
        state = str(tmp_path / 'rs.json')
        led = make_ledger(tmp_path, positions={'A': 1000}, costs={'A': 10.0})
        prices = {'A': 10.0, 'B': 10.0}
        gate = RiskGate(state_path=state)
        orders = [Order('B', 'BUY', 0.01, amount=5000), Order('A', 'SELL', 0, quantity=100)]

        # Day1：基准 -6% 触发市场熔断
        d1 = gate.review(orders, led, prices, prices, ['A', 'B'],
                         bench_df(AS_OF, chg=-0.06), as_of=AS_OF)
        assert d1.action == 'no_buy'
        # 买入拦截、卖出放行
        assert any(o.side == 'BUY' for o, _ in d1.suppressed)
        assert any(o.side == 'SELL' for o in d1.orders)

        # Day2（冷却期内，新实例验证持久化）：基准恢复正常仍禁买
        gate2 = RiskGate(state_path=state)
        d2 = gate2.review(orders, led, prices, prices, ['A', 'B'],
                          bench_df(AS_OF + timedelta(days=1), chg=0.01),
                          as_of=AS_OF + timedelta(days=1))
        assert d2.action == 'no_buy'
        assert any('冷却' in r for r in d2.reasons)

        # Day3：冷却期结束恢复
        gate3 = RiskGate(state_path=state)
        d3 = gate3.review(orders, led, prices, prices, ['A', 'B'],
                          bench_df(AS_OF + timedelta(days=2), chg=0.01),
                          as_of=AS_OF + timedelta(days=2))
        assert d3.action == 'trade'
        assert len(d3.orders) == 2

    def test_portfolio_daily_crash_no_buy(self, tmp_path):
        # 净值 100→92：单日 -8% ≤ -7%
        led = make_ledger(tmp_path, positions={'A': 1000}, costs={'A': 10.0},
                          navs=[100000, 100000, 92000])
        gate = RiskGate(state_path=str(tmp_path / 'rs.json'))
        prices = {'A': 10.0}
        d = gate.review([Order('B', 'BUY', 0.01, amount=5000)], led, prices, prices,
                        ['A'], bench_df(AS_OF), as_of=AS_OF)
        assert d.action == 'no_buy'
        assert any('组合单日' in r for r in d.reasons)


class TestNormalFlow:

    def test_orders_pass_through_unchanged(self, tmp_path):
        led = make_ledger(tmp_path, positions={'A': 1000}, costs={'A': 10.0})
        gate = RiskGate(state_path=str(tmp_path / 'rs.json'))
        o1 = Order('B', 'BUY', 0.01, amount=5000)
        o2 = Order('A', 'SELL', 0, quantity=200)
        d = gate.review([o1, o2], led, {'A': 10.0, 'B': 10.0}, {'A': 10.0, 'B': 10.0},
                        ['A', 'B'], bench_df(AS_OF), as_of=AS_OF)
        assert d.action == 'trade'
        assert d.exposure == 1.0
        assert set(id(o) for o in d.orders) == {id(o1), id(o2)}   # 原对象透传
        assert d.suppressed == [] and d.forced_sells == []

    def test_halt_skips_even_stop_loss(self, tmp_path):
        """停机优先于一切：数据不可信时连强制卖出也不执行"""
        led = make_ledger(tmp_path, positions={'A': 1000}, costs={'A': 10.0})
        gate = RiskGate(state_path=str(tmp_path / 'rs.json'))
        d = gate.review([], led, {'A': 9.0}, {'A': 9.0}, ['A'],
                        bench_df(AS_OF, stale_days=30), as_of=AS_OF)
        assert d.action == 'halt' and d.orders == []

    def test_render_markdown_layers(self, tmp_path):
        led = make_ledger(tmp_path, positions={'A': 1000}, costs={'A': 10.0})
        gate = RiskGate(state_path=str(tmp_path / 'rs.json'))
        d = gate.review([Order('A', 'BUY', 0.01, amount=5000)], led,
                        {'A': 9.4}, {'A': 9.4}, ['A'], bench_df(AS_OF), as_of=AS_OF)
        md = "\n".join(render_gate_markdown(d))
        assert 'L4' in md and '强制卖出' in md
        assert 'L0研究指标' in md and 'L1预警' in md


class TestGateConfigDefaults:

    def test_thresholds_align_backtest_risk_limits(self):
        """生产闸门阈值与回测 RiskLimits 口径对齐（回撤/熔断）"""
        cfg = GateConfig()
        assert cfg.drawdown_soft == 0.08
        assert cfg.drawdown_hard == 0.15
        assert cfg.portfolio_daily_crash == 0.07
        assert cfg.market_daily_crash == 0.05
        assert cfg.stop_loss_pct == 0.05   # 与 journal 风险扫描 cost×0.95 一致


class TestScaleDeadlockFix:
    """缩量死锁修复（2026-10-12 审计发现：深回撤→碎片单全拒→空仓→净值冻结→
    回撤永不恢复→2.5年零交易。生产闸门与重放共用本实现）"""

    def _ledger(self, tmp_path, navs, positions=None):
        led = Ledger(str(tmp_path / 'l.json'))
        led.initialize(1_000_000)
        led._d['nav_history'] = [{'date': f'2026-09-{i+1:02d}', 'nav': v}
                                 for i, v in enumerate(navs)]
        led._d['positions'] = positions or {}
        return led

    def _bench(self):
        dates = pd.bdate_range('2026-10-06', periods=3)
        return pd.DataFrame({'date': dates, 'close': [4000.0, 3990.0, 4000.0]})

    def test_budget_allocation_not_fragmented(self, tmp_path):
        """缩量预算制：exposure=30% 时足额买 ~30 只（每只1%净值），
        而非 100 只各 0.3% 的碎片（旧逻辑会全部 <min_order_amount 拒掉）"""
        # 渐进下跌至 dd≈12.5%（单日跌幅<7%，避免误触暴跌熔断）
        led = self._ledger(tmp_path, [1_000_000, 970_000, 945_000, 920_000,
                                      900_000, 885_000, 875_000])
        orders = [Order(f'S{i:03d}.SH', 'BUY', 0.01, amount=9000) for i in range(100)]
        d = RiskGate(state_path=str(tmp_path / 'rs.json')).review(
            orders, led, {f'S{i:03d}.SH': 10.0 for i in range(100)},
                              {f'S{i:03d}.SH': 10.0 for i in range(100)},
                              [], bench_df=self._bench(), as_of=date(2026, 10, 8))
        assert d.action == 'scale'
        buys = [o for o in d.orders if o.side == 'BUY']
        assert buys, "缩量时必须有足额买单放行（旧逻辑全拒=死锁成因）"
        for o in buys:
            assert o.amount >= 2000, "缩量后单票金额必须≥最低下单金额"
        # 总买入预算 ≈ exposure×nav
        total = sum(o.amount for o in buys)
        assert total <= d.exposure * 1_000_000 * 1.01   # 预算基准=cash+市值=100万

    def test_empty_position_keeps_anchor(self, tmp_path):
        """空仓不重置回撤锚（2026-10-15 修复）。

        旧版在 scale+空仓 时锚重置+恢复 trade/1.0，导致回撤按"锚分段"计算：
        每段 <15% 硬限而累计可无限深（E2E 实测 -26.3% 未熔断、十年重放
        B口径累计 -24.9% 未熔断）。10-12 的缩量死锁已由预算制单独修复，
        锚重置属过度保险，现删除：回撤锚只在显式 restart 时重置。"""
        # 渐进下跌至 dd≈14.5%（scale 且 exposure<15%；单日<7%）
        led = self._ledger(tmp_path, [1_000_000, 970_000, 945_000, 920_000,
                                      900_000, 880_000, 865_000, 855_000])
        orders = [Order('S001.SH', 'BUY', 0.01, amount=9000)]
        gate = RiskGate(state_path=str(tmp_path / 'rs.json'))
        d = gate.review(orders, led, {'S001.SH': 10.0}, {'S001.SH': 10.0},
                        [], bench_df=self._bench(), as_of=date(2026, 10, 8))
        assert any('回撤锚保持' in r for r in d.reasons)
        assert d.action == 'scale'          # 不再恢复 trade
        assert d.exposure < 0.15            # 缩量敞口保持，不回满仓
        # 锚未写入状态：新实例回撤仍按全程峰值计算（≈14.5%）
        gate2 = RiskGate(state_path=str(tmp_path / 'rs.json'))
        d2 = gate2.review(orders, led, {'S001.SH': 10.0}, {'S001.SH': 10.0},
                          [], bench_df=self._bench(), as_of=date(2026, 10, 8))
        assert d2.drawdown > 0.14


class TestLiquidateTerminalSemantics:
    """硬限清仓终态：默认进入重入协议冷却（醒目留痕）；
    restart_after_liquidation=True 立即满仓重启（研究对照）"""

    def _bench(self):
        dates = pd.bdate_range('2026-10-06', periods=3)
        return pd.DataFrame({'date': dates, 'close': [4000.0, 3990.0, 4000.0]})

    def _ledger_liquidated(self, tmp_path):
        led = Ledger(str(tmp_path / 'l.json'))
        led.initialize(1_000_000)
        # 渐进下跌至 dd>15%（硬限），已无持仓（清仓完成）
        led._d['nav_history'] = [{'date': f'2026-09-{i+1:02d}', 'nav': v}
                                 for i, v in enumerate([1_000_000, 970_000, 940_000,
                                                        910_000, 880_000, 860_000,
                                                        845_000])]
        return led

    def test_default_reenters_with_loud_reason(self, tmp_path):
        """定稿语义：清仓完成 → 立即满仓重入（醒目留痕+计数），
        不再无限期持币等人工（十年实证 -7.2% vs +59.0%），也不再冷却"""
        led = self._ledger_liquidated(tmp_path)
        orders = [Order('S001.SH', 'BUY', 0.01, amount=9000)]
        d = RiskGate(state_path=str(tmp_path / 'rs.json')).review(
            orders, led, {'S001.SH': 10.0}, {'S001.SH': 10.0}, [],
            bench_df=self._bench(), as_of=date(2026, 10, 8))
        assert d.action == 'trade' and d.exposure == 1.0
        assert any('硬限清仓完成' in r for r in d.reasons), "必须醒目留痕"
        assert any(o.side == 'BUY' for o in d.orders)

    def test_restart_flag_reenables_trading(self, tmp_path):
        led = self._ledger_liquidated(tmp_path)
        orders = [Order('S001.SH', 'BUY', 0.01, amount=9000)]
        cfg = GateConfig(restart_after_liquidation=True)
        d = RiskGate(cfg, state_path=str(tmp_path / 'rs2.json')).review(
            orders, led, {'S001.SH': 10.0}, {'S001.SH': 10.0}, [],
            bench_df=self._bench(), as_of=date(2026, 10, 8))
        assert d.action == 'trade', "restart 模式清仓后应恢复交易"
        assert any('研究对照' in r for r in d.reasons)
        assert any(o.side == 'BUY' for o in d.orders)


class TestReentryProtocol:
    """重入协议（2026-10-15 定稿：清仓完成→立即满仓重入+滚动窗口熔断）

    参数来自 trading/config：MAX_LIQUIDATIONS=3 / CIRCUIT_WINDOW_DAYS=365。
    第一轮"冷却20日+半仓梯度"设计被十年重放否决（-1.5%/-27.6% 双输于
    立即重入 +59.0%/-24.9%），定稿为立即重入+熔断计数。"""

    def _bench(self):
        dates = pd.bdate_range('2026-10-06', periods=3)
        return pd.DataFrame({'date': dates, 'close': [4000.0, 3990.0, 4000.0]})

    def _ledger(self, tmp_path, navs, name='l.json'):
        led = Ledger(str(tmp_path / name))
        led.initialize(1_000_000)
        led._d['nav_history'] = [{'date': f'2026-09-{i+1:02d}', 'nav': v}
                                 for i, v in enumerate(navs)]
        led._d['positions'] = {}
        return led

    def _liquidated(self, tmp_path):
        # 渐进下跌至 dd>15%（硬限），已无持仓（清仓完成）
        return self._ledger(tmp_path, [1_000_000, 970_000, 940_000, 910_000,
                                       880_000, 860_000, 845_000])

    def _review(self, gate, led, as_of=date(2026, 10, 8)):
        return gate.review([Order('S001.SH', 'BUY', 0.01, amount=9000)],
                           led, {'S001.SH': 10.0}, {'S001.SH': 10.0}, [],
                           bench_df=self._bench(), as_of=as_of)

    def test_liquidation_reenters_immediately(self, tmp_path):
        """清仓完成 → 立即满仓重入（现金为新锚），无冷却期"""
        led = self._liquidated(tmp_path)
        gate = RiskGate(state_path=str(tmp_path / 'rs.json'))
        d = self._review(gate, led)
        assert d.action == 'trade' and d.exposure == 1.0, "定稿语义=立即满仓重入"
        assert any('立即重入' in r for r in d.reasons)
        assert abs(gate._state['dd_anchor'] - 845_000) < 1.0   # 现金为锚
        assert gate._state['liq_dates'] == ['2026-10-08']

    def test_circuit_halts_on_excess_liquidations(self, tmp_path):
        """滚动窗口内第4次硬限（>上限3）→ 永久停机；后续 review 一律拦截"""
        gate = RiskGate(state_path=str(tmp_path / 'rs.json'))
        gate._state['liq_dates'] = ['2026-01-10', '2026-04-10', '2026-07-10']
        led = self._liquidated(tmp_path)
        d = self._review(gate, led)
        assert gate._state.get('protocol_halted') is True
        assert d.action == 'halt'
        assert any('重入熔断' in r for r in d.reasons)
        # 后续（新账本）L2.7 早期拦截一切
        led2 = self._ledger(tmp_path, [1_000_000, 1_000_000], name='l2.json')
        d2 = RiskGate(state_path=str(tmp_path / 'rs.json'))._review(
            RiskGate(state_path=str(tmp_path / 'rs.json')), led2) if False else             RiskGate(state_path=str(tmp_path / 'rs.json')).review(
                [Order('S001.SH', 'BUY', 0.01, amount=9000)],
                led2, {'S001.SH': 10.0}, {'S001.SH': 10.0}, [],
                bench_df=self._bench(), as_of=date(2026, 10, 9))
        assert d2.action == 'halt'
        assert any('协议永久停机' in i for i in d2.data_issues)

    def test_circuit_window_expires(self, tmp_path):
        """窗口外的历史清仓不计数（滚动窗口，旧事件自然失效）"""
        gate = RiskGate(state_path=str(tmp_path / 'rs.json'))
        gate._state['liq_dates'] = ['2024-01-10', '2024-06-10', '2025-01-10']
        led = self._liquidated(tmp_path)
        d = self._review(gate, led)   # as_of=2026-10-08，全部 >365 日 → 窗口清空
        assert gate._state.get('protocol_halted') is None
        assert d.action == 'trade'
        assert gate._state['liq_dates'] == ['2026-10-08'], "窗口外事件应被淘汰"

    def test_restart_flag_skips_circuit(self, tmp_path):
        """研究对照（restart=True）：计数超限也不熔断（纯立即重启）"""
        cfg = GateConfig(restart_after_liquidation=True)
        gate = RiskGate(cfg, state_path=str(tmp_path / 'rs.json'))
        gate._state['liq_dates'] = ['2026-01-10', '2026-04-10', '2026-07-10']
        led = self._liquidated(tmp_path)
        d = self._review(gate, led)
        assert d.action == 'trade', "研究对照无熔断"
        assert any('研究对照' in r for r in d.reasons)
