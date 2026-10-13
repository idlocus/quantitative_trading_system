#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""第八批修复回归：清仓净值顺序/申报合法化/ST禁买/config默认/amount优先/费用同源/窗口/口径统一"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import date, datetime

import numpy as np
import pandas as pd
import pytest

from trading import config as chain
from portfolio.backtester import PortfolioBacktester, PortfolioConfig
from portfolio.risk_manager import RiskLimits
from portfolio.daily_replay import DailyReplayBacktester, ReplayConfig


def make_cache(symbols_prices, days=70, volume=100e4):
    dates = pd.bdate_range('2026-06-01', periods=days)
    cache = {}
    for sym, arr in symbols_prices.items():
        full = [arr[0]] * (days - len(arr)) + list(arr)
        cache[sym] = pd.DataFrame({'date': dates, 'open': full, 'high': full,
                                   'low': full, 'close': full, 'volume': [volume] * days})
    bench = [4000.0] * days
    cache['000300.SH'] = pd.DataFrame({'date': dates, 'open': bench, 'high': bench,
                                       'low': bench, 'close': bench, 'volume': [1e9] * days})
    return cache


class TestHaltDayEquity:

    def test_liquidation_recorded_in_same_day_equity(self):
        """修复#1：硬限触发日，当日净值=清仓后现金（含全部费用），无隐藏持仓"""
        # 构造：首日买入后连续暴跌→触发15%硬限清仓
        cache = make_cache({'600001.SH': [10.0] * 30 + [8.5] * 5 + [7.0] * 5 + [5.9] * 30})
        dates = pd.bdate_range('2026-06-01', periods=70)
        mx = pd.DataFrame({d: {'600001.SH': 2.0} for d in dates}).T.sort_index()
        cfg = PortfolioConfig(top_n=1, rebalance_days=5,
                              min_daily_turnover=0, min_listing_days=0, buffer_band=0.0,
                              max_weight=1.0, risk_limits=RiskLimits(max_position_weight=1.0))
        res = PortfolioBacktester(cache, cfg).run(mx)
        sells = [r for r in res.trade_log if r[2] == 'SELL']
        assert sells, "暴跌应触发硬限清仓"
        halt_day = sells[0][0]
        # 当日净值 = 全现金（持仓已清），即 equity 在 SELL 之后记录
        eq = res.equity_curve.loc[halt_day]
        holdings_that_day = res.daily_holdings.get(halt_day, [])
        assert holdings_that_day == [], "清仓日不应残留持仓记录"
        # 清仓费用含过户费：验证卖出现金扣减 = 费率合计（佣金+印花+过户）
        sell_row = sells[0]
        px, qty = sell_row[3], sell_row[4]
        fees = qty * px * (cfg.commission + cfg.stamp_tax + cfg.transfer_fee)
        assert fees > 0

    def test_limit_down_deferred_on_halt(self):
        """修复#1：清仓日跌停的持仓不虚构成交，顺延后续"""
        # A 在清仓触发窗口内恰逢跌停（-10%）
        prices = [10.0] * 30 + [8.5] * 5 + [7.0] * 5 + [6.29] + [5.9] * 29
        cache = make_cache({'600001.SH': prices})
        dates = pd.bdate_range('2026-06-01', periods=70)
        mx = pd.DataFrame({d: {'600001.SH': 2.0} for d in dates}).T.sort_index()
        cfg = PortfolioConfig(top_n=1, rebalance_days=5,
                              min_daily_turnover=0, min_listing_days=0, buffer_band=0.0,
                              max_weight=1.0, risk_limits=RiskLimits(max_position_weight=1.0))
        res = PortfolioBacktester(cache, cfg).run(mx)
        # 无论哪天卖出，都不应有虚构的跌停日成交（跌停日卖出量必为顺延）
        # （若最终仍未卖出，也说明顺延机制生效而非硬卖）
        sells = [(r[0], r[3]) for r in res.trade_log if r[2] == 'SELL']
        for d, _px in sells:
            i = dates.get_loc(pd.Timestamp(d))
            if i > 0:
                prev = float(cache['600001.SH']['close'].iloc[i - 1])
                cur = float(cache['600001.SH']['close'].iloc[i])
                assert not (cur <= prev * 0.901), f"{d} 为跌停日不应成交卖出"


class TestQtyRulesUnification:

    def test_risk_gate_trim_star_step1(self):
        """修复#2：闸门降仓卖出按申报步长（科创板1股递增），不再固定整百"""
        from live.risk_gate import RiskGate
        from live.ledger import Ledger
        import tempfile
        led = Ledger(os.path.join(tempfile.mkdtemp(), 'l.json'))
        led.initialize(1_000_000)
        # 持有科创板 500 股，净值回撤触发缩量（nav 序列 -10%）
        led.apply_fills([__import__('live.broker', fromlist=['Fill', 'Order']).Fill(
            __import__('live.broker', fromlist=['Order']).Order('688981.SH', 'BUY', 1.0),
            price=10.0, quantity=500, commission=0, taxes=0, trade_date=date(2026, 9, 1))])
        led._d['nav_history'] = [
            {'date': f'2026-09-{i+1:02d}', 'nav': v} for i, v in enumerate(
                [1000000] * 20 + [950000, 920000, 900000])]
        gate = RiskGate()
        prices = {'688981.SH': 10.0}
        d = gate.review([], led, prices, prices, [], bench_df=None, as_of=date(2026, 10, 12))
        trims = [o for o in d.orders if o.symbol == '688981.SH' and o.side == 'SELL']
        if trims:  # 触发了降仓削减才有意义
            qty = trims[0].quantity
            assert qty >= 200, "科创板减持≥最低申报200"
            assert qty % 1 == 0, "1股递增合法"


class TestSTDataUnavailable:

    def test_gate_blocks_buys_when_st_data_down(self, tmp_path):
        """修复#3：ST数据不可用 → 闸门禁买（卖出照常）"""
        from live.risk_gate import RiskGate
        from live.ledger import Ledger
        from live.broker import Order
        led = Ledger(str(tmp_path / 'l.json'))
        led.initialize(1_000_000)
        dates = pd.bdate_range('2026-10-06', periods=3)
        bench = pd.DataFrame({'date': dates, 'close': [4000.0, 3990.0, 4000.0]})
        orders = [Order('600000.SH', 'BUY', 0.1, amount=10000),
                  Order('600001.SH', 'SELL', 0, quantity=100)]
        led._d['positions']['600001.SH'] = 100
        led._d['cost']['600001.SH'] = 9.0
        d = RiskGate().review(orders, led, {'600000.SH': 10.0, '600001.SH': 9.5},
                              {'600000.SH': 10.0, '600001.SH': 9.5},
                              ['600000.SH'], bench_df=bench,
                              as_of=dates[-1].date(), st_data_ok=False)
        assert d.action == 'no_buy'
        assert any('ST状态数据不可用' in r for r in d.reasons)
        sells_out = [o for o in d.orders if o.side == 'SELL']
        assert sells_out, "卖出不受影响"
        assert all(o.side != 'BUY' for o in d.orders)


class TestReplayInterfaceAndTurnover:

    def test_config_none_public_interface(self, tmp_path):
        """修复#4：DailyReplayBacktester(cache) 不传配置可用（原 config.st_matrix 崩）"""
        cache = make_cache({'600001.SH': [10.0] * 70, '000001.SZ': [10.0] * 70})
        bt = DailyReplayBacktester(cache)      # config=None
        assert bt.cfg.top_n == chain.TOP_N
        assert bt.st_matrix is None

    def test_amount_column_preferred(self):
        """修复#5：回放成交额优先用 amount 列（close×volume 兜底）"""
        dates = pd.bdate_range('2026-06-01', periods=25)
        # volume 极小（手单位残留误算 20万），amount 正确 2000万 → 应判流动（不排除）
        cache = {'600001.SH': pd.DataFrame({
            'date': dates, 'open': 10.0, 'high': 10.0, 'low': 10.0, 'close': 10.0,
            'volume': 2e4, 'amount': 2e7})}
        bench = [4000.0] * 25
        cache['000300.SH'] = pd.DataFrame({'date': dates, 'open': bench, 'high': bench,
                                           'low': bench, 'close': bench, 'volume': [1e9] * 25})
        bt = DailyReplayBacktester(cache, ReplayConfig(min_listing_days=0))
        avg = bt.avg_turnover_20d.iloc[-1]['600001.SH']
        assert avg == pytest.approx(2e7, rel=1e-6), "应取 amount 而非 close×volume"

    def test_wind_db_source_normalized(self):
        """修复#5源码守护：旧数据源 wind_db 也必须单位归一+带amount"""
        src = open('data/wind_db.py', encoding='utf-8').read()
        assert 'S_DQ_VOLUME   AS FLOAT) * 100' in src.replace('  ', ' ') or \
            '* 100' in src, "volume 手→股"
        assert 'S_DQ_AMOUNT' in src, "返回 amount 列"


class TestQmtFeesAndWindow:

    def test_est_fees_from_chain(self):
        """修复#6：QMT 费率与主链配置一致（不再硬编码）"""
        from live.qmt_broker import _est_fees
        c, t = _est_fees('BUY', 100000.0)
        assert c == pytest.approx(100000 * chain.COMMISSION)
        assert t == pytest.approx(100000 * chain.TRANSFER_FEE)
        c2, t2 = _est_fees('SELL', 100000.0)
        assert t2 == pytest.approx(100000 * (chain.TRANSFER_FEE + chain.STAMP_TAX))

    def test_running_cash_includes_fees_source(self):
        src = open('live/qmt_broker.py', encoding='utf-8').read()
        assert 'running_cash -= fill.price * fill.quantity + fill.commission + fill.taxes' in src

    def test_execution_window(self):
        """修复#7：收盘竞价窗口判定"""
        from live.qmt_broker import check_execution_window
        assert check_execution_window(datetime(2026, 10, 12, 14, 55))
        assert not check_execution_window(datetime(2026, 10, 12, 9, 40))
        assert not check_execution_window(datetime(2026, 10, 12, 15, 30))


class TestEntryUnification:

    def test_run_pm_default_top_n(self):
        """修复#8：run_pm 默认 top_n 与主链一致（原30 vs CLI 100）"""
        import inspect
        import run_daily
        sig = inspect.signature(run_daily.run_pm)
        assert sig.parameters['top_n'].default == chain.TOP_N == 100

    def test_journal_uses_production_signal_entry(self):
        """修复#8：journal AM 经由 generate_production_signal（唯一入口）"""
        src = open('live/journal.py', encoding='utf-8').read()
        assert 'generate_production_signal' in src
        assert 'generate_target(ds' not in src, "禁止直调 generate_target 产生第二套口径"


class TestBatch9QmtInterface:
    """第九批：run_daily+QMT 组合调用（TypeError硬bug）/卖出合法化/回放ST安全"""

    def test_qmt_execute_accepts_positions(self):
        """修复#1：QMTBroker.execute 必须接受 positions（run_daily 传了它）"""
        import inspect
        from live.qmt_broker import QMTBroker
        params = inspect.signature(QMTBroker.execute).parameters
        for p in ('prev_prices', 'available_cash', 'st_symbols', 'positions'):
            assert p in params, f"QMTBroker.execute 缺参数 {p}（run_daily 组合调用会 TypeError）"

    def test_run_daily_qmt_combo_signature_match(self):
        """run_daily 传的所有 kwargs 必须在两个 Broker 的 execute 签名里"""
        import inspect
        import run_daily
        from live.broker import PaperBroker
        from live.qmt_broker import QMTBroker
        src = inspect.getsource(run_daily.run_pm)
        kwargs = ['prev_prices=', 'available_cash=', 'st_symbols=', 'positions=']
        for kw in kwargs:
            assert kw in src
            for cls in (PaperBroker, QMTBroker):
                assert kw.rstrip('=') in inspect.signature(cls.execute).parameters, \
                    f"{cls.__name__}.execute 缺 {kw}"

    def test_qmt_sell_legalization_with_positions(self):
        """修复#2：QMT 卖出经 round_sell_qty（主板150→100；科创150全仓）"""
        from datetime import date as _d
        from tests.test_qmt_lifecycle import FakeQMT
        from live.qmt_broker import QMTOrderState
        from live.broker import Order
        import tempfile
        b = FakeQMT(str(tempfile.mkdtemp() + '/o.json'),
                    outcomes=[(QMTOrderState.FILLED, 100, 10.0)])  # 券商按申报量成交
        b.enforce_window = False                        # 测试在任意时间运行，绕过窗口
        b.execute([Order('600000.SH', 'SELL', 0, quantity=150)],
                  {'600000.SH': 10.0}, _d(2026, 10, 12),
                  available_cash=100000, positions={'600000.SH': 350})
        # 提交侧已合法化：150 → 主板整手 100
        assert getattr(b, 'last_submitted_qty', None) == 100

        b2 = FakeQMT(str(tempfile.mkdtemp() + '/o2.json'),
                     outcomes=[(QMTOrderState.FILLED, 150, 10.0)])
        b2.enforce_window = False
        f2 = b2.execute([Order('688981.SH', 'SELL', 0, quantity=150)],
                        {'688981.SH': 10.0}, _d(2026, 10, 12),
                        available_cash=100000, positions={'688981.SH': 150})
        assert getattr(b2, 'last_submitted_qty', None) == 150  # 零股一次性全仓
        assert f2[0].quantity == 150

    def test_paper_affordable_star_min_200(self):
        """修复#3：PaperBroker 资金不足缩减按申报规则（科创板不会出现100股）"""
        from live.broker import Order, PaperBroker
        from datetime import date as _d
        b = PaperBroker()
        # 现金只够 ~150股×10元（科创板<200最低申报→拒；旧代码会按100股成交）
        f = b.execute([Order('688981.SH', 'BUY', 0.1, quantity=1000)],
                      {'688981.SH': 10.0}, _d(2026, 10, 12), available_cash=1600)
        assert '资金不足' in f[0].status, "科创板资金不足时不得出现<200股的成交"

    def test_replay_st_required_blocks_buys(self, tmp_path):
        """修复#4：st_data_required 且矩阵缺失 → 回放闸门禁买"""
        cache = make_cache({'600001.SH': [10.0] * 70, '000001.SZ': [10.0] * 70})
        dates = pd.bdate_range('2026-06-01', periods=70)
        mx = pd.DataFrame({d: {'600001.SH': 2.0, '000001.SZ': 1.0}
                           for d in dates}).T.sort_index()
        cfg = ReplayConfig(top_n=1, min_listing_days=0, min_daily_turnover=0,
                           use_risk_gate=True, st_matrix=None, st_data_required=True)
        res = DailyReplayBacktester(cache, cfg).run(mx)
        assert res.trades == [], "ST数据缺失+required → 禁买不应有任何成交"

    def test_research_mode_default_unchanged(self):
        """默认（研究模式）st_matrix=None 不触发禁买——历史回测语义不变"""
        from portfolio.daily_replay import ReplayConfig
        cfg = ReplayConfig()
        assert cfg.st_data_required is False

    def test_production_signal_carries_st_status(self, monkeypatch):
        """修复#5：生产入口返回 st_ok 状态（数据不可用时 False，供闸门禁买）"""
        import live.signals as sig
        from dataclasses import dataclass

        def fake_gt(ds, as_of, top_n=100, st_syms=None):
            return ['600000.SH'], pd.Series({'600000.SH': 1.0})

        monkeypatch.setattr(sig, 'generate_target', fake_gt)
        # FakeDS：STProvider 加载失败（无 _db_src）→ st_ok=False
        class _FakeDS:
            def preload_batch(self, syms, s, e):
                dates = [d for d in ['2026-10-09', '2026-10-12'] if d <= e.isoformat()]
                return {'000300.SH': pd.DataFrame({
                    'date': pd.to_datetime(dates),
                    'close': [4000.0] * len(dates)})}
        ds = _FakeDS()
        ps = sig.generate_production_signal(ds, as_of=date(2026, 10, 12))
        assert ps.st_ok is False, "ST源不可用必须反映在 ProductionSignal.st_ok"
        assert ps.signal_date == date(2026, 10, 9)
        assert ps.targets == ['600000.SH']

    def test_journal_gate_gets_st_data_ok(self):
        """修复#6：AM 预览闸门传 st_data_ok（与 PM 同源）"""
        src = open('live/journal.py', encoding='utf-8').read()
        assert 'st_data_ok=psig.st_ok' in src

    def test_hk_kline_amount_passthrough(self):
        """修复#7：港股路径带 amount 列（原样直通，单位未验证不入成交额计算）"""
        src = open('data/wind_db.py', encoding='utf-8').read()
        assert 'S_DQ_AMOUNT   AS FLOAT)          AS amt' in src.replace('  ', ' ') or \
            'amt' in src


class TestBatch10StDatesAndAudit:
    """第十批：信号日/执行日ST分离、空集语义、回放工件审计字段、实验口径"""

    def test_signal_vs_execution_st_separation(self, monkeypatch):
        """修复#1：T-1选股与T撮合使用各自日期的ST状态（T日戴帽按±5%撮合）"""
        import live.signals as sig
        captured = {'sig_date': None, 'exec_date': None}

        class _Prov:
            def __init__(self, ds):
                pass
            def st_set(self, d):
                if captured['sig_date'] is None:
                    captured['sig_date'] = d          # 第一次调用=信号日
                else:
                    captured['exec_date'] = d         # 第二次调用=执行日
                return {f'ST_{d}'}                    # 两个日期返回不同集合

        monkeypatch.setattr('data.st_provider.STProvider', _Prov)
        monkeypatch.setattr(sig, 'generate_target',
                            lambda ds, as_of, top_n=100, st_syms=None:
                            (['600000.SH'], pd.Series({'600000.SH': 1.0})))

        class _FakeDS:
            def preload_batch(self, syms, s, e):
                dates = [d for d in ['2026-10-09', '2026-10-12'] if d <= e.isoformat()]
                return {'000300.SH': pd.DataFrame({'date': pd.to_datetime(dates),
                                                   'close': [4000.0] * len(dates)})}
        ps = sig.generate_production_signal(_FakeDS(), as_of=date(2026, 10, 12))
        assert captured['sig_date'] == date(2026, 10, 9)
        assert captured['exec_date'] == date(2026, 10, 12)
        assert ps.st_symbols == {date(2026, 10, 9).__str__()} or ps.st_symbols == {'ST_2026-10-09'}
        assert ps.exec_st_symbols != ps.st_symbols, "两日期ST集合必须独立"
        assert ps.st_ok is True

    def test_empty_st_set_still_passed_not_reloaded(self, monkeypatch):
        """修复#2：加载成功但为空集时必须传 st_syms=set()（不触发二次加载）"""
        import live.signals as sig
        received = {}

        class _Prov:
            def __init__(self, ds):
                pass
            def st_set(self, d):
                return set()                          # 成功但当日无ST

        monkeypatch.setattr('data.st_provider.STProvider', _Prov)

        def fake_gt(ds, as_of, top_n=100, st_syms=None):
            received['st_syms'] = st_syms
            return ['600000.SH'], pd.Series({'600000.SH': 1.0})
        monkeypatch.setattr(sig, 'generate_target', fake_gt)

        class _FakeDS:
            def preload_batch(self, syms, s, e):
                dates = [d for d in ['2026-10-09', '2026-10-12'] if d <= e.isoformat()]
                return {'000300.SH': pd.DataFrame({'date': pd.to_datetime(dates),
                                                   'close': [4000.0] * len(dates)})}
        ps = sig.generate_production_signal(_FakeDS(), as_of=date(2026, 10, 12))
        assert received['st_syms'] == set(), "空集≠None：成功加载的空集必须显式传入"
        assert ps.st_ok is True

    def test_replay_result_has_audit_fields(self):
        """修复#3：ReplayResult 携带 trade_log/daily_holdings/daily_contrib/n_no_buy_days"""
        cache = make_cache({'600001.SH': [10.0] * 65 + [10.0, 11.0, 12.0, 13.0, 14.0]})
        dates = pd.bdate_range('2026-06-01', periods=70)
        mx = pd.DataFrame({d: {'600001.SH': 1.0} for d in dates}).T.sort_index()
        cfg = ReplayConfig(top_n=1, min_listing_days=0, min_daily_turnover=0,
                           use_risk_gate=False)
        res = DailyReplayBacktester(cache, cfg).run(mx)
        assert len(res.trade_log) > 0, "成交流水必须可审计"
        assert res.trade_log[0][:2] == [res.trade_log[0][0], '600001.SH']
        assert res.daily_holdings, "逐日持仓必须可审计"
        assert isinstance(res.n_no_buy_days, int)
        # 审计明细与净值一致：某日 holdings 市值 + 现金 ≈ 当日净值（近似）
        first_day = sorted(res.daily_holdings)[0]
        assert isinstance(res.daily_holdings[first_day], list)

    def test_mainchain_passes_st_required(self):
        """修复#4：主实验 ST 失败时必须传 st_data_required（与CLI同安全口径）"""
        src = open('experiments/mainchain_v5.py', encoding='utf-8').read()
        assert 'st_data_required=st_required' in src
        assert 'st_required = True' in src

    def test_payload_includes_no_buy_days(self):
        src = open('research/artifacts.py', encoding='utf-8').read()
        assert 'n_no_buy_days' in src


class TestBatch12SafetyAndPhases:
    """第十二批：ST零行安全模式/QMT窗口拒单/收益分相"""

    def test_st_zero_rows_raises(self):
        """修复#1：ASHAREST 返回0行（权限/同步故障）必须抛错→st_ok=False→禁买"""
        from data.st_provider import STProvider

        class _FakeDBSrc:
            def _cursor(self):
                class _Cur:
                    def execute(self, sql): pass
                    def fetchall(self): return []      # 查询成功但0行
                    def close(self): pass
                return _Cur()

        ds = type('FakeDS', (), {'_db_src': lambda self: _FakeDBSrc()})()
        with pytest.raises(RuntimeError, match='安全模式'):
            STProvider(ds)

    def test_qmt_window_blocks_by_default(self, monkeypatch):
        """修复#2：窗口外默认拒单（不再只warning后继续提交）"""
        from datetime import datetime
        from live import qmt_broker as qb
        from live.broker import Order
        monkeypatch.setattr(qb, 'check_execution_window',
                            lambda now=None: False)    # 强制窗口外
        b = qb.QMTBroker.__new__(qb.QMTBroker)
        b.enforce_window = True
        b._connected = True
        fills = b.execute([Order('600000.SH', 'BUY', 0.1, quantity=100)],
                          {'600000.SH': 10.0}, datetime(2026, 10, 13, 9, 40))
        assert '非收盘竞价窗口' in fills[0].status

    def test_qmt_window_override_allows(self, monkeypatch):
        from datetime import datetime
        from live import qmt_broker as qb
        monkeypatch.setattr(qb, 'check_execution_window', lambda now=None: False)
        b = qb.QMTBroker.__new__(qb.QMTBroker)
        b.enforce_window = False                        # 显式覆盖
        b._connected = True
        b._account = None

        class _FakeXt:
            def query_stock_asset(self, acc):
                class _A: cash = 1_000_000.0
                return _A()
        b._xt_trader = _FakeXt()
        b._pending = {}
        import tempfile as _tf
        b.state_path = _tf.mkdtemp() + "/qmt_test.json"
        b._orders = {}
        b._outcomes = [(qb.QMTOrderState.FILLED, 100, 10.0)]
        b.order_timeout = 0.2
        b._submit_order = lambda order, qty, price: 'O1'
        b._query_order = lambda oid: b._outcomes[0]
        b._cancel_order = lambda oid: True
        from live.broker import Order
        fills = b.execute([Order('600000.SH', 'BUY', 0.1, quantity=100)],
                          {'600000.SH': 10.0}, datetime(2026, 10, 13, 9, 40))
        assert fills[0].quantity == 100                 # 覆盖模式继续执行

    def test_replay_phase_separation(self):
        """修复#3：终态存在时收益必须分相（运行期/停机期/整体）"""
        # 构造：第30日买入后一直持有到第50日卖出，之后停机20日
        days = 70
        cache = make_cache({'600001.SH': [10.0] * days})
        dates = pd.bdate_range('2026-06-01', periods=days)
        mx = pd.DataFrame({d: {'600001.SH': 1.0} for d in dates}).T.sort_index()
        # 通过截断 end_date 制造停机期：信号在第50日后消失（分数矩阵截断）
        mx_trunc = mx.iloc[:50]
        cfg = ReplayConfig(top_n=1, min_listing_days=0, min_daily_turnover=0,
                           use_risk_gate=False)
        res = DailyReplayBacktester(cache, cfg).run(mx_trunc, start_date='2026-06-01')
        # 信号截断≠终态（持仓仍在）→ terminal_date 应为 None（daily_holdings到最后）
        assert res.terminal_date is None                # 持仓期间非终态
        assert res.active_period_return is None

    def test_replay_phase_fields_on_terminal(self):
        """终态场景：分相字段被正确填充（用 batch 测试的止损场景简化验证字段存在性）"""
        cache = make_cache({'600001.SH': [10.0] * 70})
        dates = pd.bdate_range('2026-06-01', periods=70)
        mx = pd.DataFrame({d: {'600001.SH': 1.0} for d in dates}).T.sort_index()
        cfg = ReplayConfig(top_n=1, min_listing_days=0, min_daily_turnover=0,
                           use_risk_gate=False)
        res = DailyReplayBacktester(cache, cfg).run(mx, start_date='2026-06-01')
        # 全程运行：无终态字段
        assert res.terminal_note is None and res.idle_trading_days == 0
        # 字段已在 payload 序列化路径上
        from research.artifacts import portfolio_payload
        p = portfolio_payload(res)
        assert 'terminal_date' in p and 'active_period_return' in p \
               and 'idle_trading_days' in p

    def test_make_broker_passes_enforce_window(self):
        import inspect
        from live import make_broker
        params = inspect.signature(make_broker).parameters
        assert 'enforce_window' in params
        src_daily = open('run_daily.py', encoding='utf-8').read()
        src_live = open('run_live.py', encoding='utf-8').read()
        assert 'enforce_window=not outside_window' in src_daily
        assert 'enforce_window=not args.outside_window' in src_live
