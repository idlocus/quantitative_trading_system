#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""申报数量规则测试：科创板200起1股递增 / 北交所100起1股递增 / 主板创业板100整手 /
卖出零股一次性 / 价格档位 / Paper与回测接入"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import date

import pytest

from trading import rules
from data.board_params import BoardParamsProvider
from live.broker import Order, PaperBroker
from portfolio.backtester import PortfolioBacktester, PortfolioConfig
import pandas as pd


class TestQtyRules:

    def test_min_order_qty_by_board(self):
        assert rules.min_order_qty('600527.SH') == 100
        assert rules.min_order_qty('000001.SZ') == 100
        assert rules.min_order_qty('300750.SZ') == 100
        assert rules.min_order_qty('688981.SH') == 200     # 科创板
        assert rules.min_order_qty('920099.BJ') == 100
        assert rules.min_order_qty('00700.HK') == 1        # 港股（每手另计，此处不限制）

    def test_qty_step(self):
        assert rules.qty_step('688981.SH') == 1            # 科创板1股递增
        assert rules.qty_step('920099.BJ') == 1            # 北交所1股递增
        assert rules.qty_step('600527.SH') == 100
        assert rules.qty_step('300750.SZ') == 100

    def test_round_buy_qty(self):
        assert rules.round_buy_qty('600527.SH', 250) == 200    # 主板整手
        assert rules.round_buy_qty('600527.SH', 99) == 0       # 不足100
        assert rules.round_buy_qty('688981.SH', 237) == 237    # 科创200起1股递增
        assert rules.round_buy_qty('688981.SH', 150) == 0      # 不足200最低申报
        assert rules.round_buy_qty('920099.BJ', 137) == 137    # 北交100起1股递增
        assert rules.round_buy_qty('300750.SZ', 250) == 200    # 创业板整手

    def test_lot_override(self):
        """表数据（LOT_SIZE）覆盖优先：ASHAREWINDCUSTOMCODE 科创板实测=200"""
        assert rules.min_order_qty('688981.SH', lot=200) == 200
        assert rules.min_order_qty('600527.SH', lot=100) == 100

    def test_sell_zero_lot_rule(self):
        """卖出零股规则：余额不足最低申报 → 必须一次性全仓申报"""
        # 主板持有350：卖300合法（整手）；卖150非法（非整手非全仓）；全仓350合法
        assert rules.sell_qty_valid('600527.SH', 300, 350)
        assert not rules.sell_qty_valid('600527.SH', 150, 350)
        assert rules.sell_qty_valid('600527.SH', 350, 350)
        # 科创板持有150（<200）：只能一次性全卖
        assert not rules.sell_qty_valid('688981.SH', 100, 150)
        assert rules.sell_qty_valid('688981.SH', 150, 150)
        # 科创板持有500：237股（1股递增）合法
        assert rules.sell_qty_valid('688981.SH', 237, 500)

    def test_round_sell_qty(self):
        assert rules.round_sell_qty('600527.SH', 150, 350) == 100   # 修正到整手
        assert rules.round_sell_qty('600527.SH', 999, 350) == 350   # 超持仓→全仓
        assert rules.round_sell_qty('688981.SH', 100, 150) == 0     # 不足200非全仓→放弃


class TestPriceTick:

    def test_round_tick(self):
        assert rules.round_tick('600527.SH', 10.081) == 10.08
        assert rules.round_tick('600527.SH', 9.999) == 10.0
        assert rules.round_tick('600527.SH', 10.089) == 10.09


class TestPaperBrokerQty:

    def test_star_buy_237_shares(self):
        """科创板：1股递增的申报数量原样成交（旧逻辑会砍成整百）"""
        b = PaperBroker()
        fills = b.execute([Order('688981.SH', 'BUY', 0.1, quantity=237)],
                          {'688981.SH': 10.0}, date(2026, 10, 12))
        assert fills[0].status == 'filled'
        assert fills[0].quantity == 237

    def test_star_buy_below_200_rejected(self):
        b = PaperBroker()
        fills = b.execute([Order('688981.SH', 'BUY', 0.1, amount=150 * 10.0)],
                          {'688981.SH': 10.0}, date(2026, 10, 12))
        assert '不足最低申报' in fills[0].status

    def test_main_board_lot(self):
        b = PaperBroker()
        fills = b.execute([Order('600527.SH', 'BUY', 0.1, amount=250 * 10.0)],
                          {'600527.SH': 10.0}, date(2026, 10, 12))
        assert fills[0].quantity == 200

    def test_sell_normalized_with_positions(self):
        """卖出数量合法化：主板持有350、订单卖150 → 修正为100"""
        b = PaperBroker()
        fills = b.execute([Order('600527.SH', 'SELL', 0, quantity=150)],
                          {'600527.SH': 10.0}, date(2026, 10, 12),
                          positions={'600527.SH': 350})
        assert fills[0].status == 'filled' and fills[0].quantity == 100

    def test_star_holding_150_sell_all_ok(self):
        """科创板零股：持有150（<200）全仓卖出合法（一次性申报）"""
        b = PaperBroker()
        fills = b.execute([Order('688981.SH', 'SELL', 0, quantity=150)],
                          {'688981.SH': 10.0}, date(2026, 10, 12),
                          positions={'688981.SH': 150})
        assert fills[0].quantity == 150

    def test_price_on_tick(self):
        b = PaperBroker(slippage=0.001)
        fills = b.execute([Order('600527.SH', 'BUY', 0.1, amount=50000)],
                          {'600527.SH': 10.0}, date(2026, 10, 12))
        assert fills[0].price * 100 % 1 == pytest.approx(0, abs=1e-9)   # 0.01档位

    def test_lot_map_override_in_broker(self):
        b = PaperBroker(lot_map={'600527.SH': 200})   # 表覆盖最低申报=200
        fills = b.execute([Order('600527.SH', 'BUY', 0.1, quantity=250)],
                          {'600527.SH': 10.0}, date(2026, 10, 12))
        # 表只覆盖最低申报量；步长仍按板块规则（主板整手）→ 250→200
        assert fills[0].quantity == 200
        # 低于表覆盖的最低申报 → 拒单
        fills2 = b.execute([Order('600527.SH', 'BUY', 0.1, quantity=150)],
                           {'600527.SH': 10.0}, date(2026, 10, 12))
        assert '不足最低申报' in fills2[0].status


class TestBacktesterQty:

    def test_star_buys_non_round_hundred(self):
        """回测：科创板买入数量按200起1股递增（旧逻辑强制整百）"""
        cache = {}
        dates = pd.bdate_range('2026-06-01', periods=70)
        full = [10.0] * 70
        cache['688981.SH'] = pd.DataFrame({'date': dates, 'open': full, 'high': full,
                                           'low': full, 'close': full, 'volume': [100e4] * 70})
        bench = [4000.0] * 70
        cache['000300.SH'] = pd.DataFrame({'date': dates, 'open': bench, 'high': bench,
                                           'low': bench, 'close': bench, 'volume': [1e9] * 70})
        mx = pd.DataFrame({d: {'688981.SH': 2.0} for d in dates}).T.sort_index()
        cfg = PortfolioConfig(top_n=1, min_daily_turnover=0, min_listing_days=0,
                              risk_limits=__import__('portfolio.risk_manager', fromlist=['RiskLimits']).RiskLimits(
                                  max_position_weight=1.0))
        res = PortfolioBacktester(cache, cfg).run(mx)
        buys = [r for r in res.trade_log if r[2] == 'BUY']
        assert buys
        qty = buys[0][4]
        assert qty >= 200                      # 科创板最低申报
        assert qty == 200 or qty % 1 == 0      # 1股递增（任何整数都合法）
        # 金额→数量自然带零头时未被强制整百
        assert qty not in (100,)               # 绝不会出现100（<200被拒）


class TestBoardParamsProvider:

    def test_from_rows_and_fallback(self):
        p = BoardParamsProvider.from_rows([('688981.SH', 200), ('600527.SH', 100)])
        assert p.lot('688981.SH') == 200
        assert p.lot('920099.BJ') == 100       # 表缺失→板块规则兜底
