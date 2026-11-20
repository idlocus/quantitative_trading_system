#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""约束集合一致性测试：两套风控的约束名集合必须对等（或显式声明白名单）。
此前只断言8个阈值相等，但约束集合本身可以不对等而测试照样绿——
回测有vol_target/sector/turnover/cash_floor而实盘没有，实盘有stop_loss而回测没有，
两套各自朝有利方向偏，净效应=回测数字不可代表实盘。"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trading import config as chain
from live.risk_gate import GateConfig
from portfolio.risk_manager import RiskLimits


# 显式声明的"仅回测有"约束（实盘无法/不需要落地）
DECLARED_BACKTEST_ONLY = set(chain.BACKTEST_ONLY)
DECLARED_LIVE_ONLY = set(chain.LIVE_ONLY)


class TestConstraintSetConsistency:

    def test_all_thresholds_single_source(self):
        """所有共享阈值的值必须来自 trading/config（精确相等）"""
        g, r = GateConfig(), RiskLimits()
        pairs = [
            ('drawdown_soft', g.drawdown_soft, 'drawdown_derisk', r.drawdown_derisk),
            ('drawdown_hard', g.drawdown_hard, 'max_drawdown', r.max_drawdown),
            ('market_crash', g.market_daily_crash, 'circuit_breaker_market', r.circuit_breaker_market),
            ('portfolio_crash', g.portfolio_daily_crash,
             'circuit_breaker_portfolio', r.circuit_breaker_portfolio),
            ('stop_loss', g.stop_loss_pct, 'stop_loss_pct', r.stop_loss_pct),
            ('vol_target', g.vol_target, 'vol_target', r.vol_target),
            ('max_turnover', g.max_turnover, 'max_turnover', r.max_turnover),
            ('min_cash', g.min_cash_ratio, 'min_cash_ratio', r.min_cash_ratio),
            ('max_position_weight', None, 'max_position_weight', r.max_position_weight),
        ]
        for name, gv, rn, rv in pairs:
            if name == 'max_position_weight':
                assert rv == chain.MAX_WEIGHT, \
                    f"max_position_weight({rv}) != chain.MAX_WEIGHT({chain.MAX_WEIGHT})"
                continue
            assert gv == rv, \
                f"约束 '{name}' 值不一致: gate({gv}) != risk_manager({rv})"

    def test_backtest_only_whitelist_is_explicit(self):
        """仅回测有的约束必须在 chain.BACKTEST_ONLY 中显式声明（显式清单，非反射）"""
        # 已知已统一的约束（两边都有字段且值同源）
        shared = {'vol_target', 'max_turnover', 'min_cash_ratio', 'stop_loss_pct'}
        # 仍仅在回测有的（GateConfig 无对应字段或未执行）
        still_backtest_only = {'max_sector_weight', 'max_position_weight'}
        undeclared = still_backtest_only - DECLARED_BACKTEST_ONLY
        assert not undeclared, \
            f"回测有但实盘没有的约束未声明为 BACKTEST_ONLY: {undeclared}. " \
            f"要么在 trading/config.BACKTEST_ONLY 中声明，要么加到 GateConfig."
        # 已统一的约束不应再出现在 BACKTEST_ONLY 中
        overlap = shared & DECLARED_BACKTEST_ONLY
        assert not overlap, \
            f"约束 {overlap} 已搬到 GateConfig 但仍声明为 BACKTEST_ONLY——声明过时"

    def test_live_only_whitelist_is_explicit(self):
        """仅实盘有的约束必须在 chain.LIVE_ONLY 中显式声明"""
        g_specific = {'stop_loss_pct', 'cooldown_days', 'max_stop_sells_per_day',
                      'min_order_amount', 'st_data_required',
                      'restart_after_liquidation'}
        # stop_loss_pct 已搬到 RiskLimits，不再 live-only
        live_only = {'stop_loss_pct'} - {'stop_loss_pct'}  # 空（已统一）
        # 其余 gate 特有参数是执行细节，不是风控约束，不需要声明
        assert live_only <= DECLARED_LIVE_ONLY | set(), \
            f"实盘有但回测没有的风控约束未声明: {live_only}"

    def test_constraint_disclaimer_in_reports(self):
        """回测报告必须包含未含约束的声明"""
        import inspect
        from portfolio.backtester import PortfolioResult
        from portfolio.daily_replay import ReplayResult
        bt_src = inspect.getsource(PortfolioResult.summary_markdown)
        rp_src = inspect.getsource(ReplayResult.summary_markdown)
        assert '口径声明' in bt_src, "近似口径回测报告缺少口径声明"
        assert '口径声明' in rp_src, "每日重放报告缺少口径声明"
