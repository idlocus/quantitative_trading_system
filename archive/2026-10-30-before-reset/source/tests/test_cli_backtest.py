#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""run_portfolio_backtest CLI 测试：--all-equal 生效、报告文件名不因分支崩溃"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import inspect

import pandas as pd

import run_portfolio_backtest as cli
from portfolio.tradable_benchmark import (all_equal_scores,
                                          tradable_equal_weight)


class TestAllEqual:

    def test_flag_is_actually_used(self):
        """--all-equal 不得是死参数（此前定义后零引用，传了仍按因子选股）"""
        src = inspect.getsource(cli.main)
        assert 'if args.all_equal:' in src, "args.all_equal 应在主流程有分支处理"
        assert 'all_equal_scores' in src                    # 共享helper（同tradable_benchmark）
        # 对照组必须把 top 提到全池（否则仍按 top 截断=仍在选股）
        assert 'args.top = max(args.top' in src
        # 分支须在因子合成之前（对照组跳过因子计算）
        assert src.index('if args.all_equal:') < src.index('elif args.fundamental:')

    def test_all_equal_scores_shared_helper(self):
        dates = pd.bdate_range('2026-06-01', periods=5)
        mx = all_equal_scores(dates, ['A.SH', 'B.SZ', '000300.SH'])
        assert list(mx.columns) == ['A.SH', 'B.SZ']        # 基准排除
        assert mx.shape == (5, 2) and (mx.values == 1.0).all()


class TestTradableEqualWeightControl:
    """可交易等权对照：与统计参照不同，必须发生真实成本与再平衡"""

    @staticmethod
    def _cache(days=70):
        dates = pd.bdate_range('2026-06-01', periods=days)
        cache = {}
        for i, sym in enumerate(['600001.SH', '000001.SZ', '300001.SZ']):
            full = [10.0 + i] * days
            cache[sym] = pd.DataFrame({'date': dates, 'open': full, 'high': full,
                                       'low': full, 'close': full,
                                       'volume': [100e4] * days})
        bench = [4000.0] * days
        cache['000300.SH'] = pd.DataFrame({'date': dates, 'open': bench, 'high': bench,
                                           'low': bench, 'close': bench,
                                           'volume': [1e9] * days})
        return cache

    def test_control_trades_with_costs(self):
        """可交易对照必须产生真实成本（统计参照成本恒为0）；
        大池时资金按 ceil(N/100) 等比放大（否则单票分配<最低申报无法建仓）"""
        from portfolio.backtester import PortfolioConfig
        cfg = PortfolioConfig(top_n=2, rebalance_days=10,
                              min_daily_turnover=0, min_listing_days=0,
                              buffer_band=0.20)
        bt, res, scale = tradable_equal_weight(self._cache(), cfg,
                                               start_date='2026-06-01')
        assert scale == 1                                  # 3只池不放大
        assert res.total_cost > 0, "对照走了真实撮合必须有成本"
        assert res.config.top_n == 3                       # top 被提升到全池（3只非基准）
        assert res.n_rebalances >= 1
        assert '等权统计参照' in res.benchmark_suite

    def test_capital_scaling_for_large_pool(self):
        """大池资金放大：放大后单票分配必须可满足最低申报（主板100股）"""
        dates = pd.bdate_range('2026-06-01', periods=30)
        cache = {}
        # 250只10元股：不放大时单票4000元(1M/250)可买100股——构造更小分配场景用150只高价股
        for i in range(150):
            full = [100.0] * 30                             # 高价股：100股=1万元
            sym = f'6{i:05d}.SH'
            cache[sym] = pd.DataFrame({'date': dates, 'open': full, 'high': full,
                                       'low': full, 'close': full,
                                       'volume': [100e4] * 30})
        bench = [4000.0] * 30
        cache['000300.SH'] = pd.DataFrame({'date': dates, 'open': bench, 'high': bench,
                                           'low': bench, 'close': bench,
                                           'volume': [1e9] * 30})
        from portfolio.backtester import PortfolioConfig
        cfg = PortfolioConfig(top_n=10, rebalance_days=10,
                              min_daily_turnover=0, min_listing_days=0)
        bt, res, scale = tradable_equal_weight(cache, cfg, start_date='2026-06-01')
        assert scale == 2                                  # ceil(150/100)
        # 放大后资金2M/150只≈1.33万≥100股×100元 → 必须真实建仓
        assert res.total_cost > 0
        assert len(res.trade_log) >= 100                   # 大规模建仓发生

    def test_statistical_suite_renamed(self):
        """suite 键必须标注'统计参照'（不可交易），不再叫'等权对照'冒充分析基准"""
        from portfolio.backtester import PortfolioBacktester, PortfolioConfig
        cfg = PortfolioConfig(top_n=1, rebalance_days=10,
                              min_daily_turnover=0, min_listing_days=0)
        mx = all_equal_scores(pd.bdate_range('2026-06-01', periods=5),
                              ['600001.SH', '000001.SZ'])
        mx.loc[:, :] = 1.0
        mx.loc[:, '600001.SH'] = 2.0            # 让策略侧有排序
        res = PortfolioBacktester(self._cache(), cfg).run(mx)
        assert '等权统计参照' in res.benchmark_suite
        assert '等权对照' not in res.benchmark_suite


class TestReportFilename:

    def test_factor_names_defined_in_every_branch(self):
        """报告文件名依赖 factor_names——三个分支（all-equal/基本面/动量）都必须定义"""
        src = inspect.getsource(cli.main)
        assert 'factor_names = [\'全池等权\']' in src      # all-equal 分支
        assert 'factor_names = fnames' in src              # 基本面分支（此前缺失→NameError）
        assert 'factor_names = [f.strip()' in src          # 动量分支
        # 使用点在定义之后（分支内定义先于 save_report 段）
        assert src.index("factor_names = fnames") < src.index("'+'.join(factor_names)")
