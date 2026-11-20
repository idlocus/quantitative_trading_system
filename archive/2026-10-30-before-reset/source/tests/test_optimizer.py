#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""cvxpy 优化器测试：约束满足、alpha倾斜、风格预算、回退路径"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import pytest

from portfolio.optimizer import solve_weights, shrink_cov, OptimizerConfig


def make_data(n=40, days=120, seed=7):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range('2024-01-01', periods=days)
    rets = pd.DataFrame(rng.normal(0.0005, 0.02, (days, n)), index=dates,
                        columns=[f'S{i:03d}' for i in range(n)])
    alpha = pd.Series(rng.normal(0, 1, n), index=rets.columns)
    style = pd.DataFrame({'size': rng.normal(0, 1, n),
                          'momentum': rng.normal(0, 1, n),
                          'volatility': rng.normal(0, 1, n)}, index=rets.columns)
    return rets, alpha, style


class TestOptimizer:

    def test_constraints_hold(self):
        rets, alpha, style = make_data()
        w = solve_weights(alpha, rets, style_exp=style, bench_style={'size': 0.0})
        assert abs(sum(w.values()) - 1) < 1e-3
        assert max(w.values()) <= 0.03 + 1e-4, "单票上限"
        assert min(w.values()) >= -1e-6

    def test_alpha_tilts_weights(self):
        """高 alpha 股应系统性获得更高权重（相对等权）"""
        rets, _, style = make_data()
        alpha = pd.Series(np.linspace(2, -2, rets.shape[1]), index=rets.columns)
        w = solve_weights(alpha, rets, config=OptimizerConfig(max_weight=0.05))
        half = rets.shape[1] // 2
        top = sum(list(w.values())[:half])
        bottom = sum(list(w.values())[half:])
        assert top > bottom, "高alpha半区权重应更大"

    def test_style_budget_respected(self):
        """风格暴露偏离锚点不超预算"""
        rets, alpha, style = make_data()
        w = solve_weights(alpha, rets, style_exp=style,
                          bench_style={'size': 0.0, 'momentum': 0.0},
                          config=OptimizerConfig(style_budget=0.1))
        size_exp = (style['size'] * pd.Series(w)).sum()
        assert abs(size_exp) <= 0.1 + 0.05, f"size暴露 {size_exp:.3f} 超预算"

    def test_equal_weight_fallback(self):
        """候选过少回退等权"""
        rets, _, _ = make_data(n=3)
        alpha = pd.Series([1.0, 2.0, 3.0], index=rets.columns)
        w = solve_weights(alpha, rets)
        assert len(w) == 3
        assert abs(sum(w.values()) - min(1.0, 0.03 * 3)) < 1e-6 or abs(sum(w.values()) - 1) < 1e-6

    def test_shrink_cov_positive(self):
        rets, _, _ = make_data()
        S = shrink_cov(rets, 0.1)
        assert (np.diag(S) > 0).all()
        assert np.allclose(S, S.T)


