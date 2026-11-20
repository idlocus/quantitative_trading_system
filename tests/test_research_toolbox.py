#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""research/risk_model.py 与 research/robustness.py 行为测试（2026-11-04 补，
此前为测试覆盖缺口）。注意：不调用 register_experiment（会写真实登记册）。"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from research import risk_model, robustness


def _make_panel(n_syms=60, days=30, seed=7):
    """合成横截面面板：dates × symbols"""
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range('2026-01-05', periods=days)
    syms = [f'S{i:03d}' for i in range(n_syms)]
    return rng, pd.DataFrame(index=dates, columns=syms, dtype=float), syms


class TestNeutralize:
    def test_style_contamination_removed(self):
        """因子=纯风格暴露+噪声 → 中性化后残差与风格截面相关≈0"""
        rng, factor, syms = _make_panel()
        style = pd.DataFrame(
            rng.randn(*factor.shape), index=factor.index, columns=syms)
        noisy = style * 2.0 + pd.DataFrame(
            rng.randn(*factor.shape) * 0.1, index=factor.index, columns=syms)
        resid = risk_model.neutralize(noisy, {'style': style})
        # 取中间一天（首日 zscore 内部一致）验证相关被剥离
        day = resid.index[len(resid) // 2]
        corr_before = noisy.loc[day].corr(style.loc[day])
        corr_after = resid.loc[day].dropna().corr(style.loc[day].dropna())
        assert abs(corr_before) > 0.95, "合成数据本身应强相关（前提检查）"
        assert abs(corr_after) < 0.2, \
            f"中性化后残差与风格仍相关({corr_after:.2f})——中性化失效"

    def test_insufficient_cross_section_returns_nan(self):
        """横截面不足 max(30,(k+1)*8) 只时该日不出残差"""
        small = pd.DataFrame(
            np.random.RandomState(1).randn(5, 10),
            index=pd.bdate_range('2026-01-05', periods=5),
            columns=[f'S{i}' for i in range(10)], dtype=float)
        style = small.copy()
        resid = risk_model.neutralize(small, {'style': style})
        assert resid.isna().all().all(), "10只<30只下限应整日NaN"


class TestStyleExposures:
    def test_zscore_standardized(self):
        """暴露矩阵逐日标准化（均值≈0、有界 winsorize）"""
        rng = np.random.RandomState(3)
        dates = pd.bdate_range('2026-01-05', periods=80)
        syms = [f'S{i:03d}' for i in range(50)]
        close = pd.DataFrame(
            10 + rng.randn(80, 50).cumsum(axis=0) * 0.1, index=dates, columns=syms)
        mv = pd.DataFrame(
            1e9 + rng.randn(80, 50) * 1e8, index=dates, columns=syms)
        exp = risk_model.style_exposures(close, mv)
        assert set(exp) == {'size', 'momentum', 'volatility'}
        day = dates[-1]
        for name, mat in exp.items():
            row = mat.loc[day].dropna()
            assert len(row) == 50
            assert abs(row.mean()) < 0.1, f"{name} 标准化后均值应≈0"
            assert row.abs().max() <= 3.0 + 1e-9, f"{name} winsorize 上限3σ被突破"


class TestFactorReturns:
    def test_known_loading_recovered(self):
        """r = 0.5×size + 噪声 → size 因子收益≈0.5"""
        rng = np.random.RandomState(5)
        n = 80
        size = pd.Series(rng.randn(n))
        ret = 0.5 * size + rng.randn(n) * 0.01
        fr = risk_model.factor_returns(ret, {'size': size})
        assert abs(fr['size'] - 0.5) < 0.05, \
            f"因子收益应恢复0.5载荷（实际{fr['size']:.3f}）"

    def test_min_cross_section_gates(self):
        """<50 只有效样本时返回空（不产出脆弱估计）"""
        s = pd.Series(np.random.RandomState(2).randn(30))
        assert risk_model.factor_returns(s, {'size': s}) == {}


class TestYearlyAndScore:
    def test_yearly_breakdown_math(self):
        y25 = pd.bdate_range('2025-01-02', '2025-12-31')
        y26 = pd.bdate_range('2026-01-02', '2026-12-31')
        eq = pd.Series(np.concatenate([
            np.linspace(1.0, 1.2, len(y25)),             # 2025: +20%
            np.linspace(1.2, 1.26, len(y26))]),          # 2026: +5%
            index=y25.append(y26))
        yearly = robustness.yearly_breakdown(eq)
        assert abs(yearly[2025] - 0.2) < 1e-9
        assert abs(yearly[2026] - 0.05) < 1e-9

    def test_stability_score_flags_instability(self):
        score_ok, warns_ok = robustness.yearly_stability_score(
            {2021: 0.1, 2022: 0.12, 2023: 0.08, 2024: 0.11})
        score_bad, warns_bad = robustness.yearly_stability_score(
            {2021: 0.4, 2022: -0.3, 2023: -0.35, 2024: -0.25})
        assert score_ok > score_bad
        assert any('正收益' in w for w in warns_bad), "半数负收益年份必须告警"
        assert not any('正收益' in w for w in warns_ok)


class TestRegimeAnalysis:
    def test_vol_split_produced(self):
        idx = pd.bdate_range('2025-01-02', periods=200)
        rng = np.random.RandomState(9)
        bench = pd.Series(4000 + rng.randn(200).cumsum(), index=idx)
        eq = pd.Series(1 + np.linspace(0, 0.1, 200), index=idx)
        out = robustness.market_regime_analysis(eq, bench)
        assert any('波动' in k for k in out), "高/低波分组至少一组应产出"


class TestRobustnessReportRender:
    def test_summary_markdown_sections(self):
        rep = robustness.RobustnessReport(
            n_trials=5, dsr=0.97, stability_score=80,
            yearly_returns={2025: 0.1},
            warnings=['测试告警'])
        md = rep.summary_markdown()
        for token in ('累计实验次数', 'Deflated Sharpe', '稳定性评分',
                      '逐年表现', '测试告警'):
            assert token in md
