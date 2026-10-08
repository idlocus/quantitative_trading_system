#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
过拟合防御：Deflated Sharpe Ratio（Bailey & López de Prado）+ 实验登记

核心思想：我们尝试了 N 个策略，报告的夏普是 N 次抽样中的最大值——
即使所有策略真实夏普为 0，最大值也显著为正。DSR 回答：
"考虑到尝试次数 N，这个夏普在统计上还显著吗？"
"""

import json
import math
import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy import stats

TRIALS_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           'results', 'data', 'trial_registry.json')


@dataclass
class DeflatedSharpe:
    """Deflated Sharpe Ratio 检验结果"""
    sr_observed: float        # 观测年化夏普
    sr_expected_max: float    # N 次尝试下"真夏普=0"时的期望最大夏普
    dsr: float                # DSR ∈ (0,1)，>0.95 才认为通过检验
    n_trials: int
    n_obs: int

    def summary(self) -> str:
        verdict = "通过（可信）" if self.dsr > 0.95 else "未通过（可能只是运气）"
        return (f"Sharpe={self.sr_observed:+.2f} | 零假设期望最大={self.sr_expected_max:+.2f} "
                f"(N={self.n_trials}次尝试) | DSR={self.dsr:.3f} → {verdict}")


def expected_max_sharpe(n_trials: int, var_sr: float = 1.0) -> float:
    """
    零假设（真SR=0）下 N 次独立尝试的期望最大 SR（Euler-Mascheroni 近似）

    Args:
        n_trials: 尝试次数
        var_sr: 各次尝试 SR 的方差（跨策略SR方差，默认1即标准化）
    """
    if n_trials <= 1:
        return 0.0
    emc = 0.5772156649015329
    z1 = stats.norm.ppf(1 - 1.0 / n_trials)
    z2 = stats.norm.ppf(1 - 1.0 / (n_trials * math.e))
    return math.sqrt(var_sr) * ((1 - emc) * z1 + emc * z2)


def deflated_sharpe_ratio(returns: np.ndarray, n_trials: int = 1,
                          periods_per_year: int = 252) -> DeflatedSharpe:
    """
    计算 DSR

    Args:
        returns: 日收益率序列
        n_trials: 本研究方向累计尝试的策略数（从实验登记读取）
        periods_per_year: 年化系数
    """
    r = np.asarray(returns, dtype=float)
    r = r[~np.isnan(r)]
    n = len(r)
    if n < 20:
        return DeflatedSharpe(0.0, 0.0, 0.0, n_trials, n)

    sr = float(r.mean() / r.std() * math.sqrt(periods_per_year)) if r.std() > 0 else 0.0
    g3 = float(stats.skew(r))
    g4 = float(stats.kurtosis(r, fisher=False))  # 峰度（非超额）

    sr0 = expected_max_sharpe(max(n_trials, 1))
    # 日频尺度换算
    sr_daily = sr / math.sqrt(periods_per_year)
    sr0_daily = sr0 / math.sqrt(periods_per_year)

    denom = math.sqrt(max(1e-12, 1.0 - g3 * sr_daily + (g4 - 1) / 4.0 * sr_daily ** 2))
    z = (sr_daily - sr0_daily) * math.sqrt(n - 1) / denom
    return DeflatedSharpe(sr, sr0, float(stats.norm.cdf(z)), n_trials, n)


# =============================================================================
# 实验登记：诚实计数我们尝试过的策略（DSR 的 n_trials 来源）
# =============================================================================

class TrialRegistry:
    """
    策略级实验登记（JSON 持久化）。
    每跑一次"准备对外声称"的策略回测就 record 一次——
    DSR 用登记总数作为尝试次数，防止"只报告最好的那次"。
    """

    def __init__(self, path: str = None):
        self.path = path or TRIALS_PATH
        self._data = self._load()

    def _load(self) -> dict:
        if os.path.isfile(self.path):
            try:
                with open(self.path, encoding='utf-8') as f:
                    return json.load(f)
            except Exception:
                pass
        return {'trials': [], 'count': 0}

    def record(self, label: str, sharpe: float = None, total_return: float = None,
               extra: dict = None) -> int:
        """登记一次策略尝试，返回累计尝试次数"""
        entry = {'label': label}
        if sharpe is not None:
            entry['sharpe'] = round(float(sharpe), 3)
        if total_return is not None:
            entry['total_return'] = round(float(total_return), 4)
        if extra:
            entry.update(extra)
        self._data['trials'].append(entry)
        self._data['count'] = len(self._data['trials'])
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, 'w', encoding='utf-8') as f:
            json.dump(self._data, f, ensure_ascii=False, indent=1)
        return self._data['count']

    @property
    def count(self) -> int:
        return self._data['count']
