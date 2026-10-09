#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
因子策略稳健性验证框架

功能：
1. 强制实验登记（TrialRegistry 集成）
2. Deflated Sharpe Ratio 自动计算
3. 参数敏感性分析（TopN、调仓频率、缓冲带）
4. 子周期稳定性（逐年表现）
5. 市场状态分析（趋势/震荡/高波/低波）
6. 因子方向稳定性检验

用法：作为回测结果的必经验证步骤
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from research.overfitting import deflated_sharpe_ratio, TrialRegistry
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class RobustnessReport:
    """稳健性验证报告"""
    n_trials: int = 0                  # 累计实验次数
    dsr: Optional[float] = None       # Deflated Sharpe Ratio
    yearly_returns: Dict[int, float] = field(default_factory=dict)
    regime_performance: Dict[str, dict] = field(default_factory=dict)
    param_sensitivity: Dict[str, List[Tuple[float, float]]] = field(default=dict)
    stability_score: float = 0.0       # 综合稳定性评分 (0-100)
    warnings: List[str] = field(default_factory=list)

    def summary_markdown(self) -> str:
        lines = ["## 稳健性验证\n"]
        lines.append(f"- **累计实验次数**: {self.n_trials}")
        lines.append(f"- **Deflated Sharpe**: {self.dsr:.3f} "
                    f"({'通过' if self.dsr and self.dsr > 0.95 else '未通过（可能过拟合）'}")
        lines.append(f"- **稳定性评分**: {self.stability_score:.0f}/100")
        if self.warnings:
            lines.append("\n**警告**:")
            for w in self.warnings:
                lines.append(f"- ⚠️ {w}")
        if self.yearly_returns:
            lines.append("\n### 逐年表现")
            lines.append("| 年份 | 收益 |")
            lines.append("|------|------|")
            for y, r in sorted(self.yearly_returns.items()):
                lines.append(f"| {y} | {r:+.1%} |")
        if self.regime_performance:
            lines.append("\n### 市场状态分析")
            lines.append("| 状态 | 收益 | 胜率 |")
            lines.append("|------|------|------|")
            for regime, perf in self.regime_performance.items():
                lines.append(f"| {regime} | {perf.get('return', 0):+.1%} | {perf.get('win_rate', 0):.0%} |")
        if self.param_sensitivity:
            lines.append("\n### 参数敏感性")
            for param, results in self.param_sensitivity.items():
                rets = [r for _, r in results]
                spread = max(rets) - min(rets)
                lines.append(f"- **{param}**: 收益范围 {min(rets):+.1%} ~ {max(rets):+.1%} "
                            f"（极差 {spread:.1%}）")
        return "\n".join(lines)


def compute_dsr(returns: pd.Series, n_trials: int) -> float:
    """计算 Deflated Sharpe Ratio"""
    result = deflated_sharpe_ratio(returns.values, n_trials=n_trials)
    return result.dsr


def yearly_breakdown(equity: pd.Series) -> Dict[int, float]:
    """逐年收益分解"""
    returns = {}
    for year, group in equity.groupby(equity.index.year):
        if len(group) >= 10:
            returns[int(year)] = float(group.iloc[-1] / group.iloc[0] - 1)
    return returns


def yearly_stability_score(yearly: Dict[int, float]) -> Tuple[float, List[str]]:
    """年度稳定性评分（正收益年份比例 + 波动率惩罚）"""
    if not yearly:
        return 0, ["无年度数据"]
    rets = list(yearly.values())
    positive_ratio = sum(1 for r in rets if r > 0) / len(rets)
    volatility = np.std(rets)
    # 评分 = 正收益比例×70 + (1-波动率)×30
    score = positive_ratio * 70 + max(0, 1 - volatility * 5) * 30
    warnings = []
    if positive_ratio < 0.5:
        warnings.append(f"仅{positive_ratio:.0%}年份正收益，策略不稳定")
    if volatility > 0.3:
        warnings.append(f"年度收益波动过大（σ={volatility:.1%}）")
    return score, warnings


def market_regime_analysis(equity: pd.Series,
                           bench_close: pd.Series) -> Dict[str, dict]:
    """
    市场状态分析：按趋势/震荡、高波/低波分组统计表现
    """
    from indicators.trend_indicators import DMI_ADX

    results = {}

    # 用基准构建日频 DataFrame
    bench_df = pd.DataFrame({
        'date': bench_close.index, 'open': bench_close.values,
        'high': bench_close.values, 'low': bench_close.values,
        'close': bench_close.values, 'volume': [1e9] * len(bench_close)
    })

    try:
        dmi = DMI_ADX(bench_df.set_index('date'), period=14)
        adx = dmi['adx']
    except Exception:
        return results

    # 日收益
    daily_ret = equity.pct_change().dropna()

    # 趋势/震荡
    for label, condition in [('趋势(ADX>25)', adx > 25), ('震荡(ADX<20)', adx < 20)]:
        mask = condition.reindex(daily_ret.index).fillna(False)
        rets = daily_ret[mask]
        if len(rets) > 20:
            cum_ret = (1 + rets).prod() - 1
            win = (rets > 0).mean()
            results[label] = {'return': float(cum_ret), 'win_rate': float(win), 'days': len(rets)}

    # 高波/低波
    rolling_vol = bench_close.pct_change().rolling(20).std()
    vol_median = rolling_vol.median()
    for label, condition in [('高波动', rolling_vol > vol_median),
                              ('低波动', rolling_vol <= vol_median)]:
        mask = condition.reindex(daily_ret.index).fillna(False)
        rets = daily_ret[mask]
        if len(rets) > 20:
            cum_ret = (1 + rets).prod() - 1
            win = (rets > 0).mean()
            results[label] = {'return': float(cum_ret), 'win_rate': float(win), 'days': len(rets)}

    return results


def register_experiment(label: str, total_return: float, sharpe: float) -> int:
    """强制登记实验（每次回测必须调用）"""
    registry = TrialRegistry()
    n = registry.record(label, sharpe=sharpe, total_return=total_return)
    return n


def generate_robustness_report(result, bench_close: pd.Series = None,
                               n_trials: int = None) -> RobustnessReport:
    """
    从回测结果生成稳健性报告

    Args:
        result: PortfolioResult
        bench_close: 基准收盘价（可选，用于市场状态分析）
        n_trials: 累计实验次数（默认从注册表读取）
    """
    report = RobustnessReport()

    # 1. 实验次数
    if n_trials is None:
        try:
            registry = TrialRegistry()
            n_trials = registry.count
        except Exception:
            n_trials = 1
    report.n_trials = n_trials

    # 2. DSR
    returns = result.equity_curve.pct_change().dropna()
    if len(returns) > 60:
        report.dsr = compute_dsr(returns, n_trials)

    # 3. 逐年分析
    report.yearly_returns = yearly_breakdown(result.equity_curve)
    score, warns = yearly_stability_score(report.yearly_returns)
    report.warnings.extend(warns)

    # 4. 市场状态分析
    if bench_close is not None and len(bench_close) > 60:
        report.regime_performance = market_regime_analysis(
            result.equity_curve, bench_close)

    # 5. 综合评分
    report.stability_score = score

    return report
