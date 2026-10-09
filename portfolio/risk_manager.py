#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
组合风险管理器 — 多层风控约束

约束层级：
1. 回撤控制：软限（降仓）→ 硬限（清仓）
2. 波动率目标：超过目标时缩仓
3. 单票/行业集中度限制
4. 换手上限
5. 现金底线
6. 极端行情熔断
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class RiskLimits:
    """组合风险约束参数"""
    # 回撤控制
    max_drawdown: float = 0.15        # 硬限：超过此值完全清仓
    drawdown_derisk: float = 0.10     # 软限：超过此值开始降仓

    # 波动率目标
    vol_target: float = 0.25           # 年化波动率目标
    vol_window: int = 20                # 波动率计算窗口（交易日）

    # 集中度
    max_position_weight: float = 0.05   # 单票最大权重
    max_sector_weight: float = 0.30      # 单行业最大权重

    # 换手与现金
    max_turnover: float = 0.50           # 单次调仓最大换手
    min_cash_ratio: float = 0.05          # 最低现金比例

    # 极端行情
    circuit_breaker_market: float = 0.05   # 基准单日跌幅熔断线
    circuit_breaker_portfolio: float = 0.07  # 组合单日跌幅熔断线


@dataclass
class RiskState:
    """当前风险状态"""
    current_drawdown: float = 0.0
    peak_equity: float = 0.0
    target_exposure: float = 1.0       # 目标仓位比例（1.0=满仓）
    is_halted: bool = False            # 是否触发熔断
    risk_score: float = 0.0           # 综合风险评分（0=安全, 100=极高）
    active_constraints: List[str] = field(default_factory=list)


class PortfolioRiskManager:
    """
    组合风险管理器

    在每次调仓时评估风险约束，返回调整后的目标权重。
    集成到 PortfolioBacktester 的调仓循环中。
    """

    def __init__(self, limits: RiskLimits = None):
        self.limits = limits or RiskLimits()
        self.state = RiskState()
        self._equity_history: List[float] = []
        self._daily_returns: List[float] = []

    def update(self, equity: float, date) -> RiskState:
        """
        每日更新风险状态

        Args:
            equity: 当日总权益
            date: 当前日期
        """
        # 更新权益历史
        self._equity_history.append(equity)

        # 计算日收益
        if len(self._equity_history) >= 2:
            ret = equity / self._equity_history[-2] - 1
            self._daily_returns.append(ret)

        # 更新峰值和回撤
        self.state.peak_equity = max(self.state.peak_equity, equity)
        if self.state.peak_equity > 0:
            self.state.current_drawdown = 1 - equity / self.state.peak_equity

        # 评估约束
        self._evaluate_constraints(date)
        return self.state

    def _evaluate_constraints(self, date):
        """评估所有风险约束"""
        self.state.active_constraints = []
        self.state.is_halted = False
        self.state.target_exposure = 1.0
        risk_points = 0

        lim = self.limits

        # 1. 回撤控制
        dd = self.state.current_drawdown
        if dd >= lim.max_drawdown:
            self.state.target_exposure = 0.0
            self.state.is_halted = True
            self.state.active_constraints.append(f'回撤{dd:.1%}≥硬限{lim.max_drawdown:.0%}，清仓')
            risk_points += 60
        elif dd >= lim.drawdown_derisk:
            # 线性降仓：从derisk到max之间，仓位从1.0降到0.0
            excess = (dd - lim.drawdown_derisk) / max(lim.max_drawdown - lim.drawdown_derisk, 0.01)
            self.state.target_exposure = max(0.0, 1.0 - excess)
            self.state.active_constraints.append(
                f'回撤{dd:.1%}≥软限{lim.drawdown_derisk:.0%}，仓位降至{self.state.target_exposure:.0%}')
            risk_points += 30
        else:
            risk_points += dd / max(lim.drawdown_derisk, 0.01) * 20

        # 2. 波动率目标
        if len(self._daily_returns) >= lim.vol_window:
            recent_vol = np.std(self._daily_returns[-lim.vol_window:]) * np.sqrt(252)
            if recent_vol > lim.vol_target:
                # 超过目标，按比例缩仓
                scale = lim.vol_target / recent_vol
                new_exposure = self.state.target_exposure * scale
                self.state.target_exposure = new_exposure
                self.state.active_constraints.append(
                    f'年化波动率{recent_vol:.0%}超目标{lim.vol_target:.0%}，缩仓至{new_exposure:.0%}')
                risk_points += 20

        # 3. 组合单日跌幅熔断
        if len(self._daily_returns) >= 1:
            today_ret = self._daily_returns[-1]
            if today_ret <= -lim.circuit_breaker_portfolio:
                self.state.active_constraints.append(
                    f'组合单日跌幅{today_ret:.1%}≥熔断线{lim.circuit_breaker_portfolio:.0%}')
                # 不立即清仓，但标记警告
                risk_points += 15

        # 4. 市场单日跌幅熔断（需要基准数据）
        # 由调用方检查并传入

        self.state.risk_score = min(100, risk_points)

    def check_market_circuit_breaker(self, bench_return: float) -> bool:
        """
        检查市场熔断

        Args:
            bench_return: 基准指数当日涨跌幅

        Returns:
            True = 触发熔断，应暂停交易
        """
        if bench_return <= -self.limits.circuit_breaker_market:
            self.state.active_constraints.append(
                f'市场单日跌幅{bench_return:.1%}≥熔断线{self.limits.circuit_breaker_market:.0%}')
            self.state.risk_score = min(100, self.state.risk_score + 20)
            return True
        return False

    def adjust_weights(self, target_weights: Dict[str, float]) -> Dict[str, float]:
        """
        调整目标权重以满足风险约束

        Args:
            target_weights: 原始目标权重 {symbol: weight}

        Returns:
            调整后的权重（考虑回撤降仓、单票上限、现金底线）
        """
        if not target_weights or self.state.is_halted:
            return {}

        adjusted = {}
        exposure = self.state.target_exposure

        # 应用目标仓位比例
        for sym, w in target_weights.items():
            adjusted[sym] = w * exposure

        # 单票上限
        for sym in adjusted:
            adjusted[sym] = min(adjusted[sym], self.limits.max_position_weight)

        # 确保现金底线
        total_weight = sum(adjusted.values())
        max_total = 1.0 - self.limits.min_cash_ratio
        if total_weight > max_total:
            scale = max_total / total_weight
            adjusted = {s: w * scale for s, w in adjusted.items()}

        # 换手上限由调用方（backtester）在生成订单时检查
        return adjusted

    def check_turnover(self, old_weights: Dict[str, float],
                      new_weights: Dict[str, float]) -> Tuple[bool, float]:
        """
        检查换手率是否超限

        Returns:
            (是否允许, 实际换手率)
        """
        all_syms = set(old_weights) | set(new_weights)
        turnover = sum(abs(new_weights.get(s, 0) - old_weights.get(s, 0)) for s in all_syms) / 2
        if turnover > self.limits.max_turnover:
            self.state.active_constraints.append(
                f'换手率{turnover:.0%}超限{self.limits.max_turnover:.0%}')
            return False, turnover
        return True, turnover

    def get_risk_report(self) -> str:
        """生成风险状态报告"""
        lines = ["### 风险状态\n"]
        lines.append(f"- 当前回撤: **{self.state.current_drawdown:.1%}**"
                    f"（软限 {self.limits.drawdown_derisk:.0%} / 硬限 {self.limits.max_drawdown:.0%}）")
        lines.append(f"- 目标仓位: **{self.state.target_exposure:.0%}**")
        lines.append(f"- 风险评分: **{self.state.risk_score:.0f}/100**")
        lines.append(f"- 熔断状态: {'🔴 已熔断' if self.state.is_halted else '🟢 正常'}")
        if self.state.active_constraints:
            lines.append("\n**生效约束**:")
            for c in self.state.active_constraints:
                lines.append(f"- {c}")
        return "\n".join(lines)
