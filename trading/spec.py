#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
StrategySpec —— 策略版本治理

每次生成信号或回测结果时，自动捕获完整的策略规格：
- 因子定义、单位、PIT规则
- 质量过滤参数
- 动量参数
- 调仓规则、组合约束
- 执行假设
- 策略版本号

规格从 trading/config 自动派生（不可独立维护），确保永不漂移。
每个结果工件附带 StrategySpec，可回答"这次收益是哪个版本产生的"。

用法：
    spec = StrategySpec.current()
    spec_dict = spec.to_dict()  # 附到工件 _meta
    spec.hash                   # 参数指纹（同参数=同hash）
"""

import hashlib
import json
from dataclasses import dataclass, asdict, field
from datetime import datetime
from typing import Dict, List, Optional

from trading import config as chain


# 策略版本号（改 trading/config 中的策略参数时必须递增）
# 格式: major.minor.patch
#   major: 因子/过滤逻辑变更（收益结论不可比）
#   minor: 参数微调（阈值/权重/窗口）
#   patch: bug修复（不改变收益逻辑）
STRATEGY_VERSION = "2.1.0"

# 版本变更日志（新版本放最上面）
VERSION_LOG = {
    "2.1.0": "2026-10-14: 杠杆阈值修正(3.0→65.0, D/E百分比)；组合约束执行层",
    "2.0.0": "2026-10-13: v2策略——质量过滤+价值合成+趋势叠加；去掉size/leverage权重",
    "1.0.0": "2026-10-12之前: 旧5因子(ep/bp/sp/size/leverage)——已淘汰",
}


@dataclass(frozen=True)
class StrategySpec:
    """策略规格（不可变快照，从 trading/config 自动派生）"""

    # ---- 版本 ----
    version: str = STRATEGY_VERSION

    # ---- 因子定义 ----
    factors: Dict[str, float] = field(default_factory=lambda: dict(chain.PRODUCTION_FACTORS))
    factor_units: Dict[str, str] = field(default_factory=lambda: {
        f: chain.__dict__.get('FACTOR_CONTRACTS', {}).get(f, {}).get('unit', 'unknown')
        if hasattr(chain, 'FACTOR_CONTRACTS') else 'see data/contracts.py'
        for f in chain.PRODUCTION_FACTORS
    })

    # ---- 质量过滤 ----
    quality_filter: Dict[str, float] = field(default_factory=lambda: dict(chain.QUALITY_FILTER))

    # ---- 动量叠加 ----
    momentum_weight: float = chain.MOMENTUM_WEIGHT

    # ---- 选股 ----
    top_n: int = chain.TOP_N
    warmup_days: int = chain.WARMUP_DAYS
    min_listing_days: int = chain.MIN_LISTING_DAYS
    min_daily_turnover: float = chain.MIN_DAILY_TURNOVER
    min_factor_coverage: float = 0.30
    min_target_count: int = 20

    # ---- 调仓规则 ----
    buffer_band: float = chain.BUFFER_BAND
    signal_lag: int = chain.SIGNAL_LAG  # T-1信号 T执行
    rebalance_frequency: str = "monthly"  # 月度调仓

    # ---- 组合约束 ----
    max_weight: float = chain.MAX_WEIGHT

    # ---- 执行假设 ----
    commission: float = chain.COMMISSION
    stamp_tax: float = chain.STAMP_TAX
    transfer_fee: float = chain.TRANSFER_FEE
    slippage: float = chain.SLIPPAGE

    # ---- 风控参数 ----
    drawdown_soft: float = chain.DRAWDOWN_SOFT
    drawdown_hard: float = chain.DRAWDOWN_HARD
    stop_loss_pct: float = chain.STOP_LOSS_PCT
    vol_target: float = chain.VOL_TARGET
    max_turnover: float = chain.MAX_TURNOVER
    min_cash_ratio: float = chain.MIN_CASH_RATIO

    @classmethod
    def current(cls) -> "StrategySpec":
        """捕获当前 trading/config 状态为不可变规格"""
        return cls()

    def to_dict(self) -> dict:
        """序列化（附到工件 _meta 或信号日志）"""
        d = asdict(self)
        d['version_log'] = VERSION_LOG.get(self.version, '')
        d['captured_at'] = datetime.now().isoformat()
        d['strategy_hash'] = self.hash
        return d

    @property
    def hash(self) -> str:
        """参数指纹：相同参数=相同hash（不含时间戳）"""
        params = {k: v for k, v in asdict(self).items() if k != 'captured_at'}
        blob = json.dumps(params, sort_keys=True, default=str)
        return hashlib.sha256(blob.encode()).hexdigest()[:16]

    def summary(self) -> str:
        """一行摘要（用于日志/日报）"""
        factors_str = '+'.join(f'{k}({v:+.0f})' for k, v in self.factors.items())
        qf = self.quality_filter
        return (f"v{self.version} [{factors_str}] "
                f"QF(ROE>{qf.get('roe_min', '?')},D/E<{qf.get('leverage_max', '?')}) "
                f"Mom({self.momentum_weight}) Top{self.top_n} "
                f"Band{self.buffer_band:.0%} Vol{self.vol_target:.0%}")


def attach_spec_to_artifact(payload: dict) -> dict:
    """将当前策略规格附到工件 payload 的 _meta 中"""
    spec = StrategySpec.current()
    if '_meta' not in payload:
        payload['_meta'] = {}
    payload['_meta']['strategy_spec'] = spec.to_dict()
    payload['_meta']['strategy_hash'] = spec.hash
    return payload


def attach_spec_to_signal(signal_context: dict) -> dict:
    """将当前策略规格附到信号生成上下文"""
    spec = StrategySpec.current()
    signal_context['strategy_version'] = spec.version
    signal_context['strategy_hash'] = spec.hash
    signal_context['strategy_summary'] = spec.summary()
    return signal_context
