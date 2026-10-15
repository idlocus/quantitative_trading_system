#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
主链配置 —— 回测与实盘共用的唯一参数来源

任何一侧调整这些值 = 改变主链语义，必须两侧同步生效并重跑一致性测试
（tests/test_chain_consistency.py）。在两侧另行硬编码同一数值视为分叉缺陷。
"""

# ---- 信号（PIT 因子合成） ----
# 生产候选 v2（2026-10-13 冻结）：质量过滤 + 价值合成 + 趋势叠加(0.2)
# 验证结果（2022-2026 严格重放，参数稳定性 11/11 通过）：
#   质量+价值: +27.2%  |  质量+价值+趋势: +30.5%  |  等权对照: -15.4%
# 旧5因子（size=-1 小盘倾斜）已淘汰：-2.7%，问题在选股
PRODUCTION_FACTORS = {'ep': 1.0, 'bp': 1.0, 'sp': 1.0}
MOMENTUM_WEIGHT = 0.2          # 趋势叠加权重（0.2=价值0.8+动量0.2）
QUALITY_FILTER = {
    'roe_min': 0.08,           # ROE > 8%（PIT公告值）
    'leverage_max': 65.0,      # 负债/权益 < 65%（PIT值=百分比，≈资产/权益<2.9；
                                # 此前错用3.0当资产/权益倍数，实际PIT返回的是D/E%）
}
TOP_N = 100                  # 目标持仓数
WARMUP_DAYS = 400            # 因子 warm-up 自然日

# ---- 目标组合 ----
BUFFER_BAND = 0.20           # 权重缓冲带（20%，无漂移则零调仓）
MAX_WEIGHT = 0.05            # 单票权重上限

# ---- 选股过滤（回测 _select 与 live generate_target 同口径） ----
MIN_LISTING_DAYS = 60        # 上市天数过滤（≥60个交易日数据才可入选）
MIN_DAILY_TURNOVER = 500e4   # 20日均成交额下限（500万，流动性不足不可入选）

# ---- 成交约束 ----
LOT_SIZE = 100               # A股整手
LIMIT_TOLERANCE = 0.002      # 涨跌停判定容差（比例）
MAX_PARTICIPATION = 0.10     # 单日买入额 ≤ 日均成交额×10%（回测参与率约束）
IMPACT_COEFF = 0.10          # 市场冲击系数（sqrt模型，回测专属保守项）

# ---- 费用（三处旧定义收敛于此：backtester/PaperBroker/risk.constants 同值） ----
COMMISSION = 0.0003          # 佣金（双边，万3）
STAMP_TAX = 0.0005           # 印花税（卖出单边，2023-08 起 0.05%）
TRANSFER_FEE = 0.00001       # 过户费（双边，万0.1）
TRADING_DAYS_PER_YEAR = 252  # 年化基数（主链唯一来源；旧 risk.constants 同值但属弃用包）
SELL_PROCEEDS_DISCOUNT = 0.95  # 卖出回款折扣（费用+滑点；build_orders 预算用——
                               # 写路径口径唯一来源，原为 signals.py 内联魔数 0.95）
SLIPPAGE = 0.001             # 单边滑点（0.1%）——注意：旧引擎 risk/constants 中
                              # BacktestParams.SLIPPAGE=0.0（零滑点），与主链不可比

# ---- 调仓节奏 ----
# 回测（近似口径）：每 N 个交易日重选。注意：与生产"每日重选"【并非】严格同一策略——
# 排名变化/新进TopN/风控强制卖出都不受缓冲带控制，缓冲带只管已持仓的权重漂移。
# 严格验证生产收益用 DailyReplayBacktester（portfolio/daily_replay.py）：
# 逐日重放生产组件（build_orders/RiskGate/PaperBroker/Ledger），见 run_daily_replay.py。
REBALANCE_DAYS_BT = 5        # 近似回测默认调仓间隔（交易日）

# ---- 执行时序（前视防护） ----
# 主链唯一口径：T-1 收盘信号 → T 日成交（回测/Paper 以 T 收盘价+滑点撮合，
# QMT 在 T 日盘中/收盘竞价下单）。回测实现 = score_mx.shift(1)。
# 禁止 generate_target(当日) + 当日收盘价成交的组合（信号用了当日收盘数据，
# 却假设能以当日收盘价成交——真实交易不可实现）。
SIGNAL_LAG = 1               # 信号滞后交易日数

# ---- 执行风控（与 live/risk_gate.GateConfig、portfolio/risk_manager.RiskLimits 对齐） ----
DRAWDOWN_SOFT = 0.08         # 回撤软限（降仓）
DRAWDOWN_HARD = 0.15         # 回撤硬限（清仓）
MARKET_DAILY_CRASH = 0.05    # 基准单日跌幅熔断（禁买冷却）
PORTFOLIO_DAILY_CRASH = 0.07 # 组合单日跌幅熔断（禁买冷却）
STOP_LOSS_PCT = 0.05         # 个股硬止损（成本下方5%，全仓卖出）
VOL_TARGET = 0.25            # 年化波动率目标（超过时缩仓）
MAX_TURNOVER = 0.50          # 单次调仓最大换手（回缩目标权重）
MIN_CASH_RATIO = 0.05        # 最低现金比例

# ---- 硬限清仓后重入（2026-10-15 定稿，十年重放两轮验证） ----
# 第一轮（冷却20日+半仓×60日梯度）被数据否决：-1.5%/-27.6% 回撤双输于
# 立即满仓重入（+59.0%/-24.9%）——本策略回撤段成簇（2018/2020/2022/2024），
# 慢重入系统性错过反弹。定稿语义：清仓完成 → 立即以现金为新锚满仓重入，
# 辅以滚动窗口熔断（防 2008 式连续崩盘的无限清仓-重入循环）。
# 人工重启 = 清除 runtime/risk_state.json 的 protocol_halted/liq_dates 字段
# （流程见 docs/runbook.md）。
REENTRY_MAX_LIQUIDATIONS = 3    # 熔断窗口内硬限清仓次数上限（超过=永久停机）
REENTRY_CIRCUIT_WINDOW_DAYS = 365  # 熔断计数滚动窗口（日历日≈252交易日）


def validate():
    """配置自校验——RiskGate / build_strategy_score 入口调用，违反=启动即失败。

    背景：trading/config 是唯一真源，但此前无任何机制保证参数组合自洽
    （评审报告 P2 #15）。"""
    if not (0 < DRAWDOWN_SOFT < DRAWDOWN_HARD < 1.0):
        raise ValueError(
            f"回撤阈值不自洽: SOFT={DRAWDOWN_SOFT} HARD={DRAWDOWN_HARD}"
            f"（要求 0 < SOFT < HARD < 1）")
    if not (0.0 <= MOMENTUM_WEIGHT <= 1.0):
        raise ValueError(f"MOMENTUM_WEIGHT={MOMENTUM_WEIGHT} 超出[0,1]")
    if TOP_N < 2:
        raise ValueError(f"TOP_N={TOP_N} < 2")
    if 1.0 / TOP_N > MAX_WEIGHT + 1e-12:
        raise ValueError(
            f"等权下限 1/TOP_N={1.0/TOP_N:.4f} > 单票上限 MAX_WEIGHT={MAX_WEIGHT}"
            f"——约束不可满足")
    if REENTRY_CIRCUIT_WINDOW_DAYS < 30:
        raise ValueError("REENTRY_CIRCUIT_WINDOW_DAYS 过短（<30日历日）——"
                         "熔断窗口应覆盖一个季度以上的清仓聚集")
    if REENTRY_MAX_LIQUIDATIONS < 1:
        raise ValueError("REENTRY_MAX_LIQUIDATIONS 至少为 1（首次硬限即永久停机"
                         "应显式设 1，而不是 0）")
    if not PRODUCTION_FACTORS:
        raise ValueError("PRODUCTION_FACTORS 为空")

# ---- 约束覆盖范围声明（回测 vs 实盘的诚实对照） ----
# 回测有/实盘无的约束（回测曲线因此更平滑/回撤更浅）
# 回测有/实盘无的约束（vol_target/max_turnover/min_cash_ratio 已统一到 GateConfig，
# 但 gate 尚未主动执行缩仓/换手回缩/现金底线——值已同源，执行差异由重放口径覆盖）
BACKTEST_ONLY = ['max_sector_weight', 'max_position_weight']
# 实盘有/回测无的约束（回测因此保留本该砍掉的仓位）
LIVE_ONLY = ['stop_loss_pct']
