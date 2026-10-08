from dataclasses import dataclass, field
from typing import List, Dict, Any, Literal

Operator = Literal[">", "<", ">=", "<=", "cross_up", "cross_down", "break_upper", "break_lower", "adx_gt", "volume_ma_ratio_gt", "price_above_ma", "ma5_above_ma20", "rps_rank_gt"]
Logic = Literal["AND", "OR"]

@dataclass
class IndicatorRule:
    name: str                    # "RSI", "MACD"
    operator: Operator          # ">", "<", ">=", "<=", "cross_up", "cross_down", "break_upper", "break_lower"
    threshold: float             # 阈值
    weight: float = 1.0          # 评分权重

@dataclass
class SelectionConfig:
    name: str
    indicators: List[IndicatorRule] = field(default_factory=list)
    logic: Logic = "AND"         # "AND" 或 "OR"
    min_score: int = 60          # 最低信号评分 (0-100)
    max_positions: int = 5       # 最大持仓数
    sw_industry_filter: bool = False  # 是否启用申万行业过滤
    sw_industry_top_n: int = 10      # 选取的申万行业数量

    def __post_init__(self):
        if self.min_score < 0 or self.min_score > 100:
            raise ValueError(f"min_score must be 0-100, got {self.min_score}")
        if self.max_positions <= 0:
            raise ValueError(f"max_positions must be positive, got {self.max_positions}")

@dataclass
class ScoredStock:
    symbol: str
    name: str = ""
    score: int = 0                # 0-100
    conditions: List[Dict] = field(default_factory=list)  # 满足的条件
    indicators: Dict[str, float] = field(default_factory=dict)  # 当前指标值


@dataclass
class ScanConstraint:
    """
    选股约束条件（可选）

    用于在 scanner.scan() 时传入市场环境和行业过滤约束。
    """
    allowed_sectors: List[str] = None       # 只选这些行业
    block_sectors: List[str] = None         # 排除这些行业
    min_market_score: int = 0               # 最低市场评分

    # 市场环境信号（由 BacktestEngine 每日注入）
    breadth_signal: str = None   # 'expanding' | 'contracting' | 'neutral'
    valuation_signal: str = None # 'cheap' | 'fair' | 'expensive'
    macro_signal: str = None     # 'expanding' | 'contracting' | 'neutral'
    leading_signal: str = None   # 'expanding' | 'contracting' | 'neutral'
    seasonal_signal: str = None  # 'favorable' | 'neutral' | 'unfavorable'
    position_factor: float = 1.0 # 仓位系数（0.2-1.0）
    market_score: int = 50       # 市场综合评分（0-100）
    adx_signal: str = None       # 'trending' | 'ranging' | 'neutral'（ADX市场状态）
    rps_rank: Dict[str, float] = None  # 横截面RPS：symbol → 0-100 百分位（引擎每日注入）

    def is_trending(self) -> bool:
        """市场是否处于趋势状态（宽度扩张 + 领先指标向上）"""
        return (self.breadth_signal == 'expanding' and
                self.leading_signal in ('expanding', 'neutral'))

    def is_ranging(self) -> bool:
        """市场是否处于震荡状态"""
        return self.breadth_signal in ('neutral', 'contracting')

    def is_cheap(self) -> bool:
        """估值是否便宜"""
        return self.valuation_signal == 'cheap'

    def is_expensive(self) -> bool:
        """估值是否偏贵"""
        return self.valuation_signal == 'expensive'

    def is_seasonal_favorable(self) -> bool:
        """季节性是否有利"""
        return self.seasonal_signal == 'favorable'

    def is_confident(self) -> bool:
        """市场环境是否有信心（position_factor > 0.5）"""
        return self.position_factor >= 0.5

    def is_adx_trending(self) -> bool:
        """ADX信号是否显示趋势市场（ADX > 30）"""
        return self.adx_signal == 'trending'

    def is_adx_ranging(self) -> bool:
        """ADX信号是否显示震荡市场（ADX < 20）"""
        return self.adx_signal == 'ranging'

    def __post_init__(self):
        self.allowed_sectors = self.allowed_sectors or None
        self.block_sectors = self.block_sectors or None