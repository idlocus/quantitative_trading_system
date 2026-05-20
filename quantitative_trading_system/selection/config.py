from dataclasses import dataclass, field
from typing import List, Dict, Any

@dataclass
class IndicatorRule:
    name: str                    # "RSI", "MACD"
    operator: str                # ">", "<", ">=", "<=", "cross_up", "cross_down", "break_upper", "break_lower"
    threshold: float             # 阈值
    weight: float = 1.0          # 评分权重

@dataclass
class SelectionConfig:
    name: str
    indicators: List[IndicatorRule] = field(default_factory=list)
    logic: str = "AND"           # "AND" 或 "OR"
    min_score: int = 60          # 最低信号评分 (0-100)
    max_positions: int = 5       # 最大持仓数

@dataclass
class ScoredStock:
    symbol: str
    name: str = ""
    score: int = 0                # 0-100
    conditions: List[Dict] = field(default_factory=list)  # 满足的条件
    indicators: Dict[str, float] = field(default_factory=dict)  # 当前指标值