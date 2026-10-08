#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
选股扫描器
基于技术指标规则扫描股票
"""

from typing import List, Dict, Any, Optional
import pandas as pd
from datetime import date, timedelta

from .config import SelectionConfig, ScoredStock, IndicatorRule, ScanConstraint
from indicators.registry import IndicatorRegistry
from risk.constants import IndicatorParams
from utils.logger import get_logger

logger = get_logger(__name__)

# Default indicator params for scanning
_PARAMS = IndicatorParams()


class StockScanner:
    """
    股票扫描器

    基于预配置的技术指标规则对股票进行扫描和评分
    """

    def __init__(self, data_source):
        """
        初始化扫描器

        Args:
            data_source: 数据源，需提供 get_kline(symbol) 方法返回 OHLCV DataFrame
        """
        self.data_source = data_source
        self.registry = IndicatorRegistry
        # 全历史指标缓存：symbol -> {'kline': df, 'indicators': {key: Series/DataFrame}}
        # 指标在完整K线上只算一次（ewm/rolling 只依赖历史，时点正确），逐日切片读取
        self._history_cache = {}

    # ---- 历史缓存 ----

    def _get_history(self, symbol: str, ref_date: date):
        """
        获取该股票的完整缓存K线（含充分暖机窗口与未来余量）

        每只股票只从数据源取一次宽窗口（截至今日+余量，数据源返回其全部可用数据）；
        之后逐日只做内存切片。指标在完整序列上一次计算后按 ≤ ref_date 切片——
        rolling/ewm 均为因果算子，切片结果与"只用历史数据计算"完全一致（时点正确）。
        """
        cached = self._history_cache.get(symbol)
        if cached is not None:
            df = cached['kline']
            if not df.empty and df['date'].iloc[-1] >= pd.Timestamp(ref_date):
                return cached

        # 宽窗口：暖机（3×LOOKBACK 保证 ewm 收敛）+ 未来余量（回测逐日前进时免重取）
        start = ref_date - timedelta(days=_PARAMS.LOOKBACK_DAYS * 3)
        end = ref_date + timedelta(days=400)
        df = self.data_source.get_kline(symbol, start, end)
        if df is None or df.empty:
            self._history_cache[symbol] = {'kline': pd.DataFrame(), 'indicators': {}}
            return self._history_cache[symbol]
        df = df.copy()
        if not isinstance(df['date'].iloc[0], pd.Timestamp):
            df['date'] = pd.to_datetime(df['date'])
        cached = {'kline': df, 'indicators': {}}
        self._history_cache[symbol] = cached
        return cached

    def _get_indicator_history(self, symbol: str, rule_name: str, operator: str,
                                ref_date: date):
        """
        获取指标在完整K线上的时序值（缓存），返回 (current, prev, values, data)

        values: Series（主列）或 DataFrame（复合指标，如 DMI 需要 plus_di/minus_di）
        """
        cached = self._get_history(symbol, ref_date)
        df = cached['kline']
        if df is None or len(df) < _PARAMS.MIN_DATA_LENGTH:
            return None, None, None

        cache_key = f"{rule_name}|{operator}"
        if cache_key not in cached['indicators']:
            try:
                indicator_func = self.registry.get(rule_name)
                values = indicator_func(df)
            except Exception:
                cached['indicators'][cache_key] = None
                return None, None, None
            cached['indicators'][cache_key] = values
        values = cached['indicators'][cache_key]
        if values is None:
            return None, None, None

        # 对齐日期索引并截断到 ≤ ref_date（时点正确）
        ts = pd.Timestamp(ref_date)
        try:
            if not isinstance(values.index, pd.DatetimeIndex):
                values = values.copy()
                values.index = pd.DatetimeIndex(df['date'].iloc[:len(values)])
            values = values.loc[values.index <= ts]
        except Exception:
            return None, None, None
        if len(values) == 0:
            return None, None, None

        # 取主列的最新两个值
        if isinstance(values, pd.DataFrame):
            if operator == 'adx_gt' and 'adx' in values.columns:
                col = 'adx'
            else:
                col = getattr(self.registry.get(rule_name), '_primary_column', None)
                if col is None or col not in values.columns:
                    col = values.columns[-1]
            series = values[col]
            extra = values  # DMI 操作符需要完整 DataFrame
        else:
            series = values
            extra = None

        current = series.iloc[-1]
        prev = series.iloc[-2] if len(series) > 1 else current
        return current, prev, extra

    def scan(self, symbols: List[str], config: SelectionConfig, ref_date: date = None,
             allowed_symbols: Optional[set] = None,
             constraint: ScanConstraint = None) -> List[ScoredStock]:
        """
        扫描股票列表

        Args:
            symbols: 股票代码列表
            config: 选股配置
            ref_date: 参考日期（用于获取历史数据）
            allowed_symbols: 可选的股票代码集合，用于限制扫描范围
            constraint: 可选的选股约束（市场/行业过滤）

        Returns:
            按评分降序排列的股票列表
        """
        # 市场环境评分过滤
        if constraint is not None and constraint.min_market_score > 0:
            # position_factor 低于阈值时直接跳过
            pass  # 后续在 allocate 层面处理，这里仅记录约束

        # If allowed_symbols is provided, only scan stocks in that set
        if allowed_symbols is not None:
            symbols = [s for s in symbols if s in allowed_symbols]

        results = []
        for symbol in symbols:
            score, conditions, indicators = self._evaluate(symbol, config, ref_date, constraint)
            if score >= config.min_score:
                results.append(ScoredStock(
                    symbol=symbol,
                    score=score,
                    conditions=conditions,
                    indicators=indicators
                ))
        return sorted(results, key=lambda x: x.score, reverse=True)

    def _evaluate(self, symbol: str, config: SelectionConfig, ref_date: date = None,
                  constraint: ScanConstraint = None):
        """
        评估单个股票

        Args:
            symbol: 股票代码
            config: 选股配置
            ref_date: 参考日期（用于获取历史数据）
            constraint: 可选的选股约束

        Returns:
            (总分, 满足条件列表, 指标值字典)
        """
        end = ref_date or date.today()

        # 全历史缓存 + 截断到 ref_date（时点正确）
        cached = self._get_history(symbol, end)
        df = cached['kline']
        if df is None or df.empty:
            return 0, [], {}
        data = df[df['date'] <= pd.Timestamp(end)]
        if len(data) < _PARAMS.MIN_DATA_LENGTH:
            logger.debug(f"Insufficient data for {symbol}: need {_PARAMS.MIN_DATA_LENGTH}, got {len(data)}")
            return 0, [], {}

        # 如果有资金流向数据，同时获取并注入到 data DataFrame
        if hasattr(self.data_source, 'get_fund_flow'):
            try:
                fund_df = self.data_source.get_fund_flow(symbol, end - timedelta(days=_PARAMS.LOOKBACK_DAYS), end)
                if fund_df is not None and not fund_df.empty:
                    # 重命名 date 列以便 merge
                    if 'date' not in fund_df.columns:
                        for col in fund_df.columns:
                            if pd.api.types.is_datetime64_any_dtype(fund_df[col]):
                                fund_df = fund_df.rename(columns={col: 'date'})
                    if 'date' in fund_df.columns:
                        fund_df['date'] = pd.to_datetime(fund_df['date'], errors='coerce')
                        data = data.copy()
                        data['date'] = pd.to_datetime(data['date'], errors='coerce')
                        data = data.merge(fund_df, on='date', how='left', suffixes=('', '_ff'))
                        # 填充 NaN
                        for col in ['主力净流入', '散户净流入', '超大单净流入', '大单净流入',
                                    '中单净流入', '小单净流入', '成交额']:
                            if col in data.columns:
                                data[col] = data[col].fillna(0)
            except Exception:
                pass  # 资金流向获取失败不影响主流程

        conditions = []
        indicators = {}
        met_weight = 0.0        # 满足条件的权重合计
        strength_weight = 0.0   # 权重 × 强度 合计
        total_weight = 0.0      # 全部条件权重合计

        for rule in config.indicators:
            # 纯数据/上下文计算操作符：不需要从注册表获取指标
            if rule.operator in ("volume_ma_ratio_gt", "price_above_ma", "ma5_above_ma20"):
                values = None
                current = 0.0
                prev = 0.0
            elif rule.operator == "rps_rank_gt":
                # 横截面RPS：从约束中查本股票的全市场百分位（引擎每日注入）
                values = None
                prev = 0.0
                rps_map = constraint.rps_rank if constraint is not None else None
                current = rps_map.get(symbol, 0.0) if rps_map else 0.0
            else:
                # 指标类条件：全历史一次计算，逐日切片（大幅加速回测）
                current, prev, values = self._get_indicator_history(
                    symbol, rule.name, rule.operator, end
                )
                if current is None:
                    total_weight += rule.weight
                    continue  # 指标不可用（数据不足等）→ 条件不满足

            indicators[rule.name or rule.operator] = current

            # 评估条件（返回 是否满足 + 满足强度 0-1）
            total_weight += rule.weight
            met, strength = self._check_condition(
                current, prev, rule.operator, rule.threshold, data, values
            )
            if met:
                met_weight += rule.weight
                strength_weight += rule.weight * strength
                conditions.append({
                    'indicator': rule.name or rule.operator,
                    'operator': rule.operator,
                    'threshold': rule.threshold,
                    'value': current,
                    'strength': round(float(strength), 3)
                })

        # 逻辑门：AND 需全部满足，OR 至少一个
        if config.logic == "AND" and met_weight != total_weight:
            return 0, [], {}
        elif config.logic == "OR" and met_weight == 0:
            return 0, [], {}

        # 连续评分：60（刚好通过）+ 40 × 加权平均强度
        # 勉强通过的信号得低分（被 min_score 过滤），远超阈值的强信号接近 100
        avg_strength = strength_weight / total_weight if total_weight > 0 else 0.0
        total_score = int(60 + 40 * avg_strength)
        return total_score, conditions, indicators

    def _to_scalar(self, val):
        """Convert pandas Series/DataFrame value to scalar"""
        if isinstance(val, pd.Series):
            return val.iloc[-1] if len(val) > 0 else 0
        elif isinstance(val, pd.DataFrame):
            return val.iloc[-1, -1] if len(val) > 0 else 0
        return val

    @staticmethod
    def _last_two_values(values, indicator_func):
        """
        取指标序列的最新两个值（current, prev）

        DataFrame 型指标优先读注册时声明的主列（primary_column），
        未声明时回退到最后一列。
        """
        if isinstance(values, pd.DataFrame):
            col = getattr(indicator_func, '_primary_column', None)
            if col and col in values.columns:
                series = values[col]
            else:
                series = values.iloc[:, -1]
            current = series.iloc[-1] if len(series) > 0 else 0.0
            prev = series.iloc[-2] if len(series) > 1 else current
            return current, prev
        current = values.iloc[-1] if len(values) > 0 else 0.0
        prev = values.iloc[-2] if len(values) > 1 else current
        return current, prev

    def _check_condition(self, current: float, prev: float, operator: str, threshold,
                         data: pd.DataFrame = None, indicator_values=None) -> tuple:
        """
        检查条件是否满足，并评估满足强度

        Args:
            current: 当前值
            prev: 前一个值
            operator: 操作符
            threshold: 阈值
            data: 原始K线数据（用于break操作符计算布林带）
            indicator_values: 指标原始DataFrame（用于DMI等复合指标）

        Returns:
            (met: bool, strength: float 0-1)
            strength 表示满足程度：勉强通过 ≈ 0，远超阈值 ≈ 1
        """
        current = self._to_scalar(current)
        prev = self._to_scalar(prev)

        if operator == ">":
            return current > threshold, self._distance_strength(current, threshold)
        elif operator == "<":
            return current < threshold, self._distance_strength(current, threshold)
        elif operator == ">=":
            return current >= threshold, self._distance_strength(current, threshold)
        elif operator == "<=":
            return current <= threshold, self._distance_strength(current, threshold)
        elif operator == "cross_up":
            met = prev < threshold < current
            return met, self._distance_strength(current, threshold) if met else 0.0
        elif operator == "cross_down":
            met = prev > threshold > current
            return met, self._distance_strength(current, threshold) if met else 0.0
        elif operator == "break_upper":
            # 价格突破布林带上轨
            if data is None:
                return False, 0.0
            upper = data['close'].rolling(20).mean() + 2 * data['close'].rolling(20).std()
            band = upper.iloc[-1]
            met = current > band and not pd.isna(band)
            return met, min(1.0, abs(current - band) / band) if met else 0.0
        elif operator == "break_lower":
            # 价格突破布林带下轨
            if data is None:
                return False, 0.0
            lower = data['close'].rolling(20).mean() - 2 * data['close'].rolling(20).std()
            band = lower.iloc[-1]
            met = current < band and not pd.isna(band)
            return met, min(1.0, abs(current - band) / band) if met else 0.0
        elif operator == "dmi_golden_cross":
            # DMI 金叉: +DI 上穿 -DI
            result = self._dmi_cross(indicator_values)
            if result is None:
                return False, 0.0
            met, di_spread = result
            return met, min(1.0, abs(di_spread) / 100.0) if met else 0.0
        elif operator == "dmi_death_cross":
            # DMI 死叉: -DI 上穿 +DI
            result = self._dmi_cross(indicator_values, death=True)
            if result is None:
                return False, 0.0
            met, di_spread = result
            return met, min(1.0, abs(di_spread) / 100.0) if met else 0.0
        elif operator == "adx_gt":
            # ADX > threshold: 趋势强度足够（过滤震荡行情）
            # NaN 视为不满足（无有效趋势数据）
            if not isinstance(indicator_values, pd.DataFrame) or 'adx' not in indicator_values.columns:
                return False, 0.0
            adx_val = indicator_values['adx'].iloc[-1]
            if adx_val != adx_val:  # NaN check
                return False, 0.0
            met = adx_val > threshold
            return met, min(1.0, (adx_val - threshold) / _PARAMS.STRENGTH_ADX_EXCESS) if met else 0.0
        elif operator == "rps_rank_gt":
            # 横截面RPS > threshold: 强于全市场大部分股票（欧奈尔相对强度）
            met = current > threshold
            return met, min(1.0, (current - threshold) / _PARAMS.STRENGTH_RPS_EXCESS) if met else 0.0
        elif operator == "volume_ma_ratio_gt":
            # 成交量/MA20 > threshold: 放量确认
            if data is None:
                return False, 0.0
            vol = data['volume'].iloc[-1]
            vol_ma20 = data['volume'].iloc[-20:].mean() if len(data) >= 20 else data['volume'].mean()
            if vol_ma20 <= 0:
                return False, 0.0
            ratio = vol / vol_ma20
            met = ratio > threshold
            return met, min(1.0, ratio - threshold) if met else 0.0
        elif operator == "price_above_ma":
            # 收盘价站稳MA threshold日均线上方
            if data is None:
                return False, 0.0
            period = int(threshold)
            ma = data['close'].rolling(period).mean()
            if pd.isna(ma.iloc[-1]) or ma.iloc[-1] <= 0:
                return False, 0.0
            premium = (data['close'].iloc[-1] - ma.iloc[-1]) / ma.iloc[-1]
            met = premium > 0
            return met, min(1.0, premium / _PARAMS.STRENGTH_MA_PREMIUM) if met else 0.0
        elif operator == "ma5_above_ma20":
            # MA5 > MA20: 短期均线在长期均线上方（均线多头排列）
            if data is None:
                return False, 0.0
            ma5 = data['close'].rolling(5).mean()
            ma20 = data['close'].rolling(20).mean()
            if pd.isna(ma5.iloc[-1]) or pd.isna(ma20.iloc[-1]) or ma20.iloc[-1] <= 0:
                return False, 0.0
            spread = (ma5.iloc[-1] - ma20.iloc[-1]) / ma20.iloc[-1]
            met = spread > 0
            return met, min(1.0, spread / _PARAMS.STRENGTH_MA_CROSS_SPREAD) if met else 0.0
        return False, 0.0

    @staticmethod
    def _distance_strength(current: float, threshold: float) -> float:
        """
        相对距离强度（尺度无关）：当前值越远离阈值，强度越高

        |current - threshold| / (|current| + |threshold|)，结果截断到 [0, 1]
        """
        denom = abs(current) + abs(threshold) + 1e-9
        return min(1.0, abs(current - threshold) / denom)

    @staticmethod
    def _dmi_cross(indicator_values, death: bool = False):
        """
        DMI 交叉判断

        Returns:
            (met: bool, di_spread: +DI 与 -DI 之差) 或 None（数据不足）
        """
        if not isinstance(indicator_values, pd.DataFrame):
            return None
        if 'plus_di' not in indicator_values.columns or 'minus_di' not in indicator_values.columns:
            return None
        plus_di = indicator_values['plus_di'].iloc[-1]
        minus_di = indicator_values['minus_di'].iloc[-1]
        plus_di_prev = indicator_values['plus_di'].iloc[-2] if len(indicator_values) > 1 else plus_di
        minus_di_prev = indicator_values['minus_di'].iloc[-2] if len(indicator_values) > 1 else minus_di
        if death:
            met = minus_di_prev <= plus_di_prev and minus_di > plus_di
        else:
            met = plus_di_prev <= minus_di_prev and plus_di > minus_di
        return met, plus_di - minus_di


def load_strategy_from_yaml(strategy_name: str, yaml_path: str) -> SelectionConfig:
    """
    从YAML文件加载选股策略配置

    Args:
        strategy_name: 策略名称
        yaml_path: YAML文件路径

    Returns:
        SelectionConfig对象

    Raises:
        FileNotFoundError: YAML文件不存在
        ValueError: 策略不存在或配置格式错误
    """
    import yaml

    with open(yaml_path, 'r', encoding='utf-8') as f:
        strategies = yaml.safe_load(f)

    if strategy_name not in strategies:
        available = list(strategies.keys())
        raise ValueError(f"Strategy '{strategy_name}' not found. Available: {available}")

    config_dict = strategies[strategy_name]

    # 解析indicators
    indicator_rules = []
    for rule_dict in config_dict.get('indicators', []):
        rule = IndicatorRule(
            name=rule_dict['name'],
            operator=rule_dict['operator'],
            threshold=rule_dict['threshold'],
            weight=rule_dict.get('weight', 1.0)
        )
        indicator_rules.append(rule)

    return SelectionConfig(
        name=config_dict['name'],
        indicators=indicator_rules,
        logic=config_dict.get('logic', 'AND'),
        min_score=config_dict.get('min_score', 60),
        max_positions=config_dict.get('max_positions', 5),
        sw_industry_filter=config_dict.get('sw_industry_filter', False),
        sw_industry_top_n=config_dict.get('sw_industry_top_n', 10)
    )
