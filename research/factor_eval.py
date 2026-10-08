#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
因子评价模块 —— IC / RankIC / 分层回测 / 换手率

对齐专业量化研究流程：任何指标（因子）在进入策略前，先测量它的预测力。
核心输出：
- IC / RankIC 时间序列及其均值、IR（信息比率 = 均值/标准差）
- 分位数分层收益（Q1 最弱 → QN 最强），多空年化收益
- 分组换手率（衡量因子的交易成本敏感度）

时点正确性：因子值由指标函数在"截至 t 日"的数据上计算
（rolling/ewm 类指标天然只依赖历史），收益为 t → t+period 的前瞻收益。
"""

from dataclasses import dataclass, field
from typing import Dict, Callable, Optional

import numpy as np
import pandas as pd

from risk.constants import BacktestParams
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class FactorEvalResult:
    """因子评价结果"""
    name: str
    period: int                    # 前瞻收益期数（交易日）
    n_dates: int                   # 有效截面数
    n_symbols: int                 # 平均截面股票数

    ic_mean: float                 # IC 均数
    ic_std: float
    ic_ir: float                   # IC / IC_std
    rankic_mean: float             # RankIC（Spearman）均值
    rankic_ir: float
    ic_positive_rate: float        # IC > 0 的比例

    long_short_annual: float       # 多空年化收益（Top分位 - Bottom分位）
    quantile_mean_ret: Dict[int, float]  # 各分位平均期收益（0=最弱）
    turnover: float                # 多头组换手率（每次调仓平均换手比例）

    ic_series: pd.Series = None    # IC 时间序列
    quantile_cum: pd.DataFrame = None    # 各分位累计收益曲线

    def summary_markdown(self) -> str:
        lines = [f"## 因子评价: {self.name}\n"]
        lines.append(f"- 前瞻期: {self.period} 交易日 | 截面数: {self.n_dates} | "
                     f"平均股票数: {self.n_symbols}")
        lines.append(f"- **IC**: 均值 {self.ic_mean:+.4f}（IR={self.ic_ir:+.2f}，"
                     f"正向率 {self.ic_positive_rate:.0%}）")
        lines.append(f"- **RankIC**: 均值 {self.rankic_mean:+.4f}（IR={self.rankic_ir:+.2f}）")
        lines.append(f"- **多空年化**: {self.long_short_annual:+.2%} | 多头组换手率: {self.turnover:.1%}/期")
        lines.append("\n| 分位 | 平均期收益 | 年化 |")
        lines.append("|------|-----------|------|")
        for q, ret in sorted(self.quantile_mean_ret.items()):
            annual = ret * BacktestParams.TRADING_DAYS_PER_YEAR / self.period
            label = f"Q{q + 1}" + ("（最弱）" if q == 0 else "（最强）" if q == max(self.quantile_mean_ret) else "")
            lines.append(f"| {label} | {ret:+.4%} | {annual:+.2%} |")
        # 单调性检验：分层收益应随分位单调（专业因子的重要特征）
        rets = [self.quantile_mean_ret[q] for q in sorted(self.quantile_mean_ret)]
        monotonic = all(rets[i] <= rets[i + 1] for i in range(len(rets) - 1)) or \
            all(rets[i] >= rets[i + 1] for i in range(len(rets) - 1))
        lines.append(f"\n- 分层单调性: {'单调' if monotonic else '不单调（因子有效性存疑）'}")
        return "\n".join(lines)


def build_factor_matrix(cache: Dict[str, pd.DataFrame], factor_func: Callable) -> pd.DataFrame:
    """
    构建因子截面矩阵（dates × symbols）

    每只股票的指标在其完整历史K线上计算一次（rolling/ewm 只依赖历史，
    时点正确），再按日期对齐到统一矩阵。
    DataFrame 型指标自动取注册时声明的 primary_column。
    """
    data = {}
    for sym, df in cache.items():
        if df is None or df.empty:
            continue
        try:
            vals = factor_func(df)
            if isinstance(vals, pd.DataFrame):
                col = getattr(factor_func, '_primary_column', None)
                vals = vals[col] if col and col in vals.columns else vals.iloc[:, -1]
            vals = pd.Series(vals)
            vals = vals.astype(float)
            # 对齐日期索引
            if not isinstance(vals.index, pd.DatetimeIndex):
                if len(vals) == len(df):
                    vals.index = pd.to_datetime(df['date'])
                else:
                    continue
            # 统一时间分辨率（pandas 3 中 us/ns 混用会导致跨索引 .loc 失败）
            vals.index = pd.DatetimeIndex(vals.index).as_unit('ns')
            data[sym] = vals
        except Exception as e:
            logger.debug(f"[build_factor_matrix] {sym}: {e}")
            continue
    return pd.DataFrame(data)


class FactorEvaluator:
    """
    因子评价器

    用法:
        ev = FactorEvaluator(cache)
        result = ev.evaluate(registry.get('rsi'), name='RSI14', period=5)
        print(result.summary_markdown())
    """

    def __init__(self, cache: Dict[str, pd.DataFrame]):
        self.cache = cache
        close = {}
        for sym, df in cache.items():
            if df is not None and not df.empty:
                s = pd.to_numeric(df.set_index('date')['close'], errors='coerce')
                s.index = pd.DatetimeIndex(s.index).as_unit('ns')
                close[sym] = s
        self.close_mx = pd.DataFrame(close).sort_index()

    def evaluate(self, factor_func: Callable, name: str = '',
                 period: int = 5, quantiles: int = 5,
                 start: str = None, end: str = None) -> Optional[FactorEvalResult]:
        """
        评价函数型因子（指标在K线上计算）

        name 默认取函数注册名；其余见 evaluate_matrix。
        """
        name = name or getattr(factor_func, '_indicator_name', getattr(factor_func, '__name__', 'factor'))
        factor_mx = build_factor_matrix(self.cache, factor_func)
        return self.evaluate_matrix(factor_mx, name=name, period=period,
                                    quantiles=quantiles, start=start, end=end)

    def evaluate_matrix(self, factor_mx: pd.DataFrame, name: str = '',
                        period: int = 5, quantiles: int = 5,
                        start: str = None, end: str = None) -> Optional[FactorEvalResult]:
        """
        评价矩阵型因子（dates × symbols；基本面因子等外部矩阵直接接入）

        Args:
            factor_mx: 因子截面矩阵（时点已保证 ≤t）
            name: 因子名
            period: 前瞻收益期数（交易日）
            quantiles: 分层数
            start/end: 评价区间（yyyy-MM-dd，可选）

        Returns:
            FactorEvalResult 或 None（数据不足）
        """
        if factor_mx is None or factor_mx.empty:
            logger.warning(f"[evaluate_matrix] empty matrix for {name}")
            return None

        # 前瞻收益：t → t+period（shift(-period)，仅评价用，不进入交易）
        fwd_ret = self.close_mx.shift(-period) / self.close_mx - 1.0

        # 对齐日期与股票
        common_dates = factor_mx.index.intersection(fwd_ret.index)
        if start:
            common_dates = common_dates[common_dates >= pd.Timestamp(start)]
        if end:
            common_dates = common_dates[common_dates <= pd.Timestamp(end)]
        if len(common_dates) < 20:
            logger.warning(f"[evaluate] insufficient dates for {name}: {len(common_dates)}")
            return None

        ic_list, rankic_list = [], []
        quantile_rets = [[] for _ in range(quantiles)]
        top_members_prev, turnover_list = None, []

        for t in common_dates:
            fv = factor_mx.loc[t].dropna()
            rv = fwd_ret.loc[t]
            common = fv.index.intersection(rv.dropna().index)
            if len(common) < max(30, quantiles * 5):
                continue

            fv, rv = fv[common], rv[common]

            # IC / RankIC
            if fv.std() > 0 and rv.std() > 0:
                ic_list.append((t, np.corrcoef(fv.values, rv.values)[0, 1]))
                rankic_list.append((t, fv.rank().corr(rv.rank(), method='spearman')))

            # 分层：按因子值分位（0=最弱, quantiles-1=最强）
            try:
                q = pd.qcut(fv.rank(method='first'), quantiles, labels=False)
            except ValueError:
                continue
            for g in range(quantiles):
                members = q[q == g].index
                quantile_rets[g].append(float(rv[members].mean()))

            # 多头组（最强分位）换手率
            top_now = set(q[q == quantiles - 1].index)
            if top_members_prev is not None and top_members_prev:
                overlap = len(top_now & top_members_prev) / len(top_members_prev | top_now)
                turnover_list.append(1.0 - overlap)
            top_members_prev = top_now

        if len(ic_list) < 10:
            logger.warning(f"[evaluate] insufficient IC samples for {name}")
            return None

        ic_series = pd.Series(dict(ic_list)).sort_index()
        rankic_series = pd.Series(dict(rankic_list)).sort_index()
        n_symbols = int(factor_mx.notna().sum(axis=1).loc[common_dates].mean())

        # 多空收益（Top - Bottom 各期均值 → 年化）
        top_ret = np.mean(quantile_rets[quantiles - 1])
        bottom_ret = np.mean(quantile_rets[0])
        ls_annual = (top_ret - bottom_ret) * BacktestParams.TRADING_DAYS_PER_YEAR / period

        return FactorEvalResult(
            name=name,
            period=period,
            n_dates=len(ic_series),
            n_symbols=n_symbols,
            ic_mean=float(ic_series.mean()),
            ic_std=float(ic_series.std()),
            ic_ir=float(ic_series.mean() / ic_series.std()) if ic_series.std() > 0 else 0.0,
            rankic_mean=float(rankic_series.mean()),
            rankic_ir=float(rankic_series.mean() / rankic_series.std()) if rankic_series.std() > 0 else 0.0,
            ic_positive_rate=float((ic_series > 0).mean()),
            long_short_annual=float(ls_annual),
            quantile_mean_ret={g: float(np.mean(r)) for g, r in enumerate(quantile_rets) if len(r) > 0},
            turnover=float(np.mean(turnover_list)) if turnover_list else 0.0,
            ic_series=ic_series,
        )
