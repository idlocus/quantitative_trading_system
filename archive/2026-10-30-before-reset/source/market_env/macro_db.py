#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
宏观数据库 — 通过 akshare 获取多维度宏观指标

五大维度：增长、流动性、情绪、外生、领先
共 27 个指标，覆盖商业周期的主要维度

数据源：AkShareMarketEnvProvider（akshare 免费数据）
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import pandas as pd

from utils.logger import get_logger

logger = get_logger(__name__)


# =============================================================================
# 指标定义 — 27 个宏观指标 (与 akshare_provider 保持一致)
# =============================================================================

INDICATORS: List[dict] = [
    # ---- 增长维度 ----
    {"id": "中国GDP同比",                     "cat": "增长", "freq": "Q", "lead": 0},
    {"id": "中国制造业PMI",                   "cat": "增长", "freq": "M", "lead": 1},
    {"id": "中国非制造业PMI",                 "cat": "增长", "freq": "M", "lead": 1},
    {"id": "中国CPI同比",                    "cat": "增长", "freq": "M", "lead": 0},
    {"id": "中国PPI同比",                    "cat": "增长", "freq": "M", "lead": 2},
    {"id": "中国社会消费品零售总额同比",         "cat": "增长", "freq": "M", "lead": 1},
    {"id": "中国固定资产投资完成额同比",        "cat": "增长", "freq": "M", "lead": 2},
    {"id": "中国发电量同比",                  "cat": "增长", "freq": "M", "lead": 1},
    {"id": "中国工业企业利润总额累计同比",       "cat": "增长", "freq": "M", "lead": 3},
    # ---- 流动性维度 ----
    {"id": "中国M2同比",                    "cat": "流动", "freq": "M", "lead": 2},
    {"id": "中国社会融资规模存量同比",           "cat": "流动", "freq": "M", "lead": 3},
    {"id": "银行间质押式回购加权利率1周",        "cat": "流动", "freq": "D", "lead": 0},
    {"id": "中国10年期国债收益率",             "cat": "流动", "freq": "D", "lead": 0},
    {"id": "贷款市场报价利率(LPR)1年",          "cat": "流动", "freq": "M", "lead": 0},
    # ---- 情绪维度 ----
    {"id": "A股融资融券余额",                 "cat": "情绪", "freq": "D", "lead": 1},
    {"id": "沪深股通北向资金净流入",             "cat": "情绪", "freq": "D", "lead": 0},
    {"id": "新增投资者数量",                   "cat": "情绪", "freq": "M", "lead": 1},
    {"id": "股票型基金发行份额",                "cat": "情绪", "freq": "M", "lead": 2},
    # ---- 外生维度 ----
    {"id": "美国联邦基金利率",                "cat": "外生", "freq": "M", "lead": 0},
    {"id": "美国CPI同比",                    "cat": "外生", "freq": "M", "lead": 0},
    {"id": "CBOE波动率指数VIX",              "cat": "外生", "freq": "D", "lead": 0},
    {"id": "美元指数",                       "cat": "外生", "freq": "D", "lead": 1},
    {"id": "CRB商品指数",                    "cat": "外生", "freq": "D", "lead": 1},
    # ---- 领先指标 ----
    {"id": "中国社会融资规模存量同比",           "cat": "领先", "freq": "M", "lead": 3},
    {"id": "中国M1同比",                    "cat": "领先", "freq": "M", "lead": 2},
    {"id": "出口集装箱运价指数",               "cat": "领先", "freq": "M", "lead": 3},
    {"id": "螺纹钢价格",                      "cat": "领先", "freq": "D", "lead": 2},
    {"id": "70个大中城市住宅销售价格指数",        "cat": "领先", "freq": "M", "lead": 4},
]

# 指标类别标签（中文）
CAT_LABELS = {
    "增长": "增长",
    "流动": "流动性",
    "情绪": "情绪",
    "外生": "外生",
    "领先": "领先",
}


# =============================================================================
# 宏观数据库
# =============================================================================

class MacroDB:
    """
    宏观数据库：通过 AkShareMarketEnvProvider 获取多维度宏观指标。

    支持：
    - 单指标查询（自动缓存）
    - 批量获取（用于评分）
    - 领先指标扩散指数
    - 先行合成指数（社融+M2+PMI）
    """

    def __init__(self, akshare_provider):
        """
        Args:
            akshare_provider: AkShareMarketEnvProvider 实例
        """
        self.provider = akshare_provider
        self._cache: Dict[str, pd.Series] = {}

    def fetch(self, indicator_id: str,
              begin: str = None, end: str = None,
              freq: str = "M") -> pd.Series:
        """
        获取单个宏观指标时间序列。

        Args:
            indicator_id: 指标名称
            begin: 开始日期 yyyyMMdd
            end: 结束日期 yyyyMMdd
            freq: 数据频率 M/D/Q

        Returns:
            pd.Series，index 为 datetime，value 为 float
        """
        key = f"{indicator_id}_{begin}_{end}_{freq}"
        if key in self._cache:
            return self._cache[key]

        try:
            series = self.provider.fetch_indicator(indicator_id, begin=begin, end=end, freq=freq)
            self._cache[key] = series
            return series
        except Exception as e:
            logger.warning(f"MacroDB fetch failed for '{indicator_id}': {e}")
            return pd.Series(dtype=float)

    def fetch_all(self, end: str = None, lookback: int = 60) -> Dict[str, pd.Series]:
        """
        批量获取所有指标（用于每日评分）。

        只取月度频率指标以保证可比性。
        """
        begin = _offset_months(end, -lookback)
        results = {}
        for ind in INDICATORS:
            freq = ind["freq"]
            try:
                s = self.fetch(ind["id"], begin=begin, end=end, freq=freq)
                if len(s) > 0:
                    results[ind["id"]] = s
            except Exception as e:
                logger.debug(f"fetch_all skip '{ind['id']}': {e}")
                continue
        return results

    def get_leading_score(self, end: str = None) -> tuple[float, str]:
        """
        领先指标扩散指数。

        取所有 lead > 0 的指标，比较近月 vs 历史均值，
        计算扩张指标占比 → 0-100 扩散分 + 信号。
        """
        begin = _offset_months(end, -12)
        indicators = [ind for ind in INDICATORS if ind["lead"] > 0]

        expanding = 0
        total = 0

        for ind in indicators:
            s = self.fetch(ind["id"], begin=begin, end=end, freq=ind["freq"])
            if len(s) < 3:
                continue

            # 取最近一期的值，与历史均值比较
            recent = s.iloc[-1]
            hist_mean = s.mean()

            if recent is not None and not pd.isna(recent) and hist_mean != 0:
                total += 1
                if recent > hist_mean:
                    expanding += 1

        if total == 0:
            return 50.0, "neutral"

        ratio = expanding / total
        score = int(ratio * 100)

        if ratio > 0.6:
            signal = "expanding"
        elif ratio < 0.4:
            signal = "contracting"
        else:
            signal = "neutral"

        return score, signal

    def get_macro_score(self, end: str = None) -> tuple[float, str]:
        """
        宏观增长维度综合评分。

        用 PMI 和 GDP 同比作为代理指标：
        - PMI > 50 且趋势向上 → 扩张信号
        - GDP 同比趋势 → 增长/衰退
        """
        begin = _offset_months(end, -12)

        pmi_s = self.fetch("中国制造业PMI", begin=begin, end=end, freq="M")
        gdp_s = self.fetch("中国GDP同比", begin=begin, end=end, freq="Q")
        cpi_s = self.fetch("中国CPI同比", begin=begin, end=end, freq="M")

        score = 50.0
        signal = "neutral"

        # PMI 评分：>50 扩张，<50 收缩
        if len(pmi_s) > 0:
            pmi_recent = pmi_s.iloc[-1]
            pmi_past = pmi_s.iloc[-min(3, len(pmi_s))]
            if pmi_recent > 50 and pmi_recent > pmi_past:
                score += 15
                signal = "expanding"
            elif pmi_recent < 50:
                score -= 15
                signal = "contracting"
            elif pmi_recent > pmi_past:
                score += 5

        # GDP 趋势
        if len(gdp_s) >= 2:
            gdp_recent = gdp_s.iloc[-1]
            gdp_past = gdp_s.iloc[-2]
            if gdp_recent > gdp_past:
                score += 10
            else:
                score -= 10

        # CPI：过高或过低都偏负面
        if len(cpi_s) > 0:
            cpi = cpi_s.iloc[-1]
            if 1.0 <= cpi <= 3.0:
                score += 5   # 温和通胀，健康
            elif cpi < 0:
                score -= 10  # 通缩压力

        score = max(0, min(100, int(score)))
        return score, signal

    def get_syzm(self, end: str = None) -> tuple[float, str]:
        """
        商业周期先行合成指数。

        用社融存量同比 + M2同比 + PMI 三个指标合成，
        比较近 3 个月均值 vs 前 3 个月均值，
        变化率 > 2% → 扩张，< -2% → 收缩。
        """
        begin = _offset_months(end, -15)
        ind_ids = [
            "中国社会融资规模存量同比",
            "中国M2同比",
            "中国制造业PMI",
        ]

        aligned_series = []
        for ind_id in ind_ids:
            s = self.fetch(ind_id, begin=begin, end=end, freq="M")
            if len(s) >= 3:
                aligned_series.append(s)

        if not aligned_series:
            return 50.0, "neutral"

        # 合并取均值
        combined = pd.concat(aligned_series, axis=1).mean(axis=1).dropna()
        if len(combined) < 6:
            return 50.0, "neutral"

        # 近 3 个月均值 vs 前 3 个月均值
        recent_3m = combined.iloc[-3:].mean()
        past_3m = combined.iloc[-6:-3].mean()

        if past_3m == 0:
            return 50.0, "neutral"

        change = recent_3m / past_3m - 1

        if change > 0.02:
            return 65, "expanding"
        elif change < -0.02:
            return 35, "contracting"
        else:
            return 50, "neutral"

    def get_current_values(self, end: str = None) -> Dict[str, float]:
        """
        获取所有指标的最新值（用于诊断面板）。
        """
        result = {}
        for ind in INDICATORS:
            try:
                s = self.fetch(ind["id"], end=end, freq=ind["freq"])
                if len(s) > 0:
                    result[ind["id"]] = float(s.iloc[-1])
            except Exception:
                continue
        return result


# =============================================================================
# 工具函数
# =============================================================================

def _offset_months(date_str: str, months: int) -> str:
    """
    将 yyyyMMdd 格式日期往前推 months 个月，返回新的 yyyyMMdd。

    months > 0 = 往前推，months < 0 = 往后推
    """
    if date_str is None:
        return None
    dt = datetime.strptime(date_str, "%Y%m%d")
    # 简化：每月按 30 天估算
    dt = dt + timedelta(days=months * 30)
    return dt.strftime("%Y%m%d")