#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
akshare 市场环境数据实现

基于 akshare 免费数据源提供 IMarketEnvProvider 接口：
- 市场宽度（通过全市场每日涨跌家数计算 A/D ratio）
- 估值分位（通过 A 股历史 PE/PB 估算）
- 宏观指标（27 个宏观指标 via akshare 宏观函数）
"""

from typing import Optional, Dict
import pandas as pd
from datetime import datetime, timedelta

from utils.logger import get_logger
from .base import IMarketEnvProvider

logger = get_logger(__name__)


# =============================================================================
# 27 个宏观指标 → akshare 函数映射
# =============================================================================

# 增长维度
GROWTH = [
    {"id": "中国GDP同比",                    "ak": "macro_china_gdp",                    "col": "国内生产总值-同比增长",            "freq": "Q"},
    {"id": "中国制造业PMI",                  "ak": "macro_china_pmi",                    "col": "制造业-指数",                      "freq": "M"},
    {"id": "中国非制造业PMI",                "ak": "macro_china_pmi",                    "col": "非制造业-指数",                    "freq": "M"},
    {"id": "中国CPI同比",                    "ak": "macro_china_cpi",                    "col": "全国-同比增长",                    "freq": "M"},
    {"id": "中国PPI同比",                    "ak": "macro_china_ppi",                    "col": "当月同比增长",                     "freq": "M"},
    {"id": "中国社会消费品零售总额同比",       "ak": "macro_china_consumer_goods_retail", "col": "同比增长",                        "freq": "M"},
    {"id": "中国固定资产投资完成额同比",        "ak": None,                                "col": None,                               "freq": "M"},  # 需替代
    {"id": "中国发电量同比",                  "ak": None,                                "col": None,                               "freq": "M"},  # 需替代
    {"id": "中国工业企业利润总额累计同比",      "ak": None,                                "col": None,                               "freq": "M"},  # 需替代
]

# 流动性维度
LIQUIDITY = [
    {"id": "中国M2同比",                    "ak": "macro_china_money_supply",            "col": "货币和准货币(M2)-同比增长",        "freq": "M"},
    {"id": "中国社会融资规模存量同比",          "ak": "macro_china_shrzgm",                 "col": None,                               "freq": "M"},  # 用社融增量替代
    {"id": "银行间质押式回购加权利率1周",      "ak": "macro_china_bank_financing",        "col": "最新值",                           "freq": "D"},
    {"id": "中国10年期国债收益率",             "ak": "bond_china_yield",                  "col": "10年",                             "freq": "D"},
    {"id": "贷款市场报价利率(LPR)1年",         "ak": "macro_china_lpr",                   "col": "LPR1Y",                           "freq": "M"},
]

# 情绪维度
SENTIMENT = [
    {"id": "A股融资融券余额",               "ak": "stock_margin_detail_sse",           "col": "融资余额",                         "freq": "D", "agg": "sum"},  # 需要按日汇总
    {"id": "沪深股通北向资金净流入",          "ak": "stock_hsgt_hist_em",               "col": "当日成交净买额",                   "freq": "D"},
    {"id": "新增投资者数量",                  "ak": None,                                "col": None,                               "freq": "M"},  # 需替代
    {"id": "股票型基金发行份额",               "ak": "fund_aum_em",                      "col": None,                               "freq": "M"},  # 基金规模替代（无历史序列）
]

# 外生维度
EXOGENOUS = [
    {"id": "美国联邦基金利率",              "ak": None,                                "col": None,                               "freq": "M"},  # 需替代
    {"id": "美国CPI同比",                  "ak": None,                                "col": None,                               "freq": "M"},  # 需替代
    {"id": "CBOE波动率指数VIX",            "ak": None,                                "col": None,                               "freq": "D"},  # 需替代
    {"id": "美元指数",                      "ak": None,                                "col": None,                               "freq": "D"},  # 需替代
    {"id": "CRB商品指数",                  "ak": "macro_china_commodity_price_index", "col": "最新值",                           "freq": "D"},
]

# 领先维度
LEADING = [
    {"id": "中国社会融资规模存量同比",          "ak": "macro_china_shrzgm",               "col": None,                               "freq": "M"},
    {"id": "中国M1同比",                    "ak": "macro_china_money_supply",           "col": "货币(M1)-同比增长",               "freq": "M"},
    {"id": "出口集装箱运价指数",              "ak": None,                                "col": None,                               "freq": "M"},  # 需替代
    {"id": "螺纹钢价格",                    "ak": None,                                "col": None,                               "freq": "D"},  # 需替代
    {"id": "70个大中城市住宅销售价格指数",      "ak": "macro_china_new_house_price",      "col": "新建商品住宅价格指数-同比",         "freq": "M"},
]

ALL_INDICATORS = GROWTH + LIQUIDITY + SENTIMENT + EXOGENOUS + LEADING


# =============================================================================
# 数据获取器
# =============================================================================

class _AkShareFetcher:
    """封装 akshare 宏观函数调用（全量序列缓存 + 内存时点切片）"""

    def __init__(self):
        import akshare as ak
        self._ak = ak
        self._series_cache: Dict[tuple, pd.Series] = {}  # (ak_func, col, agg) -> 全量序列

    def fetch(self, ak_func: str, col: str = None, begin: str = None, end: str = None,
              freq: str = "M", agg: str = None, date_col_idx: int = None) -> pd.Series:
        """
        调用 akshare 函数并返回日期切片后的 Series

        网络只拉取一次全量序列并缓存；begin/end 过滤在内存中完成，
        保证回测中按历史日期评分时既时点正确又不重复请求。

        Args:
            ak_func: akshare 函数名
            col: 值列名（None=第一列）
            begin/end: 日期过滤（yyyyMMdd）
            freq: 频率（M/D/Q，仅作记录）
            agg: 聚合方式（None=无聚合, "sum"=按日期求和）
            date_col_idx: 日期列索引（默认自动检测）
        """
        if not ak_func:
            return pd.Series(dtype=float)

        series = self._load_full_series(ak_func, col, agg, date_col_idx)
        if series is None or series.empty:
            return pd.Series(dtype=float)

        # 内存时点切片
        if begin:
            begin_dt = _parse_date(begin)
            series = series[series.index >= begin_dt]
        if end:
            end_dt = _parse_date(end)
            series = series[series.index <= end_dt]

        return series

    def _load_full_series(self, ak_func: str, col, agg, date_col_idx: int = None) -> Optional[pd.Series]:
        """拉取并缓存指标的全量序列（同一指标只请求一次网络）"""
        cache_key = (ak_func, col, agg)
        if cache_key in self._series_cache:
            return self._series_cache[cache_key]

        try:
            func = getattr(self._ak, ak_func)
            df = func()
            if df is None or df.empty:
                self._series_cache[cache_key] = pd.Series(dtype=float)
                return self._series_cache[cache_key]

            # 找日期列
            if date_col_idx is not None:
                date_col = df.columns[date_col_idx]
            else:
                date_col = None
                for i, c in enumerate(df.columns):
                    if any(k in c for k in ['日期', '月份', '季度', 'trade_date', 'DATE', 'date', '交易日期']):
                        date_col = c
                        break
                if date_col is None:
                    self._series_cache[cache_key] = pd.Series(dtype=float)
                    return self._series_cache[cache_key]

            # 特殊处理：国债收益率只取"中债国债收益率曲线"
            if 'bond_china_yield' == ak_func and '曲线名称' in df.columns:
                df = df[df['曲线名称'] == '中债国债收益率曲线']

            # 取值列
            val_col = col if (col and col in df.columns) else df.columns[1] if len(df.columns) > 1 else df.columns[0]

            # 聚合（融资融券余额按日汇总全市场）
            if agg == "sum":
                grouped = df.groupby(date_col)[val_col].sum()
                series = grouped
            else:
                df_indexed = df.set_index(date_col)
                series = df_indexed[val_col]

            # 解析日期索引
            series.index = series.index.map(lambda x: _parse_akshare_date(x))
            series = series[series.index.notna()].sort_index()
            series = pd.to_numeric(series, errors='coerce').dropna()

            self._series_cache[cache_key] = series
            return series

        except Exception as e:
            logger.debug(f"[_AkShareFetcher] {ak_func}: {e}")
            self._series_cache[cache_key] = pd.Series(dtype=float)
            return self._series_cache[cache_key]


def _parse_date(s: str) -> datetime:
    """将 yyyyMMdd 或 yyyy-MM-dd 解析为 datetime"""
    if len(s) == 8:
        return datetime.strptime(s[:4] + "-" + s[4:6] + "-" + s[6:], "%Y-%m-%d")
    return pd.to_datetime(s, errors='coerce')


def _parse_akshare_date(date_str) -> datetime:
    """
    解析 akshare 返回的各种日期格式：
    - "2026年05月份" → datetime(2026, 5, 1)
    - "2026年第1季度" → datetime(2026, 3, 31)
    - "2025年第1-4季度" → datetime(2025, 12, 31)
    - "2026-05-29" → datetime(2026, 5, 29)
    - "201501" (int/str) → datetime(2015, 1, 1)  # YYYYMM
    - "20230922" (int/str) → datetime(2023, 9, 22)  # YYYYMMDD
    """
    s = str(date_str)

    # 月份格式: "2026年05月份"
    m = re.match(r'(\d{4})年(\d{1,2})月份?', s)
    if m:
        return datetime(int(m.group(1)), int(m.group(2)), 1)

    # 季度格式: "2026年第1季度" 或 "2025年第1-4季度"
    m = re.match(r'(\d{4})年第1-?\d季度?', s)
    if m:
        year = int(m.group(1))
        return datetime(year, 3, 31)

    m = re.match(r'(\d{4})年第\d-(\d)季度?', s)
    if m:
        year = int(m.group(1))
        end_month = int(m.group(2)) * 3
        return datetime(year, end_month, 1)

    # YYYYMM 格式（社融等）
    if len(s) == 6 and s.isdigit():
        return datetime(int(s[:4]), int(s[4:6]), 1)

    # YYYYMMDD 格式（融资融券等）
    if len(s) == 8 and s.isdigit():
        return datetime(int(s[:4]), int(s[4:6]), int(s[6:8]))

    # 标准格式或 ISO
    return pd.to_datetime(s, errors='coerce')


import re


# =============================================================================
# akshare 市场环境数据提供者
# =============================================================================

class AkShareMarketEnvProvider:
    """
    基于 akshare 的市场环境数据提供方。

    实现 IMarketEnvProvider 接口。
    """

    def __init__(self):
        self._fetcher = _AkShareFetcher()
        self._valuation_cache: Dict[str, dict] = {}

    def get_market_breadth(self, end_date: str) -> Optional[pd.DataFrame]:
        """
        获取市场宽度数据（近30日 A/D ratio）

        通过 stock_board_change_em（无参数，返回全市场板块涨跌数据）
        汇总计算全市场涨跌家数近似值。
        """
        try:
            df = self._fetcher._ak.stock_board_change_em()
            if df is None or df.empty:
                return None

            # 构建近似涨跌家数 DataFrame
            # stock_board_change_em 每行是一个板块（概念/行业），有涨跌幅
            # 用日期分组统计（但这个函数返回的是当前快照，不是时间序列）
            # 需要用历史数据，这里暂时用均值估算
            return None  # 暂时无法获取历史A/D，用None跳过

        except Exception as e:
            logger.warning(f"[get_market_breadth] {e}")
            return None

    def get_market_valuation(self, index_code: str = "000985.SH", date: str = None) -> Optional[dict]:
        """
        获取指数估值分位（时点正确：只使用 ≤ date 的数据）。

        akshare 没有直接的指数历史 PE/PB，用以下方式估算：
        1. 获取中证指数历史估值（全量缓存一次，逐日切片）
        2. 失败时用宏观指标推导估值温度
        """
        try:
            end_dt = _parse_date(date) if date else datetime.now()
            val_df = self._load_csindex_valuation()

            if val_df is not None and not val_df.empty:
                # 5年窗口，时点过滤到 ≤ end_dt（避免使用未来估值）
                start_dt = end_dt - timedelta(days=365*5)
                window = val_df[(val_df.index >= start_dt) & (val_df.index <= end_dt)]
                if not window.empty:
                    pe_col = [c for c in window.columns if 'pe' in c.lower()][0] if any('pe' in c.lower() for c in window.columns) else window.columns[0]
                    pe_vals = pd.to_numeric(window[pe_col], errors='coerce').dropna()
                    latest_pe = pe_vals.iloc[-1]  # end_dt 当日的最新 PE
                    pe_pct = float((pe_vals <= latest_pe).mean() * 100)
                    return {'pe_percentile': pe_pct, 'pb_percentile': 50}

            # 备选：用宏观指标推导估值温度（综合 M2/GDP/PMI）
            pe_pct, pb_pct = self._estimate_valuation_from_macro(end_dt)

            return {
                'pe_percentile': pe_pct,
                'pb_percentile': pb_pct,
            }

        except Exception as e:
            logger.warning(f"[get_market_valuation] {e}")
            return None

    def _load_csindex_valuation(self) -> Optional[pd.DataFrame]:
        """拉取并缓存中证指数估值全量序列（网络只请求一次）"""
        if self._valuation_cache.get('_full_df') is not None:
            return self._valuation_cache['_full_df']
        try:
            val_df = self._fetcher._ak.stock_zh_index_value_csindex(symbol="沪深300")
            if val_df is None or val_df.empty:
                return None
            date_col = [c for c in val_df.columns if 'date' in c.lower() or '日期' in c][0]
            val_df = val_df.set_index(date_col)
            if val_df.index.dtype == 'object':
                val_df.index = pd.to_datetime(val_df.index, errors='coerce')
            val_df = val_df.dropna(how='all').sort_index()
            self._valuation_cache['_full_df'] = val_df
            return val_df
        except Exception as e:
            logger.debug(f"[_load_csindex_valuation] {e}")
            return None

    def _estimate_valuation_from_macro(self, end_dt: datetime) -> tuple:
        """
        用宏观指标估算市场估值分位（0-100）

        PMI > 50 → 经济好 → 估值偏高（分位高）
        M2 增速高 → 流动性宽松 → 估值偏高
        CPI 低 → 估值偏高（防御性强）

        返回 (pe_percentile, pb_percentile)
        """
        try:
            begin_dt = end_dt - timedelta(days=365)

            pmi_s = self._fetcher.fetch("macro_china_pmi", col="制造业-指数", begin=begin_dt.strftime("%Y%m%d"), end=end_dt.strftime("%Y%m%d"), freq="M")
            m2_s = self._fetcher.fetch("macro_china_money_supply", col="货币和准货币(M2)-同比增长", begin=begin_dt.strftime("%Y%m%d"), end=end_dt.strftime("%Y%m%d"), freq="M")
            cpi_s = self._fetcher.fetch("macro_china_cpi", col="全国-同比增长", begin=begin_dt.strftime("%Y%m%d"), end=end_dt.strftime("%Y%m%d"), freq="M")

            scores = []
            if len(pmi_s) > 0:
                # PMI > 50 且趋势向上 → 估值偏高
                recent_pmi = pmi_s.iloc[-1]
                pmi_score = (recent_pmi - 50) / 50 * 100  # 50→0分, 60→20分, 70→40分
                scores.append(pmi_score)

            if len(m2_s) > 0:
                recent_m2 = m2_s.iloc[-1]
                m2_score = (recent_m2 - 8) / 8 * 100  # 8%→0分, 12%→50分, 16%→100分
                scores.append(m2_score)

            if len(cpi_s) > 0:
                recent_cpi = cpi_s.iloc[-1]
                if recent_cpi < 0:  # 通缩
                    cpi_score = 20
                elif recent_cpi < 2:  # 偏低
                    cpi_score = 40
                elif recent_cpi <= 3:  # 温和
                    cpi_score = 60
                else:  # 偏高
                    cpi_score = 80
                scores.append(cpi_score)

            if not scores:
                return 50.0, 50.0

            # 综合评分映射到 PE 分位（经验公式：宏观温度→估值分位）
            composite = sum(scores) / len(scores)
            # 宏观好(PMI高/M2高/CPI温和) → 市场估值偏高 → PE分位高
            pe_percentile = max(10, min(90, 50 + (composite - 40)))
            return float(pe_percentile), float(pe_percentile * 0.9)

        except Exception as e:
            logger.debug(f"[_estimate_valuation_from_macro] {e}")
            return 50.0, 50.0

    def get_macro_indicators(self, end_date: str) -> Optional[Dict[str, float]]:
        """获取当前宏观指标值（用于诊断面板）"""
        result = {}
        for ind in ALL_INDICATORS:
            if not ind.get("ak"):
                continue
            try:
                agg = ind.get("agg")
                s = self._fetcher.fetch(ind["ak"], col=ind.get("col"), end=end_date,
                                        freq=ind["freq"], agg=agg)
                if len(s) > 0:
                    result[ind["id"]] = float(s.iloc[-1])
            except Exception:
                continue
        return result if result else None

    def get_leading_indicators(self, end_date: str) -> Optional[Dict[str, float]]:
        """获取领先指标值"""
        result = {}
        for ind in LEADING:
            if not ind.get("ak"):
                continue
            try:
                agg = ind.get("agg")
                s = self._fetcher.fetch(ind["ak"], col=ind.get("col"), end=end_date,
                                        freq=ind["freq"], agg=agg)
                if len(s) > 0:
                    result[ind["id"]] = float(s.iloc[-1])
            except Exception:
                continue
        return result if result else None

    def fetch_indicator(self, indicator_id: str, begin: str = None, end: str = None, freq: str = "M") -> pd.Series:
        """获取单个宏观指标时间序列（供 MacroDB 使用）"""
        for ind in ALL_INDICATORS:
            if ind["id"] == indicator_id:
                agg = ind.get("agg")
                return self._fetcher.fetch(ind["ak"], col=ind.get("col"), begin=begin, end=end,
                                          freq=freq, agg=agg)
        return pd.Series(dtype=float)