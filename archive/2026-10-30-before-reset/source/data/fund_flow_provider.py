#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
资金流向数据源 — 机构资金（主力）/ 游资（散户）分离

优先级：
1. Wind Oracle AShareMoneyFlow（最稳定，推荐）
2. akshare stock_individual_fund_flow（东方财富，备选）

使用方式：
    provider = FundFlowProvider()
    df = provider.get_fund_flow('300433.SZ', date(2026,1,1), date(2026,6,2))
"""

import os
import pandas as pd
from datetime import date, timedelta
from typing import Optional

import akshare as ak
from utils.logger import get_logger

logger = get_logger(__name__)


class FundFlowProvider:
    """
    个股资金流向提供者

    区分：
    - 主力资金 = 超大单 + 大单资金（机构行为）
    - 游资 = 中单 + 小单资金（散户/投机行为）

    数据源优先级：WindDB > akshare 东财
    """

    _ORACLE_DSN = os.getenv('ORACLE_DSN', '')
    _ORACLE_USER = os.getenv('ORACLE_USER', '')
    _ORACLE_PASSWORD = os.getenv('ORACLE_PASSWORD', '')

    _MARKET_MAP = {
        'sh': '沪股通',
        'sz': '深股通',
        'bj': '北交所',
    }

    def __init__(self):
        self._cache = {}
        self._wind_conn = None

    @staticmethod
    def _parse_symbol(symbol: str) -> tuple[str, str]:
        """解析股票代码，返回 (stock_code, market)"""
        symbol = symbol.upper()
        if symbol.endswith('.SH') or symbol.endswith('.SH'):
            return symbol.replace('.SH', '').replace('.SH', ''), 'sh'
        elif symbol.endswith('.SZ') or symbol.endswith('.SZ'):
            return symbol.replace('.SZ', '').replace('.SZ', ''), 'sz'
        elif symbol.endswith('.BJ'):
            return symbol.replace('.BJ', ''), 'bj'
        else:
            # 默认深交所（A股大多在此）
            return symbol, 'sz'

    def get_fund_flow(self, symbol: str, start: date, end: date) -> pd.DataFrame:
        """
        获取个股资金流向

        Args:
            symbol: 股票代码，如 '300433.SZ' 或 '600519.SH'
            start: 开始日期
            end: 结束日期

        Returns:
            DataFrame with columns:
            date, 主力净流入, 散户净流入, 超大单净流入, 大单净流入,
            中单净流入, 小单净流入, 主力净流入占比, 收盘价, 涨跌幅
            （按 date 排序）
        """
        cache_key = f"{symbol}_{start.isoformat()}_{end.isoformat()}"

        if cache_key in self._cache:
            return self._cache[cache_key].copy()

        # 优先用 Wind Oracle
        wind_df = self._get_wind_money_flow(symbol, start, end)
        if wind_df is not None and not wind_df.empty:
            self._cache[cache_key] = wind_df
            return wind_df.copy()

        # 回退到 akshare 东财
        stock_code, market = self._parse_symbol(symbol)
        import akshare.stock.stock_fund_em as sfe

        try:
            raw = sfe.stock_individual_fund_flow(stock=stock_code, market=market)
            return self._parse_and_normalize(raw, start, end, cache_key)
        except Exception as e:
            logger.warning(f"stock_individual_fund_flow failed for {symbol}: {e}")
            return pd.DataFrame()

    def _get_wind_money_flow(self, symbol: str, start: date, end: date) -> Optional[pd.DataFrame]:
        """从 Wind Oracle AShareMoneyFlow 获取资金流向"""
        try:
            import oracledb
            if self._wind_conn is None:
                oracledb.init_oracle_client()
                self._wind_conn = oracledb.connect(
                    user=self._ORACLE_USER,
                    password=self._ORACLE_PASSWORD,
                    dsn=self._ORACLE_DSN
                )

            sql = """
                SELECT
                    TRADE_DT                          AS dt,
                    VOLUME_DIFF_INSTITUTE              AS inst_vol,
                    VALUE_DIFF_INSTITUTE              AS inst_val,
                    VOLUME_DIFF_LARGE_TRADER           AS large_vol,
                    VALUE_DIFF_LARGE_TRADER            AS large_val,
                    VOLUME_DIFF_SMALL_TRADER           AS small_vol,
                    VALUE_DIFF_SMALL_TRADER            AS small_val,
                    S_MFD_INFLOW                       AS net_flow,
                    NET_INFLOW_RATE_VALUE             AS net_rate
                FROM AShareMoneyFlow
                WHERE S_INFO_WINDCODE = :symbol
                  AND TRADE_DT >= :start_dt
                  AND TRADE_DT <= :end_dt
                ORDER BY TRADE_DT
            """
            cursor = self._wind_conn.cursor()
            cursor.execute(sql, {
                "symbol": symbol,
                "start_dt": start.strftime("%Y%m%d"),
                "end_dt": end.strftime("%Y%m%d"),
            })
            cols = [d[0].lower() for d in cursor.description]
            rows = cursor.fetchall()
            cursor.close()

            if not rows:
                return pd.DataFrame()

            df = pd.DataFrame(rows, columns=cols)
            df['date'] = pd.to_datetime(df['dt'], format="%Y%m%d")
            df.set_index('date', inplace=True)

            # 主力净流入 = 机构净流入 + 大单净流入（成交量）
            # 散户净流入用 SMALL_TRADER
            df['主力净流入'] = df.get('inst_vol', 0) + df.get('large_vol', 0)
            df['散户净流入'] = df.get('small_vol', 0)

            return df.reset_index()
        except Exception as e:
            logger.warning(f"Wind AShareMoneyFlow failed for {symbol}: {e}")
            return None

    def _parse_and_normalize(self, raw: pd.DataFrame, start: date, end: date,
                              cache_key: str) -> pd.DataFrame:
        """解析 akshare 返回的原始 DataFrame"""
        if raw is None or raw.empty:
            logger.debug(f"Empty fund flow data for cache_key={cache_key}")
            return pd.DataFrame()

        # 找到日期列（通常是第一列）
        date_col = raw.columns[0]
        result = raw.copy()

        # 重命名列为英文
        col_map = {}
        for col in result.columns:
            col_lower = col.lower()
            if '日期' in col or 'date' in col_lower:
                col_map[col] = 'date'
            elif '主力' in col and '净流入' in col and '占比' not in col:
                col_map[col] = '主力净流入'
            elif '散户' in col and '净流入' in col and '占比' not in col:
                col_map[col] = '散户净流入'
            elif '超大单' in col and '净流入' in col and '占比' not in col:
                col_map[col] = '超大单净流入'
            elif '大单' in col and '净流入' in col and '占比' not in col:
                col_map[col] = '大单净流入'
            elif '中单' in col and '净流入' in col and '占比' not in col:
                col_map[col] = '中单净流入'
            elif '小单' in col and '净流入' in col and '占比' not in col:
                col_map[col] = '小单净流入'
            elif '主力' in col and '占比' in col:
                col_map[col] = '主力净流入占比'
            elif '收盘' in col:
                col_map[col] = '收盘价'
            elif '涨跌幅' in col:
                col_map[col] = '涨跌幅'
            elif '成交额' in col:
                col_map[col] = '成交额'

        result.rename(columns=col_map, inplace=True)

        # 确保有 date 列
        if 'date' not in result.columns and date_col in result.columns:
            result.rename(columns={date_col: 'date'}, inplace=True)

        # 解析日期
        if 'date' in result.columns:
            result['date'] = pd.to_datetime(result['date'], errors='coerce')
            result.dropna(subset=['date'], inplace=True)
            result.set_index('date', inplace=True)
            # 按日期过滤
            result = result[(result.index >= pd.Timestamp(start)) &
                            (result.index <= pd.Timestamp(end))]
            result = result.reset_index().sort_values('date')
            result.set_index('date', inplace=True)

        # 数值列清理
        for col in result.columns:
            if result[col].dtype == object:
                result[col] = pd.to_numeric(
                    result[col].astype(str).str.replace(',', '').str.replace('%', ''),
                    errors='coerce'
                )

        # 计算 主力净流入 = 超大单 + 大单
        if '主力净流入' not in result.columns:
            if '超大单净流入' in result.columns and '大单净流入' in result.columns:
                result['主力净流入'] = (
                    result['超大单净流入'].fillna(0) + result['大单净流入'].fillna(0)
                )

        # 计算 游资净流入 = 中单 + 小单
        if '散户净流入' not in result.columns:
            if '中单净流入' in result.columns and '小单净流入' in result.columns:
                result['散户净流入'] = (
                    result['中单净流入'].fillna(0) + result['小单净流入'].fillna(0)
                )

        if result.empty:
            return pd.DataFrame()

        self._cache[cache_key] = result
        return result.copy()

    def _get_from_rank_fallback(self, stock_code: str, market: str,
                                 start: date, end: date,
                                 cache_key: str) -> pd.DataFrame:
        """
        通过主力资金流向排行获取单只数据（回退方案）
        获取全市场排行后筛选单只股票
        """
        import akshare.stock.stock_fund_em as sfe

        try:
            # 获取今日排行（只返回今天的数据，用于最新数据点）
            today_rank = sfe.stock_main_fund_flow()
            # 尝试找这只股票
            for col in today_rank.columns:
                if stock_code in str(col) or stock_code in str(today_rank[col].values):
                    pass
        except Exception as e:
            logger.debug(f"Fallback also failed: {e}")

        # 无论如何，返回空（避免报错）
        logger.warning(f"All fund flow sources failed for {stock_code}, returning empty")
        return pd.DataFrame()

    def _compute_fund_flow_signal(self, merged: pd.DataFrame) -> Optional[dict]:
        """
        基于合并后的 DataFrame 计算资金流向信号和结论

        merged 需包含 '主力净流入'(元) 和 '散户净流入'(元) 列
        """
        try:
            inst_col = '主力净流入'
            retail_col = '散户净流入'

            if inst_col not in merged.columns or retail_col not in merged.columns:
                return None

            n = min(5, len(merged))
            inst_vals = merged[inst_col].iloc[-n:].dropna()
            retail_vals = merged[retail_col].iloc[-n:].dropna()

            if len(inst_vals) < 3 or len(retail_vals) < 3:
                return None

            inst_net_5d = inst_vals.sum()
            retail_net_5d = retail_vals.sum()
            total_abs = abs(inst_net_5d) + abs(retail_net_5d) + 1e-10

            inst_pct = inst_net_5d / total_abs * 100
            retail_pct = retail_net_5d / total_abs * 100

            inst_ratio = inst_net_5d / (abs(merged['close'].iloc[-1]) * merged['volume'].iloc[-1] + 1)
            retail_ratio = retail_net_5d / (abs(merged['close'].iloc[-1]) * merged['volume'].iloc[-1] + 1)

            QUIET_THRESHOLD = 3e4

            if abs(inst_net_5d) < QUIET_THRESHOLD and abs(retail_net_5d) < QUIET_THRESHOLD:
                verdict = '资金平衡，无明确方向'
                signal_icon = '⚪'
                signal_color = 'yellow'
            elif inst_net_5d > 0 and retail_net_5d < 0:
                verdict = '机构吸筹，游资离场'
                signal_icon = '🟢'
                signal_color = 'green'
            elif inst_net_5d < 0 and retail_net_5d > 0:
                verdict = '机构派发，游资接盘'
                signal_icon = '🔴'
                signal_color = 'red'
            elif inst_net_5d > 0 and retail_net_5d > 0 and inst_pct > retail_pct:
                verdict = '机构主导，多方占优'
                signal_icon = '🟢'
                signal_color = 'green'
            elif inst_net_5d > 0 and retail_net_5d > 0:
                verdict = '游资活跃，机构参与'
                signal_icon = '🟡'
                signal_color = 'yellow'
            elif inst_net_5d < 0 and retail_net_5d < 0:
                verdict = '资金全面外逃'
                signal_icon = '🔴'
                signal_color = 'red'
            else:
                verdict = '资金平衡'
                signal_icon = '⚪'
                signal_color = 'yellow'

            if inst_net_5d > 0 and inst_pct > 50 and retail_net_5d < 0:
                signal = 'institutional_accumulation'
            elif inst_net_5d > 0 and retail_net_5d < 0:
                signal = 'inst_takes_retail'
            elif inst_net_5d < 0 and retail_net_5d > 0:
                signal = 'inst_distributes'
            elif retail_net_5d > 0 and inst_pct < 30:
                signal = 'retail_hot'
            elif abs(inst_net_5d) < QUIET_THRESHOLD and abs(retail_net_5d) < QUIET_THRESHOLD:
                signal = 'neutral'
            elif inst_net_5d > 0:
                signal = 'institutional_buy'
            else:
                signal = 'neutral'

            return {
                'signal': signal,
                'verdict': verdict,
                'signal_icon': signal_icon,
                'signal_color': signal_color,
                'inst_net_5d': float(inst_net_5d),
                'retail_net_5d': float(retail_net_5d),
                'inst_pct': float(inst_pct),
                'retail_pct': float(retail_pct),
                'inst_ratio': float(inst_ratio),
                'retail_ratio': float(retail_ratio),
            }
        except Exception as e:
            logger.debug(f"_compute_fund_flow_signal error: {e}")
            return None

    def get_latest(self, symbol: str) -> Optional[dict]:
        """
        获取最新一日资金流向数据

        Returns:
            dict with keys: 主力净流入, 散户净流入, 超大单净流入,
            大单净流入, 中单净流入, 小单净流入, 主力净流入占比,
            收盘价, 涨跌幅
        """
        df = self.get_fund_flow(symbol, date.today() - timedelta(days=30), date.today())
        if df is None or df.empty:
            return None
        return df.iloc[-1].to_dict()

    def close(self):
        """关闭Wind Oracle连接"""
        if self._wind_conn is not None:
            self._wind_conn.close()
            self._wind_conn = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class CachedFundFlowSource:
    """
    资金流向数据的缓存封装

    包装 FundFlowProvider，提供 IKlineProvider 风格的接口，
    方便在 scanner 中通过 data_source.get_fund_flow() 调用
    """

    def __init__(self, provider: FundFlowProvider):
        self._provider = provider
        self._cache = {}

    def get_fund_flow(self, symbol: str, start: date, end: date) -> pd.DataFrame:
        """
        获取个股资金流向（带缓存）
        """
        cache_key = f"{symbol}_{start.isoformat()}_{end.isoformat()}"
        if cache_key not in self._cache:
            self._cache[cache_key] = self._provider.get_fund_flow(symbol, start, end)
        return self._cache[cache_key].copy()

    def has_fund_flow(self) -> bool:
        """该数据源是否支持资金流向"""
        return True


def create_fund_flow_provider() -> FundFlowProvider:
    """工厂方法：创建资金流向Provider"""
    return FundFlowProvider()