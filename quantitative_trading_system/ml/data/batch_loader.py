#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分批数据加载器 - 从Wind数据库分批加载股票数据，控制内存使用
"""

import logging
import gc
import random
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional, Callable
from dataclasses import dataclass

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class BatchLoadResult:
    """批次加载结果"""
    symbol_data: Dict[str, pd.DataFrame]  # {symbol: price_data}
    loaded_count: int                      # 成功加载的股票数
    failed_symbols: List[str]              # 加载失败的股票
    skipped_symbols: List[str]              # 跳过的股票（数据不足）


class WindBatchLoader:
    """Wind数据库分批加载器"""

    def __init__(
        self,
        connection_params: dict = None,
        days: int = 1250,  # 5年约1250交易日
        min_days: int = 100,  # 最少需要的数据天数
        sequence_length: int = 60,
    ):
        self.connection_params = connection_params or {
            'user': 'wind',
            'password': 'windPrd22',
            'dsn': '10.1.33.123:1521/info'
        }
        self.days = days
        self.min_days = min_days
        self.sequence_length = sequence_length

    def _get_connection(self):
        """获取数据库连接"""
        try:
            import oracledb
            return oracledb.connect(**self.connection_params)
        except ImportError:
            import cx_Oracle as oracledb
            dsn = oracledb.makedsn(
                self.connection_params['dsn'].split(':')[0],
                1521,
                service_name='info'
            )
            return oracledb.connect(
                self.connection_params['user'],
                self.connection_params['password'],
                dsn=dsn
            )

    def get_all_stocks(self, exclude_codes: Tuple[str, ...] = ('88%', '89%', '87%')) -> List[str]:
        """获取全量A股列表"""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()

            where_clause = " AND ".join([f"S_INFO_WINDCODE NOT LIKE '{c}'" for c in exclude_codes])

            cursor.execute(f"""
                SELECT DISTINCT S_INFO_WINDCODE
                FROM ASHAREEODPRICES
                WHERE S_INFO_WINDCODE IS NOT NULL
                AND (S_INFO_WINDCODE LIKE '%.SH' OR S_INFO_WINDCODE LIKE '%.SZ')
                AND {where_clause}
                ORDER BY S_INFO_WINDCODE
            """)

            stocks = [row[0] for row in cursor.fetchall()]
            cursor.close()
            logger.info(f"从Wind数据库获取到 {len(stocks)} 只股票")
            return stocks
        finally:
            conn.close()

    def load_stock_data(self, conn, symbol: str) -> Optional[pd.DataFrame]:
        """加载单只股票数据"""
        end_date = datetime.now().strftime('%Y%m%d')
        start_date = (datetime.now() - timedelta(days=self.days * 2)).strftime('%Y%m%d')

        try:
            with conn.cursor() as cursor:
                sql = f"""
                    SELECT TRADE_DT, S_DQ_OPEN, S_DQ_HIGH, S_DQ_LOW, S_DQ_CLOSE, S_DQ_VOLUME
                    FROM ASHAREEODPRICES
                    WHERE S_INFO_WINDCODE = '{symbol}'
                    AND TRADE_DT >= '{start_date}'
                    AND TRADE_DT <= '{end_date}'
                    AND S_DQ_CLOSE IS NOT NULL
                    AND S_DQ_OPEN IS NOT NULL
                    AND S_DQ_HIGH IS NOT NULL
                    AND S_DQ_LOW IS NOT NULL
                    AND S_DQ_VOLUME IS NOT NULL
                    ORDER BY TRADE_DT ASC
                """
                cursor.execute(sql)
                rows = cursor.fetchall()
        except Exception as e:
            logger.warning(f"Failed to load stock data for {symbol}: {e}")
            return None

        if not rows or len(rows) < self.min_days:
            return None

        df = pd.DataFrame(rows, columns=['date', 'open', 'high', 'low', 'close', 'volume'])
        df['date'] = pd.to_datetime(df['date'].astype(str), format='%Y%m%d')
        df = df.set_index('date')

        # 只保留最近5年数据
        df = df.tail(self.days)

        return df[['open', 'high', 'low', 'close', 'volume']]

    def load_batch(
        self,
        symbols: List[str],
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> BatchLoadResult:
        """加载一批股票数据"""
        conn = self._get_connection()
        symbol_data = {}
        failed_symbols = []
        skipped_symbols = []

        try:
            for i, symbol in enumerate(symbols):
                if progress_callback and i % 50 == 0:
                    progress_callback(i, len(symbols))

                df = self.load_stock_data(conn, symbol)
                if df is None:
                    skipped_symbols.append(symbol)
                else:
                    symbol_data[symbol] = df

            if progress_callback:
                progress_callback(len(symbols), len(symbols))

            logger.info(f"批次加载完成: 成功={len(symbol_data)}, 跳过={len(skipped_symbols)}, 失败={len(failed_symbols)}")

            return BatchLoadResult(
                symbol_data=symbol_data,
                loaded_count=len(symbol_data),
                failed_symbols=failed_symbols,
                skipped_symbols=skipped_symbols
            )
        finally:
            conn.close()

    def create_batches(
        self,
        stocks: List[str],
        batch_size: int,
        shuffle: bool = True,
        seed: int = 42
    ) -> List[List[str]]:
        """将股票列表分成批次"""
        if shuffle:
            rng = random.Random(seed)
            stocks = stocks.copy()
            rng.shuffle(stocks)

        batches = []
        for i in range(0, len(stocks), batch_size):
            batches.append(stocks[i:i + batch_size])

        return batches

    def load_and_release(
        self,
        symbols: List[str],
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> BatchLoadResult:
        """
        加载一批股票数据（包装方法，确保内存管理）
        训练完成后调用方应释放返回值
        """
        result = self.load_batch(symbols, progress_callback)
        return result
