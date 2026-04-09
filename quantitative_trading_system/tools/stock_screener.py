#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A股智能选股器

基于技术分析框架、基本面筛选、新闻情绪综合筛选最值得投资的股票
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import os
from datetime import datetime
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from indicators.technical_framework import TechnicalInvestmentFramework
from signals import NewsSignalGenerator
import oracledb

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class AStockScreener:
    """A股智能选股器"""

    def __init__(self):
        self.dsn = oracledb.connect(user="wind", password="windPrd22", dsn="10.1.33.123:1521/info")
        self.news_gen = NewsSignalGenerator()
        self.tech_framework = TechnicalInvestmentFramework()

        # 筛选标准
        self.criteria = {
            # 基本面标准
            'min_pe': 0,
            'max_pe': 60,            # PE不超过60
            'min_roe': 3,            # ROE不低于3%
            'max_debt_ratio': 80,    # 资产负债率不超过80%
            'min_gross_margin': 5,   # 毛利率不低于5%

            # 技术面标准
            'min_trend_score': 30,   # 趋势评分不低于30
            'min_rps': 30,           # RPS不低于30

            # 新闻情绪
            'min_sentiment': -0.3,   # 情感评分不低于-0.3
        }

    def get_a_share_list(self, limit=None):
        """获取A股股票列表"""
        logger.info("获取A股股票列表...")

        # 获取沪深A股列表（排除ST、退市等）
        sql = """
        SELECT S_INFO_WINDCODE, S_INFO_NAME
        FROM AShareDescription
        WHERE (S_INFO_WINDCODE LIKE '%.SZ' OR S_INFO_WINDCODE LIKE '%.SH')
        AND S_INFO_NAME NOT LIKE 'ST%'
        AND S_INFO_NAME NOT LIKE '%*ST%'
        AND S_INFO_NAME NOT LIKE '%退市%'
        ORDER BY S_INFO_WINDCODE
        """

        if limit:
            sql += f" FETCH FIRST {limit} ROWS ONLY"

        try:
            df = pd.read_sql(sql, self.dsn)
            logger.info(f"获取到 {len(df)} 只A股")
            return df
        except Exception as e:
            logger.error(f"获取股票列表失败: {e}")
            return pd.DataFrame()

    def get_stock_data(self, symbol):
        """获取单只股票数据"""
        try:
            # 获取日行情
            sql = f"""
            SELECT TRADE_DT, S_DQ_OPEN, S_DQ_HIGH, S_DQ_LOW, S_DQ_CLOSE, S_DQ_VOLUME
            FROM AShareEODPrices
            WHERE S_INFO_WINDCODE = '{symbol}'
            AND TRADE_DT >= '20250101'
            ORDER BY TRADE_DT ASC
            """
            df = pd.read_sql(sql, self.dsn)

            if df.empty or len(df) < 60:
                return None

            df['date'] = pd.to_datetime(df['TRADE_DT'], format='%Y%m%d')
            df = df.rename(columns={
                'S_DQ_OPEN': 'open', 'S_DQ_HIGH': 'high',
                'S_DQ_LOW': 'low', 'S_DQ_CLOSE': 'close',
                'S_DQ_VOLUME': 'volume'
            })
            df = df[['date', 'open', 'high', 'low', 'close', 'volume']].drop_duplicates()
            df.set_index('date', inplace=True)
            return df

        except Exception as e:
            return None

    def get_fundamental_data(self, symbol):
        """获取基本面数据"""
        result = {}

        try:
            # 1. 获取估值指标（PE, PB, PE_TTM）
            valuation_sql = f"""
            SELECT S_VAL_PE, S_VAL_PB_NEW, S_VAL_PE_TTM
            FROM AShareEODDerivativeIndicator
            WHERE S_INFO_WINDCODE = '{symbol}'
            AND S_VAL_PE IS NOT NULL
            ORDER BY TRADE_DT DESC
            FETCH FIRST 1 ROWS ONLY
            """
            val_df = pd.read_sql(valuation_sql, self.dsn)

            if not val_df.empty:
                row = val_df.iloc[0]
                result['pe'] = float(row['S_VAL_PE']) if pd.notna(row.get('S_VAL_PE')) else None
                result['pe_ttm'] = float(row['S_VAL_PE_TTM']) if pd.notna(row.get('S_VAL_PE_TTM')) else None
                result['pb'] = float(row['S_VAL_PB_NEW']) if pd.notna(row.get('S_VAL_PB_NEW')) else None

            # 2. 获取财务指标（毛利率、净利率、ROE、资产负债率）
            fin_sql = f"""
            SELECT S_FA_GROSSPROFITMARGIN, S_FA_NETPROFITMARGIN,
                   S_FA_ROE, S_FA_DEBTTOASSETS
            FROM AShareFinancialIndicator
            WHERE S_INFO_WINDCODE = '{symbol}'
            ORDER BY REPORT_PERIOD DESC
            FETCH FIRST 1 ROWS ONLY
            """
            fin_df = pd.read_sql(fin_sql, self.dsn)

            if not fin_df.empty:
                row = fin_df.iloc[0]
                result['gross_margin'] = float(row['S_FA_GROSSPROFITMARGIN']) if pd.notna(row.get('S_FA_GROSSPROFITMARGIN')) else None
                result['net_margin'] = float(row['S_FA_NETPROFITMARGIN']) if pd.notna(row.get('S_FA_NETPROFITMARGIN')) else None
                result['roe'] = float(row['S_FA_ROE']) if pd.notna(row.get('S_FA_ROE')) else None
                result['debt_ratio'] = float(row['S_FA_DEBTTOASSETS']) if pd.notna(row.get('S_FA_DEBTTOASSETS')) else None

            return result if result else {}

        except Exception as e:
            logger.error(f"获取基本面数据失败 {symbol}: {e}")
            return {}

    def analyze_stock(self, symbol, name):
        """综合分析单只股票"""
        result = {
            'symbol': symbol,
            'name': name,
            'tech_score': 0,
            'fund_score': 0,
            'news_score': 50,
            'total_score': 0,
            'passed': False,
            'reason': [],
            'data': {}
        }

        # 1. 技术分析
        market_data = self.get_stock_data(symbol)
        if market_data is None or len(market_data) < 60:
            result['reason'].append('数据不足')
            return result

        tech_result = self.tech_framework.analyze(market_data)
        result['tech_score'] = tech_result['signals']['composite']
        result['data']['tech'] = {
            'trend': tech_result['trend'],
            'momentum': tech_result['momentum'],
            'volatility': tech_result['volatility'],
            'signal': tech_result['signals']['recommendation']
        }

        # 2. 基本面分析
        fund_data = self.get_fundamental_data(symbol)
        result['data']['fundamental'] = fund_data

        fund_score = 0
        fund_count = 0

        if fund_data.get('pe') and 0 < fund_data['pe'] < self.criteria['max_pe']:
            fund_score += 40
            fund_count += 1
        elif fund_data.get('pe_ttm') and 0 < fund_data['pe_ttm'] < self.criteria['max_pe']:
            fund_score += 35
            fund_count += 1
        else:
            result['reason'].append(f"PE不满足({fund_data.get('pe') or fund_data.get('pe_ttm')})")

        if fund_data.get('roe') and fund_data['roe'] >= self.criteria['min_roe']:
            fund_score += 30
            fund_count += 1
        else:
            result['reason'].append(f"ROE不满足({fund_data.get('roe')})")

        if fund_data.get('debt_ratio') and fund_data['debt_ratio'] <= self.criteria['max_debt_ratio']:
            fund_score += 15
            fund_count += 1
        else:
            result['reason'].append(f"负债率不满足({fund_data.get('debt_ratio')})")

        if fund_data.get('gross_margin') and fund_data['gross_margin'] >= self.criteria['min_gross_margin']:
            fund_score += 15
            fund_count += 1
        else:
            result['reason'].append(f"毛利率不满足({fund_data.get('gross_margin')})")

        result['fund_score'] = fund_score if fund_count > 0 else 0

        # 3. 新闻情绪
        try:
            news_signal = self.news_gen.generate_signal(symbol)
            result['news_score'] = 50 + news_signal.sentiment_score * 50
            result['data']['news'] = {
                'sentiment': news_signal.sentiment_score,
                'news_count': news_signal.news_count,
                'signal': news_signal.signal_type.value
            }
        except:
            result['news_score'] = 50

        # 4. 综合评分
        result['total_score'] = (
            result['tech_score'] * 0.3 +
            result['fund_score'] * 0.4 +
            result['news_score'] * 0.3
        )

        # 5. 判断是否通过筛选
        result['passed'] = (
            result['tech_score'] >= self.criteria['min_trend_score'] and
            result['fund_score'] >= 50 and
            result['news_score'] >= 40
        )

        # 同时计算宽松版通过（至少满足基本面和技术面）
        result['passed_relaxed'] = (
            result['tech_score'] >= self.criteria['min_trend_score'] and
            result['fund_score'] >= 50
        )

        return result

    def screen(self, max_stocks=None, top_n=10):
        """执行选股"""
        logger.info(f"开始选股筛选，目标: {top_n}只")

        # 1. 获取股票列表（不限制数量，获取全部A股）
        stock_list = self.get_a_share_list(limit=max_stocks)

        if stock_list.empty:
            logger.error("获取股票列表失败")
            return []

        results = []

        # 2. 并行分析每只股票
        logger.info(f"开始分析 {len(stock_list)} 只股票...")

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = {
                executor.submit(self.analyze_stock, row['S_INFO_WINDCODE'], row['S_INFO_NAME']): row
                for _, row in stock_list.iterrows()
            }

            for i, future in enumerate(as_completed(futures)):
                if (i + 1) % 50 == 0:
                    logger.info(f"已分析 {i + 1}/{len(stock_list)} 只股票")

                try:
                    result = future.result()
                    results.append(result)

                    # 如果找到足够多的通过筛选的股票，可以提前结束
                    passed = [r for r in results if r['passed']]
                    if len(passed) >= top_n * 3:
                        logger.info(f"已找到足够多候选股票，提前结束")
                        break
                except Exception as e:
                    logger.error(f"分析失败: {e}")

        # 3. 按综合评分排序
        results.sort(key=lambda x: x['total_score'], reverse=True)

        # 4. 获取通过筛选的股票
        passed_stocks = [r for r in results if r['passed']][:top_n]

        logger.info(f"筛选完成，通过筛选的股票: {len(passed_stocks)} 只")

        return passed_stocks, results[:50]  # 返回通过筛选的和前50名

    def print_result(self, passed_stocks, all_results):
        """打印结果"""
        print("\n" + "=" * 80)
        print(" " * 20 + "A股智能选股结果")
        print("=" * 80)

        # 通过筛选的股票
        print(f"\n{'='*80}")
        print(f"  精选 TOP 10 ({len(passed_stocks)} 只通过筛选)")
        print(f"{'='*80}\n")

        if passed_stocks:
            print(f"{'序号':<4} {'代码':<10} {'名称':<12} {'综合':<6} {'技术':<6} {'基本面':<6} {'情绪':<6}")
            print("-" * 60)
            for i, stock in enumerate(passed_stocks, 1):
                d = stock['data']
                fund = d.get('fundamental', {})
                print(f"{i:<4} {stock['symbol']:<10} {stock['name']:<12} "
                      f"{stock['total_score']:<6.1f} {stock['tech_score']:<6.1f} "
                      f"{stock['fund_score']:<6.1f} {stock['news_score']:<6.1f}")
                # 打印关键指标
                pe_val = fund.get('pe') or fund.get('pe_ttm') or 0
                roe_val = fund.get('roe') or 0
                print(f"       PE={pe_val:.1f} ROE={roe_val:.1f}% "
                      f"趋势={d['tech']['trend']['direction']} "
                      f"信号={d['tech']['signal']}")
                print()
        else:
            print("没有股票通过筛选条件")

        # 前50名概览
        print(f"\n{'='*80}")
        print("  热门 TOP 50 (按综合评分)")
        print(f"{'='*80}\n")
        print(f"{'排名':<4} {'代码':<10} {'名称':<12} {'综合':<6} {'技术':<6} {'基本面':<6} {'情绪':<6} {'通过'}")
        print("-" * 75)
        for i, stock in enumerate(all_results[:50], 1):
            passed = "✓" if stock['passed'] else "✗"
            print(f"{i:<4} {stock['symbol']:<10} {stock['name']:<12} "
                  f"{stock['total_score']:<6.1f} {stock['tech_score']:<6.1f} "
                  f"{stock['fund_score']:<6.1f} {stock['news_score']:<6.1f} {passed}")

    def save_result(self, passed_stocks, all_results):
        """保存结果"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = 'results/reports'
        os.makedirs(output_dir, exist_ok=True)

        # 保存通过筛选的股票
        if passed_stocks:
            df_passed = pd.DataFrame([{
                'symbol': s['symbol'],
                'name': s['name'],
                'total_score': s['total_score'],
                'tech_score': s['tech_score'],
                'fund_score': s['fund_score'],
                'news_score': s['news_score'],
                'pe': s['data']['fundamental'].get('pe') or s['data']['fundamental'].get('pe_ttm'),
                'roe': s['data']['fundamental'].get('roe'),
                'debt_ratio': s['data']['fundamental'].get('debt_ratio'),
                'gross_margin': s['data']['fundamental'].get('gross_margin'),
                'trend': s['data']['tech']['trend']['direction'],
                'signal': s['data']['tech']['signal'],
            } for s in passed_stocks])

            passed_path = os.path.join(output_dir, f'top_stocks_{timestamp}.csv')
            df_passed.to_csv(passed_path, index=False, encoding='utf-8-sig')
            logger.info(f"精选结果已保存到: {passed_path}")

        # 保存完整排名
        df_all = pd.DataFrame([{
            'rank': i + 1,
            'symbol': s['symbol'],
            'name': s['name'],
            'total_score': s['total_score'],
            'tech_score': s['tech_score'],
            'fund_score': s['fund_score'],
            'news_score': s['news_score'],
            'passed': s['passed'],
            'reason': '; '.join(s['reason']),
        } for i, s in enumerate(all_results)])

        all_path = os.path.join(output_dir, f'stock_ranking_{timestamp}.csv')
        df_all.to_csv(all_path, index=False, encoding='utf-8-sig')
        logger.info(f"完整排名已保存到: {all_path}")

        return passed_path, all_path


def main():
    screener = AStockScreener()

    print("\n" + "=" * 80)
    print(" " * 15 + "A股智能选股系统启动...")
    print(" " * 15 + f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    # 执行筛选（获取全部A股，不限制数量）
    passed, all_results = screener.screen(top_n=10)

    # 打印结果
    screener.print_result(passed, all_results)

    # 保存结果
    screener.save_result(passed, all_results)

    screener.dsn.close()


if __name__ == '__main__':
    main()
