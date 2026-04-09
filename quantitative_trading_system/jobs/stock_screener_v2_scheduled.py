#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A股智能选股器 V2 - 定时任务版

每天早上7点运行，发送企业微信消息
"""

import sys
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

# 导入企业微信发送模块
from utils.wechat_utils import SendJcsMessage

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class StockScreenerV2Scheduled:
    """V2选股器定时任务版"""

    def __init__(self):
        self.dsn = oracledb.connect(user="wind", password="windPrd22", dsn="10.1.33.123:1521/info")
        self.tech_framework = TechnicalInvestmentFramework()

        # V2筛选标准
        self.criteria = {
            'max_pe': 40,
            'min_roe': 10,
            'max_debt_ratio': 60,
            'min_gross_margin': 15,
            'min_rps': 70,
            'min_momentum_3m': 0,
            'min_momentum_6m': 5,
            'min_trend_score': 40,
            'min_sentiment': -0.1,
        }

        self.weights = {
            'momentum': 0.50,
            'fundamental': 0.30,
            'technical': 0.20,
        }

    def get_a_share_list(self):
        """获取A股股票列表"""
        sql = """
        SELECT S_INFO_WINDCODE, S_INFO_NAME
        FROM AShareDescription
        WHERE (S_INFO_WINDCODE LIKE '%.SZ' OR S_INFO_WINDCODE LIKE '%.SH')
        AND S_INFO_NAME NOT LIKE 'ST%'
        AND S_INFO_NAME NOT LIKE '%*ST%'
        AND S_INFO_NAME NOT LIKE '%退市%'
        ORDER BY S_INFO_WINDCODE
        """
        try:
            df = pd.read_sql(sql, self.dsn)
            logger.info(f"获取到 {len(df)} 只A股")
            return df
        except Exception as e:
            logger.error(f"获取股票列表失败: {e}")
            return pd.DataFrame()

    def get_stock_data(self, symbol, days=250):
        """获取股票历史数据"""
        try:
            sql = f"""
            SELECT TRADE_DT, S_DQ_OPEN, S_DQ_HIGH, S_DQ_LOW, S_DQ_CLOSE, S_DQ_VOLUME
            FROM AShareEODPrices
            WHERE S_INFO_WINDCODE = '{symbol}'
            AND TRADE_DT >= '20250101'
            ORDER BY TRADE_DT ASC
            """
            df = pd.read_sql(sql, self.dsn)
            if df.empty or len(df) < 120:
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
            return {}

    def calculate_momentum(self, data):
        """计算动量指标"""
        if data is None or len(data) < 120:
            return {}
        close = data['close']
        momentum_1m = (close.iloc[-1] / close.iloc[-20] - 1) * 100 if len(close) >= 20 else 0
        momentum_3m = (close.iloc[-1] / close.iloc[-60] - 1) * 100 if len(close) >= 60 else 0
        momentum_6m = (close.iloc[-1] / close.iloc[-120] - 1) * 100 if len(close) >= 120 else 0
        rps = min(100, max(0, momentum_6m * 2 + 50))
        momentum_acceleration = momentum_3m - momentum_6m if momentum_6m != 0 else 0
        return {
            'momentum_1m': momentum_1m,
            'momentum_3m': momentum_3m,
            'momentum_6m': momentum_6m,
            'rps': rps,
            'momentum_acceleration': momentum_acceleration,
        }

    def calculate_momentum_score(self, momentum):
        """计算动量评分"""
        if not momentum:
            return 0
        score = 0
        rps = momentum.get('rps', 0)
        if rps >= 90: score += 35
        elif rps >= 80: score += 30
        elif rps >= 70: score += 25
        elif rps >= 60: score += 20
        elif rps >= 50: score += 15
        else: score += 10

        mom_6m = momentum.get('momentum_6m', 0)
        if mom_6m >= 30: score += 30
        elif mom_6m >= 20: score += 25
        elif mom_6m >= 10: score += 20
        elif mom_6m >= 5: score += 15
        elif mom_6m >= 0: score += 10

        acc = momentum.get('momentum_acceleration', 0)
        if acc >= 10: score += 20
        elif acc >= 5: score += 15
        elif acc >= 0: score += 10

        mom_1m = momentum.get('momentum_1m', 0)
        if mom_1m >= 10: score += 15
        elif mom_1m >= 5: score += 10
        elif mom_1m >= 0: score += 5
        return min(100, score)

    def calculate_fundamental_score(self, fund_data):
        """计算基本面评分"""
        if not fund_data:
            return 0
        score = 0
        count = 0
        pe = fund_data.get('pe') or fund_data.get('pe_ttm')
        if pe and pe > 0:
            if pe <= 15: score += 40
            elif pe <= 20: score += 35
            elif pe <= 25: score += 30
            elif pe <= 30: score += 25
            elif pe <= 40: score += 20
            else: score += 0
            count += 1
        roe = fund_data.get('roe')
        if roe:
            if roe >= 20: score += 30
            elif roe >= 15: score += 25
            elif roe >= 12: score += 20
            elif roe >= 10: score += 15
            elif roe >= 5: score += 10
            count += 1
        gm = fund_data.get('gross_margin')
        if gm:
            if gm >= 40: score += 15
            elif gm >= 30: score += 12
            elif gm >= 20: score += 10
            elif gm >= 15: score += 8
            count += 1
        debt = fund_data.get('debt_ratio')
        if debt:
            if debt <= 30: score += 15
            elif debt <= 40: score += 12
            elif debt <= 50: score += 10
            elif debt <= 60: score += 8
            count += 1
        return score if count > 0 else 0

    def analyze_stock(self, symbol, name):
        """分析单只股票"""
        result = {
            'symbol': symbol,
            'name': name,
            'momentum_score': 0,
            'fund_score': 0,
            'tech_score': 0,
            'total_score': 0,
            'passed': False,
            'failed_reasons': [],
            'data': {}
        }
        market_data = self.get_stock_data(symbol)
        if market_data is None or len(market_data) < 120:
            result['failed_reasons'].append('数据不足')
            return result

        momentum = self.calculate_momentum(market_data)
        result['data']['momentum'] = momentum
        tech_result = self.tech_framework.analyze(market_data)
        result['tech_score'] = tech_result.get('signals', {}).get('composite', 0)
        result['data']['tech'] = {'trend': tech_result['trend'], 'signal': tech_result['signals'].get('recommendation', 'hold')}
        fund_data = self.get_fundamental_data(symbol)
        result['data']['fundamental'] = fund_data
        result['fund_score'] = self.calculate_fundamental_score(fund_data)
        result['momentum_score'] = self.calculate_momentum_score(momentum)
        result['total_score'] = (
            result['momentum_score'] * self.weights['momentum'] +
            result['fund_score'] * self.weights['fundamental'] +
            result['tech_score'] * self.weights['technical']
        )

        # 检查筛选标准
        pe = fund_data.get('pe') or fund_data.get('pe_ttm')
        roe = fund_data.get('roe')
        debt = fund_data.get('debt_ratio')
        gm = fund_data.get('gross_margin')
        rps = momentum.get('rps', 0)
        mom_3m = momentum.get('momentum_3m', 0)
        mom_6m = momentum.get('momentum_6m', 0)
        tech = result['tech_score']

        if pe and (pe <= 0 or pe > self.criteria['max_pe']):
            result['failed_reasons'].append(f'PE({pe:.1f})>40')
        elif roe and roe < self.criteria['min_roe']:
            result['failed_reasons'].append(f'ROE({roe:.1f})<10%')
        elif debt and debt > self.criteria['max_debt_ratio']:
            result['failed_reasons'].append(f'负债率({debt:.1f})>60%')
        elif gm and gm < self.criteria['min_gross_margin']:
            result['failed_reasons'].append(f'毛利率({gm:.1f})<15%')
        elif rps < self.criteria['min_rps']:
            result['failed_reasons'].append(f'RPS({rps:.1f})<70')
        elif mom_3m < self.criteria['min_momentum_3m']:
            result['failed_reasons'].append(f'3M动量({mom_3m:.1f})<0')
        elif mom_6m < self.criteria['min_momentum_6m']:
            result['failed_reasons'].append(f'6M动量({mom_6m:.1f})<5%')
        elif tech < self.criteria['min_trend_score']:
            result['failed_reasons'].append(f'技术评分({tech:.1f})<40')
        else:
            result['passed'] = True
        return result

    def screen(self, top_n=10):
        """执行选股"""
        logger.info("开始V2选股...")
        stock_list = self.get_a_share_list()
        if stock_list.empty:
            return [], []
        results = []
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = {
                executor.submit(self.analyze_stock, row['S_INFO_WINDCODE'], row['S_INFO_NAME']): row
                for _, row in stock_list.iterrows()
            }
            for i, future in enumerate(as_completed(futures)):
                if (i + 1) % 1000 == 0:
                    logger.info(f"已分析 {i + 1}/{len(stock_list)} 只股票")
                try:
                    result = future.result()
                    results.append(result)
                    passed = [r for r in results if r['passed']]
                    if len(passed) >= top_n * 5:
                        logger.info(f"已找到足够候选股票，提前结束")
                        break
                except Exception as e:
                    pass
        results.sort(key=lambda x: x['total_score'], reverse=True)
        passed_stocks = [r for r in results if r['passed']][:top_n]
        logger.info(f"筛选完成，通过筛选: {len(passed_stocks)} 只")
        return passed_stocks, results[:50]

    def send_wechat_message(self, passed_stocks, all_results):
        """发送企业微信消息"""
        if not passed_stocks:
            msg = f"【A股选股】{datetime.now().strftime('%Y-%m-%d')} V2选股结果\n\n今日未找到符合条件的股票"
            SendJcsMessage('A股智能选股V2', msg, 'hua.guo', 'ST_EMAIL;ST_WECHAT', '选股结果推送')
            return

        # 构建消息
        msg_lines = [
            f"【A股智能选股 V2】{datetime.now().strftime('%Y-%m-%d %H:%M')}",
            f"━━━━━━━━━━━━━━━",
            f"今日精选 TOP {len(passed_stocks)} 只:",
            f"",
        ]

        for i, stock in enumerate(passed_stocks, 1):
            d = stock['data']
            fund = d.get('fundamental', {})
            mom = d.get('momentum', {})
            pe = fund.get('pe') or fund.get('pe_ttm') or 0
            roe = fund.get('roe') or 0
            rps = mom.get('rps', 0)
            mom_3m = mom.get('momentum_3m', 0)

            msg_lines.append(
                f"{i}. {stock['name']}({stock['symbol']})"
            )
            msg_lines.append(
                f"   评分:{stock['total_score']:.1f} | PE:{pe:.0f} ROE:{roe:.1f}%"
            )
            msg_lines.append(
                f"   RPS:{rps:.0f} | 3M动量:{mom_3m:.1f}% | {d['tech']['trend']['direction']}"
            )
            msg_lines.append(f"")

        # 添加统计信息
        msg_lines.append(f"━━━━━━━━━━━━━━━")
        msg_lines.append(f"选股标准:")
        msg_lines.append(f"• PE≤40, ROE≥10%, 负债率≤60%")
        msg_lines.append(f"• RPS≥70, 3M动量>0%, 6M动量>5%")
        msg_lines.append(f"• 趋势评分≥40")
        msg_lines.append(f"权重: 动量50% + 基本面30% + 技术20%")

        msg = '\n'.join(msg_lines)
        logger.info(f"发送企业微信消息，长度: {len(msg)}")

        try:
            SendJcsMessage('A股智能选股V2', msg, 'hua.guo', 'ST_EMAIL;ST_WECHAT', '每日选股结果推送')
            logger.info("企业微信消息发送成功")
        except Exception as e:
            logger.error(f"发送失败: {e}")

    def save_result(self, passed_stocks, all_results):
        """保存结果"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = 'results/reports'
        os.makedirs(output_dir, exist_ok=True)

        if passed_stocks:
            df_passed = pd.DataFrame([{
                'symbol': s['symbol'],
                'name': s['name'],
                'total_score': s['total_score'],
                'momentum_score': s['momentum_score'],
                'fund_score': s['fund_score'],
                'tech_score': s['tech_score'],
                'pe': s['data']['fundamental'].get('pe') or s['data']['fundamental'].get('pe_ttm'),
                'roe': s['data']['fundamental'].get('roe'),
                'rps': s['data']['momentum'].get('rps', 0),
                'momentum_3m': s['data']['momentum'].get('momentum_3m', 0),
                'trend': s['data']['tech']['trend']['direction'],
            } for s in passed_stocks])
            passed_path = os.path.join(output_dir, f'top_stocks_v2_{timestamp}.csv')
            df_passed.to_csv(passed_path, index=False, encoding='utf-8-sig')
            logger.info(f"结果已保存: {passed_path}")

    def run(self):
        """执行定时任务"""
        logger.info("="*50)
        logger.info("开始执行V2选股定时任务")
        logger.info("="*50)

        try:
            # 1. 执行选股
            passed, all_results = self.screen(top_n=10)

            # 2. 保存结果
            self.save_result(passed, all_results)

            # 3. 发送企业微信
            self.send_wechat_message(passed, all_results)

            logger.info("定时任务执行完成")
        except Exception as e:
            logger.error(f"任务执行失败: {e}")
            msg = f"【A股选股】{datetime.now().strftime('%Y-%m-%d')} 选股任务执行失败\n\n错误: {e}"
            try:
                SendJcsMessage('A股智能选股V2', msg, 'hua.guo', 'ST_EMAIL;ST_WECHAT', '任务失败告警')
            except:
                pass
        finally:
            self.dsn.close()


def main():
    task = StockScreenerV2Scheduled()
    task.run()


if __name__ == '__main__':
    main()
