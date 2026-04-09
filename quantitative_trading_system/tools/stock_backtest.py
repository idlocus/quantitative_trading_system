#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A股策略回测

假设在2026年1月1日使用同样的选股策略筛选股票，
计算到今天的收益率
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


# 回测参数
BACKTEST_START_DATE = '20260101'  # 回测开始日期
CURRENT_DATE = datetime.now().strftime('%Y%m%d')


class StockBacktest:
    """股票策略回测"""

    def __init__(self):
        self.dsn = oracledb.connect(user="wind", password="windPrd22", dsn="10.1.33.123:1521/info")
        self.news_gen = NewsSignalGenerator()
        self.tech_framework = TechnicalInvestmentFramework()

        # 同样的筛选标准
        self.criteria = {
            'min_pe': 0,
            'max_pe': 60,
            'min_roe': 3,
            'max_debt_ratio': 80,
            'min_gross_margin': 5,
            'min_trend_score': 30,
            'min_rps': 30,
            'min_sentiment': -0.3,
        }

    def get_a_share_list(self, limit=None):
        """获取A股股票列表"""
        logger.info("获取A股股票列表...")

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

    def get_stock_data(self, symbol, start_date=None, end_date=None):
        """获取股票历史数据"""
        try:
            start = start_date or '20250101'
            end = end_date or CURRENT_DATE

            sql = f"""
            SELECT TRADE_DT, S_DQ_OPEN, S_DQ_HIGH, S_DQ_LOW, S_DQ_CLOSE, S_DQ_VOLUME
            FROM AShareEODPrices
            WHERE S_INFO_WINDCODE = '{symbol}'
            AND TRADE_DT >= '{start}'
            AND TRADE_DT <= '{end}'
            ORDER BY TRADE_DT ASC
            """
            df = pd.read_sql(sql, self.dsn)

            if df.empty or len(df) < 20:
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
            logger.error(f"获取股票数据失败 {symbol}: {e}")
            return None

    def get_fundamental_data(self, symbol):
        """获取基本面数据（使用回测开始前的最新数据）"""
        result = {}

        try:
            # 使用回测开始前的最新数据
            valuation_sql = f"""
            SELECT S_VAL_PE, S_VAL_PB_NEW, S_VAL_PE_TTM
            FROM AShareEODDerivativeIndicator
            WHERE S_INFO_WINDCODE = '{symbol}'
            AND TRADE_DT <= '{BACKTEST_START_DATE}'
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
            AND REPORT_PERIOD <= '{BACKTEST_START_DATE}'
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
        """分析单只股票"""
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

        # 1. 技术分析（使用回测开始前的数据）
        market_data = self.get_stock_data(symbol, start_date='20250901', end_date=BACKTEST_START_DATE)
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

        # 3. 新闻情绪（设为默认值，因为回测时无法获取历史新闻）
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
            result['fund_score'] >= 50
        )

        return result

    def screen(self, max_stocks=None, top_n=10):
        """执行选股"""
        logger.info(f"开始回测选股，假设日期: {BACKTEST_START_DATE}，目标: {top_n}只")

        stock_list = self.get_a_share_list(limit=max_stocks)
        if stock_list.empty:
            logger.error("获取股票列表失败")
            return [], []

        results = []

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = {
                executor.submit(self.analyze_stock, row['S_INFO_WINDCODE'], row['S_INFO_NAME']): row
                for _, row in stock_list.iterrows()
            }

            for i, future in enumerate(as_completed(futures)):
                if (i + 1) % 500 == 0:
                    logger.info(f"已分析 {i + 1}/{len(stock_list)} 只股票")

                try:
                    result = future.result()
                    results.append(result)

                    passed = [r for r in results if r['passed']]
                    if len(passed) >= top_n * 3:
                        logger.info(f"已找到足够多候选股票，提前结束")
                        break
                except Exception as e:
                    logger.error(f"分析失败: {e}")

        results.sort(key=lambda x: x['total_score'], reverse=True)
        passed_stocks = [r for r in results if r['passed']][:top_n]

        logger.info(f"筛选完成，通过筛选的股票: {len(passed_stocks)} 只")

        return passed_stocks, results[:100]

    def calculate_returns(self, selected_stocks):
        """计算选定股票的收益率"""
        logger.info("计算回测收益率...")

        returns_data = []

        for stock in selected_stocks:
            symbol = stock['symbol']
            name = stock['name']

            # 获取从回测开始到现在全部数据
            full_data = self.get_stock_data(symbol)

            if full_data is None or len(full_data) < 20:
                logger.warning(f"{symbol} 数据不足，跳过")
                continue

            # 找到回测开始日期的数据
            start_price = None
            start_date = None
            end_price = None
            end_date = None

            for date, row in full_data.iterrows():
                if date.strftime('%Y%m%d') >= BACKTEST_START_DATE:
                    start_price = row['close']
                    start_date = date
                    break

            if start_price is None:
                logger.warning(f"{symbol} 在回测开始日期后无数据")
                continue

            # 获取最新价格
            end_price = full_data.iloc[-1]['close']
            end_date = full_data.index[-1]

            # 计算收益率
            total_return = (end_price - start_price) / start_price * 100

            # 计算持有天数
            holding_days = (end_date - start_date).days

            # 计算年化收益率
            if holding_days > 0:
                annualized_return = ((1 + total_return / 100) ** (365 / holding_days) - 1) * 100
            else:
                annualized_return = 0

            # 获取回测开始日的价格序列（用于计算最大回撤）
            price_series = full_data[full_data.index >= start_date]['close']

            # 计算最大回撤
            max_price = price_series.expanding().max()
            drawdown = (price_series - max_price) / max_price * 100
            max_drawdown = drawdown.min()

            returns_data.append({
                'symbol': symbol,
                'name': name,
                'start_date': start_date.strftime('%Y-%m-%d'),
                'end_date': end_date.strftime('%Y-%m-%d'),
                'start_price': start_price,
                'end_price': end_price,
                'total_return': total_return,
                'annualized_return': annualized_return,
                'max_drawdown': max_drawdown,
                'holding_days': holding_days,
                'tech_score': stock['tech_score'],
                'fund_score': stock['fund_score'],
                'total_score': stock['total_score'],
                'fundamental': stock['data'].get('fundamental', {}),
                'price_data': full_data[full_data.index >= start_date]
            })

            logger.info(f"{symbol} {name}: 收益率={total_return:.2f}%, 年化={annualized_return:.2f}%, 最大回撤={max_drawdown:.2f}%")

        return returns_data

    def generate_report(self, returns_data):
        """生成回测报告"""
        if not returns_data:
            print("没有足够的回测数据")
            return

        # 打印报告
        print("\n" + "=" * 100)
        print(" " * 30 + "A股策略回测报告")
        print("=" * 100)
        print(f"\n回测假设日期: {BACKTEST_START_DATE}")
        print(f"回测结束日期: {datetime.now().strftime('%Y-%m-%d')}")
        print(f"选股策略: 技术面30% + 基本面40% + 新闻情绪30%")

        print("\n" + "-" * 100)
        print("选股结果及收益率")
        print("-" * 100)

        # 按收益率排序
        sorted_returns = sorted(returns_data, key=lambda x: x['total_return'], reverse=True)

        print(f"\n{'排名':<4} {'代码':<12} {'名称':<10} {'评分':<6} {'收益率':<10} {'年化收益':<10} {'最大回撤':<10} {'持有天数':<8}")
        print("-" * 80)

        for i, stock in enumerate(sorted_returns, 1):
            print(f"{i:<4} {stock['symbol']:<12} {stock['name']:<10} {stock['total_score']:<6.1f} "
                  f"{stock['total_return']:>8.2f}% {stock['annualized_return']:>8.2f}% {stock['max_drawdown']:>8.2f}% {stock['holding_days']:>6}天")

        # 计算组合统计
        total_returns = [s['total_return'] for s in returns_data]
        avg_return = np.mean(total_returns)
        max_return = max(total_returns)
        min_return = min(total_returns)

        # 盈利股票数
        profitable_count = len([r for r in total_returns if r > 0])
        win_rate = profitable_count / len(total_returns) * 100

        print("\n" + "-" * 100)
        print("组合统计")
        print("-" * 100)
        print(f"平均收益率: {avg_return:.2f}%")
        print(f"最高收益率: {max_return:.2f}%")
        print(f"最低收益率: {min_return:.2f}%")
        print(f"盈利股票数: {profitable_count}/{len(total_returns)}")
        print(f"胜率: {win_rate:.1f}%")

        # 如果有10只票，计算等权组合收益
        if len(returns_data) >= 5:
            equal_weight_return = avg_return
            print(f"\n等权组合收益率: {equal_weight_return:.2f}%")

        print("\n" + "=" * 100)

        return sorted_returns

    def create_visualization(self, returns_data):
        """创建可视化图表"""
        try:
            import matplotlib.pyplot as plt
            import matplotlib.dates as mdates

            # 设置中文字体
            plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
            plt.rcParams['axes.unicode_minus'] = False

            fig, axes = plt.subplots(2, 2, figsize=(16, 12))

            # 1. 各股票收益率柱状图
            ax1 = axes[0, 0]
            stocks = [f"{s['name']}\n({s['symbol'][-4:]})" for s in sorted(returns_data, key=lambda x: x['total_return'], reverse=True)]
            returns = [s['total_return'] for s in sorted(returns_data, key=lambda x: x['total_return'], reverse=True)]
            colors = ['green' if r > 0 else 'red' for r in returns]
            bars = ax1.bar(range(len(stocks)), returns, color=colors, alpha=0.7)
            ax1.set_xticks(range(len(stocks)))
            ax1.set_xticklabels(stocks, rotation=45, ha='right', fontsize=8)
            ax1.set_ylabel('Return (%)')
            ax1.set_title('Stock Returns Since 2026-01-01')
            ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            ax1.grid(True, alpha=0.3)

            # 添加数值标签
            for bar, ret in zip(bars, returns):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{ret:.1f}%', ha='center', va='bottom' if height > 0 else 'top', fontsize=8)

            # 2. 累计收益率曲线（如果有价格数据）
            ax2 = axes[0, 1]
            for stock in sorted(returns_data, key=lambda x: x['total_return'], reverse=True)[:5]:
                if 'price_data' in stock and not stock['price_data'].empty:
                    price_data = stock['price_data']
                    # 归一化到起始值100
                    normalized = price_data['close'] / price_data['close'].iloc[0] * 100
                    ax2.plot(price_data.index, normalized, label=f"{stock['name']} ({stock['total_return']:.1f}%)", linewidth=1.5)

            ax2.set_ylabel('Normalized Price (Start=100)')
            ax2.set_title('Top 5 Stocks Price Performance')
            ax2.legend(loc='best', fontsize=8)
            ax2.grid(True, alpha=0.3)
            ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
            ax2.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))

            # 3. 收益率 vs 最大回撤散点图
            ax3 = axes[1, 0]
            for stock in returns_data:
                ax3.scatter(stock['total_return'], stock['max_drawdown'],
                           s=100, c='green' if stock['total_return'] > 0 else 'red', alpha=0.7)
                ax3.annotate(stock['name'][:4], (stock['total_return'], stock['max_drawdown']),
                            fontsize=7, xytext=(5, 5), textcoords='offset points')

            ax3.set_xlabel('Total Return (%)')
            ax3.set_ylabel('Max Drawdown (%)')
            ax3.set_title('Return vs Max Drawdown')
            ax3.axvline(x=0, color='black', linestyle='--', linewidth=0.5)
            ax3.grid(True, alpha=0.3)

            # 4. 评分 vs 收益率
            ax4 = axes[1, 1]
            for stock in returns_data:
                ax4.scatter(stock['total_score'], stock['total_return'],
                           s=100, c='blue' if stock['total_return'] > 0 else 'red', alpha=0.7)
                ax4.annotate(stock['name'][:4], (stock['total_score'], stock['total_return']),
                            fontsize=7, xytext=(5, 5), textcoords='offset points')

            ax4.set_xlabel('Composite Score')
            ax4.set_ylabel('Total Return (%)')
            ax4.set_title('Score vs Return')
            ax4.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
            ax4.grid(True, alpha=0.3)

            plt.tight_layout()

            # 保存图表
            output_dir = 'results/backtests'
            os.makedirs(output_dir, exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            save_path = os.path.join(output_dir, f'backtest_report_{timestamp}.png')
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"回测图表已保存到: {save_path}")
            plt.close()

            return save_path

        except Exception as e:
            logger.error(f"生成可视化失败: {e}")
            return None

    def save_results(self, returns_data):
        """保存回测结果"""
        output_dir = 'results/backtests'
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # 保存为CSV
        df = pd.DataFrame([{
            'rank': i + 1,
            'symbol': s['symbol'],
            'name': s['name'],
            'start_date': s['start_date'],
            'end_date': s['end_date'],
            'start_price': s['start_price'],
            'end_price': s['end_price'],
            'total_return': s['total_return'],
            'annualized_return': s['annualized_return'],
            'max_drawdown': s['max_drawdown'],
            'holding_days': s['holding_days'],
            'tech_score': s['tech_score'],
            'fund_score': s['fund_score'],
            'total_score': s['total_score'],
            'pe': s['fundamental'].get('pe') or s['fundamental'].get('pe_ttm'),
            'roe': s['fundamental'].get('roe'),
            'debt_ratio': s['fundamental'].get('debt_ratio'),
        } for i, s in enumerate(returns_data)])

        csv_path = os.path.join(output_dir, f'backtest_results_{timestamp}.csv')
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        logger.info(f"回测结果已保存到: {csv_path}")

        return csv_path

    def run(self):
        """执行回测"""
        print("\n" + "=" * 80)
        print(" " * 20 + f"A股策略回测 (假设日期: {BACKTEST_START_DATE})")
        print(" " * 20 + f"执行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)

        # 1. 筛选股票
        selected_stocks, all_results = self.screen(top_n=10)

        if not selected_stocks:
            print("没有找到通过筛选的股票")
            return

        print(f"\n通过筛选的股票数量: {len(selected_stocks)}")

        # 2. 计算收益率
        returns_data = self.calculate_returns(selected_stocks)

        # 3. 生成报告
        sorted_returns = self.generate_report(returns_data)

        # 4. 创建可视化
        chart_path = self.create_visualization(returns_data)

        # 5. 保存结果
        csv_path = self.save_results(returns_data)

        print(f"\n回测图表: {chart_path}")
        print(f"回测结果: {csv_path}")

        self.dsn.close()

        return returns_data


def main():
    backtest = StockBacktest()
    backtest.run()


if __name__ == '__main__':
    main()
