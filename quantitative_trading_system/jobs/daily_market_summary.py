#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
每日市场总结任务
收集今日A股市场重大消息，分析市场走势，给出总结

每天下午3点后运行（收盘后）
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict

# 数据库
import oracledb

# 指标模块
from indicators.technical_framework import TechnicalInvestmentFramework
from indicators.market_regime import MarketRegime, MarketRegimeAnalyzer

# 飞书发送
from utils.feishu_utils import send_feishu_message
from utils.message_tracker import should_send_message, mark_message_sent


class DailyMarketSummary:
    """每日市场总结"""

    def __init__(self):
        self.tech_framework = TechnicalInvestmentFramework()
        self.market_analyzer = MarketRegimeAnalyzer()
        self.dsn = None

    def connect_db(self):
        """连接数据库"""
        if self.dsn is None:
            self.dsn = oracledb.connect(user="wind", password="windPrd22", dsn="10.1.33.123:1521/info")
        return self.dsn

    def get_index_data(self, symbol: str, days: int = 250) -> pd.DataFrame:
        """获取指数数据"""
        conn = self.connect_db()
        # 计算起始日期
        start_date = (datetime.now() - timedelta(days=days + 30)).strftime('%Y%m%d')
        sql = f"""
        SELECT TRADE_DT, S_DQ_OPEN, S_DQ_HIGH, S_DQ_LOW, S_DQ_CLOSE, S_DQ_VOLUME, S_DQ_AMOUNT
        FROM AINDEXEODPRICES
        WHERE S_INFO_WINDCODE = '{symbol}'
        AND TRADE_DT >= '{start_date}'
        ORDER BY TRADE_DT ASC
        """
        df = pd.read_sql(sql, conn)
        df['date'] = pd.to_datetime(df['TRADE_DT'], format='%Y%m%d')
        df = df.rename(columns={
            'S_DQ_OPEN': 'open', 'S_DQ_HIGH': 'high',
            'S_DQ_LOW': 'low', 'S_DQ_CLOSE': 'close',
            'S_DQ_VOLUME': 'volume', 'S_DQ_AMOUNT': 'amount'
        })
        return df[['date', 'open', 'high', 'low', 'close', 'volume', 'amount']].tail(days)

    def get_market_stats(self) -> Dict:
        """获取今日市场统计数据"""
        conn = self.connect_db()

        # 获取最新交易日期
        cursor = conn.cursor()
        cursor.execute("SELECT MAX(TRADE_DT) FROM AShareEODPrices")
        latest_date = cursor.fetchone()[0]
        cursor.close()

        if latest_date is None:
            return {
                'limit_up': 0, 'limit_down': 0, 'total_amount': 0,
                'up_count': 0, 'down_count': 0, 'up_ratio': 50
            }

        today = latest_date.strftime('%Y%m%d') if hasattr(latest_date, 'strftime') else str(latest_date)

        # 涨跌停统计
        sql_limit = f"""
        SELECT COUNT(*) as limit_up_count
        FROM AShareEODPrices
        WHERE S_DQ_CLOSE = S_DQ_HIGH
        AND S_DQ_PCTCHANGE > 9.5
        AND TRADE_DT = '{today}'
        """
        df_limit = pd.read_sql(sql_limit, conn)

        sql_down = f"""
        SELECT COUNT(*) as limit_down_count
        FROM AShareEODPrices
        WHERE S_DQ_CLOSE = S_DQ_LOW
        AND S_DQ_PCTCHANGE < -9.5
        AND TRADE_DT = '{today}'
        """
        df_down = pd.read_sql(sql_down, conn)

        # 成交额统计
        sql_amount = f"""
        SELECT SUM(S_DQ_AMOUNT) as total_amount
        FROM AShareEODPrices
        WHERE TRADE_DT = '{today}'
        AND S_DQ_AMOUNT IS NOT NULL
        """
        df_amount = pd.read_sql(sql_amount, conn)

        # 处理可能的NULL值
        total_amount_val = df_amount.iloc[0, 0] if not df_amount.empty and df_amount.iloc[0, 0] is not None else 0

        # 上涨下跌家数
        sql_up = f"""
        SELECT COUNT(*) as up_count
        FROM AShareEODPrices
        WHERE S_DQ_PCTCHANGE > 0
        AND TRADE_DT = '{today}'
        AND S_DQ_CLOSE IS NOT NULL
        """
        df_up = pd.read_sql(sql_up, conn)

        sql_down_count = f"""
        SELECT COUNT(*) as down_count
        FROM AShareEODPrices
        WHERE S_DQ_PCTCHANGE < 0
        AND TRADE_DT = '{today}'
        AND S_DQ_CLOSE IS NOT NULL
        """
        df_down_count = pd.read_sql(sql_down_count, conn)

        total = df_up.iloc[0, 0] + df_down_count.iloc[0, 0]
        up_ratio = df_up.iloc[0, 0] / total * 100 if total > 0 else 0

        return {
            'limit_up': int(df_limit.iloc[0, 0]) if not df_limit.empty else 0,
            'limit_down': int(df_down.iloc[0, 0]) if not df_down.empty else 0,
            'total_amount': float(total_amount_val),
            'up_count': int(df_up.iloc[0, 0]) if not df_up.empty else 0,
            'down_count': int(df_down_count.iloc[0, 0]) if not df_down.empty else 0,
            'up_ratio': up_ratio
        }

    def get_sector_performance(self) -> List[Dict]:
        """获取行业板块表现"""
        conn = self.connect_db()
        today = datetime.now().strftime('%Y%m%d')

        sql = f"""
        SELECT * FROM (
            SELECT S_INFO_WINDCODE, S_DQ_CLOSE, S_DQ_PCTCHANGE
            FROM AShareEODPrices
            WHERE TRADE_DT = '{today}'
            AND S_DQ_AMOUNT IS NOT NULL
            ORDER BY S_DQ_AMOUNT DESC
        ) WHERE ROWNUM <= 20
        """
        df = pd.read_sql(sql, conn)
        return df.to_dict('records')

    def get_news(self) -> List[Dict]:
        """获取今日财经新闻"""
        news_list = []

        try:
            import requests

            # 使用新浪财经快讯API
            url = 'https://feed.mix.sina.com.cn/api/roll/get'
            params = {
                'pageid': 153,
                'lid': 2516,
                'k': '',
                'num': 10,
                'page': 1,
                'r': 0.5
            }
            resp = requests.get(url, params=params, timeout=10)
            data = resp.json()

            if 'result' in data and 'data' in data['result']:
                for item in data['result']['data'][:10]:
                    title = item.get('title', '')
                    ctime = item.get('ctime', '')
                    # 转换时间戳
                    if ctime and ctime.isdigit():
                        import time
                        time_str = time.strftime('%Y-%m-%d %H:%M', time.localtime(int(ctime)))
                    else:
                        time_str = ctime

                    if title:
                        news_list.append({
                            'title': title[:100],
                            'content': '',
                            'datetime': time_str,
                            'source': '新浪财经'
                        })

        except Exception as e:
            print(f"获取新浪新闻失败: {e}")

        # 如果没有获取到新闻，尝试akshare
        if not news_list:
            try:
                import akshare as ak
                df = ak.news_cctv()
                for _, row in df.head(5).iterrows():
                    news_list.append({
                        'title': str(row['title'])[:100],
                        'content': str(row['content'])[:200] if 'content' in row else '',
                        'datetime': str(row.get('date', '')),
                        'source': 'CCTV'
                    })
            except:
                pass

        # 如果仍然没有新闻，返回默认
        if not news_list:
            news_list.append({
                'title': '今日市场交易活跃',
                'content': '市场整体交易平稳',
                'datetime': datetime.now().strftime('%Y-%m-%d'),
                'source': '系统'
            })

        return news_list

    def analyze_market(self, index_data: pd.DataFrame, market_stats: Dict) -> Dict:
        """分析市场走势"""
        if index_data.empty:
            return {}

        today = index_data.iloc[-1]
        yesterday = index_data.iloc[-2] if len(index_data) > 1 else today

        # 计算涨跌幅
        change_pct = (today['close'] - yesterday['close']) / yesterday['close'] * 100

        # 市场状态分析
        regime_result = self.market_analyzer.analyze(index_data)

        # 市场强度判断
        if market_stats['up_ratio'] > 60:
            strength = "强势"
        elif market_stats['up_ratio'] > 50:
            strength = "偏强"
        elif market_stats['up_ratio'] > 40:
            strength = "偏弱"
        else:
            strength = "弱势"

        # 成交量分析
        if len(index_data) >= 2:
            vol_change = (today['volume'] - yesterday['volume']) / yesterday['volume'] * 100
        else:
            vol_change = 0

        return {
            'close': today['close'],
            'change_pct': change_pct,
            'volume': today['volume'],
            'amount': today['amount'],
            'vol_change': vol_change,
            'regime': regime_result.regime.value,
            'regime_confidence': regime_result.confidence,
            'composite_score': regime_result.composite_score,
            'strength': strength,
            'up_ratio': market_stats['up_ratio'],
            'limit_up': market_stats['limit_up'],
            'limit_down': market_stats['limit_down']
        }

    def generate_summary(self, market_data: Dict, news: List[Dict], stats: Dict) -> str:
        """生成市场总结"""
        now = datetime.now()
        lines = []

        # 标题
        lines.append(f"【A股每日市场总结】{now.strftime('%Y-%m-%d %H:%M')}")
        lines.append("=" * 50)

        # 1. 今日市场概况
        change_emoji = "↑" if market_data['change_pct'] >= 0 else "↓"
        regime_emoji = {
            'BULLISH': '【多头】',
            'BEARISH': '【空头】',
            'NEUTRAL': '【中性】',
            'VOLATILE': '【波动】'
        }

        lines.append(f"\n【今日市场概况】")
        lines.append(f"  沪深300: {market_data['close']:.0f} {change_emoji}{abs(market_data['change_pct']):.2f}%")
        lines.append(f"  市场状态: {regime_emoji.get(market_data['regime'], '')}{market_data['regime']}")
        lines.append(f"  状态置信度: {market_data['regime_confidence']:.0f}%")
        lines.append(f"  综合评分: {market_data['composite_score']:.1f}/100")
        lines.append(f"  市场强度: {market_data['strength']}")

        # 2. 市场宽度
        lines.append(f"\n【市场宽度】")
        lines.append(f"  上涨家数: {stats['up_count']} ({market_data['up_ratio']:.1f}%)")
        lines.append(f"  下跌家数: {stats['down_count']} ({100-market_data['up_ratio']:.1f}%)")
        lines.append(f"  涨停: {market_data['limit_up']} 家")
        lines.append(f"  跌停: {market_data['limit_down']} 家")

        # 3. 成交量
        amount_billion = market_data['amount'] / 1e8 if market_data['amount'] else 0
        vol_emoji = "↑" if market_data['vol_change'] >= 0 else "↓"
        lines.append(f"\n【成交量】")
        lines.append(f"  沪深300成交额: {amount_billion:.0f} 亿元")
        lines.append(f"  量能变化: {vol_emoji}{abs(market_data['vol_change']):.1f}%")

        # 4. 重大新闻
        if news:
            lines.append(f"\n【重大消息】")
            for i, item in enumerate(news[:5], 1):
                title = item.get('title', '无标题')[:50]
                source = item.get('source', '未知')
                lines.append(f"  {i}. [{source}] {title}")

        # 5. 市场分析
        lines.append(f"\n【市场分析】")
        if market_data['regime'] == 'BULLISH':
            lines.append("  今日市场表现强势，趋势向上，短线可适当参与")
        elif market_data['regime'] == 'BEARISH':
            lines.append("  今日市场表现弱势，趋势向下，建议观望为主")
        elif market_data['regime'] == 'VOLATILE':
            lines.append("  今日市场波动较大，注意控制风险，等待趋势明朗")
        else:
            if market_data['up_ratio'] > 55:
                lines.append("  今日市场整体偏强，但趋势不明显，建议区间操作")
            elif market_data['up_ratio'] < 45:
                lines.append("  今日市场整体偏弱，但未形成明显趋势，谨慎操作")
            else:
                lines.append("  今日市场震荡整理，涨跌家数基本平衡")

        # 6. 操作建议
        lines.append(f"\n【操作建议】")
        if market_data['regime'] == 'BULLISH' and market_data['composite_score'] > 60:
            lines.append("  建议仓位: 60-80%")
            lines.append("  策略: 顺势而为，可关注近期强势板块")
        elif market_data['regime'] == 'BEARISH':
            lines.append("  建议仓位: 20%以下")
            lines.append("  策略: 轻仓或空仓，等待止跌信号")
        elif market_data['regime'] == 'VOLATILE':
            lines.append("  建议仓位: 30%以下")
            lines.append("  策略: 严格止损，避免追涨杀跌")
        else:
            lines.append("  建议仓位: 40-50%")
            lines.append("  策略: 区间操作，高抛低吸")

        lines.append("\n" + "=" * 50)
        return '\n'.join(lines)

    def run(self) -> str:
        """执行每日市场总结"""
        print("=" * 60)
        print("  每日市场总结任务启动")
        print("=" * 60)

        try:
            # 1. 获取指数数据
            print("\n[1/5] 获取沪深300数据...")
            index_data = self.get_index_data('000300.SH', days=250)
            if index_data.empty:
                print("  获取数据失败")
                return ""
            print(f"  获取到 {len(index_data)} 天数据")

            # 2. 获取市场统计
            print("[2/5] 获取市场统计...")
            market_stats = self.get_market_stats()
            print(f"  上涨: {market_stats['up_count']} 下跌: {market_stats['down_count']}")
            print(f"  涨停: {market_stats['limit_up']} 跌停: {market_stats['limit_down']}")

            # 3. 获取新闻
            print("[3/5] 获取财经新闻...")
            news = self.get_news()
            print(f"  获取到 {len(news)} 条新闻")

            # 4. 分析市场
            print("[4/5] 分析市场走势...")
            market_data = self.analyze_market(index_data, market_stats)
            print(f"  市场状态: {market_data.get('regime', 'N/A')}")

            # 5. 生成总结
            print("[5/5] 生成市场总结...")
            summary = self.generate_summary(market_data, news, market_stats)

            # 保存报告
            report_dir = 'analysis_reports'
            os.makedirs(report_dir, exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            report_file = os.path.join(report_dir, f'market_summary_{timestamp}.txt')
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write(summary)

            # 发送飞书（检查是否需要发送）
            if should_send_message('market_summary', summary):
                success = send_feishu_message(summary)
                if success:
                    print(f"  飞书消息发送成功")
                    mark_message_sent('market_summary', summary)
                else:
                    print(f"  飞书消息发送失败")
            else:
                print(f"  跳过发送（今日已发送）")

            print(f"  报告已保存: {report_file}")
            return summary

        except Exception as e:
            print(f"\n执行出错: {e}")
            import traceback
            traceback.print_exc()
            return ""


def main():
    """主函数"""
    summary = DailyMarketSummary()
    result = summary.run()
    if result:
        print("\n" + "=" * 50)
        print(result)


if __name__ == '__main__':
    main()
