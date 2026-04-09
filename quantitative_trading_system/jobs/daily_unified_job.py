#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
每日统一投资任务
整合市场状态分析 + 大盘预测 + 个股选股 + 企业微信推送

每天早上7点运行
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

# 数据库和指标模块
import oracledb
from indicators.technical_framework import TechnicalInvestmentFramework
from indicators.market_regime import (
    MarketRegime, MarketRegimeAnalyzer, get_strategy_by_regime
)
from indicators.volatility_indicators import ATR, BollingerBands
from indicators.trend_indicators import SMA

# 企业微信发送
from utils.wechat_utils import SendJcsMessage
# 飞书发送
from utils.feishu_utils import send_feishu_message
# 消息追踪
from utils.message_tracker import should_send_message, mark_message_sent


class DailyUnifiedJob:
    """每日统一投资任务"""

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
        sql = f"""
        SELECT TRADE_DT, S_DQ_OPEN, S_DQ_HIGH, S_DQ_LOW, S_DQ_CLOSE, S_DQ_VOLUME
        FROM AINDEXEODPRICES
        WHERE S_INFO_WINDCODE = '{symbol}'
        AND TRADE_DT >= '20240101'
        ORDER BY TRADE_DT ASC
        """
        df = pd.read_sql(sql, conn)
        df['date'] = pd.to_datetime(df['TRADE_DT'], format='%Y%m%d')
        df = df.rename(columns={
            'S_DQ_OPEN': 'open', 'S_DQ_HIGH': 'high',
            'S_DQ_LOW': 'low', 'S_DQ_CLOSE': 'close',
            'S_DQ_VOLUME': 'volume'
        })
        return df[['date', 'open', 'high', 'low', 'close', 'volume']].tail(days).set_index('date')

    def get_stock_data(self, symbol: str, days: int = 250) -> pd.DataFrame:
        """获取股票数据"""
        conn = self.connect_db()
        sql = f"""
        SELECT TRADE_DT, S_DQ_OPEN, S_DQ_HIGH, S_DQ_LOW, S_DQ_CLOSE, S_DQ_VOLUME
        FROM AShareEODPrices
        WHERE S_INFO_WINDCODE = '{symbol}'
        AND TRADE_DT >= '20240101'
        ORDER BY TRADE_DT ASC
        """
        df = pd.read_sql(sql, conn)
        df['date'] = pd.to_datetime(df['TRADE_DT'], format='%Y%m%d')
        df = df.rename(columns={
            'S_DQ_OPEN': 'open', 'S_DQ_HIGH': 'high',
            'S_DQ_LOW': 'low', 'S_DQ_CLOSE': 'close',
            'S_DQ_VOLUME': 'volume'
        })
        return df[['date', 'open', 'high', 'low', 'close', 'volume']].tail(days).set_index('date')

    def get_a_share_list(self, limit: int = 200) -> pd.DataFrame:
        """获取A股列表"""
        conn = self.connect_db()
        sql = f"""
        SELECT S_INFO_WINDCODE, S_INFO_NAME
        FROM AShareDescription
        WHERE (S_INFO_WINDCODE LIKE '%.SZ' OR S_INFO_WINDCODE LIKE '%.SH')
        AND S_INFO_NAME NOT LIKE 'ST%'
        AND S_INFO_NAME NOT LIKE '%*ST%'
        AND S_INFO_NAME NOT LIKE '%退市%'
        ORDER BY S_INFO_WINDCODE
        """
        df = pd.read_sql(sql, conn)
        return df.head(limit)

    def calculate_support_resistance(self, data: pd.DataFrame) -> dict:
        """计算支撑阻力位"""
        close = data['close']
        high = data['high']
        low = data['low']
        current_price = close.iloc[-1]

        atr = ATR(data, 14).iloc[-1]
        bb = BollingerBands(data, 20, 2)

        return {
            'current_price': current_price,
            'high_60': high.tail(60).max(),
            'low_60': low.tail(60).min(),
            'atr': atr,
            'bb_upper': bb['upper'].iloc[-1],
            'bb_lower': bb['lower'].iloc[-1],
            'ma20': SMA(data, 20).iloc[-1],
            'ma60': SMA(data, 60).iloc[-1],
        }

    def predict_market_direction(self, index_data: pd.DataFrame, regime: MarketRegime) -> dict:
        """预测市场方向"""
        sr = self.calculate_support_resistance(index_data)
        current = sr['current_price']

        prediction = {'current_price': current, 'targets': {}, 'supports': [], 'resistances': [], 'outlook': ''}

        if regime == MarketRegime.BULLISH:
            prediction['targets'] = {
                '短期 (1-2周)': f"{current * 1.03:.0f} ({((current*1.03/current)-1)*100:+.1f}%)",
                '中期 (1个月)': f"{current * 1.05:.0f} ({((current*1.05/current)-1)*100:+.1f}%)",
            }
            prediction['supports'] = [f"{sr['ma20']:.0f}", f"{sr['bb_lower']:.0f}"]
            prediction['resistances'] = [f"{sr['bb_upper']:.0f}", f"{sr['high_60']:.0f}"]
            prediction['outlook'] = "市场上升趋势，建议逢低买入，仓位60-80%"

        elif regime == MarketRegime.BEARISH:
            prediction['targets'] = {
                '短期 (1-2周)': f"{current * 0.97:.0f} ({((current*0.97/current)-1)*100:+.1f}%)",
                '中期 (1个月)': f"{current * 0.95:.0f} ({((current*0.95/current)-1)*100:+.1f}%)",
            }
            prediction['supports'] = [f"{sr['ma60']:.0f}", f"{sr['bb_lower']:.0f}"]
            prediction['resistances'] = [f"{sr['ma20']:.0f}", f"{sr['bb_upper']:.0f}"]
            prediction['outlook'] = "市场下降趋势，建议轻仓或空仓，等待止跌信号"

        elif regime == MarketRegime.VOLATILE:
            mid = (sr['high_60'] + sr['low_60']) / 2
            prediction['targets'] = {
                '上行目标': f"{sr['high_60']:.0f} ({((sr['high_60']/current)-1)*100:+.1f}%)",
                '区间中值': f"{mid:.0f} ({((mid/current)-1)*100:+.1f}%)",
                '下行目标': f"{sr['low_60']:.0f} ({((sr['low_60']/current)-1)*100:+.1f}%)",
            }
            prediction['supports'] = [f"{sr['bb_lower']:.0f}", f"{sr['low_60']:.0f}"]
            prediction['resistances'] = [f"{sr['bb_upper']:.0f}", f"{sr['high_60']:.0f}"]
            prediction['outlook'] = "市场波动剧烈，建议观望，等待趋势明确"

        else:  # NEUTRAL
            prediction['targets'] = {
                '上行空间': f"{sr['bb_upper']:.0f} ({((sr['bb_upper']/current)-1)*100:+.1f}%)",
                '下行空间': f"{sr['bb_lower']:.0f} ({((sr['bb_lower']/current)-1)*100:+.1f}%)",
            }
            prediction['supports'] = [f"{sr['ma20']:.0f}", f"{sr['ma60']:.0f}"]
            prediction['resistances'] = [f"{sr['high_60']:.0f}", f"{sr['bb_upper']:.0f}"]
            prediction['outlook'] = "市场横盘整理，区间操作，高抛低吸"

        return prediction

    def calculate_momentum(self, data: pd.DataFrame) -> dict:
        """计算动量"""
        close = data['close']
        returns = close.pct_change()
        return {
            'momentum_1m': (close.iloc[-1] / close.iloc[-20] - 1) * 100 if len(close) >= 20 else 0,
            'momentum_3m': (close.iloc[-1] / close.iloc[-60] - 1) * 100 if len(close) >= 60 else 0,
            'rps': (returns.tail(250).rank(pct=True).iloc[-1] * 100) if len(returns) >= 250 else 50
        }

    def screen_stocks(self, top_n: int = 5) -> list:
        """筛选股票"""
        stock_list = self.get_a_share_list(limit=150)
        results = []
        count = 0

        for _, row in stock_list.iterrows():
            symbol = row['S_INFO_WINDCODE']
            name = row['S_INFO_NAME']
            count += 1

            if count % 30 == 0:
                print(f"  已分析 {count} 只股票...")

            data = self.get_stock_data(symbol, days=250)
            if data is None or len(data) < 120:
                continue

            try:
                tech_result = self.tech_framework.analyze(data)
                signal = tech_result.get('signals', {})
                composite = signal.get('composite_adjusted', signal.get('composite', 0))
                recommendation = signal.get('recommendation_adjusted', signal.get('recommendation', 'hold'))

                if composite < 50:
                    continue

                momentum = self.calculate_momentum(data)
                if momentum['rps'] < 50:
                    continue

                results.append({
                    'symbol': symbol,
                    'name': name,
                    'composite': composite,
                    'recommendation': recommendation,
                    'trend': tech_result['trend']['direction'],
                    'rps': momentum['rps'],
                    'momentum_3m': momentum['momentum_3m'],
                    'stop_loss': tech_result['risk']['stop_loss'],
                    'risk_reward': tech_result['risk']['risk_reward_ratio'],
                    'position': tech_result['risk']['position_size_recommendation'],
                })

                if len(results) >= top_n * 3:
                    break

            except Exception:
                continue

        results.sort(key=lambda x: x['composite'], reverse=True)
        return results[:top_n]

    def build_wechat_message(self, regime, regime_analysis, prediction, top_stocks) -> str:
        """构建企业微信消息"""
        now = datetime.now().strftime('%Y-%m-%d %H:%M')
        lines = []

        # 标题
        lines.append(f"【每日投资建议】{now}")
        lines.append("=" * 50)

        # 市场状态
        regime_emoji = {
            'BULLISH': '【多头】',
            'BEARISH': '【空头】',
            'NEUTRAL': '【中性】',
            'VOLATILE': '【高波动】'
        }
        emoji = regime_emoji.get(regime.value, '')
        lines.append(f"\n{emoji}大盘状态: {regime.value.upper()}")
        lines.append(f"   趋势: {regime_analysis['trend'].value}")
        lines.append(f"   评分: {regime_analysis['composite_score']:.1f}/100")
        lines.append(f"   置信度: {regime_analysis['confidence']:.0f}%")
        lines.append(f"   情绪: {regime_analysis['sentiment']}")

        # 大盘预测
        lines.append(f"\n【大盘预测】沪深300: {prediction['current_price']:.0f}")
        for period, target in prediction['targets'].items():
            lines.append(f"   {period}: {target}")

        # 支撑阻力
        lines.append(f"\n【关键位置】")
        lines.append(f"   支撑: {', '.join(prediction['supports'])}")
        lines.append(f"   阻力: {', '.join(prediction['resistances'])}")
        lines.append(f"   展望: {prediction['outlook']}")

        # 策略建议
        strategy = get_strategy_by_regime(regime)
        lines.append(f"\n【操作策略】")
        lines.append(f"   {strategy['name']}")
        lines.append(f"   建议仓位: {strategy['position']}")
        lines.append(f"   止损: {strategy['stop_loss']}")

        # 个股推荐
        if top_stocks:
            lines.append(f"\n【精选个股 TOP {len(top_stocks)}】")
            lines.append("-" * 50)
            for i, s in enumerate(top_stocks, 1):
                lines.append(f"{i}. {s['name']}({s['symbol']})")
                lines.append(f"   评分:{s['composite']:.0f} RPS:{s['rps']:.0f} 3M动量:{s['momentum_3m']:+.1f}%")
                lines.append(f"   建议仓位:{s['position']*100:.0f}% 止损:{s['stop_loss']:.2f} 风险收益比:{s['risk_reward']:.1f}:1")
        else:
            lines.append(f"\n【市场环境提示】")
            lines.append("  当前市场不适合买入，建议保持低仓位")

        lines.append("\n" + "=" * 50)
        return '\n'.join(lines)

    def run(self):
        """执行每日任务"""
        print("=" * 60)
        print("  每日统一投资任务启动")
        print("=" * 60)

        try:
            # 1. 获取指数数据
            print("\n[1/5] 获取沪深300数据...")
            index_data = self.get_index_data('000300.SH', days=250)
            print(f"  获取到 {len(index_data)} 条数据")

            # 2. 分析市场状态
            print("[2/5] 分析市场状态...")
            regime_result = self.market_analyzer.analyze(index_data)
            regime = regime_result.regime
            regime_analysis = {
                'regime': regime,
                'trend': regime_result.trend,
                'confidence': regime_result.confidence,
                'composite_score': regime_result.composite_score,
                'sentiment': regime_result.sentiment,
                'risk_level': regime_result.risk_level,
            }
            print(f"  市场状态: {regime.value} ({regime_result.confidence:.0f}%置信度)")

            # 3. 预测市场方向
            print("[3/5] 预测大盘方向...")
            prediction = self.predict_market_direction(index_data, regime)

            # 4. 筛选个股
            print("[4/5] 筛选个股...")
            top_stocks = []
            if regime in [MarketRegime.BULLISH, MarketRegime.NEUTRAL]:
                top_stocks = self.screen_stocks(top_n=5)
                print(f"  找到 {len(top_stocks)} 只候选股票")
            else:
                print(f"  当前市场({regime.value})不适合买入，跳过选股")

            # 5. 发送消息
            print("[5/5] 发送飞书消息...")
            message = self.build_wechat_message(regime, regime_analysis, prediction, top_stocks)

            # 保存报告
            report_dir = 'analysis_reports'
            os.makedirs(report_dir, exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            report_file = os.path.join(report_dir, f'unified_daily_{timestamp}.txt')
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write(message)

            # 发送飞书（检查是否需要发送）
            if should_send_message('daily_advisor', message):
                feishu_success = send_feishu_message(message)
                if feishu_success:
                    print(f"  飞书消息发送成功")
                    mark_message_sent('daily_advisor', message)
                else:
                    print(f"  飞书消息发送失败")
            else:
                print(f"  跳过发送（今日已发送）")

            # 同时发送企业微信（备用，不再检查重复）
            try:
                SendJcsMessage(
                    '每日投资建议',
                    message,
                    'hua.guo',
                    'ST_EMAIL;ST_WECHAT',
                    '每日投资建议推送'
                )
            except Exception as e:
                print(f"  企业微信发送失败: {e}")

            print(f"  报告已保存: {report_file}")

            return True

        except Exception as e:
            print(f"\n执行出错: {e}")
            import traceback
            traceback.print_exc()
            return False


def main():
    """主函数"""
    job = DailyUnifiedJob()
    success = job.run()
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
