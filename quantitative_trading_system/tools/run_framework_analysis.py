#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
技术分析投资框架 - 通用分析入口
用法:
  python run_framework_analysis.py [标的代码] [类型]
  股票示例: python run_framework_analysis.py 002602.SZ stock
  期货示例: python run_framework_analysis.py lh2607 futures
  默认: 300308.SZ stock
"""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import os
import subprocess
from datetime import datetime
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from indicators.technical_framework import TechnicalInvestmentFramework
from indicators.market_regime import MarketRegime, MarketRegimeAnalyzer, get_strategy_by_regime
from indicators.trend_indicators import SMA, EMA, MACD, DMI_ADX
from indicators.momentum_indicators import RSI, Stochastic, RPS
from indicators.volatility_indicators import BollingerBands, ATR
from indicators.volume_indicators import OBV
from reports.generator import AnalysisReportGenerator, generate_framework_report
import pandas as pd


def get_stock_data(symbol, days=250):
    """从Wind数据库获取股票数据"""
    import oracledb
    try:
        dsn = oracledb.connect(user="wind", password="windPrd22", dsn="10.1.33.123:1521/info")
        sql = f"""
        SELECT TRADE_DT, S_DQ_OPEN, S_DQ_HIGH, S_DQ_LOW, S_DQ_CLOSE, S_DQ_VOLUME
        FROM AShareEODPrices
        WHERE S_INFO_WINDCODE = '{symbol}'
        AND TRADE_DT >= '20240101'
        ORDER BY TRADE_DT ASC
        """
        df = pd.read_sql(sql, dsn)
        dsn.close()

        if df.empty:
            return None

        df['date'] = pd.to_datetime(df['TRADE_DT'], format='%Y%m%d')
        df = df.rename(columns={
            'S_DQ_OPEN': 'open', 'S_DQ_HIGH': 'high',
            'S_DQ_LOW': 'low', 'S_DQ_CLOSE': 'close',
            'S_DQ_VOLUME': 'volume'
        })
        return df[['date', 'open', 'high', 'low', 'close', 'volume']].tail(days).set_index('date')
    except Exception as e:
        print(f"获取股票数据失败: {e}")
        return None


def get_index_data(symbol, days=250):
    """从Wind数据库获取指数数据"""
    import oracledb
    try:
        dsn = oracledb.connect(user="wind", password="windPrd22", dsn="10.1.33.123:1521/info")
        sql = f"""
        SELECT TRADE_DT, S_DQ_OPEN, S_DQ_HIGH, S_DQ_LOW, S_DQ_CLOSE, S_DQ_VOLUME
        FROM AINDEXEODPRICES
        WHERE S_INFO_WINDCODE = '{symbol}'
        AND TRADE_DT >= '20240101'
        ORDER BY TRADE_DT ASC
        """
        df = pd.read_sql(sql, dsn)
        dsn.close()

        if df.empty:
            return None

        df['date'] = pd.to_datetime(df['TRADE_DT'], format='%Y%m%d')
        df = df.rename(columns={
            'S_DQ_OPEN': 'open', 'S_DQ_HIGH': 'high',
            'S_DQ_LOW': 'low', 'S_DQ_CLOSE': 'close',
            'S_DQ_VOLUME': 'volume'
        })
        return df[['date', 'open', 'high', 'low', 'close', 'volume']].tail(days).set_index('date')
    except Exception as e:
        print(f"获取指数数据失败: {e}")
        return None


def get_futures_data(symbol, days=250):
    """获取期货数据（使用akshare）"""
    try:
        import akshare as ak
        df = ak.futures_zh_daily_sina(symbol=symbol)
        if df is None or df.empty:
            return None

        df['date'] = pd.to_datetime(df['date'])
        df = df.rename(columns={
            'open': 'open', 'high': 'high', 'low': 'low',
            'close': 'close', 'volume': 'volume'
        })
        df = df[['date', 'open', 'high', 'low', 'close', 'volume']].tail(days)
        df.set_index('date', inplace=True)
        return df
    except Exception as e:
        print(f"获取期货数据失败: {e}")
        return None


def get_market_index():
    """获取大盘指数（沪深300）用于市场状态分析"""
    return get_index_data('000300.SH', days=250)


def main():
    # 解析命令行参数
    if len(sys.argv) > 1:
        symbol = sys.argv[1]
    else:
        symbol = '300308.SZ'

    asset_type = 'stock'  # 默认股票
    if len(sys.argv) > 2:
        if sys.argv[2].lower() in ['futures', 'future', '期货']:
            asset_type = 'futures'

    # 判断资产类型
    is_futures = asset_type == 'futures' or '.' not in symbol

    print("=" * 60)
    print(f"技术分析投资框架 - {'期货' if is_futures else '股票'} {symbol} 综合分析")
    print("=" * 60)

    # 获取数据
    if is_futures:
        print(f"\n获取 {symbol} 期货数据...")
        data = get_futures_data(symbol, days=250)
        market_data = get_market_index()  # 同时获取大盘
    else:
        print(f"\n获取 {symbol} 股票数据...")
        data = get_stock_data(symbol, days=250)
        market_data = None

    if data is None or data.empty:
        print("获取数据失败!")
        return

    print(f"获取到 {len(data)} 条数据")
    print(f"数据区间: {data.index[0]} ~ {data.index[-1]}")

    # 初始化框架
    framework = TechnicalInvestmentFramework()

    # 运行综合分析
    print("\n运行技术分析...")
    analysis = framework.analyze(data)

    # 打印报告
    print(generate_framework_report(analysis))

    # 详细指标
    print("\n" + "=" * 60)
    print("详细技术指标")
    print("=" * 60)

    close = data['close']

    print(f"\n【移动平均线】")
    print(f"  SMA(10):  ¥{SMA(data, 10).iloc[-1]:.2f}")
    print(f"  SMA(50):  ¥{SMA(data, 50).iloc[-1]:.2f}")
    print(f"  SMA(200): ¥{SMA(data, 200).iloc[-1]:.2f}")
    print(f"  EMA(12):  ¥{EMA(data, 12).iloc[-1]:.2f}")
    print(f"  EMA(26):  ¥{EMA(data, 26).iloc[-1]:.2f}")

    print(f"\n【MACD】")
    macd = MACD(data)
    print(f"  MACD线:    {macd['macd'].iloc[-1]:.4f}")
    print(f"  Signal线:  {macd['signal'].iloc[-1]:.4f}")
    print(f"  Histogram: {macd['histogram'].iloc[-1]:.4f}")
    macd_cross = "金叉" if macd['macd'].iloc[-1] > macd['signal'].iloc[-1] else "死叉"
    print(f"  状态: {macd_cross}")

    print(f"\n【RSI】")
    rsi = RSI(data)
    print(f"  RSI(14): {rsi.iloc[-1]:.1f}")
    if rsi.iloc[-1] > 70:
        print(f"  状态: 超买区域")
    elif rsi.iloc[-1] < 30:
        print(f"  状态: 超卖区域")
    else:
        print(f"  状态: 中性区域")

    print(f"\n【随机指标】")
    stoch = Stochastic(data)
    print(f"  %K: {stoch['k'].iloc[-1]:.1f}")
    print(f"  %D: {stoch['d'].iloc[-1]:.1f}")
    stoch_cross = "多头交叉" if stoch['k'].iloc[-1] > stoch['d'].iloc[-1] else "空头交叉"
    print(f"  状态: {stoch_cross}")

    print(f"\n【RPS 相对价格强度】")
    rps = RPS(data, period=20)
    rps_val = rps.iloc[-1]
    print(f"  RPS(20日): {rps_val:.1f}")
    if rps_val >= 80:
        print(f"  状态: 强势股(>80)")
    elif rps_val >= 60:
        print(f"  状态: 偏强(60-80)")
    elif rps_val >= 40:
        print(f"  状态: 中性(40-60)")
    elif rps_val >= 20:
        print(f"  状态: 偏弱(20-40)")
    else:
        print(f"  状态: 弱势股(<20)")

    print(f"\n【布林带】")
    bb = BollingerBands(data)
    print(f"  上轨:  ¥{bb['upper'].iloc[-1]:.2f}")
    print(f"  中轨:  ¥{bb['middle'].iloc[-1]:.2f}")
    print(f"  下轨:  ¥{bb['lower'].iloc[-1]:.2f}")
    print(f"  %B:   {bb['percent_b'].iloc[-1]:.2f}")
    print(f"  带宽:  {bb['bandwidth'].iloc[-1]:.2f}")

    print(f"\n【ATR】")
    atr = ATR(data)
    print(f"  ATR(14): {atr.iloc[-1]:.2f}")
    print(f"  ATR%:   {(atr.iloc[-1]/close.iloc[-1]*100):.2f}%")

    print(f"\n【DMI+ADX】")
    dmi = DMI_ADX(data)
    print(f"  +DI: {dmi['plus_di'].iloc[-1]:.1f}")
    print(f"  -DI: {dmi['minus_di'].iloc[-1]:.1f}")
    print(f"  ADX: {dmi['adx'].iloc[-1]:.1f}")
    if dmi['adx'].iloc[-1] > 25:
        print(f"  状态: 趋势市场 (ADX>25)")
    else:
        print(f"  状态: 震荡市场 (ADX<25)")

    print(f"\n【OBV】")
    obv = OBV(data)
    obv_change = (obv.iloc[-1] - obv.iloc[-20]) / abs(obv.iloc[-20]) * 100 if obv.iloc[-20] != 0 else 0
    print(f"  OBV: {obv.iloc[-1]:,.0f}")
    print(f"  20日变化: {obv_change:+.1f}%")

    # 投资决策
    print("\n" + "=" * 60)
    print("投资决策总结")
    print("=" * 60)

    rec = analysis['signals']['recommendation']
    composite = analysis['signals']['composite']

    if rec == 'strong_buy':
        action = "强烈买入"
    elif rec == 'buy':
        action = "买入"
    elif rec == 'sell':
        action = "卖出"
    elif rec == 'strong_sell':
        action = "强烈卖出"
    else:
        action = "持有观望"

    trend = analysis['trend']['direction']
    momentum = analysis['momentum']['signal']
    risk = analysis['risk']['risk_reward_ratio']

    print(f"\n综合评分: {composite:.1f}/100")
    print(f"建议操作: {action}")
    print(f"趋势状态: {trend}")
    print(f"动量信号: {momentum}")
    print(f"风险收益比: {risk:.2f}:1")

    if composite >= 70 and risk >= 2:
        print(f"\n技术面支持买入信号")
        print(f"  - 趋势向好，{trend}")
        print(f"  - 风险收益比 {risk:.2f}:1 满足要求")
        print(f"  - 建议仓位: {analysis['risk']['position_size_recommendation']*100:.0f}%")
        print(f"  - 止损位: ¥{analysis['risk']['stop_loss']:.2f}")
        print(f"  - 止盈位: ¥{analysis['risk']['take_profit']:.2f}")
    elif composite < 40:
        print(f"\n技术面偏弱，建议观望或减仓")
    else:
        print(f"\n技术面中性，建议谨慎操作")

    # 生成报告文档
    print("\n" + "=" * 60)
    print("生成分析报告...")
    print("=" * 60)

    # 使用新的报告生成器
    report_gen = AnalysisReportGenerator()
    report_path = report_gen.generate(symbol, data, analysis)

    print(f"\n报告已生成: {report_path}")

    # 自动打开报告
    print(f"\n正在打开报告...")
    try:
        if sys.platform == 'win32':
            os.startfile(report_path)
        elif sys.platform == 'darwin':
            subprocess.run(['open', report_path])
        else:
            subprocess.run(['xdg-open', report_path])
    except Exception as e:
        print(f"自动打开失败，请手动打开: {report_path}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
