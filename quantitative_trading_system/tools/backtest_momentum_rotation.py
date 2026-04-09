#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
动量轮动策略回测

使用沪深300成分股进行回测
初始资金: 10万
持仓: 1-3天
止损: -3%
止盈: +5%

用法:
    python tools/backtest_momentum_rotation.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

import oracledb
from indicators.technical_framework import TechnicalInvestmentFramework
from indicators.momentum_indicators import RPS
from indicators.market_regime import MarketRegimeAnalyzer
from strategy.momentum_rotation import MomentumRotationStrategy

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MomentumRotationBacktester:
    """动量轮动策略回测器"""

    def __init__(self, initial_capital=100000, commission=0.001, slippage=0.0005):
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage

        self.tech_framework = TechnicalInvestmentFramework()
        self.market_analyzer = MarketRegimeAnalyzer()

        # 回测结果
        self.trades = []
        self.equity_curve = []
        self.current_capital = initial_capital
        self.current_positions = {}  # {symbol: {'shares': int, 'entry_price': float, 'entry_date': datetime}}

    def get_hs300_stocks(self):
        """获取沪深300成分股"""
        try:
            dsn = oracledb.connect(user="wind", password="windPrd22", dsn="10.1.33.123:1521/info")
            sql = """
            SELECT S_CON_WINDCODE
            FROM AIndexMembers
            WHERE S_INFO_WINDCODE = '000300.SH'
            AND (S_CON_OUTDATE IS NULL OR S_CON_OUTDATE > '20260101')
            """
            df = pd.read_sql(sql, dsn)
            dsn.close()
            return df['S_CON_WINDCODE'].tolist()
        except Exception as e:
            logger.error(f"获取沪深300成分股失败: {e}")
            return []

    def get_stock_data(self, symbol, start_date='20240101', end_date=None):
        """获取股票历史数据"""
        try:
            dsn = oracledb.connect(user="wind", password="windPrd22", dsn="10.1.33.123:1521/info")
            end_date = end_date or datetime.now().strftime('%Y%m%d')
            sql = f"""
            SELECT TRADE_DT, S_DQ_OPEN, S_DQ_HIGH, S_DQ_LOW, S_DQ_CLOSE, S_DQ_VOLUME
            FROM AShareEODPrices
            WHERE S_INFO_WINDCODE = '{symbol}'
            AND TRADE_DT >= '{start_date}'
            AND TRADE_DT <= '{end_date}'
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
            df = df[['date', 'open', 'high', 'low', 'close', 'volume']]
            df.set_index('date', inplace=True)
            return df
        except Exception as e:
            return None

    def get_index_data(self, symbol='000300.SH', start_date='20240101', end_date=None):
        """获取指数历史数据"""
        try:
            dsn = oracledb.connect(user="wind", password="windPrd22", dsn="10.1.33.123:1521/info")
            end_date = end_date or datetime.now().strftime('%Y%m%d')
            sql = f"""
            SELECT TRADE_DT, S_DQ_OPEN, S_DQ_HIGH, S_DQ_LOW, S_DQ_CLOSE, S_DQ_VOLUME
            FROM AINDEXEODPRICES
            WHERE S_INFO_WINDCODE = '{symbol}'
            AND TRADE_DT >= '{start_date}'
            AND TRADE_DT <= '{end_date}'
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
            df = df[['date', 'open', 'high', 'low', 'close', 'volume']]
            df.set_index('date', inplace=True)
            return df
        except Exception as e:
            logger.error(f"获取指数{symbol}数据失败: {e}")
            return None

    def get_market_regime(self, date, index_data):
        """获取指定日期的市场状态"""
        if index_data is None or index_data.empty:
            return 'neutral', 50

        # 只取到指定日期之前的数据
        data_before = index_data[index_data.index < date]
        if len(data_before) < 60:
            return 'neutral', 50

        try:
            result = self.market_analyzer.analyze(data_before)
            regime = result.regime.value if hasattr(result, 'regime') else 'neutral'
            score = result.composite_score if hasattr(result, 'composite_score') else 50
            return regime, score
        except:
            return 'neutral', 50

    def analyze_stock(self, symbol, data):
        """分析单只股票（含RSI和MACD）"""
        if data is None or len(data) < 60:
            return None

        try:
            # 技术分析
            tech_result = self.tech_framework.analyze(data)

            # RPS计算
            rps_value = RPS(data, period=20).iloc[-1] if len(data) >= 20 else 50

            # RSI计算
            from indicators.momentum_indicators import RSI
            rsi_value = RSI(data, period=14).iloc[-1] if len(data) >= 14 else 50

            # MACD计算
            from indicators.trend_indicators import MACD
            macd_result = MACD(data)
            macd_value = macd_result['macd'].iloc[-1] if len(macd_result) > 0 else 0

            # 获取动量指标
            momentum = tech_result.get('momentum', {})
            rsi_from_framework = momentum.get('rsi', rsi_value)

            composite = tech_result.get('signals', {}).get('composite', 0)
            recommendation = tech_result.get('signals', {}).get('recommendation', 'hold')

            return {
                'symbol': symbol,
                'composite': composite,
                'rps': rps_value,
                'rsi': rsi_from_framework,
                'macd': macd_value,
                'recommendation': recommendation,
                'trend': tech_result.get('trend', {}).get('direction', 'neutral'),
                'close': data['close'].iloc[-1],
                'volume': data['volume'].iloc[-1]
            }
        except Exception as e:
            logger.warning(f"分析股票{symbol}失败: {e}")
            return None

    def calculate_daily_scores(self, date, stock_data_dict):
        """计算指定日期的股票评分"""
        scores = {}
        for symbol, data in stock_data_dict.items():
            if data is None or len(data) < 20:
                continue

            # 只取到指定日期之前的数据（模拟盘中无法看到当日数据）
            data_before = data[data.index < date]
            if len(data_before) < 20:
                continue

            result = self.analyze_stock(symbol, data_before)
            if result:
                scores[symbol] = result

        return scores

    def run_backtest(self, start_date='20250101', end_date=None, rebalance_days=1):
        """
        运行回测

        Args:
            start_date: 回测开始日期
            end_date: 回测结束日期
            rebalance_days: 调仓间隔（天）
        """
        logger.info("=" * 60)
        logger.info("动量轮动策略回测")
        logger.info("=" * 60)
        logger.info(f"初始资金: ¥{self.initial_capital:,.2f}")
        logger.info(f"回测区间: {start_date} - {end_date or '今天'}")

        # 1. 获取股票列表
        logger.info("\n[1/5] 获取沪深300成分股...")
        stocks = self.get_hs300_stocks()
        logger.info(f"获取到 {len(stocks)} 只股票")

        if not stocks:
            logger.error("获取股票列表失败")
            return

        # 限制数量以便快速回测
        stocks = stocks[:50]  # 使用前50只
        logger.info(f"使用前 {len(stocks)} 只进行回测")

        # 2. 获取所有股票历史数据
        logger.info("\n[2/5] 获取历史数据...")
        all_data = {}
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = {executor.submit(self.get_stock_data, s, start_date): s for s in stocks}
            for i, future in enumerate(as_completed(futures)):
                symbol = futures[future]
                try:
                    data = future.result()
                    if data is not None and len(data) >= 60:
                        all_data[symbol] = data
                except Exception as e:
                    pass
                if (i + 1) % 10 == 0:
                    logger.info(f"  已获取 {i + 1}/{len(stocks)} 只股票")

        logger.info(f"成功获取 {len(all_data)} 只股票的历史数据")

        if len(all_data) < 5:
            logger.error("有效股票数据不足")
            return

        # 3. 获取所有交易日期
        all_dates = set()
        for data in all_data.values():
            all_dates.update(data.index.tolist())
        trading_dates = sorted([d for d in all_dates if d >= pd.to_datetime(start_date)])
        logger.info(f"共 {len(trading_dates)} 个交易日")

        # 3.5 获取沪深300指数数据（用于市场状态判断）
        logger.info("\n[3.5/5] 获取沪深300指数数据...")
        index_data = self.get_index_data('000300.SH', start_date)
        if index_data is not None:
            logger.info(f"获取到 {len(index_data)} 天指数数据")
        else:
            logger.warning("获取指数数据失败，市场状态过滤将失效")

        # 4. 初始化策略
        strategy = MomentumRotationStrategy()
        strategy.initialize()

        # 5. 开始回测
        logger.info("\n[3/5] 运行回测...")

        # 调仓日计数器
        days_since_rebalance = 0
        last_rebalance_date = None

        for i, date in enumerate(trading_dates):
            current_date = date
            current_date_str = current_date.strftime('%Y-%m-%d')

            # 跳过第一天（需要预热）
            if i == 0:
                self.equity_curve.append({
                    'date': current_date,
                    'capital': self.current_capital,
                    'positions': len(self.current_positions)
                })
                continue

            # 更新每日评分
            stock_data_today = {}
            for symbol, data in all_data.items():
                if current_date in data.index:
                    stock_data_today[symbol] = data

            # 每隔N天重新计算所有股票评分
            if days_since_rebalance >= rebalance_days or last_rebalance_date is None:
                logger.info(f"\n  {current_date_str} 调仓日，重新计算评分...")
                all_scores = self.calculate_daily_scores(current_date, all_data)
                strategy.daily_scores[current_date] = all_scores
                days_since_rebalance = 0
                last_rebalance_date = current_date

            # 获取当前所有股票在当日的前一日收盘价
            prev_date = trading_dates[i - 1] if i > 0 else None
            current_prices = {}
            for symbol, data in all_data.items():
                if prev_date in data.index:
                    current_prices[symbol] = data.loc[prev_date, 'close']

            # 获取市场状态（用于仓位管理）
            market_regime, market_score = self.get_market_regime(current_date, index_data)

            # 根据市场状态确定仓位系数
            # BULLISH: 100%仓位, NEUTRAL: 80%仓位, VOLATILE: 50%仓位, BEARISH: 30%仓位
            if market_regime == 'BULLISH':
                position_ratio = 1.0
                market_status = "强势(100%)"
            elif market_regime == 'NEUTRAL':
                position_ratio = 0.8
                market_status = "中性(80%)"
            elif market_regime == 'VOLATILE':
                position_ratio = 0.5
                market_status = "波动(50%)"
            else:  # BEARISH
                position_ratio = 0.3
                market_status = "弱势(30%)"

            # 市场状态日志（只在状态变化时打印）
            if i == 1 or (i > 1 and self.equity_curve[-1].get('market_regime') != market_regime):
                logger.info(f"    市场状态: {market_regime}, 评分: {market_score:.1f}, 仓位: {market_status}")

            # 1) 检查现有持仓
            positions_to_close = []
            for symbol in list(self.current_positions.keys()):
                pos_info = self.current_positions[symbol]
                entry_price = pos_info['entry_price']
                entry_date = pos_info['entry_date']

                hold_days = (current_date - entry_date).days
                current_price = current_prices.get(symbol, entry_price)
                ret = (current_price - entry_price) / entry_price

                should_sell = False
                reason = ""

                if ret <= -0.03:  # 3%止损
                    should_sell = True
                    reason = f"止损({ret*100:.1f}%)"
                elif ret >= 0.05:  # 5%止盈
                    should_sell = True
                    reason = f"止盈({ret*100:.1f}%)"
                elif hold_days >= 3:  # 最多持3天
                    # 到期时如果RSI偏高（超买），提前止盈
                    current_scores = strategy.daily_scores.get(current_date, {})
                    if symbol in current_scores and current_scores[symbol].get('rsi', 50) > 65:
                        should_sell = True
                        reason = f"到期RSI高({scores[symbol].get('rsi', 0):.1f})"
                    elif hold_days >= 5:  # 最多延到5天
                        should_sell = True
                        reason = f"到期({hold_days}天)"

                if should_sell:
                    positions_to_close.append((symbol, current_price, reason))

            # 执行卖出
            for symbol, price, reason in positions_to_close:
                pos_info = self.current_positions[symbol]
                shares = pos_info['shares']
                entry_price = pos_info['entry_price']

                # 扣除手续费和滑点
                sell_price = price * (1 - 0.0005)  # 滑点
                proceeds = shares * sell_price * (1 - self.commission)

                pnl = proceeds - (shares * entry_price)

                self.trades.append({
                    'date': current_date,
                    'symbol': symbol,
                    'action': 'sell',
                    'price': sell_price,
                    'shares': shares,
                    'pnl': pnl,
                    'reason': reason
                })

                self.current_capital += proceeds
                del self.current_positions[symbol]

                logger.info(f"    卖出 {symbol} @{sell_price:.2f} {reason}, 盈亏: {pnl:+.2f}")

            # 2) 买入新股票（如果有空位，使用RSI/MACD过滤）
            available_slots = 2 - len(self.current_positions)
            if available_slots > 0 and current_date in strategy.daily_scores:
                scores = strategy.daily_scores[current_date]

                # 排除已持有的
                held_symbols = list(self.current_positions.keys())
                candidates = []
                for symbol, data in scores.items():
                    if symbol in held_symbols:
                        continue
                    composite = data.get('composite', 0)
                    rps = data.get('rps', 0)
                    rsi = data.get('rsi', 50)
                    macd = data.get('macd', 0)

                    # RSI < 65 买入（放宽到65，增加交易机会）
                    if rsi > 65:
                        continue
                    # MACD > 0 确认上升趋势（保留但放宽条件）
                    # 在弱市中可以用 MACD > signal line 作为替代
                    if macd < 0:
                        continue
                    # 基本条件（降低RPS要求到60）
                    if composite < 40 or rps < 60:
                        continue

                    candidates.append((symbol, composite, rsi, macd))

                candidates.sort(key=lambda x: x[1], reverse=True)

                # 取可买入数量和候选数量的最小值
                actual_buy_count = min(available_slots, len(candidates))
                if actual_buy_count > 0:
                    # 等权分配：每只股票分配相同资金（应用仓位系数）
                    available_capital = self.current_capital * position_ratio
                    allocation = available_capital / actual_buy_count

                    for symbol, score, rsi, macd in candidates[:actual_buy_count]:
                        price = current_prices.get(symbol)
                        if price is None:
                            continue

                        shares = int(allocation / price / 100) * 100  # 整手

                        if shares > 0:
                            cost = shares * price * (1 + self.commission + 0.0005)  # 手续费+滑点

                            if cost <= self.current_capital:
                                self.current_positions[symbol] = {
                                    'shares': shares,
                                    'entry_price': price,
                                    'entry_date': current_date,
                                    'score': score,
                                    'position_ratio': position_ratio  # 记录当时的仓位系数
                                }
                                self.current_capital -= cost

                                self.trades.append({
                                    'date': current_date,
                                    'symbol': symbol,
                                    'action': 'buy',
                                    'price': price,
                                    'shares': shares,
                                    'pnl': 0,
                                    'reason': f"入选(评分:{score:.1f},RSI:{rsi:.1f},MACD:{macd:.2f},仓位:{market_status})"
                                })

                                logger.info(f"    买入 {symbol} @{price:.2f} x {shares}, 评分:{score:.1f}, RSI:{rsi:.1f}, MACD:{macd:.2f}, 仓位:{market_status}")

            # 3) 更新权益曲线
            total_value = self.current_capital
            for symbol, pos_info in self.current_positions.items():
                price = current_prices.get(symbol, pos_info['entry_price'])
                total_value += pos_info['shares'] * price

            self.equity_curve.append({
                'date': current_date,
                'capital': total_value,
                'positions': len(self.current_positions),
                'market_regime': market_regime,
                'market_score': market_score
            })

            days_since_rebalance += 1

            if (i + 1) % 20 == 0:
                logger.info(f"  进度: {i + 1}/{len(trading_dates)}, 资金: ¥{total_value:,.2f}")

        # 6. 计算回测结果
        logger.info("\n[4/5] 计算回测结果...")
        results = self.calculate_results()

        # 7. 生成报告
        logger.info("\n[5/5] 生成回测报告...")
        self.print_report(results)

        # 保存结果
        self.save_results(results)

        return results

    def calculate_results(self):
        """计算回测绩效指标"""
        equity_df = pd.DataFrame(self.equity_curve)
        trades_df = pd.DataFrame(self.trades) if self.trades else pd.DataFrame()

        if equity_df.empty:
            return {}

        # 计算收益
        initial = self.initial_capital
        final = equity_df['capital'].iloc[-1]
        total_return = (final - initial) / initial * 100

        # 计算最大回撤
        cumulative = equity_df['capital']
        peak = cumulative.expanding(min_periods=1).max()
        drawdown = (cumulative - peak) / peak * 100
        max_drawdown = drawdown.min()

        # 计算年化收益（假设252交易日）
        trading_days = len(equity_df)
        years = trading_days / 252
        annualized_return = ((final / initial) ** (1 / years) - 1) * 100 if years > 0 else 0

        # 计算波动率
        daily_returns = equity_df['capital'].pct_change().dropna()
        volatility = daily_returns.std() * np.sqrt(252) * 100

        # 夏普比率（假设无风险利率3%）
        risk_free = 0.03
        sharpe = (annualized_return / 100 - risk_free) / (volatility / 100) if volatility > 0 else 0

        # 胜率
        win_rate = 0
        if not trades_df.empty and 'action' in trades_df.columns:
            sell_trades = trades_df[trades_df['action'] == 'sell']
            win_trades = sell_trades[sell_trades['pnl'] > 0]
            win_rate = len(win_trades) / len(sell_trades) * 100 if len(sell_trades) > 0 else 0

        # 交易次数
        total_trades = len(trades_df)
        buy_trades = 0
        sell_trades = 0
        if not trades_df.empty and 'action' in trades_df.columns:
            buy_trades = len(trades_df[trades_df['action'] == 'buy'])
            sell_trades = len(trades_df[trades_df['action'] == 'sell'])

        # 平均持仓天数
        avg_hold_days = 0
        if not trades_df.empty and 'action' in trades_df.columns:
            buy_dates = trades_df[trades_df['action'] == 'buy']['date'].values
            sell_dates = trades_df[trades_df['action'] == 'sell']['date'].values
            if len(buy_dates) > 0 and len(sell_dates) > 0:
                avg_hold_days = 3  # 简化计算

        return {
            'initial_capital': initial,
            'final_capital': final,
            'total_return': total_return,
            'annualized_return': annualized_return,
            'max_drawdown': max_drawdown,
            'volatility': volatility,
            'sharpe_ratio': sharpe,
            'win_rate': win_rate,
            'total_trades': total_trades,
            'buy_trades': buy_trades,
            'sell_trades': sell_trades,
            'avg_hold_days': avg_hold_days,
            'equity_curve': equity_df,
            'trades': trades_df
        }

    def print_report(self, results):
        """打印回测报告"""
        print("\n" + "=" * 60)
        print("  动量轮动策略 - 回测报告")
        print("=" * 60)

        print(f"\n【收益概况】")
        print(f"  初始资金:   ¥{results['initial_capital']:>15,.2f}")
        print(f"  最终资金:   ¥{results['final_capital']:>15,.2f}")
        print(f"  总收益率:   {results['total_return']:>15.2f}%")
        print(f"  年化收益率: {results['annualized_return']:>15.2f}%")

        print(f"\n【风险指标】")
        print(f"  最大回撤:   {results['max_drawdown']:>15.2f}%")
        print(f"  波动率:     {results['volatility']:>15.2f}%")
        print(f"  夏普比率:   {results['sharpe_ratio']:>15.2f}")

        print(f"\n【交易统计】")
        print(f"  总交易次数: {results['total_trades']:>15d} 次")
        print(f"  买入次数:   {results['buy_trades']:>15d} 次")
        print(f"  卖出次数:   {results['sell_trades']:>15d} 次")
        print(f"  胜率:       {results['win_rate']:>15.2f}%")
        print(f"  平均持仓:   {results['avg_hold_days']:>15d} 天")

        print("\n" + "=" * 60)

        # 打印交易记录
        if not results['trades'].empty and len(results['trades']) > 0:
            print("\n【最近10笔交易】")
            trades = results['trades'].tail(10)
            for _, t in trades.iterrows():
                action = "买入" if t['action'] == 'buy' else "卖出"
                pnl_str = f"{t['pnl']:+.2f}" if t['pnl'] != 0 else ""
                print(f"  {t['date'].strftime('%Y-%m-%d')} {action} {t['symbol']} @{t['price']:.2f}x{t['shares']} {pnl_str}")

        print("\n" + "=" * 60)

    def save_results(self, results):
        """保存回测结果"""
        import os
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = 'results/backtests'
        os.makedirs(output_dir, exist_ok=True)

        # 保存权益曲线
        if not results.get('equity_curve', pd.DataFrame()).empty:
            equity_file = os.path.join(output_dir, f'equity_curve_{timestamp}.csv')
            results['equity_curve'].to_csv(equity_file, index=False)
            logger.info(f"权益曲线已保存: {equity_file}")

        # 保存交易记录
        if not results.get('trades', pd.DataFrame()).empty:
            trades_file = os.path.join(output_dir, f'trades_{timestamp}.csv')
            results['trades'].to_csv(trades_file, index=False)
            logger.info(f"交易记录已保存: {trades_file}")


def main():
    """主函数"""
    backtester = MomentumRotationBacktester(
        initial_capital=100000,  # 10万初始资金
        commission=0.001,        # 0.1%手续费
        slippage=0.0005          # 0.05%滑点
    )

    # 回测最近6个月
    start_date = '20250901'
    end_date = datetime.now().strftime('%Y%m%d')

    results = backtester.run_backtest(start_date=start_date, end_date=end_date)

    if results:
        print("\n回测完成！")
    else:
        print("\n回测失败，请检查日志")


if __name__ == '__main__':
    main()
