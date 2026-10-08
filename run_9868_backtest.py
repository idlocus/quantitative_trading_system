#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""9868.HK 纯趋势跟随策略回测"""

import sys, os
sys.path.insert(0, '.')

from data.wind_db import WindDBDataSource
from datetime import date, datetime
import pandas as pd
from indicators.trend_indicators import DMI_ADX
from selection.scanner import load_strategy_from_yaml
from backtesting.engine import BacktestEngine, BacktestConfig, MarketData

print('=' * 60)
print('9868.HK 策略回测')
print('=' * 60)

# 连接数据库（凭据从 .env / 环境变量读取）
from dotenv import load_dotenv
load_dotenv()
dsn = os.getenv('ORACLE_DSN')
user = os.getenv('ORACLE_USER')
password = os.getenv('ORACLE_PASSWORD')
if not (dsn and user and password):
    raise SystemExit('请在 .env 中配置 ORACLE_DSN / ORACLE_USER / ORACLE_PASSWORD')

db = WindDBDataSource(oracle_dsn=dsn, user=user, password=password)

# 获取9868.HK K线
end = date(2026, 6, 2)
start = date(2024, 6, 3)
df = db.get_kline('9868.HK', start, end)
print(f'数据: {len(df)} 行, {df["date"].iloc[0]} 至 {df["date"].iloc[-1]}')

# 买入持有收益率
buy_hold = (df['close'].iloc[-1] / df['close'].iloc[0] - 1)
print(f'买入持有收益: {buy_hold*100:+.1f}%')

# 计算ADX
dmi = DMI_ADX(df, period=14)
adx_series = dmi['adx']
print(f'ADX范围: {adx_series.min():.1f} ~ {adx_series.max():.1f}, 均值: {adx_series.mean():.1f}')
print(f'ADX>25天数: {(adx_series > 25).sum()} / {len(adx_series)}')

# 准备市场数据（单只股票）
class SingleStockData:
    def __init__(self, kline_df):
        self._df = kline_df.set_index('date')
        self._symbols = ['9868.HK']
    def get_kline(self, symbol, start, end):
        return self._df[(self._df.index >= pd.Timestamp(start)) & (self._df.index <= pd.Timestamp(end))].reset_index()
    def get_close(self, symbol, trading_date):
        idx = self._df.index[self._df.index == pd.Timestamp(trading_date)]
        if len(idx) == 0:
            return 0.0
        return float(self._df.loc[idx[0], 'close'])
    def get_tradable_symbols(self, date):
        return ['9868.HK']

single_ds = SingleStockData(df)

# 交易日列表（直接用实际数据日期，排除节假日）
md = MarketData(single_ds)
md.add_trading_dates(df['date'].tolist())
md._symbols = ['9868.HK']

# 使用纯趋势跟随策略
yaml_path = 'selection/strategies.yml'
cfg = load_strategy_from_yaml('pure_trend_follow', yaml_path)
print(f'\n策略: {cfg.name}')
for r in cfg.indicators:
    print(f'  - {r.name or "(op)"} {r.operator} {r.threshold} (weight={r.weight})')

from strategy.momentum import MomentumStrategy
strategy = MomentumStrategy(cfg, SingleStockData(df))

# 尝试加载其他策略
configs = {
    'pure_trend_follow': load_strategy_from_yaml('pure_trend_follow', yaml_path),
    'enhanced_momentum': load_strategy_from_yaml('enhanced_momentum', yaml_path),
    'short_term_momentum': load_strategy_from_yaml('short_term_momentum', yaml_path),
}
strategies = {}
for name, c in configs.items():
    strategies[name] = MomentumStrategy(c, SingleStockData(df))

def run_backtest(strategy_obj, name):
    engine_cfg = BacktestConfig(
        initial_capital=100000,
        commission=0.0003,
        stop_loss=0.05,
        take_profit=0.10,
        trailing_stop=0.0,       # 单标的趋势策略关闭移动止损
        max_holding_days=0,      # 关闭时间止损（趋势策略需要长持仓）
        max_positions=1
    )
    engine = BacktestEngine(engine_cfg)
    result = engine.run(strategy_obj, md)
    return result

print()
print('=' * 60)
results = {}
for name, strat in strategies.items():
    r = run_backtest(strat, name)
    results[name] = r
    print(f'\n策略: {name}')
    print(f'  收益率:   {r.total_return*100:+.1f}%')
    print(f'  夏普:     {r.sharpe_ratio:.2f}')
    print(f'  最大回撤: {r.max_drawdown*100:.1f}%')
    print(f'  交易次数: {r.trade_count}')

print()
print('=' * 60)
print('买入持有: {:+.1f}%'.format(buy_hold*100))
print('=' * 60)

db.close()