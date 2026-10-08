#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Full rolling window backtest 2020-2024"""

import sys, os
sys.path.insert(0, '.')




from config.system_config import SystemConfig
from data.base import DataLoader
from datetime import date, datetime
import pandas as pd
from risk.constants import RiskParams
from risk.manager import RiskManager
from strategy import MomentumStrategy
from selection.scanner import load_strategy_from_yaml
from backtesting.engine import BacktestEngine, BacktestConfig, MarketData
from backtesting.reporter import BacktestReporter

print('=' * 60)
print('滚动窗口回测 - 2020-2024')
print('=' * 60)

config = SystemConfig()
loader = DataLoader.from_winddb(config.oracle_dsn, config.oracle_user, config.oracle_password)
all_symbols = loader.load_symbols()
print('Total symbols:', len(all_symbols))

test_symbols = all_symbols[:500]
print('Testing with', len(test_symbols), 'symbols')

start = date(2019, 7, 1)
end = date(2024, 12, 31)
print('Loading klines', start, 'to', end, '...')
cache = loader.preload_klines(test_symbols, start, end)
print('Cached:', len(cache._cache), 'symbols')

yaml_path = 'selection/strategies.yml'
cfg = load_strategy_from_yaml('short_term_momentum', yaml_path)
print('Strategy:', cfg.name)
strategy = MomentumStrategy(cfg, cache)

md = MarketData(cache)
for td in pd.bdate_range(start=start, end=end):
    md.add_trading_date(td.date())
md._symbols = test_symbols
print('Trading dates:', len(md.trading_dates))

engine_cfg = BacktestConfig(
    initial_capital=1000000,
    commission=0.0003,
    stop_loss=0.03,
    take_profit=0.06,
    max_positions=5
)
risk_params = RiskParams(STOP_LOSS=0.03, TAKE_PROFIT=0.06)
risk_mgr = RiskManager(params=risk_params, data_source=cache)

engine = BacktestEngine.__new__(BacktestEngine)
engine.config = engine_cfg
engine.risk_manager = risk_mgr
engine.trades = []
engine.positions = {}
engine.cash = engine_cfg.initial_capital
engine.equity_curve = []
engine._trade_id = 0
engine._position_entry_dates = {}
engine._data_source = cache
engine._market_env_provider = None
engine._sector_provider = None
engine._market_env_scorer = None
engine._sector_rotator = None
engine._current_market_env = None
engine._current_allowed_sectors = None
engine._logger = None
engine._current_phase = 'IS'
engine._data = md

print('Running rolling window backtest...')
result = engine.run_rolling(strategy, md, train_window_days=252, oos_ratio=0.2)

print()
print('=' * 60)
print('回测结果汇总')
print('=' * 60)
print('总交易次数:', result.trade_count)
print('总收益率:  ', '{:+.2%}'.format(result.total_return))
print('年化收益率:', '{:+.2%}'.format(result.annual_return))
print('夏普比率:  ', '{:+.2f}'.format(result.sharpe_ratio))
print('最大回撤:  ', '{:+.2%}'.format(result.max_drawdown))
print('胜率:      ', '{:.2%}'.format(result.win_rate))
print('平均持仓:  ', '{:.1f} 天'.format(result.avg_holding_days))

SEP = '-' * 90
print()
print(SEP)
hdr = '| {:<12} | {:<12} | {:<12} | {:>8} | {:>7} | {:>9} | {:>6} |'.format(
    '阶段', '开始', '结束', '收益率', '夏普', '最大回撤', '交易数')
print(hdr)
print(SEP)
for ph in result.phase_results:
    row = '| {:<12} | {:<12} | {:<12} | {:>+7.2%} | {:>+6.2f} | {:>+8.2%} | {:>6} |'.format(
        ph.phase_name, ph.start_date, ph.end_date,
        ph.total_return, ph.sharpe_ratio, ph.max_drawdown, ph.trade_count)
    print(row)

is_results = [ph for ph in result.phase_results if '_IS' in ph.phase_name]
oos_results = [ph for ph in result.phase_results if '_OOS' in ph.phase_name]
if is_results and oos_results:
    avg_is_ret = sum(ph.total_return for ph in is_results) / len(is_results)
    avg_oos_ret = sum(ph.total_return for ph in oos_results) / len(oos_results)
    gap = abs(avg_is_ret - avg_oos_ret)
    print()
    print('样本内(IS)平均收益率: {:+.2%}'.format(avg_is_ret))
    print('样本外(OOS)平均收益率: {:+.2%}'.format(avg_oos_ret))
    print('IS/OOS差距: {:.2%}'.format(gap))
    if avg_oos_ret > 0:
        print('结论: OOS盈利，策略有参考价值')
    else:
        print('结论: OOS亏损，策略过拟合风险较高')

reporter = BacktestReporter()
report_dir = os.path.join(os.path.dirname(os.path.abspath('.')), 'results')
os.makedirs(report_dir, exist_ok=True)
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

md_path = os.path.join(report_dir, 'rolling_report_{}.md'.format(timestamp))
with open(md_path, 'w', encoding='utf-8') as f:
    f.write(reporter.generate_markdown(result))
print()
print('Markdown报告:', md_path)

html_path = os.path.join(report_dir, 'rolling_report_{}.html'.format(timestamp))
with open(html_path, 'w', encoding='utf-8') as f:
    f.write(reporter.generate_html(result))
print('HTML报告:', html_path)

phase_md = os.path.join(report_dir, 'rolling_phases_{}.md'.format(timestamp))
with open(phase_md, 'w', encoding='utf-8') as f:
    f.write('# 滚动窗口回测 - 各阶段表现 (2020-2024)\n\n')
    f.write('| 阶段 | 开始 | 结束 | 收益率 | 夏普比率 | 最大回撤 | 交易次数 |\n')
    f.write('|------|------|------|--------|---------|---------|----------|\n')
    for ph in result.phase_results:
        f.write('| {} | {} | {} | {:+.2%} | {:+.2f} | {:+.2%} | {} |\n'.format(
            ph.phase_name, ph.start_date, ph.end_date,
            ph.total_return, ph.sharpe_ratio, ph.max_drawdown, ph.trade_count))
    f.write('\n## 汇总\n\n')
    f.write('- IS平均收益率: {:+.2%}\n'.format(avg_is_ret))
    f.write('- OOS平均收益率: {:+.2%}\n'.format(avg_oos_ret))
    f.write('- IS/OOS差距: {:.2%}\n'.format(gap))
print('阶段报告:', phase_md)
print()
print('=' * 60)
print('回测完成！')
print('=' * 60)
