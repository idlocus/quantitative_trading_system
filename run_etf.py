#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ETF 趋势 sleeve 每日运行入口（paper）—— 与股票主链（run_daily）并行的第二引擎

设计与股票主链的边界：
  - 独立账本 live/ledger_etf.json（初始资金 chain.ETF_SLEEVE_CAPITAL）
  - 信号/订单：唯一实现 trading/etf_strategy.py（R1）
  - 月度调仓日才可能交易；非调仓日仅 mark-to-market
  - 风控：三态总闸注入（与股票同源、共享一个开关状态）；
    回撤保护=策略自身（负趋势资产移出）——不套用股票的 15% 硬限协议
    （趋势策略的退出机制就是它的风控，双重闸门语义不同）
  - Paper 撮合费用近似：PaperBroker 按股票费率（含卖出印花税 0.05%）——
    ETF 无印花税，paper 成本略保守（高估），实盘切换时修正

用法：
  python run_etf.py                 # 日常（paper）
  python run_etf.py --status        # 查看 sleeve 状态
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import json
from datetime import date, timedelta

from data.layered import LayeredDataSource
from live.ledger import Ledger
from live.broker import PaperBroker
from trading import config as chain
from trading.etf_strategy import build_etf_signal, build_etf_orders, is_rebalance_day

LEDGER_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           'live', 'ledger_etf.json')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--status', action='store_true', help='仅查看状态')
    args = ap.parse_args()

    ledger = Ledger(LEDGER_PATH)
    if not ledger._d.get('initialized'):
        ledger.initialize(chain.ETF_SLEEVE_CAPITAL)
        print(f"ETF sleeve 账本初始化: {chain.ETF_SLEEVE_CAPITAL:,.0f}")

    ds = LayeredDataSource()
    as_of = date.today()

    # 最新交易日价格（T-1 收盘口径）；MTM/成交=不复权 close（R2：复权价
    # 水平值会使分红型 ETF 净值虚增），趋势信号内部用复权收益（etf_strategy）
    cache = ds.preload_batch(list(chain.ETF_ASSETS) + ['000300.SH'],
                             as_of - timedelta(days=30), as_of)
    etf_df = {c: cache.get(c) for c in chain.ETF_ASSETS if cache.get(c) is not None}
    trading_dates = max((df['date'] for df in etf_df.values()), key=len)
    last_day = trading_dates.iloc[-1]
    prices = {c: float(df['close'].iloc[-1]) for c, df in etf_df.items()}

    ledger.mark_to_market(prices, date.fromisoformat(str(last_day)[:10]))
    nav = ledger.last_nav() or ledger.cash
    cur_w = ledger.current_weights(prices) if ledger.positions else {}
    print(f"ETF sleeve @ {last_day} | NAV {nav:,.0f} | 现金 {ledger.cash:,.0f} | "
          f"持仓 {len(ledger.positions)}")
    for s, q in ledger.positions.items():
        print(f"  {s} {chain.ETF_ASSETS.get(s, '')} × {q}")

    if args.status:
        return

    # 三态总闸（与股票 sleeve 共享同一开关状态，fail-safe 读取）
    from scripts.trading_switch import effective_state
    sw = effective_state()
    if sw == 'HALT_ALL':
        print("交易控制=HALT_ALL——ETF sleeve 不交易")
        return

    # 月度调仓判定 + T-1 信号
    if not is_rebalance_day(trading_dates, last_day):
        print("非月度调仓日——仅估值")
        return
    sig_day = trading_dates.iloc[-2] if len(trading_dates) > 1 else last_day
    targets, _, trend = build_etf_signal(ds, date.fromisoformat(str(sig_day)[:10]))
    print(f"信号日 {sig_day} 趋势: " +
          ", ".join(f"{c}({chain.ETF_ASSETS.get(c, c)}): {r:+.1%}"
                    for c, r in trend.items()))
    print(f"目标: {targets if targets else '100% 现金（无正趋势资产）'}")

    if sw == 'CLOSE_ONLY':
        targets = {}          # 只卖不买：目标清空（build_etf_orders 生成卖出）
    orders = build_etf_orders(targets, cur_w, nav)
    if not orders:
        print("缓冲带内，无订单")
        return

    broker = PaperBroker()
    fills = broker.execute(orders, prices, as_of)
    booked, dupes, blocked = ledger.apply_fills(fills)
    print(f"执行: {len(orders)} 单 → 入账 {len(booked)}（重复{len(dupes)} 阻断{len(blocked)}）")
    ledger.mark_to_market(prices, as_of)
    print(f"执行后 NAV {ledger.last_nav() or ledger.cash:,.0f}")

    # 状态留痕（journal/日报由股票主链外的轻量记录承担）
    state = {'date': str(as_of), 'nav': ledger.last_nav() or ledger.cash,
             'positions': dict(ledger.positions), 'targets': targets,
             'trend': {c: round(r, 4) for c, r in trend.items()}}
    os.makedirs('runtime', exist_ok=True)
    with open('runtime/etf_sleeve_state.json', 'w', encoding='utf-8') as f:
        json.dump(state, f, ensure_ascii=False, indent=1)
    print("状态: runtime/etf_sleeve_state.json")


if __name__ == '__main__':
    main()
