#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实盘驱动 CLI —— 每日收盘后运行

用法:
    python run_live.py --init 1000000              # 初始入金
    python run_live.py --signal                    # 生成目标持仓+调仓指令（仅打印）
    python run_live.py --signal --execute paper    # Paper 模式模拟成交并入账
    python run_live.py --nav                       # 按最新收盘更新净值
    python run_live.py --status                    # 持仓/净值/累计收益概览

QMT 就位后: --execute qmt --account XXXX（见 live/broker.py 前置条件）
"""

import argparse
import os
import sys
from datetime import date, timedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

LEDGER_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'live', 'ledger.json')


def get_prices(ds, symbols, as_of):
    """最新收盘价（缓存优先）"""
    cache = ds.preload_batch(list(symbols), as_of - timedelta(days=10), as_of)
    prices = {}
    prev = {}
    for sym, df in cache.items():
        if df is None or df.empty:
            continue
        d = df[df['date'] <= __import__('pandas').Timestamp(as_of)]
        if d.empty:
            continue
        prices[sym] = float(d['close'].iloc[-1])
        if len(d) > 1:
            prev[sym] = float(d['close'].iloc[-2])
    return prices, prev


def main():
    parser = argparse.ArgumentParser(description='Live trading driver (paper/QMT)')
    parser.add_argument('--init', type=float, default=None, help='初始入金')
    parser.add_argument('--signal', action='store_true', help='生成目标持仓与指令')
    parser.add_argument('--execute', choices=['paper', 'qmt'], default=None)
    parser.add_argument('--top', type=int, default=30)
    parser.add_argument('--nav', action='store_true')
    parser.add_argument('--status', action='store_true')
    parser.add_argument('--ledger', default=LEDGER_PATH)
    args = parser.parse_args()

    from live import Ledger, PaperBroker
    ledger = Ledger(args.ledger)

    if args.init is not None:
        ledger.initialize(args.init)
        print(f"账本初始化：入金 {args.init:,.0f} → {args.ledger}")
        return

    if args.status:
        nav = ledger.last_nav()
        n_trades = len(ledger._d['trades'])
        print(f"持仓 {len(ledger.positions)} 只 | 现金 {ledger.cash:,.0f} | "
              f"成交 {n_trades} 笔 | 最新净值 {nav if nav else '未记录'}")
        for s, qty in list(ledger.positions.items())[:10]:
            print(f"  {s}: {qty} 股 @成本 {ledger._d['cost'].get(s, 0):.2f}")
        return

    if args.nav:
        from data.layered import LayeredDataSource
        ds = LayeredDataSource()
        as_of = date.today()
        prices, _ = get_prices(ds, list(ledger.positions), as_of)
        nav = ledger.mark_to_market(prices, as_of)
        init = ledger._d['cash'] + sum(
            ledger._d['cost'].get(s, 0) * q for s, q in ledger._d['positions'].items())
        print(f"净值 @{as_of}: {nav:,.2f}")
        if init > 0:
            print(f"累计投入约 {init:,.0f}，盈亏 {nav - init:+,.0f} ({(nav/init-1):+.2%})")
        return

    if args.signal:
        from data.layered import LayeredDataSource
        from live.signals import generate_target, build_orders
        ds = LayeredDataSource()
        as_of = date.today()

        print(f"生成生产信号 @{as_of}（Top{args.top}，5因子）...")
        targets, scores = generate_target(ds, as_of, top_n=args.top)
        if not targets:
            print("信号为空（检查数据）")
            return
        print(f"目标持仓 {len(targets)} 只，评分区间 "
              f"[{scores[targets[-1]]:.2f}, {scores[targets[0]]:.2f}]")

        prices, prev = get_prices(ds, set(targets) | set(ledger.positions), as_of)
        nav = ledger.last_nav() or ledger.cash
        cur_w = ledger.current_weights(prices)
        orders = build_orders(targets, cur_w, nav, top_n=args.top)

        print(f"\n调仓指令 {len(orders)} 条：")
        for o in orders:
            amt = o.amount or 0
            px = prices.get(o.symbol, 0)
            print(f"  {o.side:4s} {o.symbol:10s} ~{amt:,.0f}元 @{px:.2f} [{o.note}]")

        if args.execute == 'paper':
            broker = PaperBroker()
            fills = broker.execute(orders, prices, as_of, prev_prices=prev)
            filled = [f for f in fills if f.status == 'filled']
            print(f"\nPaper 成交 {len(filled)}/{len(orders)}：")
            for f in fills:
                mark = 'OK' if f.status == 'filled' else f.status
                print(f"  {mark:16s} {f.order.symbol} {f.quantity}股 @ {f.price:.2f}")
            ledger.apply_fills(fills)
            nav_now = ledger.mark_to_market(prices, as_of)
            print(f"\n账本已更新，净值 {nav_now:,.2f}")
        elif args.execute == 'qmt':
            print("QMT 通道待券商环境就位（见 live/broker.py 前置条件）")
        else:
            print("\n（仅生成指令未执行；加 --execute paper 模拟入账）")


if __name__ == '__main__':
    main()
