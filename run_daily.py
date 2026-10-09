#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
每日交易流程 CLI —— 生成完整交易日志（journal/YYYY-MM-DD.md）

用法:
    python run_daily.py --phase am           # 早盘：市场展望+决策依据
    python run_daily.py --phase pm --execute paper   # 收盘后：成交+持仓归因+点评
    python run_daily.py                      # 收盘后一次跑完整天（am+pm）
"""

import argparse
import os
import sys
from datetime import date

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

from live import Ledger, PaperBroker
from live.journal import JournalWriter

LEDGER_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'live', 'ledger.json')


def run_am(ds, ledger, writer):
    ctx = writer.write_am(ds)
    print(f"盘前日志已写入: {writer.path}")
    print(f"目标持仓 {len(ctx['targets'])} 只 | 调仓指令 {len(ctx['orders'])} 条")
    return ctx


def run_pm(ds, ledger, writer, ctx, execute=None, top_n=30):
    from live.signals import generate_target, build_orders
    # 用今日最新信号执行（收盘后运行，今日收盘价撮合）
    targets, scores = generate_target(ds, date.today(), top_n=top_n)
    symbols = set(targets) | set(ledger.positions)
    cache = ds.preload_batch(list(symbols), date.today() - __import__('datetime').timedelta(days=15), date.today())
    prices, prev = {}, {}
    for sym, df in cache.items():
        if df is None or df.empty:
            continue
        d = df[df['date'] <= __import__('pandas').Timestamp(date.today())]
        if d.empty:
            continue
        prices[sym] = float(d['close'].iloc[-1])
        if len(d) > 1:
            prev[sym] = float(d['close'].iloc[-2])

    nav = ledger.last_nav() or ledger.cash
    cur_w = ledger.current_weights(prices) if ledger.positions else {}
    orders = build_orders(targets, cur_w, nav, top_n=top_n)

    fills = []
    if orders and execute == 'paper':
        broker = PaperBroker()
        fills = broker.execute(orders, prices, date.today(), prev_prices=prev)
        ledger.apply_fills(fills)
    path = writer.write_pm(ds, fills, ctx)
    print(f"盘后日志已追加: {path}")
    if fills:
        print(f"成交 {sum(1 for f in fills if f.status=='filled')}/{len(orders)} 笔（Paper）")
    return fills


def main():
    parser = argparse.ArgumentParser(description='Daily trading journal')
    parser.add_argument('--phase', choices=['am', 'pm', 'full'], default='full')
    parser.add_argument('--execute', choices=['paper', 'qmt'], default=None)
    parser.add_argument('--top', type=int, default=100)
    parser.add_argument('--ledger', default=LEDGER_PATH)
    args = parser.parse_args()

    from data.layered import LayeredDataSource
    ds = LayeredDataSource()
    ledger = Ledger(args.ledger)
    writer = JournalWriter(ledger, top_n=args.top)

    if args.phase in ('am', 'full'):
        ctx = run_am(ds, ledger, writer)
    else:
        ctx = {'targets': [], 'scores': None, 'orders': [], 'prev_trading': None}

    if args.phase in ('pm', 'full'):
        run_pm(ds, ledger, writer, ctx, execute=args.execute, top_n=args.top)

    print(f"\n完整日志: journal/{date.today().isoformat()}.md")


if __name__ == '__main__':
    main()
