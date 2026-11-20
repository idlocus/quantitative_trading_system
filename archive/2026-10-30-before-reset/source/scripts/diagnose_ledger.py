#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
账本价格基准诊断：检测BUY/SELL价格是否在同一复权基准上。
如果比率差异>2x，成本基准已损坏，建议重置账本。
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
from collections import defaultdict


def diagnose(ledger_path='live/ledger.json'):
    d = json.load(open(ledger_path, encoding='utf-8'))
    trades = d.get('trades', [])
    positions = d.get('positions', {})
    costs = d.get('cost', {})

    # 按symbol分组：BUY均价 vs SELL均价
    by_sym = defaultdict(lambda: {'buy_px': [], 'sell_px': [], 'buy_qty': [], 'sell_qty': []})
    for t in trades:
        rec = by_sym[t['symbol']]
        if t['side'] == 'BUY':
            rec['buy_px'].append(t['price'])
            rec['buy_qty'].append(t['quantity'])
        else:
            rec['sell_px'].append(t['price'])
            rec['sell_qty'].append(t['quantity'])

    print("=" * 80)
    print("账本价格基准诊断")
    print("=" * 80)
    corrupted = []
    for sym in sorted(by_sym.keys()):
        rec = by_sym[sym]
        n_buy = len(rec['buy_px'])
        n_sell = len(rec['sell_px'])
        if n_buy == 0 or n_sell == 0:
            continue
        avg_buy = sum(p * q for p, q in zip(rec['buy_px'], rec['buy_qty'])) / max(sum(rec['buy_qty']), 1)
        avg_sell = sum(p * q for p, q in zip(rec['sell_px'], rec['sell_qty'])) / max(sum(rec['sell_qty']), 1)
        ratio = avg_sell / avg_buy if avg_buy > 0 else 0
        cost = costs.get(sym, 0)
        holding = positions.get(sym, 0)

        flag = ""
        if ratio > 2.0 or ratio < 0.5:
            flag = " ⚠️ 基准断裂"
            corrupted.append(sym)
        print(f"  {sym:12s} 买{n_buy:3d}笔@{avg_buy:8.2f} 卖{n_sell:3d}笔@{avg_sell:8.2f} "
              f"比{ratio:6.2f}x 成本{cost:8.2f} 持{holding:6d}{flag}")

    print(f"\n结论：{len(corrupted)}/{len(by_sym)} 只存在价格基准断裂（BUY/SELL比率>2x）")
    if corrupted:
        print(f"损坏标的：{corrupted[:10]}...")
        print("\n⚠️ 成本基准已不可修复（BUY在一个宇宙、SELL在另一个宇宙）")
        print("建议：重置账本 → python run_live.py --init <资金额>")
        print("（trade history 保留在 JSON 备份中供审计）")
    return corrupted


if __name__ == '__main__':
    diagnose()
