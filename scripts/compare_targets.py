#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
目标 diff 工具 —— 策略/参数变更上实盘前的"影子关"核心（实盘变更管理缺口#1）。

同一账本状态下，对比两个代码/参数版本各自的调仓目标，回答：
  改这个版本会让实盘在下一个调仓日发生什么交易、花多少迁移成本？

用法（任何代码/参数变更，含 git 切换）：
  # 版本A下：
  python scripts/compare_targets.py --save A
  # 切到版本B（git checkout / 改 config）后：
  python scripts/compare_targets.py --save B
  python scripts/compare_targets.py --compare A B

sleeve：--sleeve stock（默认，走 generate_target 生产信号路径）
        --sleeve etf（走 build_etf_signal 唯一实现）
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import subprocess
from datetime import date

SNAP_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        'runtime', 'target_snapshots')


def git_head():
    try:
        return subprocess.check_output(
            ['git', 'rev-parse', '--short', 'HEAD'],
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return 'unknown'


def spec_hash():
    try:
        from trading.spec import StrategySpec
        return StrategySpec.current().hash
    except Exception:
        return 'n/a'


def snapshot(sleeve: str) -> dict:
    from data.layered import LayeredDataSource
    ds = LayeredDataSource()
    as_of = date.today()
    if sleeve == 'etf':
        from trading.etf_strategy import build_etf_signal
        targets, _, trend = build_etf_signal(ds, as_of)
        detail = {c: round(w, 4) for c, w in targets.items()}
    else:
        from live.signals import generate_target
        targets, _scores = generate_target(ds, as_of)
        detail = {s: '' for s in targets}
    return {'at': str(as_of), 'sleeve': sleeve, 'git_head': git_head(),
            'spec_hash': spec_hash(), 'n': len(targets), 'targets': detail}


def compare(a: dict, b: dict):
    ta, tb = set(a['targets']), set(b['targets'])
    print(f"""
{'=' * 68}
调仓目标 diff（{a['sleeve']} sleeve）
{'=' * 68}
  版本A: {a['n']} 只  git={a['git_head']}  spec={a['spec_hash']}  @{a['at']}
  版本B: {b['n']} 只  git={b['git_head']}  spec={b['spec_hash']}  @{b['at']}
  重叠:  {len(ta & tb)} 只（{len(ta & tb) / max(len(ta), 1):.0%} of A）
""")
    if a['sleeve'] == 'etf':
        for c in sorted(ta | tb):
            wa, wb = a['targets'].get(c), b['targets'].get(c)
            if wa != wb:
                print(f"  {c}: {wa} → {wb}")
        return

    entries = sorted(tb - ta)
    exits = sorted(ta - tb)
    print(f"  版本B 新入选（{len(entries)}）: {', '.join(entries[:30])}"
          f"{' …' if len(entries) > 30 else ''}")
    print(f"  版本B 移出（{len(exits)}）: {', '.join(exits[:30])}"
          f"{' …' if len(exits) > 30 else ''}")

    # 相对当前实盘持仓的真实迁移交易与成本
    from live.ledger import Ledger
    led_path = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), 'live',
        'ledger_etf.json' if a['sleeve'] == 'etf' else 'ledger.json')
    if os.path.isfile(led_path):
        led = Ledger(led_path)
        holds = set(led.positions)
        if holds:
            sells = holds - tb
            buys = tb - holds
            one_way = (len(sells) + len(buys)) / max(len(holds), 1)
            from trading.rules import buy_fee_rate, sell_fee_rate
            from trading import config as chain
            cost_rate = (len(sells) * (sell_fee_rate() + chain.SLIPPAGE) +
                         len(buys) * (buy_fee_rate() + chain.SLIPPAGE)) / 100
            nav = led.last_nav() or led.cash
            print(f"""
  相对当前持仓（{len(holds)} 只, NAV {nav:,.0f}）:
    卖出 {len(sells)} 只 · 买入 {len(buys)} 只 · 单边换手 ≈ {one_way:.0%}
    预计迁移成本 ≈ {nav * cost_rate:,.0f} 元
    （影子关判据：换手与成本应在预期内且可解释；异常→停下排查）""")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--save', metavar='NAME', help='保存当前版本的目标快照')
    ap.add_argument('--compare', nargs=2, metavar=('A', 'B'),
                    help='对比两个快照')
    ap.add_argument('--sleeve', default='stock', choices=['stock', 'etf'])
    args = ap.parse_args()

    if args.save:
        os.makedirs(SNAP_DIR, exist_ok=True)
        snap = snapshot(args.sleeve)
        path = os.path.join(SNAP_DIR, f"{args.save}.json")
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(snap, f, ensure_ascii=False, indent=1)
        print(f"快照已保存: {path}（{snap['n']} 只, git={snap['git_head']}, "
              f"spec={snap['spec_hash']}）")
    if args.compare:
        snaps = []
        for name in args.compare:
            path = os.path.join(SNAP_DIR, f"{name}.json")
            with open(path, encoding='utf-8') as f:
                snaps.append(json.load(f))
        compare(*snaps)
    if not args.save and not args.compare:
        ap.print_help()


if __name__ == '__main__':
    main()
