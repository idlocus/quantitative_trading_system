#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
上线状态机 —— RESEARCH → PAPER → CANARY → LIVE

控制谁能创建 QMT Broker（实盘通道）。
RESEARCH/PAPER 状态下 make_broker('qmt') 直接拒绝。

使用：
  python scripts/deployment.py status
  python scripts/deployment.py promote --to PAPER --confirm
  python scripts/deployment.py promote --to CANARY --confirm
  python scripts/deployment.py promote --to LIVE --confirm
  python scripts/deployment.py demote --to PAPER --reason "回滚"
"""

import json
import os
import sys
from datetime import datetime, date, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

RUNTIME_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'runtime')
STATE_FILE = os.path.join(RUNTIME_DIR, 'deployment_state.json')

# 状态层级（只能向上或向下显式移动）
STATES = ['RESEARCH', 'PAPER', 'CANARY', 'LIVE']

# 进入 CANARY 的最低 Paper 天数
MIN_PAPER_DAYS = 30

# 进入 LIVE 的最低 CANARY 天数
MIN_CANARY_DAYS = 14


def get_deployment_state() -> str:
    if not os.path.isfile(STATE_FILE):
        return 'PAPER'  # 默认 Paper（当前运行模式）
    try:
        with open(STATE_FILE, encoding='utf-8') as f:
            return json.load(f).get('state', 'PAPER')
    except Exception:
        return 'PAPER'


def get_detail() -> dict:
    if not os.path.isfile(STATE_FILE):
        return {'state': 'PAPER', 'history': []}
    try:
        with open(STATE_FILE, encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return {'state': 'PAPER', 'history': []}


def set_state(new_state: str, reason: str = '', operator: str = 'cli'):
    if new_state not in STATES:
        raise ValueError(f"无效状态: {new_state}")
    os.makedirs(RUNTIME_DIR, exist_ok=True)
    detail = get_detail()
    history = detail.get('history', [])
    history.append({'timestamp': datetime.now().isoformat(),
                    'from': detail.get('state'), 'to': new_state,
                    'reason': reason, 'operator': operator})
    new = {'state': new_state, 'history': history[-50:],
           'updated_at': datetime.now().isoformat()}
    with open(STATE_FILE, 'w', encoding='utf-8') as f:
        json.dump(new, f, ensure_ascii=False, indent=1)
    return new


def check_canary_requirements(ledger) -> tuple:
    """进入 CANARY 的前置条件检查"""
    issues = []

    # 连续 Paper 天数（从 nav_history 推断）
    nav_hist = ledger.nav_series()
    if len(nav_hist) < MIN_PAPER_DAYS:
        issues.append(f"Paper运行仅{len(nav_hist)}天 < 最低{MIN_PAPER_DAYS}天")

    # 无未处理异常
    if ledger.open_exceptions():
        issues.append(f"{len(ledger.open_exceptions())}项未处理异常")

    # 上次对账
    last_recon = ledger._d.get('meta', {}).get('last_recon', {})
    if last_recon and not last_recon.get('ok', True):
        issues.append(f"上次对账失败({last_recon.get('date')})")

    return len(issues) == 0, issues


def check_live_requirements() -> tuple:
    """进入 LIVE 的前置条件检查"""
    issues = []
    detail = get_detail()
    if detail.get('state') != 'CANARY':
        issues.append(f"当前状态{detail.get('state')}，必须先进入CANARY")
    # CANARY 持续时间
    canary_since = None
    for h in reversed(detail.get('history', [])):
        if h.get('to') == 'CANARY':
            canary_since = h.get('timestamp')
            break
    if canary_since:
        days = (datetime.now() - datetime.fromisoformat(canary_since)).days
        if days < MIN_CANARY_DAYS:
            issues.append(f"CANARY仅运行{days}天 < 最低{MIN_CANARY_DAYS}天")
    else:
        issues.append("无CANARY运行记录")
    return len(issues) == 0, issues


# ================= CLI =================

def main():
    if len(sys.argv) < 2:
        print(f"当前状态: {get_deployment_state()}")
        print(f"用法: python {sys.argv[0]} <status|promote|demote>")
        return

    cmd = sys.argv[1].lower()

    if cmd == 'status':
        detail = get_detail()
        print(f"部署状态: {detail['state']}")
        print(f"可创建QMT: {'是' if detail['state'] in ('CANARY', 'LIVE') else '否（仅CANARY/LIVE）'}")
        if detail.get('history'):
            print("\n最近变更:")
            for h in detail['history'][-5:]:
                print(f"  {h['timestamp'][:19]} {h['from']}→{h['to']} ({h['reason'] or '无原因'})")
        return

    if cmd == 'promote':
        if '--confirm' not in sys.argv:
            print("需要 --confirm 才能执行")
            return

        to_idx = sys.argv.index('--to') if '--to' in sys.argv else -1
        target = sys.argv[to_idx + 1] if to_idx >= 0 else None
        if target not in STATES:
            print(f"必须指定 --to <{'|'.join(STATES)}>")
            return

        current = get_deployment_state()
        if STATES.index(target) <= STATES.index(current):
            print(f"目标{target}不高于当前{current}——请用demote")
            return

        # 检查前置条件
        if target == 'CANARY':
            from live import Ledger
            led_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'live', 'ledger.json')
            led = Ledger(led_path)
            ok, issues = check_canary_requirements(led)
            if not ok:
                print("CANARY前置条件不满足:")
                for i in issues:
                    print(f"  [X] {i}")
                return

        if target == 'LIVE':
            ok, issues = check_live_requirements()
            if not ok:
                print("LIVE前置条件不满足:")
                for i in issues:
                    print(f"  [X] {i}")
                return

        reason = ''
        if '--reason' in sys.argv:
            ri = sys.argv.index('--reason')
            reason = sys.argv[ri + 1] if ri + 1 < len(sys.argv) else ''

        set_state(target, reason=reason)
        print(f"[OK] {current} → {target}")
        if target == 'CANARY':
            print(f"    小额实盘通道已开放（建议资金≤20%）")

    elif cmd == 'demote':
        if '--confirm' not in sys.argv:
            print("需要 --confirm")
            return
        to_idx = sys.argv.index('--to') if '--to' in sys.argv else -1
        target = sys.argv[to_idx + 1] if to_idx >= 0 else 'PAPER'
        reason = sys.argv[sys.argv.index('--reason') + 1] if '--reason' in sys.argv else 'demote'
        current = get_deployment_state()
        set_state(target, reason=reason)
        print(f"[OK] {current} → {target} ({reason})")
    else:
        print(f"未知命令: {cmd}")


if __name__ == '__main__':
    main()
