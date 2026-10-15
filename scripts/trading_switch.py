#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
交易控制系统（三态）—— 升级版 Kill Switch

三种状态：
  NORMAL      正常交易
  CLOSE_ONLY  禁止买入，允许卖出（降低风险）
  HALT_ALL    禁止一切新订单，撤销在途

使用：
  python scripts/trading_switch.py halt "原因"     → HALT_ALL
  python scripts/trading_switch.py close            → CLOSE_ONLY
  python scripts/trading_switch.py normal           → NORMAL
  python scripts/trading_switch.py status           → 查看状态

状态文件：runtime/switch_state（JSON，含操作人/时间/原因）
RiskGate 在 review() 最优先检查；QMTBroker 在 execute() 前检查。
"""

import json
import os
import sys
import logging
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = logging.getLogger(__name__)

RUNTIME_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'runtime')
STATE_FILE = os.path.join(RUNTIME_DIR, 'switch_state.json')

NORMAL = 'NORMAL'
CLOSE_ONLY = 'CLOSE_ONLY'
HALT_ALL = 'HALT_ALL'

VALID_STATES = {NORMAL, CLOSE_ONLY, HALT_ALL}


def get_state() -> str:
    """读取当前交易控制状态。

    fail-safe（2026-10-15 定稿，评审报告 §2.1）：状态文件存在但损坏/不可读
    → HALT_ALL。读不出状态=不知道该不该停，唯一安全的解释是"停"——
    误停的代价（一天不交易 + 人工修复）远小于失灵放行的代价。
    文件不存在 → NORMAL（未设置开关=正常运行，与损坏不同）。"""
    if not os.path.isfile(STATE_FILE):
        return NORMAL
    try:
        with open(STATE_FILE, encoding='utf-8') as f:
            d = json.load(f)
        return d.get('state', NORMAL)
    except Exception as e:
        logger.error(f"[switch] 状态文件损坏，按 HALT_ALL 处理（fail-safe）: {e}")
        return HALT_ALL


def get_state_detail() -> dict:
    """读取完整状态详情（含操作记录）"""
    if not os.path.isfile(STATE_FILE):
        return {'state': NORMAL, 'history': []}
    try:
        with open(STATE_FILE, encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return {'state': NORMAL, 'history': []}


def set_state(new_state: str, reason: str = '', operator: str = 'cli') -> dict:
    """设置交易控制状态"""
    if new_state not in VALID_STATES:
        raise ValueError(f"无效状态 '{new_state}'（可选: {', '.join(VALID_STATES)}）")

    os.makedirs(RUNTIME_DIR, exist_ok=True)
    detail = get_state_detail()
    history = detail.get('history', [])
    history.append({
        'timestamp': datetime.now().isoformat(),
        'from': detail.get('state', NORMAL),
        'to': new_state,
        'reason': reason,
        'operator': operator,
    })
    # 保留最近50条
    history = history[-50:]

    new_detail = {'state': new_state, 'history': history,
                  'updated_at': datetime.now().isoformat()}
    with open(STATE_FILE, 'w', encoding='utf-8') as f:
        json.dump(new_detail, f, ensure_ascii=False, indent=1)

    level = logging.CRITICAL if new_state == HALT_ALL else logging.WARNING
    logger.log(level, f"[Switch] {detail.get('state', NORMAL)} → {new_state} ({reason})")
    return new_detail


def allows_buy() -> bool:
    """当前状态是否允许买入"""
    return get_state() == NORMAL


def allows_sell() -> bool:
    """当前状态是否允许卖出（HALT_ALL 也禁止新卖单，只允许撤单）"""
    return get_state() in (NORMAL, CLOSE_ONLY)


def allows_any_order() -> bool:
    """当前状态是否允许任何新订单"""
    return get_state() != HALT_ALL


# 兼容旧 Kill Switch（runtime/KILL 存在 = HALT_ALL）
def _check_legacy_kill() -> bool:
    kill_file = os.path.join(RUNTIME_DIR, 'KILL')
    return os.path.isfile(kill_file)


def effective_state() -> str:
    """有效状态（旧 Kill 文件也视为 HALT_ALL）"""
    if _check_legacy_kill():
        return HALT_ALL
    return get_state()


# ================= CLI =================

def main():
    if len(sys.argv) < 2:
        print(f"当前状态: {effective_state()}")
        print(f"用法: python {sys.argv[0]} <normal|close|halt|status> [原因]")
        sys.exit(0)

    cmd = sys.argv[1].lower()

    if cmd == 'status':
        detail = get_state_detail()
        state = effective_state()
        print(f"当前状态: {state}")
        if detail.get('history'):
            print("\n最近操作:")
            for h in detail['history'][-5:]:
                print(f"  {h['timestamp'][:19]} {h['from']}→{h['to']} ({h['reason'] or '无原因'})")
        return

    if cmd == 'normal':
        set_state(NORMAL, reason=' '.join(sys.argv[2:]) or 'manual')
        print(f"[OK] 状态 → NORMAL（正常交易）")
    elif cmd == 'close':
        set_state(CLOSE_ONLY, reason=' '.join(sys.argv[2:]) or 'manual')
        print(f"[!] 状态 → CLOSE_ONLY（禁止买入，允许卖出）")
    elif cmd == 'halt':
        reason = ' '.join(sys.argv[2:]) or 'manual'
        set_state(HALT_ALL, reason=reason)
        print(f"[X] 状态 → HALT_ALL（停止一切交易）")
        print(f"    原因: {reason}")
        print(f"    解除: python {sys.argv[0]} normal")
    else:
        print(f"未知命令: {cmd}")
        print(f"用法: python {sys.argv[0]} <normal|close|halt|status> [原因]")
        sys.exit(1)


if __name__ == '__main__':
    main()
