#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
一键停止交易（Kill Switch）

激活：  touch runtime/KILL
解除：  rm runtime/KILL

激活后所有交易立即停止（RiskGate review() 返回 halt），包括：
- 新买入
- 新卖出（止损/调仓/清仓）
- Paper 和 QMT 均受控

RiskGate 在每次 review() 前检查此文件。
"""

import os
import logging

logger = logging.getLogger(__name__)

KILL_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'runtime', 'KILL')


def is_killed() -> bool:
    """检查是否处于停止状态"""
    return os.path.isfile(KILL_FILE)


def kill(reason: str = 'manual'):
    """激活停止"""
    os.makedirs(os.path.dirname(KILL_FILE), exist_ok=True)
    with open(KILL_FILE, 'w', encoding='utf-8') as f:
        f.write(f"reason: {reason}\n")
    logger.critical(f"[KILL] 交易已停止: {reason}")
    print(f"⛔ 交易已停止: {reason}")
    print(f"   激活文件: {KILL_FILE}")
    print(f"   解除方法: rm {KILL_FILE}")


def unkill():
    """解除停止"""
    if os.path.isfile(KILL_FILE):
        os.remove(KILL_FILE)
        logger.info("[KILL] 交易停止已解除")
        print(f"✅ 交易停止已解除")
    else:
        print("当前未处于停止状态")


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == 'off':
        unkill()
    else:
        reason = ' '.join(sys.argv[1:]) if len(sys.argv) > 1 else 'manual'
        kill(reason)
