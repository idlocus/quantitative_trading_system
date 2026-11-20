#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实盘前检查（Pre-flight Check）

Paper → 小额实盘前的自动化检查清单。全部通过才能上线。
用法: python scripts/preflight_check.py [--verbose]
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import subprocess


CHECKS = []


def check(name):
    """注册检查项"""
    def decorator(fn):
        CHECKS.append((name, fn))
        return fn
    return decorator


@check("Kill Switch 状态")
def check_kill():
    from scripts.kill_switch import is_killed, KILL_FILE
    if is_killed():
        return False, f"[X] 停止中（{KILL_FILE} 存在），需先解除"
    return True, "[OK] 未停止"


@check("策略版本定义")
def check_spec():
    from trading.spec import StrategySpec, STRATEGY_VERSION
    spec = StrategySpec.current()
    return True, f"[OK] v{STRATEGY_VERSION} ({spec.hash[:8]}) {spec.summary()}"


@check("杠杆阈值单位")
def check_leverage():
    from trading import config as chain
    lm = chain.QUALITY_FILTER.get('leverage_max', 0)
    if lm < 10:
        return False, f"[FAIL] 杠杆阈值{lm}疑似倍数而非百分比（PIT值中位数~50）"
    return True, f"[OK] D/E < {lm}%（百分比口径）"


@check("账本完整性")
def check_ledger():
    from live import Ledger
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         'live', 'ledger.json')
    led = Ledger(path)
    issues = []
    if not led._d.get('nav_history'):
        issues.append("无净值历史")
    if led.open_exceptions():
        issues.append(f"{len(led.open_exceptions())}项未处理异常")
    last_recon = led._d.get('meta', {}).get('last_recon', {})
    if last_recon and not last_recon.get('ok', True):
        issues.append(f"上次对账失败({last_recon.get('date')})")
    if issues:
        return False, "[FAIL] " + "; ".join(issues)
    n = len(led.positions)
    nav = led.last_nav() or led.cash
    return True, f"[OK] 持仓{n}只 | 净值{nav:,.0f} | 无异常"


@check("配置一致性")
def check_config():
    from trading import config as chain
    from live.risk_gate import GateConfig
    from portfolio.risk_manager import RiskLimits
    g, r = GateConfig(), RiskLimits()
    mismatches = []
    if g.drawdown_soft != r.drawdown_derisk:
        mismatches.append("回撤软限")
    if g.stop_loss_pct != r.stop_loss_pct:
        mismatches.append("止损")
    if g.vol_target != r.vol_target:
        mismatches.append("波动率目标")
    if mismatches:
        return False, f"[FAIL] 不一致: {', '.join(mismatches)}"
    return True, "[OK] 生产/回测参数一致"


@check("Git 工作区")
def check_git():
    try:
        r = subprocess.run(['git', 'status', '--porcelain'],
                          capture_output=True, text=True, timeout=10,
                          cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        dirty = bool(r.stdout.strip())
        if dirty:
            n = len(r.stdout.strip().split('\n'))
            return True, f"[!] 工作区有{n}个未提交文件（正式上线前应提交）"
        return True, "[OK] 干净"
    except Exception:
        return True, "[!] 无法检查"


@check("数据源可达")
def check_db():
    try:
        from data.layered import LayeredDataSource
        ds = LayeredDataSource()
        db = ds._db_src()
        if db:
            return True, "[OK] Oracle DB 可达"
        return True, "[!] DB 不可达（将使用缓存+客户端）"
    except Exception as e:
        return True, f"[!] {e}"


@check("QMT 环境（仅实盘需要）")
def check_qmt():
    from live.qmt_broker import QMTBroker
    env = QMTBroker.check_env()
    if env['ready']:
        return True, "[OK] QMT 就绪"
    issues = "; ".join(env['issues'])
    return True, f"[!] QMT 未就绪: {issues}（Paper 不受影响）"


def main():
    verbose = '--verbose' in sys.argv
    print("=" * 60)
    print("实盘前检查（Pre-flight Check）")
    print("=" * 60)

    all_pass = True
    for name, fn in CHECKS:
        try:
            ok, msg = fn()
        except Exception as e:
            ok, msg = False, f"[FAIL] 检查异常: {e}"
        if not ok:
            all_pass = False
        print(f"  {msg:60s} [{name}]")
        if verbose:
            print(f"    → {fn.__doc__ or name}")

    print("=" * 60)
    if all_pass:
        print("[OK] 全部通过——可以进入灰度上线流程")
        print("   下一步: docs/go_live.md")
    else:
        print("[FAIL] 存在阻断项——修复后再试")
        sys.exit(1)


if __name__ == '__main__':
    main()
