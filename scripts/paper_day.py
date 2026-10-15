#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Paper 验证期每日运行器 —— 计划任务的唯一入口（保证持续性）。

一条命令完成：股票 sleeve 全流程 + ETF sleeve + 健康视图 + 心跳留痕。
各步独立容错（一步失败不阻断其余），任一关键失败退出码非零（调度可见）。

用法：python scripts/paper_day.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# ^ 计划任务默认工作目录=System32：run_daily/layered 的相对路径
#   （live/ledger.json、data_cache）依赖项目根 cwd，入口统一 chdir 兜底

import json
import subprocess
from datetime import date, datetime

STEPS = [
    ('股票 sleeve（信号/闸门/撮合/日报）',
     [sys.executable, 'run_daily.py', '--phase', 'full', '--execute', 'paper']),
    ('ETF sleeve（月度调仓/估值）',
     [sys.executable, 'run_etf.py']),
]


def main():
    results = {}
    for name, cmd in STEPS:
        print(f"\n===== {name} =====", flush=True)
        try:
            r = subprocess.run(cmd, capture_output=True, text=True,
                               timeout=45 * 60, encoding='utf-8', errors='replace')
            results[name] = r.returncode
            tail = '\n'.join((r.stdout or '').splitlines()[-5:])
            print(tail or '(无输出)')
            if r.returncode != 0:
                print(f"[失败 rc={r.returncode}] {'/'.join((r.stderr or '').splitlines()[-3:])}")
        except Exception as e:
            results[name] = -1
            print(f"[异常] {e}")

    # 心跳留痕（watchdog 依据；含双 sleeve NAV 快照）
    hb = {'date': str(date.today()), 'at': datetime.now().isoformat()[:19],
          'results': results,
          'stock_nav': None, 'etf_nav': None}
    try:
        from live.ledger import Ledger
        led = Ledger(os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), 'live', 'ledger.json'))
        hb['stock_nav'] = led.last_nav() or led.cash
        le = Ledger(os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), 'live', 'ledger_etf.json'))
        hb['etf_nav'] = le.last_nav() or le.cash
    except Exception as e:
        hb['ledger_error'] = str(e)
    os.makedirs('runtime', exist_ok=True)
    with open('runtime/paper_heartbeat.json', 'w', encoding='utf-8') as f:
        json.dump(hb, f, ensure_ascii=False, indent=1)
    print(f"\n心跳: runtime/paper_heartbeat.json | {hb['results']} | "
          f"stock={hb['stock_nav']} etf={hb['etf_nav']}")

    failed = [n for n, rc in results.items() if rc != 0]
    if failed:
        print(f"⚠️ 失败步骤: {failed}")
        sys.exit(1)
    print("✅ Paper 验证期当日运行完成")


if __name__ == '__main__':
    main()
