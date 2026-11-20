#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一入口 —— 一个命令管理整个系统的日常运行

用法:
    python main.py daily                # ★ 每日完整流程（数据→信号→Paper成交→账本→日志→净值）
    python main.py am / pm              # 分阶段（早盘/收盘后）
    python main.py status               # 账户与持仓概览
    python main.py scan [--top 100]      # 每日选股扫描
    python main.py dashboard [--port 5000]  # 研究仪表盘（常驻）
    python main.py report               # 重建研究仪表盘工件

日常只需 daily；注册 Windows 计划任务后全自动（见 scripts/install_daily_task.ps1）。
"""

import argparse
import os
import subprocess
import sys
import traceback
from datetime import date, datetime

ROOT = os.path.dirname(os.path.abspath(__file__))
PYTHON = sys.executable


def _run(script, *args):
    """以项目根目录运行子脚本"""
    cmd = [PYTHON, os.path.join(ROOT, script)] + list(args)
    return subprocess.call(cmd, cwd=ROOT)


def cmd_daily(args):
    """每日完整流程：一次运行完成数据同步→信号→成交→账本→日志→净值"""
    from dotenv import load_dotenv
    load_dotenv()
    try:
        from data.layered import LayeredDataSource
        from live import Ledger, PaperBroker
        from live.journal import JournalWriter

        t0 = datetime.now()
        ds = LayeredDataSource()
        ledger = Ledger(os.path.join(ROOT, 'live', 'ledger.json'))
        writer = JournalWriter(ledger, top_n=args.top)

        # 1) 盘前章节（展望+决策，T-1口径）
        from run_daily import run_am, run_pm
        ctx = run_am(ds, ledger, writer)

        # 2) 盘后章节（成交+归因+点评），paper 撮合
        run_pm(ds, ledger, writer, ctx, execute=args.execute or 'paper', top_n=args.top)

        print(f"\n[daily] 完成，耗时 {(datetime.now()-t0).total_seconds()/60:.1f} 分钟")
        print(f"[daily] 日志: journal/{date.today().isoformat()}.md")
    except Exception:
        err = traceback.format_exc()
        print(f"[daily] 失败:\n{err}")
        # 失败也留痕（当日日志末尾记录，不静默）
        try:
            jpath = os.path.join(ROOT, 'journal', f"{date.today().isoformat()}.md")
            os.makedirs(os.path.dirname(jpath), exist_ok=True)
            with open(jpath, 'a', encoding='utf-8') as f:
                f.write(f"\n## ⚠ 运行异常\n\n```\n{err[-2000:]}\n```\n")
        except Exception:
            pass
        return 1
    return 0


def cmd_status(args):
    return _run('run_live.py', '--status')


def cmd_scan(args):
    cmd = ['--top', str(args.top)]
    return _run('run_scan.py', *cmd)


def cmd_dashboard(args):
    return subprocess.call([PYTHON, '-m', 'webapp.app'], cwd=ROOT)


def cmd_report(args):
    return _run(os.path.join('experiments', 'rebuild_artifacts.py'))


def main():
    parser = argparse.ArgumentParser(description='量化系统统一入口')
    sub = parser.add_subparsers(dest='cmd')

    p = sub.add_parser('daily', help='每日完整流程（推荐注册计划任务）')
    p.add_argument('--top', type=int, default=100)
    p.add_argument('--execute', choices=['paper', 'qmt'], default=None)
    p.set_defaults(func=cmd_daily)

    for name, help_ in (('am', '仅盘前（展望+决策）'), ('pm', '仅盘后（成交+归因+点评）')):
        p = sub.add_parser(name, help=help_)
        p.add_argument('--top', type=int, default=100)
        p.add_argument('--execute', choices=['paper', 'qmt'], default='paper')
        p.set_defaults(func=lambda a, n=name: _run('run_daily.py', '--phase', n,
                                                   '--execute', a.execute,
                                                   '--top', str(a.top)))

    p = sub.add_parser('status', help='账户概览')
    p.set_defaults(func=cmd_status)

    p = sub.add_parser('scan', help='每日选股扫描')
    p.add_argument('--top', type=int, default=10)
    p.add_argument('--universe', default='hs300')
    p.add_argument('--strategy', default='pure_trend_follow')
    p.set_defaults(func=lambda a: _run('run_scan.py', '--top', str(a.top),
                                       '--universe', a.universe, '--strategy', a.strategy))

    p = sub.add_parser('dashboard', help='研究仪表盘（常驻）')
    p.add_argument('--port', type=int, default=5000)
    p.set_defaults(func=cmd_dashboard)

    p = sub.add_parser('report', help='重建研究工件')
    p.set_defaults(func=cmd_report)

    args = parser.parse_args()
    if not getattr(args, 'func', None):
        parser.print_help()
        return 1
    return args.func(args)


if __name__ == '__main__':
    sys.exit(main())
