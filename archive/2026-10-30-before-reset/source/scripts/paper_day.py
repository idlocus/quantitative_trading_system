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
    # 周末守卫（F2，2026-10-20）：非交易日不执行——曾发生周六以周五收盘价
    # 成交7笔、日期戳记在非交易日的污染（价格对但审计链错）
    if date.today().weekday() >= 5:
        print(f"今日为周末（{date.today()}）——不执行，跳过")
        os.makedirs('runtime', exist_ok=True)
        with open('runtime/paper_heartbeat.json', 'w', encoding='utf-8') as f:
            json.dump({'date': str(date.today()),
                       'at': datetime.now().isoformat()[:19],
                       'results': {}, 'weekend': True}, f, ensure_ascii=False)
        return

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

    # 数据新鲜度守卫（F1，2026-10-20）：当日 14:45 的价格只有 WindPy 客户端
    # 能给（DB EOD 收盘后1-2h才落、缓存止于T-1）。终端没开 → 净值静默冻结
    # 而运行"成功"。检测：抽样缓存K线最大日期；非当日 → 标记+P1告警。
    try:
        import pandas as _pd
        kc = 'data_cache/kline'
        # 监控盲区修复（第七轮审查追加）：不再抽样前3个文件（排序靠前=
        # 000001等大盘蓝筹，数据最不容易滞后），改为检查实际持仓的标的
        # + 随机抽样5只——持仓冻结的才是真正影响净值的
        from live.ledger import Ledger
        _led = Ledger(os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), 'live', 'ledger.json'))
        _held = list(_led.positions.keys())[:5]
        import random as _rnd
        _all = [f for f in os.listdir(kc) if f.endswith('.parquet')]
        _rnd.seed(42)  # 确定性抽样
        _sample_names = _rnd.sample(_all, min(3, len(_all))) if _all else []
        _check_files = []
        for sym in _held:
            fname = sym.replace('.', '_') + '.parquet'
            if os.path.exists(os.path.join(kc, fname)):
                _check_files.append(fname)
        _check_files.extend(f for f in _sample_names if f not in _check_files)
        _check_files = _check_files[:8]   # 持仓5只+随机3只
        if not _check_files:
            _check_files = sorted(os.listdir(kc))[:3]  # 无持仓时退化为旧逻辑
        maxd = max(str(_pd.read_parquet(os.path.join(kc, f),
                                        columns=['date'])['date'].iloc[-1])[:10]
                   for f in _check_files if f.endswith('.parquet'))
        min_date = min(str(_pd.read_parquet(os.path.join(kc, f),
                                            columns=['date'])['date'].iloc[-1])[:10]
                      for f in _check_files if f.endswith('.parquet'))
        hb['data_max_date'] = maxd
        hb['data_min_date'] = min_date
        if min_date < str(date.today()):  # 用最旧的判断（任一持仓滞后=净值失真）
            hb['stale_data'] = True
            try:
                from monitoring.alerts import alert
                alert('PAPER_STALE_DATA', 'P1',
                      f"Paper标记用陈旧数据：缓存最新 {maxd} < 今日 {date.today()}"
                      f"——Wind终端未运行？当日14:45价格缺失，净值冻结（F1守卫）")
            except Exception:
                pass
            print(f"⚠️ 数据陈旧：缓存最新 {maxd} < 今日（Wind 终端未运行？）")
    except Exception as e:
        hb['data_check_error'] = str(e)
        hb['stale_data'] = True   # 检查本身失败也标记为陈旧（不能静默）
        try:
            from monitoring.alerts import alert
            alert('PAPER_DATA_CHECK_FAIL', 'P1',
                  f"数据新鲜度检查异常: {e}——视为数据不可信")
        except Exception:
            pass

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
