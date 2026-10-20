#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Paper 验证期看门狗 —— 每日收盘后由计划任务运行（保证持续性可观测）。

检查当日心跳是否落盘；缺失/失败 → P1 告警（webhook + 本地审计）。
连续缺失 ≥2 个交易日 → P0 告警。周末不告警。
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# ^ 同 paper_day：计划任务 cwd 兜底

import json
from datetime import date, timedelta

HB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                  'runtime', 'paper_heartbeat.json')


def latest_weekday(d=None):
    d = d or date.today()
    while d.weekday() >= 5:
        d -= timedelta(days=1)
    return d


def main():
    today = latest_weekday()
    if date.today().weekday() >= 5:
        print("周末——不检查")
        return

    hb = {}
    if os.path.isfile(HB):
        try:
            hb = json.load(open(HB, encoding='utf-8'))
        except Exception:
            hb = {}

    from monitoring.alerts import alert

    if hb.get('date') != str(today):
        # 看昨天心跳是否也缺（连续性判断）
        yest = latest_weekday(today - timedelta(days=1))
        hb_y = {}
        if os.path.isfile(HB):
            try:
                hb_y = json.load(open(HB, encoding='utf-8'))
            except Exception:
                pass
        consecutive = hb_y.get('date') != str(yest)
        alert('PAPER_RUN_MISSED',
              'P0' if consecutive else 'P1',
              f"Paper 验证期心跳缺失：最近心跳={hb.get('date', '无')}，"
              f"期望={today}" + ("（连续缺失≥2日）" if consecutive else ""),
              last_heartbeat=hb.get('at'))
        print(f"⚠️ 告警已发：心跳缺失（最近 {hb.get('date')}）")
        sys.exit(1)

    failed = [n for n, rc in (hb.get('results') or {}).items() if rc != 0]
    if failed:
        alert('PAPER_STEP_FAILED', 'P1',
              f"Paper 当日运行有失败步骤: {failed}", results=hb.get('results'))
        print(f"⚠️ 告警已发：失败步骤 {failed}")
        sys.exit(1)

    if hb.get('stale_data'):
        alert('PAPER_STALE_DATA', 'P1',
              f"心跳标记数据陈旧（data_max_date={hb.get('data_max_date')}）——"
              f"净值冻结风险，检查 Wind 终端")
        print(f"⚠️ 告警已发：数据陈旧 {hb.get('data_max_date')}")
        sys.exit(1)

    print(f"✅ 心跳正常 @{hb.get('at')} | stock={hb.get('stock_nav')} "
          f"etf={hb.get('etf_nav')}")


if __name__ == '__main__':
    main()
