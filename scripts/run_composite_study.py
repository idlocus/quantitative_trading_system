#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""阶段二入口：多因子共同作用（预注册 FACTOR_PREREG_2026-11-06.md 阶段二节）。"""
import sys, os
if __name__ == '__main__':
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import json
from data.preparation import atomic_json
from research.composite_study import (DEST, PORTFOLIO_PROTOCOL, PROTOCOL,
                                      portfolio_layer, signal_layer)


def main() -> None:
    composite, ic_report, eligible, d16, base, evidence = signal_layer(PROTOCOL)

    print('== 信号层：合成 vs 最优单因子 D16 ==')
    best_single = ic_report['D16_benchmark']['validation']['mean_ic']
    lines = [f"D16_benchmark  验证IC {best_single:+.4f} "
             f"(t={ic_report['D16_benchmark']['validation']['hac_t_stat']:+.2f})"]
    beat = []
    for name in ('S1_all_dir_ew', 'S2_cluster_reps', 'S3_positive_ew', 'S4_ic_weighted'):
        va = ic_report[name]['validation']
        mi, t = va['mean_ic'], va['hac_t_stat']
        lines.append(f"{name:<16} 验证IC {mi:+.4f} (t={t:+.2f})"
                     if mi is not None else f"{name:<16} 验证IC 无有效日")
        if mi is not None and t is not None and mi > best_single and abs(t) >= 2:
            beat.append(name)
    print('\n'.join(lines))
    signal_pass = bool(beat)
    print(f"信号层判定（合成IC>D16且t>=2）: {'PASS ' + str(beat) if signal_pass else 'FAIL'}")

    print('\n== 组合层：N=50 月调仓 25bp 双15%（guard=风险预算）==')
    reports = portfolio_layer(composite, eligible, d16, base, evidence, PORTFOLIO_PROTOCOL)
    bench = reports['D16_benchmark_budget_guard_25bps']['metrics']
    bench_q = reports['D16_benchmark_budget_guard_25bps'].get('qualifies_research_candidate')
    port_pass = []
    for name in ('S1_all_dir_ew', 'S2_cluster_reps', 'S3_positive_ew', 'S4_ic_weighted'):
        full = reports[f'{name}_budget_guard_25bps']
        r = full['metrics']
        better = (r['geometric_monthly_return'] > 0
                  and r['geometric_monthly_return'] >= bench['geometric_monthly_return'])
        print(f"{name:<16} 月收益 {r['geometric_monthly_return']:+.2%} | "
              f"maxDD {abs(r['max_drawdown']):.1%} | 最差月 {r['worst_calendar_month']:+.1%} | "
              f"qualifies={full.get('qualifies_research_candidate')}")
        if better and full.get('qualifies_research_candidate'):
            port_pass.append(name)
    print(f"D16_benchmark     月收益 {bench['geometric_monthly_return']:+.2%} | "
          f"maxDD {abs(bench['max_drawdown']):.1%} | qualifies={bench_q}")
    print(f"组合层判定: {'PASS ' + str(port_pass) if port_pass else 'FAIL'}")

    atomic_json(DEST / 'verdict.json', {
        'signal_layer_pass': signal_pass, 'signal_beat': beat,
        'portfolio_layer_pass': bool(port_pass), 'portfolio_pass': port_pass,
        'ic_report': {k: v for k, v in ic_report.items()},
    })
    verdict = ('PASS' if (signal_pass and port_pass)
               else ('排序层增益(组合层未达标)' if signal_pass else 'FAIL'))
    print(f"\n==> 阶段二（共同作用）: {verdict} | 证据: {DEST}")

    from research.factor_screen2 import register
    register(f'composite S1-S4: 信号层{signal_pass}/组合层{bool(port_pass)}', label='composite_study')


if __name__ == '__main__':
    main()
