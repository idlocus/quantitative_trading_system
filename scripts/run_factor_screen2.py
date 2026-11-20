#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""第二轮因子筛选入口（预注册 docs/FACTOR_PREREG_2026-11-06.md）。

只读数据、只写 ai-created/research/factor_screen2_2026-11-06/ 与登记册。
"""
import sys, os
if __name__ == '__main__':
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from research.factor_screen2 import main, register, DEST   # noqa: E402


def render(summary: dict) -> str:
    lines = []
    sig = summary['signals']
    lines.append(f"信号 {len(sig)} 个 | 检验 {summary['n_tests']} | FDR q={summary['protocol']['fdr_q']}")
    lines.append('')
    lines.append(f"{'信号':<24}{'训练IC':>9}{'验证IC':>9}{'验证t':>8}{'FDR严格':>8}")
    for name in sorted(sig, key=lambda k: -(sig[k]['validation']['mean_ic'] or 0)):
        tr = sig[name]['train']; va = sig[name]['validation']
        mi = va['mean_ic']; t = va['hac_t_stat']
        lines.append(f"{name:<24}"
                     f"{(tr['mean_ic'] or 0):>+9.4f}{(mi or 0):>+9.4f}{(t or 0):>+8.2f}"
                     f"{'PASS' if summary['fdr_strict'].get(name) else '-':>8}")
    lines.append('')
    lines.append(f"严格口径存活: {summary['fdr_strict_survivors'] or '无'}")
    lines.append(f"簇数: {len(summary['clusters'])} | 簇视图存活: {summary['cluster_view_survivors'] or '无'}")
    if summary['downgrade_list']:
        lines.append(f"降级复核名单(不进阶段二): {summary['downgrade_list']}")
    inc = summary['d_residual_incremental']
    strong_inc = [n for n, v in inc.items()
                  if v['validation_t'] is not None and abs(v['validation_t']) >= 2]
    lines.append(f"D组对基准集的残差增量显著(|t|>=2): {strong_inc or '无'}")
    return '\n'.join(lines)


if __name__ == '__main__':
    summary = main()
    report = render(summary)
    (DEST / 'report.md').write_text(
        '# 第二轮因子筛选（32 检验，预注册 2026-11-06）\n\n```\n' + report + '\n```\n',
        encoding='utf-8')
    print(report)
    register(f"严格存活{len(summary['fdr_strict_survivors'])}/"
             f"簇视图{len(summary['cluster_view_survivors'])}——见证据目录")
    print(f"\n证据: {DEST} | report.md 已写入")
