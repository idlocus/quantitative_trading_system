#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""L 系完整报告：全部 32 因子 1 日口径验证结果（IC/t/胜率/FDR/每因子成本覆盖）。"""


def main():
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

    from pathlib import Path
    import numpy as np
    import pandas as pd

    from research import technical_signals as ts
    from research.factor_screen2 import PROTOCOL, load_ohlc_panel
    from research.signal_screen import daily_statistics

    DEST = Path('ai-created/research/l_h1_screen_2026-11-11')

    panel, base, evidence = load_ohlc_panel(PROTOCOL)
    price = base['price']
    eligible = (base['usable']
                & price.notna().rolling(PROTOCOL['minimum_history_sessions']).sum()
                .eq(PROTOCOL['minimum_history_sessions'])
                & base['amount'].rolling(PROTOCOL['liquidity_window'], min_periods=PROTOCOL['liquidity_window'])
                .median().ge(PROTOCOL['minimum_median_daily_amount_local_currency']))
    labels = (panel['open'].shift(-2) / panel['open'].shift(-1) - 1)
    labels = labels.where(np.isfinite(labels))
    signals = ts.build_all_signals(panel)
    summary = __import__('json').loads((DEST / 'summary.json').read_text(encoding='utf-8'))

    label_end = pd.Series(labels.index, index=labels.index).shift(-2)
    rows = []
    for name, sig in signals.items():
        daily = daily_statistics(sig, labels, eligible, 30)
        val = daily.loc[(daily.index > '2021-01-01') & (daily.index <= '2023-12-31')
                        & label_end.le(pd.Timestamp('2023-12-31'))]
        top = val['top_decile_forward_mean'].dropna()      # top10% 组日均收益
        bot = val['bottom_decile_forward_mean'].dropna()
        r = summary['signals'][name]
        rows.append({
            'factor': name,
            'train_ic': r['train']['mean_ic'], 'val_ic': r['validation']['mean_ic'],
            'val_t': r['validation']['t'], 'win_rate': r['validation'].get('win_rate'),
            'fdr_pass': summary['fdr'].get(name, False),
            'top10_daily_bp': float(top.mean() * 10000) if len(top) else None,
            'bot10_daily_bp': float(bot.mean() * 10000) if len(bot) else None,
            'long_short_bp': float((top - bot.reindex(top.index)).mean() * 10000) if len(top) else None,
        })

    df = pd.DataFrame(rows).sort_values('val_ic', ascending=False)
    df.to_csv(DEST / 'full_h1_report.csv', index=False, encoding='utf-8-sig')

    # 正式文档
    lines = ['# 全部因子·1 日口径验证完整报告（2026-11-11）', '',
             '标签：T 日收盘信号 → T+1 开盘 至 T+2 开盘收益（含隔夜）。验证期 2021-2023。',
             'top10%/bot10% = 按因子分数每十分位的日均收益（bp）；换仓成本参考：单边 25bp。', '',
             '| 因子 | 训练IC | 验证IC | 验证t | IC胜率 | FDR | top10%日均 | bot10%日均 | 多空日价差 | 成本覆盖 |',
             '|---|---:|---:|---:|---:|:-:|---:|---:|---:|:-:|']
    for _, x in df.iterrows():
        cover = '未验证换手'
        lines.append(f"| {x.factor} | {x.train_ic:+.4f} | {x.val_ic:+.4f} | {x.val_t:+.2f} | "
                     f"{x.win_rate:.0%} | {'✓' if x.fdr_pass else '✗'} | {x.top10_daily_bp:+.1f} | "
                     f"{x.bot10_daily_bp:+.1f} | {x.long_short_bp:+.1f} | {cover} |")
    Path('docs/H1_FACTOR_REPORT_2026-11-11.md').write_text('\n'.join(lines), encoding='utf-8')
    print('\n'.join(lines[6:]))
    print(f"\n完整报告: docs/H1_FACTOR_REPORT_2026-11-11.md | 数据: {DEST}/full_h1_report.csv")


if __name__ == '__main__':
    main()
