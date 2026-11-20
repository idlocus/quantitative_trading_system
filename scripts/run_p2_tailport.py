#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P2：尾盘因子组合化（预注册见 P2 节）。"""
import sys, os
if __name__ == '__main__':
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from pathlib import Path
import glob
import numpy as np
import pandas as pd

DEST = Path('ai-created/research/p2_tailport_2026-11-11')



def main():
    DEST.mkdir(parents=True, exist_ok=True)
    """审计 2026-11-19 四-h：数据读取/目录创建/实验执行全部收进 main()——
    此前模块级执行，import 即触发整段研究并覆盖输出。"""
    files = sorted(glob.glob('data_cache/jq30/*.parquet'))
    frames = []
    for f in files:
        sym = f.replace(chr(92), '/').split('/')[-1].replace('_30m.parquet', '')
        df = pd.read_parquet(f)
        if df.empty:
            continue
        idx = df.index
        frames.append(pd.DataFrame({'sym': sym, 'date': idx.strftime('%Y-%m-%d'),
                                    'bar': idx.strftime('%H:%M'), 'close': df['close'].values}))
    big = pd.concat(frames, ignore_index=True)
    big = big[big['bar'].isin(['10:00', '14:30', '15:00'])]
    mat = big.pivot_table(index=['date', 'bar'], columns='sym', values='close')
    px = {bt: mat.xs(bt, level='bar') for bt in ['10:00', '14:30', '15:00']}
    for bt in px:
        px[bt] = px[bt].sort_index()
    dates = px['15:00'].index
    print(f'{len(files)} 只 × {len(dates)} 天', flush=True)

    s7 = px['15:00'] / px['14:30'] - 1          # 当日尾盘收益
    s7_lag = s7.shift(1)                          # 昨日尾盘 = 信号
    rank = s7_lag.rank(axis=1, pct=True)

    # 三版收益
    ret_a = px['14:30'] / px['10:00'] - 1         # 今开→今尾
    ret_b = px['14:30'].shift(-1) / px['14:30'] - 1  # 今尾→明尾
    ret_c = px['10:00'].shift(-1) / px['10:00'] - 1  # 今开→明开

    train_end = '2021-12-31'
    results = {}
    for name, ret in (('P2a_open2tail', ret_a), ('P2b_tail2tail', ret_b), ('P2c_open2open', ret_c)):
        daily = []
        prev_set = None
        for i in range(1, len(dates) - 1):
            d = dates[i]
            r = rank.loc[d].dropna()
            rr = ret.loc[d]
            if len(r) < 300:
                daily.append({'date': d, 'gross': np.nan, 'turnover': np.nan})
                prev_set = None
                continue
            top = set(r.nlargest(50).index)
            gross = rr.reindex(sorted(top)).mean()
            turn = 1.0 if prev_set is None else 1 - len(top & prev_set) / 50.0
            prev_set = top
            daily.append({'date': d, 'gross': float(gross) if np.isfinite(gross) else np.nan,
                          'turnover': float(turn)})
        df = pd.DataFrame(daily).set_index('date').dropna()
        for cost_bp in (0, 25):
            net = df['gross'] - df['turnover'] * 2 * cost_bp / 10000
            tr = net[net.index <= train_end]; va = net[net.index > train_end]
            def stat(x):
                if len(x) < 100:
                    return (np.nan, np.nan, 0)
                return (float(x.mean() * 10000),
                        float(x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))),
                        int((x > 0).mean() * 100))
            results[f'{name}_{cost_bp}bp'] = {
                'train_bp': stat(tr), 'val_bp': stat(va),
                'gross_val_bp': float(df.loc[df.index > train_end, 'gross'].mean() * 10000),
                'avg_turnover': float(df['turnover'].mean())}
        df.to_csv(DEST / f'{name}_daily.csv', encoding='utf-8-sig')

    print('\n== P2：尾盘因子 top50 组合（日均 bp；可行候选线=毛+15bp）==')
    for k, v in results.items():
        tr, va = v['train_bp'], v['val_bp']
        print(f"{k:<22} 训练 {tr[0]:+6.1f}bp(t{tr[1]:+5.1f}) | 验证 {va[0]:+6.1f}bp"
              f"(t{va[1]:+5.1f}, 胜率{va[2]}%) | 毛验证 {v['gross_val_bp']:+.1f}bp"
              f" | 日换手 {v['avg_turnover']:.0%}")
    from data.preparation import atomic_json
    atomic_json(DEST / 'summary.json', {k: {kk: (list(vv) if isinstance(vv, tuple) else vv)
                                            for kk, vv in v.items()} for k, v in results.items()})
    from research.factor_screen2 import register
    register('P2尾盘组合6run: 见p2_tailport证据', label='p2_tailport')
    print(f'\n证据: {DEST}')


if __name__ == '__main__':
    main()
