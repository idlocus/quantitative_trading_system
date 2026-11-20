#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""S 系：涨停前买入假设（预注册见 S 系节）。"""


def main():
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

    from pathlib import Path
    import glob
    import numpy as np
    import pandas as pd

    DEST = Path('ai-created/research/s_pre_limitup_2026-11-11')
    DEST.mkdir(parents=True, exist_ok=True)

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
    bars = ['10:00', '10:30', '11:00', '11:30', '14:00', '14:30', '15:00']
    big = big[big['bar'].isin(bars)]
    mat = big.pivot_table(index=['date', 'bar'], columns='sym', values='close')
    px = {bt: mat.xs(bt, level='bar').sort_index() for bt in bars}
    prev_close = px['15:00'].shift(1)
    close_ret = px['15:00'] / prev_close - 1     # 当日全天涨幅（判断最终是否涨停）
    print(f'{len(files)} 只 × {len(px["15:00"])} 天', flush=True)

    train_end = '2021-12-31'
    groups = [('A_温和+2~4%', 0.02, 0.04), ('B_较强+4~6%', 0.04, 0.06),
              ('C_很强+6~8%', 0.06, 0.08), ('D_即将涨停+8~9.7%', 0.08, 0.097),
              ('E_已涨停≥9.7%', 0.097, 1.0)]
    checkpoints = ['10:00', '10:30', '11:00', '14:00']

    # 输出: {checkpoint: {group: {mean_bp, t, n, limitup_rate}}}
    all_results = {}
    for cp in checkpoints:
        cum = px[cp] / prev_close - 1   # 截至cp的当日累计涨幅
        rem = px['15:00'] / px[cp] - 1  # cp→收盘剩余收益
        final = close_ret                # 全天涨幅（判涨停）
        res = {}
        for gname, lo, hi in groups:
            mask = (cum > lo) & (cum <= hi)
            rems, lims = [], []
            for d in cum.index:
                if d <= '2016-12-05':
                    continue
                m = mask.loc[d]
                selected = m[m].index
                if len(selected) < 5:
                    continue
                r = rem.loc[d].reindex(selected).dropna()
                if len(r) > 0:
                    rems.append((d, float(r.mean())))
                # 该组中最终收盘涨停的比例
                f = final.loc[d].reindex(selected).dropna()
                if len(f) > 0:
                    lims.append((d, float((f >= 0.093).mean())))  # 9.3%=考虑容差
            r_s = pd.Series(dict(rems))
            l_s = pd.Series(dict(lims))
            va_r = r_s[r_s.index > train_end].dropna()
            va_l = l_s[l_s.index > train_end].dropna()
            if len(va_r) >= 100:
                res[gname] = {
                    'mean_bp': float(va_r.mean() * 10000),
                    't': float(va_r.mean() / (va_r.std(ddof=1) / np.sqrt(len(va_r)))),
                    'days': int(len(va_r)),
                    'limitup_rate': float(va_l.mean()) if len(va_l) > 0 else None}
            else:
                res[gname] = {'mean_bp': None, 'days': len(va_r)}
        all_results[cp] = res

    from data.preparation import atomic_json
    # 转换不可序列化的 None
    atomic_json(DEST / 'summary.json', all_results)

    print('\n== S 系：涨停前买入假设（验证期 2022-2026；0bp；剩余收益=追入到收盘）==')
    for cp in checkpoints:
        print(f'\n--- 截至{cp} ---')
        print(f"{'组':<18}{'日均bp':>8}{'t':>7}{'日均样本':>8}{'最终涨停率':>9}")
        for gname, _, _ in groups:
            r = all_results[cp].get(gname, {})
            if r.get('mean_bp') is not None:
                print(f"{gname:<18}{r['mean_bp']:>+8.1f}{r['t']:>+7.1f}"
                      f"{r['days']:>8d}{r.get('limitup_rate', 0):>9.1%}")
            else:
                print(f"{gname:<18}  样本不足({r.get('days',0)}天)")

    from research.factor_screen2 import register
    register('S系涨停前买入5组×4时点: 见s_pre_limitup证据', label='s_pre_limitup')
    print(f"\n证据: {DEST}")


if __name__ == '__main__':
    main()
