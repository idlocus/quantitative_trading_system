#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P 系：日内段收益结构（预注册见 P 系节）。"""


def main():
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

    from pathlib import Path
    import glob
    import numpy as np
    import pandas as pd
    from scipy import stats as sps
    from scipy.stats import false_discovery_control

    DEST = Path('ai-created/research/p_intraday_2026-11-11')
    DEST.mkdir(parents=True, exist_ok=True)

    # 1) 加载：每只票 → (date, bar_idx) close 矩阵
    files = sorted(glob.glob('data_cache/jq30/*.parquet'))
    syms, panels = [], []
    for f in files:
        sym = f.replace(chr(92), '/').split('/')[-1].replace('_30m.parquet', '')
        df = pd.read_parquet(f)
        if df.empty:
            continue
        idx = df.index
        day = idx.strftime('%Y-%m-%d')
        bars = idx.strftime('%H:%M')
        panels.append(pd.DataFrame({'sym': sym, 'date': day, 'bar': bars,
                                    'close': df['close'].values}))
    big = pd.concat(panels, ignore_index=True)
    print(f'{len(files)} 只加载完成', flush=True)

    # 每天的 bar 时序（标准 8 根）
    bar_times = ['10:00', '10:30', '11:00', '11:30', '13:30', '14:00', '14:30', '15:00']
    pivot = big[big['bar'].isin(bar_times)]
    mat = pivot.pivot_table(index=['date', 'bar'], columns='sym', values='close')
    mat = mat.reindex(pd.MultiIndex.from_product([mat.index.levels[0], bar_times],
                                                 names=['date', 'bar']))
    wide = mat.unstack('bar')                       # date × (sym, bar)
    print(f'面板: {len(wide)} 天 × {mat.columns.nlevels} 只', flush=True)

    # 2) 9 段收益：S0=隔夜(昨15:00→今10:00) S1..S7=相邻bar差 S8=无（8bar→8段日内+隔夜=9段）
    def seg_returns(wide):
        segs = {}
        prev_close = wide.xs('15:00', axis=1, level=1)
        for i, bt in enumerate(bar_times):
            cur = wide.xs(bt, axis=1, level=1)
            if bt == '10:00':
                segs['S0_overnight'] = cur / prev_close - 1
            else:
                prev_bt = bar_times[i - 1]
                prev = wide.xs(prev_bt, axis=1, level=1)
                segs[f'S{i}_{bt}'] = cur / prev - 1
        daily = wide.xs('15:00', axis=1, level=1) / wide.xs('10:00', axis=1, level=1) - 1
        return segs, daily

    segs, daily_ret = seg_returns(wide)
    dates = wide.index

    # 3) IC：昨日段收益 rank → 次日全日收益 rank（Spearman）
    def daily_ic(sig, target):
        out = {}
        rk_s = sig.rank(axis=1, pct=True)
        rk_t = target.rank(axis=1, pct=True)
        for i in range(len(dates) - 1):
            d, d1 = dates[i], dates[i + 1]
            s = rk_s.loc[d].dropna()
            t = rk_t.loc[d1].reindex(s.index).dropna()
            common = s.index.intersection(t.index)
            if len(common) < 100:
                continue
            out[d1] = float(s[common].corr(t[common], method='spearman'))
        return pd.Series(out)

    train_end = '2021-12-31'
    main_results, pvals = {}, {}
    for name, seg in segs.items():
        ic = daily_ic(seg, daily_ret).dropna()
        tr = ic[ic.index <= train_end]; va = ic[ic.index > train_end]
        def summary(x):
            if len(x) < 100:
                return {'days': len(x), 'mean_ic': None, 't': None}
            return {'days': len(x), 'mean_ic': float(x.mean()),
                    't': float(x.mean() / (x.std(ddof=1) / np.sqrt(len(x))))}
        main_results[name] = {'train': summary(tr), 'validation': summary(va)}
        vt = main_results[name]['validation']['t']
        pvals[name] = 1.0 if vt is None else float(2 * sps.norm.sf(abs(vt)))

    names_sorted = sorted(pvals, key=pvals.get)
    p = np.array([pvals[n] for n in names_sorted])
    fdr = dict(zip(names_sorted, map(bool, false_discovery_control(p, method='bh') <= 0.10)))

    # 4) 副报告：段→次日同段
    sub_results = {}
    for name, seg in segs.items():
        ic = daily_ic(seg, seg).dropna()
        va = ic[ic.index > train_end]
        if len(va) >= 100:
            sub_results[name] = {'val_ic': float(va.mean()),
                                 't': float(va.mean() / (va.std(ddof=1) / np.sqrt(len(va))))}
        else:
            sub_results[name] = {'val_ic': None, 't': None}

    from data.preparation import atomic_json
    atomic_json(DEST / 'summary.json', {'main': main_results, 'fdr': fdr,
                                        'same_segment': sub_results,
                                        'n_symbols': int(mat.columns.nlevels),
                                        'note': '695只口径限制'})
    print('\n== P 系：昨日段收益 → 次日全日收益（验证期 2022-01~2026-10）==')
    print(f"{'段':<14}{'训练IC':>9}{'验证IC':>9}{'验证t':>8}{'FDR':>5}")
    for n in sorted(main_results, key=lambda k: -(main_results[k]['validation']['mean_ic'] or -9)):
        r = main_results[n]
        if r['validation']['mean_ic'] is None:
            continue
        print(f"{n:<14}{r['train']['mean_ic']:>+9.4f}{r['validation']['mean_ic']:>+9.4f}"
              f"{r['validation']['t']:>+8.2f}{'✓' if fdr.get(n) else '✗':>5}")
    print('\n== 副报告：段→次日同段（延续性）==')
    for n, r in sub_results.items():
        if r['val_ic'] is not None:
            print(f"{n:<14} val IC {r['val_ic']:+.4f} (t={r['t']:+.2f})")
    from research.factor_screen2 import register
    survivors = [n for n, ok in fdr.items() if ok]
    register(f'P系分钟线段结构18检验: 主检验FDR存活{len(survivors)}', label='p_intraday')
    print(f'\n证据: {DEST}')


if __name__ == '__main__':
    main()
