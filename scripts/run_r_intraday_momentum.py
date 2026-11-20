#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""R 系：分钟级强势因子同日检验（预注册见 R 系节）。"""


def main():
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

    from pathlib import Path
    import glob
    import numpy as np
    import pandas as pd

    DEST = Path('ai-created/research/r_intraday_mom_2026-11-11')
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
    print(f'{len(files)} 只 × {len(px["15:00"])} 天', flush=True)

    train_end = '2021-12-31'
    checkpoints = ['10:00', '10:30', '11:00', '11:30', '14:00', '14:30']
    results = {}
    for cp in checkpoints:
        # 因子：截至 cp 的当日累计涨幅
        factor = px[cp] / prev_close - 1
        # 标签：cp → 15:00 的剩余收益
        label = px['15:00'] / px[cp] - 1

        rk_f = factor.rank(axis=1, pct=True)
        rk_l = label.rank(axis=1, pct=True)

        daily_ic, top_rets, bot_rets = [], [], []
        dates = factor.index
        for i, d in enumerate(dates):
            if d <= '2016-12-05':
                continue
            f = rk_f.loc[d].dropna()
            l = label.loc[d]
            common = f.index.intersection(l.dropna().index)
            if len(common) < 200:
                continue
            ic = f[common].corr(l[common], method='spearman')
            if np.isfinite(ic):
                daily_ic.append((d, ic))
            top50 = f.nlargest(50).index
            bot50 = f.nsmallest(50).index
            top_rets.append((d, float(l.reindex(top50).mean())))
            bot_rets.append((d, float(l.reindex(bot50).mean())))

        ic_s = pd.Series(dict(daily_ic))
        top_s = pd.Series(dict(top_rets))
        bot_s = pd.Series(dict(bot_rets))
        spread = top_s - bot_s

        def summary(x):
            x = x.dropna()
            if len(x) < 100:
                return {}
            return {'mean_bp': float(x.mean() * 10000),
                    't': float(x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))),
                    'days': len(x)}

        tr_ic = ic_s[ic_s.index <= train_end]
        va_ic = ic_s[ic_s.index > train_end]
        tr_top = top_s[top_s.index <= train_end]
        va_top = top_s[top_s.index > train_end]
        va_bot = bot_s[bot_s.index > train_end]
        va_sp = spread[spread.index > train_end]

        results[cp] = {
            'ic_train': summary(tr_ic), 'ic_val': summary(va_ic),
            'top_train': summary(tr_top), 'top_val': summary(va_top),
            'bot_val': summary(va_bot), 'spread_val': summary(va_sp)}

    from data.preparation import atomic_json
    atomic_json(DEST / 'summary.json', results)

    print('\n== R 系：截至各时点"今天涨了多少" → 到收盘剩余收益（验证期）==')
    print(f"{'时点':<8}{'IC':>8}{'IC_t':>8}{'top50bp':>9}{'bot50bp':>9}{'多空bp':>8}")
    for cp in checkpoints:
        r = results[cp]
        ic, t = r['ic_val'].get('mean_bp', 0)/10000, r['ic_val'].get('t', 0)
        top = r['top_val'].get('mean_bp', 0)
        bot = r['bot_val'].get('mean_bp', 0)
        sp = r['spread_val'].get('mean_bp', 0)
        print(f"{cp:<8}{ic:>+8.4f}{t:>+8.1f}{top:>+9.1f}{bot:>+9.1f}{sp:>+8.1f}")
    print("\n注：top50bp = 追入当日最强50只到收盘的平均收益（0bp诊断口径）")

    from research.factor_screen2 import register
    register('R系分钟级追强6检验: 见r_intraday_mom证据', label='r_intraday_momentum')
    print(f"证据: {DEST}")


if __name__ == '__main__':
    main()
