#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Z5a：国内定价品种画像（预注册见 CTA_PREREG Z5 节）。"""


def main():
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

    from pathlib import Path
    import numpy as np
    import pandas as pd
    import akshare as ak
    import time

    DEST = Path('ai-created/research/z5_domestic_2026-11-17')
    DEST.mkdir(parents=True, exist_ok=True)

    SYMS = {
        'LH0': '生猪', 'P0': '棕榈油', 'SA0': '纯碱', 'FG0': '玻璃',
        'JM0': '焦煤', 'J0': '焦炭', 'M0': '豆粕', 'RB0': '螺纹钢',
    }

    closes = {}
    for sym, name in SYMS.items():
        try:
            df = ak.futures_main_sina(symbol=sym, start_date='20150101', end_date='20261112')
            df['日期'] = pd.to_datetime(df['日期'])
            closes[sym] = df.set_index('日期')['收盘价'].astype(float).sort_index()
            print(f'{sym}({name}): {len(closes[sym])} 条 {closes[sym].index[0].date()}~{closes[sym].index[-1].date()}', flush=True)
            time.sleep(0.3)
        except Exception as e:
            print(f'{sym}({name}): 失败 {str(e)[:40]}', flush=True)

    print('\n===== Z5a：品种画像 =====')
    print(f"{'品种':<8}{'天数':>6}{'年化波动':>8}{'趋势度':>8}{'自相关':>8}{'季节性':>8}{'最佳策略提示':>12}")
    for sym, name in SYMS.items():
        if sym not in closes or len(closes[sym]) < 500:
            print(f'{name:<8} 数据不足')
            continue
        px = closes[sym]
        ret = px.pct_change().dropna()
        n = len(ret)
    
        # 1) 年化波动率
        ann_vol = ret.std() * np.sqrt(252)
    
        # 2) 趋势度：价格序列的 Hurst 近似（简化：方差比检验）
        # VR(q) = Var(q期收益) / (q × Var(1期收益))
        # VR > 1 = 趋势性；VR < 1 = 均值回归
        var1 = ret.var()
        var5 = ret.rolling(5).sum().var()  # 5日累计收益方差
        vr5 = var5 / (5 * var1) if var1 > 0 else 1.0
        var20 = ret.rolling(20).sum().var()
        vr20 = var20 / (20 * var1) if var1 > 0 else 1.0
    
        # 3) 日收益自相关（ lag1 ）
        ac1 = ret.autocorr(lag=1)
    
        # 4) 季节性：月度收益的ANOVA F统计量（简化：月度均值的组间方差/组内方差）
        monthly = ret.groupby(ret.index.month).mean()
        seasonal_strength = monthly.std() / (ret.std() / np.sqrt(21)) if ret.std() > 0 else 0
    
        # 综合判断
        if vr20 > 1.1:
            style = '趋势型'
        elif vr20 < 0.9:
            style = '均值回归型'
        else:
            style = '随机游走型'
    
        print(f'{name:<8}{n:>6}{ann_vol:>7.1%}{vr20:>8.2f}{ac1:>+8.3f}{seasonal_strength:>8.2f}{style:>10}')

    # 各品种的月度季节性
    print('\n===== 月度季节性（各月平均日收益 bp）=====')
    print(f"{'品种':<8}", end='')
    for m in range(1, 13):
        print(f'{m:>3}月', end='')
    print()
    for sym, name in SYMS.items():
        if sym not in closes or len(closes[sym]) < 500:
            continue
        ret = closes[sym].pct_change().dropna()
        monthly = ret.groupby(ret.index.month).mean() * 10000
        print(f'{name:<8}', end='')
        for m in range(1, 13):
            v = monthly.get(m, 0)
            print(f'{v:>+4.0f}', end='')
        print()

    # 5日/20日均值回归测试
    print('\n===== 均值回归测试（过去N日收益→未来N日收益的相关性）=====')
    print(f"{'品种':<8}{'5日':>8}{'10日':>8}{'20日':>8}{'解读':>12}")
    for sym, name in SYMS.items():
        if sym not in closes or len(closes[sym]) < 500:
            continue
        ret = closes[sym].pct_change().dropna()
        row = f'{name:<8}'
        for lb in (5, 10, 20):
            past = ret.rolling(lb).sum()
            future = ret.rolling(lb).sum().shift(-lb)
            corr = past.corr(future)
            row += f'{corr:>+8.3f}'
        # 判断
        ret5 = ret.rolling(5).sum()
        ret5_fwd = ret5.shift(-5)
        c = ret5.corr(ret5_fwd)
        if c < -0.05:
            row += f'{"显著回归":>12}'
        elif c > 0.05:
            row += f'{"显著趋势":>12}'
        else:
            row += f'{"不显著":>12}'
        print(row)

    from research.factor_screen2 import register
    register('Z5a 国内品种画像: 见z5证据', label='z5_domestic_profile')
    print(f'\n证据: {DEST}')


if __name__ == '__main__':
    main()
