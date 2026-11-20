#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Z2：多周期CTA趋势跟随（预注册见 CTA_PREREG Z2 节）。"""


def main():
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

    from pathlib import Path
    import numpy as np
    import pandas as pd

    DEST = Path('ai-created/research/z2_cta_multima_2026-11-16')
    DEST.mkdir(parents=True, exist_ok=True)

    SYMBOLS = {
        'RB0': '螺纹钢', 'CU0': '沪铜', 'AU0': '黄金', 'AG0': '白银',
        'I0': '铁矿石', 'M0': '豆粕', 'TA0': 'PTA', 'CF0': '棉花',
        'FG0': '玻璃', 'SC0': '原油',
    }
    COST = 0.00025
    TARGET_VOL = 0.10
    PARAM_SETS = [(10, 30), (20, 60), (50, 200)]  # 短/中/长三套

    import akshare as ak
    closes = {}
    for sym in SYMBOLS:
        df = ak.futures_main_sina(symbol=sym, start_date='20150101', end_date='20261112')
        df['日期'] = pd.to_datetime(df['日期'])
        closes[sym] = df.set_index('日期')['收盘价'].astype(float).sort_index()
    close_mx = pd.DataFrame(closes).sort_index()
    ret_mx = close_mx.pct_change()

    # 三套信号
    all_signals = []
    for fast, slow in PARAM_SETS:
        ma_f = close_mx.rolling(fast).mean()
        ma_s = close_mx.rolling(slow).mean()
        sig = (ma_f > ma_s).astype(float) - (ma_f < ma_s).astype(float)
        all_signals.append(sig)
    # 平均信号（-1到+1之间的连续值）
    avg_signal = sum(all_signals) / len(all_signals)
    avg_signal = avg_signal.shift(1)  # T→T+1

    # 仓位
    vol = ret_mx.rolling(60).std() * np.sqrt(252)
    inv_vol = 1 / vol
    w_raw = inv_vol.div(inv_vol.sum(axis=1), axis=0)
    port_vol_est = (w_raw * vol * avg_signal.abs()).sum(axis=1)
    leverage = (TARGET_VOL / port_vol_est).clip(upper=3.0)
    weights = w_raw.mul(leverage, axis=0) * avg_signal

    # 收益
    position = weights.shift(1)
    gross = (position * ret_mx).sum(axis=1)
    pos_change = weights.diff().abs().sum(axis=1)
    cost = pos_change * COST
    net = gross - cost.shift(1)
    valid = net.dropna()
    nav = (1 + valid).cumprod()

    print('===== Z2：多周期CTA（10/30+20/60+50/200 组合，10品种）=====')
    split1 = '2020-12-31'
    for name, r in [('全期 2015-2026', valid),
                    ('前半 2015-2020', valid[valid.index <= split1]),
                    ('后半 2021-2026', valid[valid.index > split1])]:
        if len(r) < 100:
            continue
        years = len(r) / 252
        ann = (1 + r).prod() ** (1/years) - 1
        sharpe = r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else 0
        nav_p = (1 + r).cumprod()
        mdd = float((nav_p / nav_p.cummax() - 1).min())
        wr = (r > 0).mean()
        print(f'{name:<16} 年化{ann:>+7.1%} 夏普{sharpe:>6.2f} maxDD{mdd:>7.1%} 胜率{wr:>5.0%}')

    print('\n--- 分品种年化 ---')
    for sym in SYMBOLS:
        sr = (position[sym] * ret_mx[sym]).dropna()
        if len(sr) > 500:
            print(f'  {SYMBOLS[sym]:<6}: {sr.mean()*252:>+7.2%}/年')

    pd.DataFrame({'net_ret': valid, 'nav': nav}).to_csv(DEST / 'z2_daily_nav.csv', encoding='utf-8-sig')

    from research.factor_screen2 import register
    register('Z2 CTA多周期组合: 见z2_cta证据', label='z2_cta_multima')
    print(f'\n证据: {DEST}')


if __name__ == '__main__':
    main()
