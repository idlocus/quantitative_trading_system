#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Z6：持仓量×价格四象限信号（预注册见 CTA_PREREG Z6 节）。"""


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

    DEST = Path('ai-created/research/z6_oi_price_2026-11-17')
    DEST.mkdir(parents=True, exist_ok=True)

    SYMS = ['AU0','AG0','CU0','CF0','RB0','I0','M0','TA0','FG0','SC0',
            'J0','JM0','P0','SR0','HC0','NI0','AL0','ZN0','V0','PP0']

    COST = 0.00025
    OI_LOOKBACK = 5
    OI_THRESHOLD = 0.01  # 5日OI变化>1% = 显著
    TARGET_VOL = 0.15

    print('拉取数据（含持仓量）...', flush=True)
    data = {}
    for sym in SYMS:
        try:
            df = ak.futures_main_sina(symbol=sym, start_date='20150101', end_date='20261112')
            df['日期'] = pd.to_datetime(df['日期'])
            df = df.set_index('日期').sort_index()
            data[sym] = {'close': df['收盘价'].astype(float),
                         'oi': df['持仓量'].astype(float),
                         'volume': df['成交量'].astype(float)}
            time.sleep(0.2)
        except Exception:
            pass
    print(f'{len(data)} 品种', flush=True)

    def oi_signal(close, oi):
        """四象限信号"""
        ret5 = close.pct_change(OI_LOOKBACK)
        oi_chg5 = oi.pct_change(OI_LOOKBACK)
        oi_up = oi_chg5 > OI_THRESHOLD
        oi_dn = oi_chg5 < -OI_THRESHOLD
        px_up = ret5 > 0
        # Q1(新多): OI↑+价↑ → +1
        # Q3(新空): OI↑+价↓ → -1
        # Q2(空回): OI↓+价↑ → -0.5（弱反弹）
        # Q4(多离): OI↓+价↓ → +0.5（卖压枯竭）
        sig = pd.Series(0.0, index=close.index)
        sig[oi_up & px_up] = 1.0     # Q1 新多
        sig[oi_up & ~px_up] = -1.0   # Q3 新空
        sig[oi_dn & px_up] = -0.5    # Q2 空回补（弱）
        sig[oi_dn & ~px_up] = 0.5    # Q4 多离场（卖压枯竭）
        return sig

    # 两种信号版本
    def oi_signal_trend(close, oi):
        """顺势确认型：只做Q1/Q3"""
        ret5 = close.pct_change(OI_LOOKBACK)
        oi_chg5 = oi.pct_change(OI_LOOKBACK)
        sig = pd.Series(0.0, index=close.index)
        sig[(oi_chg5 > OI_THRESHOLD) & (ret5 > 0)] = 1.0    # Q1
        sig[(oi_chg5 > OI_THRESHOLD) & (ret5 < 0)] = -1.0   # Q3
        return sig

    def oi_signal_reversal(close, oi):
        """反转预判型：只做Q2/Q4"""
        ret5 = close.pct_change(OI_LOOKBACK)
        oi_chg5 = oi.pct_change(OI_LOOKBACK)
        sig = pd.Series(0.0, index=close.index)
        sig[(oi_chg5 < -OI_THRESHOLD) & (ret5 > 0)] = -1.0  # Q2 空回补→看空
        sig[(oi_chg5 < -OI_THRESHOLD) & (ret5 < 0)] = 1.0   # Q4 多离场→看多
        return sig

    # 通用回测
    def backtest(signal_func, label):
        all_daily = {}
        for sym in data:
            close = data[sym]['close']
            oi = data[sym]['oi']
            ret = close.pct_change()
            sig = signal_func(close, oi).shift(1)
            # vol weight
            # 简单等权（品种间不做vol target，组合层面做）
            daily = sig * ret
            all_daily[sym] = daily
    
        mx = pd.DataFrame(all_daily)
        # 组合等权 + vol target
        port_gross = mx.mean(axis=1)
        port_vol = port_gross.rolling(60).std() * np.sqrt(252)
        lev = (TARGET_VOL / port_vol).clip(upper=3).shift(1)
        net = port_gross * lev - COST * 0.1  # 信号变化不频繁，成本较低
        valid = net.dropna()
        nav = (1 + valid).cumprod()
    
        split = '2020-12-31'
        print(f'\n--- {label} ---')
        for name, r in [('全期', valid), ('前半', valid[valid.index<=split]),
                        ('后半', valid[valid.index>split])]:
            if len(r) < 100:
                continue
            yrs = len(r)/252
            ann = (1+r).prod()**(1/yrs)-1
            sh = r.mean()/r.std()*np.sqrt(252) if r.std()>0 else 0
            nav_p = (1+r).cumprod()
            mdd = float((nav_p/nav_p.cummax()-1).min())
            print(f'  {name}: 年化{ann:>+7.1%} 夏普{sh:>5.2f} maxDD{mdd:>7.1%}')
        return valid, nav

    print('\n===== Z6：持仓量×价格四象限信号 =====')
    v1, n1 = backtest(oi_signal, 'Z6a 四象限完整版（Q1/Q3主信号+Q2/Q4副信号）')
    v2, n2 = backtest(oi_signal_trend, 'Z6b 顺势确认型（只做Q1新多/Q3新空）')
    v3, n3 = backtest(oi_signal_reversal, 'Z6c 反转预判型（只做Q2空回/Q4多离）')

    # 分品种贡献（Z6b 顺势版）
    print('\n--- Z6b 分品种年化贡献 ---')
    for sym in data:
        close = data[sym]['close']
        oi = data[sym]['oi']
        ret = close.pct_change()
        sig = oi_signal_trend(close, oi).shift(1)
        daily = (sig * ret).dropna()
        if len(daily) > 500:
            print(f'  {sym:>5}: {daily.mean()*252:>+7.2%}/年')

    from research.factor_screen2 import register
    register('Z6 OI×价格四象限: 见z6证据', label='z6_oi_price')
    print(f'\n证据: {DEST}')


if __name__ == '__main__':
    main()
