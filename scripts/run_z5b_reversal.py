#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Z5b：国内品种趋势反转策略回测 + 当前位置诊断。"""


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

    SYMS = {
        'LH0': '生猪', 'P0': '棕榈油', 'SA0': '纯碱', 'FG0': '玻璃',
        'JM0': '焦煤', 'J0': '焦炭', 'M0': '豆粕', 'RB0': '螺纹钢',
    }
    COST = 0.00025

    closes = {}
    for sym in SYMS:
        try:
            df = ak.futures_main_sina(symbol=sym, start_date='20150101', end_date='20261112')
            df['日期'] = pd.to_datetime(df['日期'])
            closes[sym] = df.set_index('日期')['收盘价'].astype(float).sort_index()
            time.sleep(0.2)
        except Exception:
            pass

    print('===== 第一部分：当前位置诊断（截至2026-11-12）=====')
    print(f"{'品种':<8}{'现价':>8}{'3年百分位':>9}{'偏离200日线':>11}{'z-score':>8}{'RSI14':>7}{'60日收益':>9}{'信号':>6}")
    for sym, name in SYMS.items():
        if sym not in closes or len(closes[sym]) < 300:
            continue
        px = closes[sym]
        ret = px.pct_change()
    
        # 3年百分位
        hist_3y = px[px.index >= px.index[-1] - pd.Timedelta(days=1095)]
        pctile = float((hist_3y < px.iloc[-1]).mean())
    
        # 偏离200日均线
        ma200 = px.rolling(200).mean()
        dev = px.iloc[-1] / ma200.iloc[-1] - 1
    
        # z-score vs 200日均线
        resid = (px - ma200) / px.rolling(200).std()
        z = float(resid.iloc[-1])
    
        # RSI 14
        delta = px.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rsi = float((100 * gain / (gain + loss)).iloc[-1])
    
        # 60日收益 z-score
        ret60 = px.iloc[-1] / px.iloc[-61] - 1
        ret60_hist = px.pct_change(60).dropna()
        z60 = float((ret60 - ret60_hist.mean()) / ret60_hist.std())
    
        # 综合信号
        signals = []
        if z >= 2: signals.append('超买')
        elif z <= -2: signals.append('超卖')
        if rsi >= 70: signals.append('RSI超买')
        elif rsi <= 30: signals.append('RSI超卖')
        if z60 >= 2: signals.append('60日过高')
        elif z60 <= -2: signals.append('60日过低')
        sig = '/'.join(signals) if signals else '中性'
    
        print(f'{name:<8}{px.iloc[-1]:>8.0f}{pctile:>9.0%}{dev:>+10.1%}{z:>+8.2f}{rsi:>7.0f}{ret60:>+9.1%}{sig:>8}')

    # ── 第二部分：反转策略回测 ──
    print('\n===== 第二部分：反转策略回测（z-score + RSI + 60日收益 三信号）=====')
    print(f"{'品种':<8}{'交易次数':>8}{'胜率':>7}{'平均收益/次':>10}{'年化':>8}{'累计':>8}")

    all_trades = []
    for sym, name in SYMS.items():
        if sym not in closes or len(closes[sym]) < 500:
            continue
        px = closes[sym]
        ret = px.pct_change()
    
        # 信号计算
        ma200 = px.rolling(200).mean()
        std200 = px.rolling(200).std()
        z_sig = -(px - ma200) / std200  # 反转方向：高→做空(-1)，低→做多(+1)
        z_sig = z_sig.clip(-1, 1) * ((np.abs(z_sig) >= 2).astype(float))
    
        delta = px.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rsi = 100 * gain / (gain + loss)
        rsi_sig = np.where(rsi >= 70, -1, np.where(rsi <= 30, 1, 0))
        rsi_sig = pd.Series(rsi_sig, index=px.index)
    
        ret60_z = (px.pct_change(60) - px.pct_change(60).rolling(500).mean()) / px.pct_change(60).rolling(500).std()
        sig60 = np.where(ret60_z >= 2, -1, np.where(ret60_z <= -2, 1, 0))
        sig60 = pd.Series(sig60, index=px.index)
    
        # 三信号等权
        signal = ((z_sig + rsi_sig + sig60) / 3).shift(1)  # T→T+1
    
        # 收益
        daily = signal * ret
        # 成本
        cost = signal.diff().abs() * COST
        net = (daily - cost).dropna()
    
        if len(net) < 100:
            continue
    
        yrs = len(net) / 252
        ann = (1 + net).prod() ** (1/yrs) - 1
        cum = (1 + net).prod() - 1
    
        # 交易统计（信号非零的连续段为一笔交易）
        active = signal[signal != 0]
        if len(active) > 0:
            # 简化：按月统计胜率
            monthly = net.resample('ME').sum()
            active_months = monthly[monthly != 0]
            wr = (active_months > 0).mean() if len(active_months) > 0 else 0
            avg = active_months.mean() if len(active_months) > 0 else 0
        else:
            wr, avg = 0, 0
    
        print(f'{name:<8}{len(active_months):>8}{wr:>7.0%}{avg:>+10.2%}{ann:>+8.1%}{cum:>+8.1%}')
        all_trades.append({'sym': name, 'net': net})

    from research.factor_screen2 import register
    register('Z5b 反转策略+位置诊断: 见z5证据', label='z5b_reversal')
    print(f'\n证据: {DEST}')


if __name__ == '__main__':
    main()
