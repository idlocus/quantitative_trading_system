#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Z4：全品种CTA + TSMOM + 回撤控制（预注册见 CTA_PREREG Z4 节）。"""


def main():
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

    from pathlib import Path
    import numpy as np
    import pandas as pd
    import pickle

    DEST = Path('ai-created/research/z4_cta_full_2026-11-16')
    DEST.mkdir(parents=True, exist_ok=True)

    COST = 0.00025
    TARGET_VOL = 0.20
    LOOKBACK_TSMOM = 252

    import akshare as ak
    import time

    # ── 拉全品种数据（缓存）──
    cache_file = DEST / 'all_closes.pkl'
    if cache_file.exists():
        closes = pickle.load(open(cache_file, 'rb'))
        print(f'缓存: {len(closes)} 品种', flush=True)
    else:
        ALL_SYMS = ['RB0','CU0','AU0','AG0','I0','M0','TA0','CF0','FG0','SC0',
                    'A0','C0','P0','Y0','OI0','RM0','J0','JM0','ZN0','AL0',
                    'NI0','PB0','SN0','V0','PP0','EG0','SR0','AP0','HC0','FU0',
                    'BU0','RU0','SP0']
        closes = {}
        for i, sym in enumerate(ALL_SYMS):
            try:
                df = ak.futures_main_sina(symbol=sym, start_date='20150101', end_date='20261112')
                df['日期'] = pd.to_datetime(df['日期'])
                closes[sym] = df.set_index('日期')['收盘价'].astype(float).sort_index()
                if (i+1) % 10 == 0:
                    print(f'  {i+1}/{len(ALL_SYMS)}', flush=True)
                time.sleep(0.3)  # 限速
            except Exception as e:
                print(f'  {sym} 失败: {str(e)[:30]}', flush=True)
        pickle.dump(closes, open(cache_file, 'wb'))
        print(f'拉取完成: {len(closes)} 品种', flush=True)

    close_mx = pd.DataFrame(closes).sort_index()
    ret_mx = close_mx.pct_change()
    n_syms = close_mx.shape[1]
    print(f'面板: {close_mx.shape[0]} 天 × {n_syms} 品种', flush=True)

    def run_cta(signal_func, vol_target, dd_control=False, label=''):
        """通用CTA回测框架"""
        # 信号
        signal = signal_func(close_mx).shift(1)  # T→T+1
    
        # 仓位
        vol = ret_mx.rolling(60).std() * np.sqrt(252)
        inv_vol = 1 / vol
        w_raw = inv_vol.div(inv_vol.sum(axis=1), axis=0)
        port_vol = (w_raw * vol * signal.abs()).sum(axis=1)
        lev = (vol_target / port_vol).clip(upper=3.0)
        weights = w_raw.mul(lev, axis=0) * signal
    
        # 回测
        position = weights.shift(1)
        gross = (position * ret_mx).sum(axis=1)
        cost = weights.diff().abs().sum(axis=1) * COST
        net = gross - cost.shift(1)
        valid = net.dropna()
        nav = (1 + valid).cumprod()
    
        # 回撤控制
        if dd_control:
            dd = nav / nav.cummax() - 1
            scale = pd.Series(1.0, index=nav.index)
            scale[dd < -0.15] = 0.5
            scale[dd < -0.20] = 0.0
            # 应用缩放（简化：直接缩放日收益）
            net_adj = net * scale.shift(1).reindex(net.index).fillna(1.0)
            valid = net_adj.dropna()
            nav = (1 + valid).cumprod()
    
        return valid, nav

    def ma_signal(close_mx):
        """多周期均线信号"""
        sigs = []
        for f, s in [(10,30),(20,60),(50,200)]:
            ma_f = close_mx.rolling(f).mean()
            ma_s = close_mx.rolling(s).mean()
            sigs.append((ma_f > ma_s).astype(float) - (ma_f < ma_s).astype(float))
        return sum(sigs) / len(sigs)

    def tsmom_signal(close_mx):
        """时间序列动量：过去252日收益>0做多，<0做空"""
        mom = close_mx / close_mx.shift(LOOKBACK_TSMOM) - 1
        return np.sign(mom)

    def report(valid, nav, label):
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
            wr = (r>0).mean()
            print(f'  {name}: 年化{ann:>+7.1%} 夏普{sh:>5.2f} maxDD{mdd:>7.1%} 胜率{wr:.0%}')

    # Z4a: 全品种+多均线
    v_a, n_a = run_cta(ma_signal, TARGET_VOL, dd_control=False, label='Z4a')
    report(v_a, n_a, f'Z4a: {n_syms}品种 多周期均线 vol{TARGET_VOL:.0%}')

    # Z4b: 全品种+TSMOM
    v_b, n_b = run_cta(tsmom_signal, TARGET_VOL, dd_control=False, label='Z4b')
    report(v_b, n_b, f'Z4b: {n_syms}品种 TSMOM(252日) vol{TARGET_VOL:.0%}')

    # Z4c: 全品种+TSMOM+DD控制
    v_c, n_c = run_cta(tsmom_signal, TARGET_VOL, dd_control=True, label='Z4c')
    report(v_c, n_c, f'Z4c: {n_syms}品种 TSMOM vol{TARGET_VOL:.0%} +DD控制')

    from research.factor_screen2 import register
    register('Z4 全品种CTA 3run: 见z4证据', label='z4_cta_full')
    print(f'\n证据: {DEST}')


if __name__ == '__main__':
    main()
