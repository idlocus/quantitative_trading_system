#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""试炼 015：波动率目标风险层（011 C口径净值上的叠加近似，预注册见日志）"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import numpy as np
import pandas as pd

eq = pd.read_csv('ai-created/trial011_equity_c.csv', index_col=0, parse_dates=True).iloc[:, 0]
eq = eq.sort_index()
ret = eq.pct_change().dropna()

ERA1_END = '2021-12-31'
C_FULL, C_SHARPE, C_MDD = 1.118, 0.51, -0.310

def ann(x):
    return float((1 + x).prod() ** (252 / len(x)) - 1) if len(x) else np.nan

def sharpe(x):
    return float(x.mean() / x.std() * np.sqrt(252)) if x.std() > 0 else np.nan

def mdd(x):
    c = (1 + x).cumprod()
    return float((c / c.cummax() - 1).min())

def era(x, a, b):
    return x[(x.index >= a) & (x.index <= b)]

print(f'基准C曲线: {eq.index[0].date()} ~ {eq.index[-1].date()} | 年化 {ann(ret):+.2%} 夏普 {sharpe(ret):.2f} 回撤 {mdd(ret):.1%}')

for win in (60, 40, 120):   # 修正（评审#2）：60日为注册主判据，40/120为敏感性
    rv = ret.rolling(win).std() * np.sqrt(252)
    # 修正（评审#2）：目标波动率=扩展窗均值（仅用过去数据，无未来信息）
    target_t = rv.expanding().mean()
    expo = (target_t / rv).clip(upper=1.0)          # 敞口≤1（无杠杆，参数声明）
    expo_m = expo.resample('ME').last().shift(1).reindex(ret.index).ffill().fillna(1.0)
    vt = ret * expo_m
    # 成本：月度敞口变化双边0.25%
    de = expo_m.resample('ME').last().diff().abs().dropna()
    cost = 0.0025 * de
    vt = vt - cost.reindex(vt.index).fillna(0)
    # 对照（评审#2）：等平均敞口固定仓位（分离"择时价值"与"平均降敞口"）
    fixed_expo = float(expo_m.mean())
    ctrl = ret * fixed_expo
    e1, e2 = era(vt, '2017-01-01', ERA1_END), era(vt, '2022-01-01', '2099-01-01')
    tag = '主' if win == 60 else '敏'
    print(f'\n[{tag} 窗{win}日] 平均敞口 {fixed_expo:.2f} | 等敞口对照: 年化 {ann(ctrl):+.2%} 夏普 {sharpe(ctrl):.2f} 回撤 {mdd(ctrl):.1%}')
    print(f'  vol目标: 年化 {ann(vt):+.2%} 夏普 {sharpe(vt):.2f} 回撤 {mdd(vt):.1%}')
    print(f'  两期: {ann(e1):+.2%} / {ann(e2):+.2%}')
    ok = mdd(vt) > C_MDD and sharpe(vt) >= C_SHARPE and ann(e1) > 0 and ann(e2) > 0
    print(f'  判据(回撤收紧✓ 夏普≥0.51✓ 两期正✓): {"PASS" if ok else "FAIL"}')

from datetime import datetime
reg_p = 'results/data/trial_registry.json'
reg = json.load(open(reg_p, encoding='utf-8'))
reg['trials'].append({'label': '015-波动率目标风险层', 'ran_at': datetime.now().isoformat(timespec='seconds'),
                      'verdict': '见日志'})
reg['count'] = len(reg['trials'])
tmp = reg_p + '.tmp'
with open(tmp, 'w', encoding='utf-8') as f:
    json.dump(reg, f, ensure_ascii=False, indent=1)
os.replace(tmp, reg_p)
print(f'\n登记册: {reg["count"]} 条')
