#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Z10 concentration backtest v2 (execution-semantics corrected, registry 181).

The v1 backtest (registry 180) credited a newly-decided position with the full
close-to-close return of the execution day -- i.e. decisions executed at the
SIGNAL-day close, including an overnight gap the deployed rule (next-day open,
as instructed by the daily report) cannot capture. This script recomputes all
variants under the deployment-consistent next-open execution
(research.cta_signal.portfolio_daily) and prints the old close-execution
numbers alongside for transparency.

Variants (all reported): BASE (Z8 form, fixed 1/6 slots), 1/n_active,
TOP1 (deployed), TOP2, TOP3. Vol overlay identical (15%/60d/cap 2).
"""
import sys, os
if __name__ == '__main__':
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from pathlib import Path
import pickle

import numpy as np
import pandas as pd

from research import cta_signal as cs

DEST = Path('ai-created/research/z10_units_v1')

SYMS = ['AU0', 'CU0', 'RB0', 'I0', 'P0', 'HC0']
NAMES = {'AU0': 'Gold', 'CU0': 'Copper', 'RB0': 'Rebar',
         'I0': 'Iron', 'P0': 'Palm', 'HC0': 'HRC'}
SPLIT = '2020-12-31'


def build_frames(data):
    frames = {}
    for sym in SYMS:
        df = data[sym]
        frames[sym] = cs.signal_frame(df['收盘价'].astype(float),
                                      open_=df['开盘价'].astype(float))
    return frames


def levered_net(frames, kk, execution='next_open'):
    """敞口空间完整口径：未加杠杆收益序列 → 杠杆序列 → 按实际敞口记账。

    杠杆变化本身构成换手并按 |Δ敞口| 扣费；旧仓隔夜按昨日敞口、
    新仓日内按新敞口（见 cta_signal.portfolio_daily 的 leverage 分支）。
    """
    raw = cs.portfolio_daily(frames, k=kk, execution=execution, charge_cost=True)
    lev = cs.vol_target_leverage(raw)
    net = cs.portfolio_daily(frames, k=kk, execution=execution,
                             charge_cost=True, leverage=lev)
    return net.dropna(), raw


def report(net, label):
    print(f'  {label}')
    for name, r in [('Full', net), ('2015-20', net[net.index <= SPLIT]),
                    ('2021-26', net[net.index > SPLIT])]:
        r = r.dropna()
        if len(r) < 100:
            continue
        yrs = len(r) / 252
        ann = (1 + r).prod() ** (1 / yrs) - 1
        sh = r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else 0
        mdd = cs.max_drawdown_from_returns(r)   # 审计 2026-11-19 一-7：与验收共用实现（含初始本金）
        print(f'    {name:<10} ann{ann:>+7.1%} sh{sh:>5.2f} dd{mdd:>7.1%} '
              f'wr={(r > 0).mean():.0%}')


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    data = pickle.load(open('ai-created/research/z8_full_stop/data.pkl', 'rb'))
    frames = build_frames(data)
    print('=' * 76)
    print('主力连续序列代理回测：未提供逐日合约映射与换月成交价，真实换月收益未验证。')
    print('Z10 v4：敞口空间复算（杠杆变化计换手成本；次日开盘=部署口径）')
    print(f'部署变体（单一来源 cta_signal.DEPLOY_K={cs.DEPLOY_K}，'
          f'{cs.DEPLOY_LABEL}）')
    print('=' * 76)

    results = {}
    for label, kk in [
            ('BASE = Z8复现(固定1/6槽)', 0),
            ('等权摊活跃(1/n_active)', 6),
            ('TOP1 = 全给最强', 1),
            ('TOP2 = 最强两个', 2),
            ('TOP3 = 最强三个', 3)]:
        net_open, raw_open = levered_net(frames, kk, 'next_open')
        net_close, _ = levered_net(frames, kk, 'close')
        results[label] = net_open

        print(f'\n--- {label} ---')
        print('  [次日开盘执行 = 部署口径，含成本+波动率目标层(敞口空间)]')
        report(net_open, '净收益')
        print('  [信号日收盘执行 = 旧登记口径对照]')
        report(net_close, '净收益(对照)')

    pd.DataFrame(results).to_csv(DEST / 'daily_returns.csv')

    # criteria check on the DEPLOYMENT variant (single source: cta_signal.DEPLOY_K)
    deploy_label = {0: 'BASE = Z8复现(固定1/6槽)', 1: 'TOP1 = 全给最强',
                    2: 'TOP2 = 最强两个', 3: 'TOP3 = 最强三个',
                    6: '等权摊活跃(1/n_active)'}[cs.DEPLOY_K]
    base = results['BASE = Z8复现(固定1/6槽)']
    v = cs.criteria_verdict(results[deploy_label], base, SPLIT)
    from data.preparation import atomic_json
    atomic_json(DEST / 'verdict.json', dict(v, deploy_k=cs.DEPLOY_K,
                annual_return=float((1 + results[deploy_label]).prod() ** (252 / len(results[deploy_label])) - 1),
                instrument_basis='continuous_proxy', roll_execution_verified=False))
    print('\n' + '=' * 76)
    print(f'部署口径判据核对（{deploy_label}，k={cs.DEPLOY_K}，次日开盘执行，'
          f'敞口空间含杠杆换手成本）')
    print('=' * 76)
    print(f'  ① 夏普 {v["sharpe"]:.2f} ≥ BASE {v["sharpe_base"]:.2f}: '
          f'{"PASS" if v["c1_sharpe_beats_base"] else "FAIL"}')
    print(f'  ② 两期 {v["ann_p1"]:+.1%} / {v["ann_p2"]:+.1%} 同号为正: '
          f'{"PASS" if v["c2_both_periods_positive"] else "FAIL"}')
    print(f'  ③ maxDD幅度 {abs(v["max_dd"]):.1%} ≤ 25%: '
          f'{"PASS" if v["c3_dd_within_cap"] else "FAIL"}')
    print(f'  → 总判定: {"PASS" if v["pass_all"] else "FAIL"}')
    print(f'\nEvidence: {DEST}')


if __name__ == '__main__':
    main()
