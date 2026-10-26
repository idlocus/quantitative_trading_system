#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
试炼 009 两融余额择时 + 012 资产层趋势（共享择时框架，各自独立判据）
预注册见 docs/研究日志/009 与 012（012 内容源自资产层趋势原始登记，升池执行）

009：融资余额（两所合计）20日净变化>0 → 次月持有300，否则现金；月判定
     判据：两期净超额vs持有300均>0；回撤<持有的70%；窗10/20/40三档同号
012：510300/510500/512100 收盘>自身MA_N → 持有（在场者等权），否则现金；
     月判定（MA200主判据，160/240敏感性）
     判据：两期年化>0；夏普>买入持有等权标的池；回撤<持有的60%；三档同号
成本：单边 0.25%×切换；区间：持有月 2017-01 ~ 2026-10
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import numpy as np
import pandas as pd

from data.layered import LayeredDataSource

COST = 0.0025
HOLD_START, HOLD_END = pd.Period('2017-01', 'M'), pd.Period('2026-10', 'M')
ERA1_END = pd.Period('2021-12', 'M')

# ── 数据 ──
ds = LayeredDataSource()
cur = ds._db_src()._cursor()
cur.execute("SELECT TRADE_DT, SUM(S_MARSUM_TRADINGBALANCE) FROM ASHAREMARGINTRADESUM "
            "WHERE TRADE_DT >= '20160601' GROUP BY TRADE_DT ORDER BY TRADE_DT")
mg = pd.DataFrame(cur.fetchall(), columns=['date', 'bal'])
mg['date'] = pd.to_datetime(mg['date'], format='%Y%m%d')
mg = mg.set_index('date')['bal'].astype(float)
print(f'融资余额: {len(mg)} 日 | {mg.index.min().date()} ~ {mg.index.max().date()}', flush=True)

idx = pd.read_parquet('data_cache/kline/000300_SH.parquet')
idx['date'] = pd.to_datetime(idx['date'])
px300 = idx.set_index('date')['close']
px300 = px300[(px300.index >= '2016-06-01') & (px300.index <= '2026-10-23')]

# ETF 池（012）
etfs = {}
for c in ('510300', '510500', '512100'):
    df = pd.read_parquet(f'data_cache/kline/{c}_SH.parquet')
    df['date'] = pd.to_datetime(df['date'])
    s = df.set_index('date')['close'].astype(float)
    etfs[c] = s[(s.index >= '2016-06-01') & (s.index <= '2026-10-23')]
etf_px = pd.DataFrame(etfs).sort_index()

cal = px300.index
months = sorted({d.to_period('M') for d in cal})
me300 = px300.groupby(px300.index.to_period('M')).last()
meE = etf_px.groupby(etf_px.index.to_period('M')).last()
mg_m = mg.groupby(mg.index.to_period('M')).last()

def timing_eval(name, monthly_ret, bench_ret):
    """monthly_ret/bench_ret: 已含成本的月度净收益与对照（Series，月度索引）"""
    r = pd.Series(monthly_ret)
    b = pd.Series(bench_ret)

    def ann(x):
        return float((1 + x).prod() ** (12 / len(x)) - 1) if len(x) else np.nan

    def seg(x, a, z):
        return x[(x.index >= a) & (x.index <= z)]

    def mdd(x):
        c = (1 + x).cumprod()
        return float((c / c.cummax() - 1).min())

    e = {'full': ann(r), 'bench': ann(b), 'mdd': mdd(r), 'bench_mdd': mdd(b)}
    for era, a, z in (('e1', HOLD_START, ERA1_END), ('e2', pd.Period('2022-01', 'M'), r.index[-1])):
        e[f'{era}'] = ann(seg(r, a, z))
        e[f'{era}_excess'] = ann(seg(r, a, z)) - ann(seg(b, a, z))
    e['sharpe'] = (r.mean() / r.std() * np.sqrt(12)) if r.std() > 0 else 0
    e['bench_sharpe'] = (b.mean() / b.std() * np.sqrt(12)) if b.std() > 0 else 0
    print(f"  [{name}] 全期 {e['full']:+.2%}(夏普{e['sharpe']:.2f}) vs 持有 {e['bench']:+.2%}({e['bench_sharpe']:.2f}) "
          f"| 回撤 {e['mdd']:.1%} vs {e['bench_mdd']:.1%} | e1超额 {e['e1_excess']:+.2%} e2超额 {e['e2_excess']:+.2%}")
    return e

def run_monthly(signals, asset_me, cost_sym='300'):
    """signals: dict month->bool（该月是否在场，由上月末信号决定）"""
    rets, benchs, poss = [], [], []
    prev_in = None
    for m in months:
        if not (HOLD_START <= m <= HOLD_END):
            continue
        i_m = asset_me.index.get_loc(m)
        nxt_i = i_m + 1
        if nxt_i >= len(asset_me):
            break
        r_next = asset_me.iloc[nxt_i] / asset_me.iloc[i_m] - 1
        in_mkt = signals.get(m, True)
        cost = 0 if prev_in is None or prev_in == in_mkt else COST
        rets.append(float(r_next) * (1 if in_mkt else 0) - cost)
        benchs.append(float(r_next))
        poss.append(m)
        prev_in = in_mkt
    return pd.Series(rets, index=poss), pd.Series(benchs, index=poss)

print('\n== 009 两融余额择时（信号=余额W日变化>0）==', flush=True)
r9 = {}
for w in (10, 20, 40):
    chg = mg_m / mg_m.shift(1) - 1          # 月度采样近似W日（W/21≈月数）
    # 精确：用日频算W日变化再取月末
    chg_d = mg / mg.shift(w) - 1
    chg_m = chg_d.groupby(chg_d.index.to_period('M')).last()
    sig = {}
    for m in months:
        if m in chg_m.index and HOLD_START <= m <= HOLD_END:
            sig[m] = bool(chg_m.loc[m] > 0)
    r, b = run_monthly(sig, me300)
    r9[w] = timing_eval(f'{w}日', r, b)

print('\n== 012 资产层趋势（510300/510500/512100，MA_N，在场者等权）==', flush=True)
r12 = {}
for n in (160, 200, 240):
    ma = etf_px.rolling(n).mean()
    above = (etf_px > ma)
    above_m = above.groupby(above.index.to_period('M')).last()
    # 月度策略收益：在场标的等权（次月），全现金则0
    rets, benchs, idxs = [], [], []
    prev_set = None
    for m in months:
        if not (HOLD_START <= m <= HOLD_END):
            continue
        i_m = meE.index.get_loc(m)
        nxt_i = i_m + 1
        if nxt_i >= len(meE):
            break
        cur_set = above_m.loc[m] if m in above_m.index else None
        if cur_set is None:
            continue
        sel = [c for c in meE.columns if cur_set.get(c, False)]
        r_next_sel = meE.iloc[nxt_i][sel] / meE.iloc[i_m][sel] - 1 if sel else None
        r_bench = float((meE.iloc[nxt_i] / meE.iloc[i_m] - 1).mean())
        g = float(r_next_sel.mean()) if sel else 0.0
        cur_set_keys = set(sel)
        traded = 0.0 if prev_set is None else len(cur_set_keys ^ prev_set) / max(len(meE.columns), 1) / 2
        cost = COST * traded * 2 + (COST * len(sel) / len(meE.columns) if prev_set is None else 0)
        rets.append(g - cost); benchs.append(r_bench); idxs.append(m)
        prev_set = cur_set_keys
    r12[n] = timing_eval(f'MA{n}', pd.Series(rets, index=idxs), pd.Series(benchs, index=idxs))

# 判据
ok9 = all(r9[w]['e1_excess'] > 0 and r9[w]['e2_excess'] > 0 for w in (10, 20, 40)) \
      and r9[20]['mdd'] > 0.7 * r9[20]['bench_mdd']
ok12 = all(r12[n]['e1'] > 0 and r12[n]['e2'] > 0 for n in (160, 200, 240)) \
       and r12[200]['sharpe'] > r12[200]['bench_sharpe'] \
       and r12[200]['mdd'] > 0.6 * r12[200]['bench_mdd']
print(f'\n判据 009（三档同号+回撤改善）: {"PASS" if ok9 else "FAIL"}')
print(f'判据 012（三档两期为正+夏普胜持有+回撤<60%持有）: {"PASS" if ok12 else "FAIL"}')

from datetime import datetime
reg_p = 'results/data/trial_registry.json'
reg = json.load(open(reg_p, encoding='utf-8'))
reg['trials'].append({
    'label': '009-两融余额择时', 'ran_at': datetime.now().isoformat(timespec='seconds'),
    'e1_excess': r9[20]['e1_excess'], 'e2_excess': r9[20]['e2_excess'],
    'verdict': 'PASS' if ok9 else 'FAIL'})
reg['trials'].append({
    'label': '012-资产层趋势(MA200)', 'ran_at': datetime.now().isoformat(timespec='seconds'),
    'full': r12[200]['full'], 'e1': r12[200]['e1'], 'e2': r12[200]['e2'],
    'sharpe': r12[200]['sharpe'], 'bench_sharpe': r12[200]['bench_sharpe'],
    'verdict': 'PASS' if ok12 else 'FAIL'})
reg['count'] = len(reg['trials'])
tmp = reg_p + '.tmp'
with open(tmp, 'w', encoding='utf-8') as f:
    json.dump(reg, f, ensure_ascii=False, indent=1)
os.replace(tmp, reg_p)
print(f'\n登记册: {reg["count"]} 条')

