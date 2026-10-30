#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""试炼 017-P2：情绪周期 × 涨板事件溢价（预注册见 docs/研究日志/017_实战策略族.md）

主判据指标 E4=昨日涨停今日表现（赚钱效应）；E1/E2/E3/E5 辅助发现性报告。
统计纪律：组间显著性按日聚合（同日事件横截面相关，按事件检验虚高）。
自足计算（与 P1 同宇宙同逻辑重算事件），不依赖 P1 的 CSV 产物。
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import numpy as np
import pandas as pd
from scipy import stats

from trading import rules

START, END = '2015-01-01', '2026-10-23'
ERA_SPLIT = '2021-01-01'
COST = 0.0015   # 单边（与 P1 同口径）

# ---------- 加载日线（宇宙与 P1 一致：沪深主板+创业板，剔除上市<250日） ----------
cache = {}
for fn in os.listdir('data_cache/kline'):
    if not fn.endswith('.parquet'):
        continue
    code, suf = fn[:-8].split('_')
    if (suf == 'SH' and code[0] == '6') or (suf == 'SZ' and code[0] in '03'):
        sym = code + '.' + suf
        try:
            df = pd.read_parquet(f'data_cache/kline/{fn}')
            if {'open', 'high', 'low', 'close', 'date'} <= set(df.columns) and len(df):
                df['date'] = pd.to_datetime(df['date'])
                df = df[(df['date'] >= START) & (df['date'] <= END)]
                if len(df) > 250:
                    cache[sym] = df.reset_index(drop=True)
        except Exception:
            continue
print(f'股票: {len(cache)}', flush=True)

idx = pd.read_parquet('data_cache/kline/000300_SH.parquet')
idx['date'] = pd.to_datetime(idx['date'])
c300 = idx.set_index('date')['close']
c300 = c300[(c300.index >= START) & (c300.index <= END)]
cal = c300.index
calpos = {d: i for i, d in enumerate(cal)}

# ---------- 逐日情绪指标 + 事件（单遍扫描） ----------
sent_rows = []    # (date, n_zt, max_streak, n_touched, n_broken)
e4_rows = []      # (T, ret) —— T-1 涨停股在 T 的收益（E4 原料）
events = []       # P1 同逻辑事件
for sym, df in cache.items():
    o, h, c = df['open'].values, df['high'].values, df['close'].values
    dates = df['date'].values
    n = len(df)
    if n < 260:
        continue
    ratio = rules.price_limit_ratio(sym)
    limit_up = ratio - 0.003
    prev = np.roll(c, 1); prev[0] = np.nan
    chg = c / prev - 1
    is_zt = chg >= limit_up                      # 收盘封板
    touched = (h / prev - 1) >= limit_up         # 盘中触及（含炸板）
    streak = 0
    streaks = np.zeros(n, dtype=int)
    for i in range(n):
        streak = streak + 1 if is_zt[i] else 0
        streaks[i] = streak
    d_ts = [pd.Timestamp(x) for x in dates]
    for i in range(1, n):
        if is_zt[i - 1] and i < n:               # E4 原料：昨日涨停今日收益
            e4_rows.append((d_ts[i], c[i] / c[i - 1] - 1))
    for i in range(n):
        if touched[i]:
            sent_rows.append((d_ts[i], int(is_zt[i]), int(streaks[i])))
    for i in range(1, n - 6):
        if not is_zt[i] or streaks[i] > 5:
            continue
        t1 = i + 1
        chg_o = o[t1] / c[i] - 1
        if chg_o >= ratio - 0.005:               # T+1 开盘涨停不可买
            continue
        r1 = c[t1] / o[t1] - 1
        r5 = c[min(i + 5, n - 1)] / o[t1] - 1    # 注：跨日含除权噪声（与P1同口径）
        d = d_ts[i]
        events.append({'sym': sym, 'date': d, 'board': min(int(streaks[i]), 3),
                       'era': '2015-2020' if d < pd.Timestamp(ERA_SPLIT) else '2021-2026',
                       'r1': r1, 'r5': r5,
                       'amt': float(df['amount'].iloc[t1]) if 'amount' in df else np.nan})

ev = pd.DataFrame(events)
print(f'事件: {len(ev)}（已剔 T+1 开盘涨停不可买）', flush=True)

# 日级情绪表
sdf = pd.DataFrame(sent_rows, columns=['date', 'sealed', 'streak'])
g = sdf.groupby('date')
daily = pd.DataFrame({'n_zt': g['sealed'].sum(), 'n_touched': g.size(),
                      'max_streak': g['streak'].max()})
daily['break_rate'] = 1 - daily['n_zt'] / daily['n_touched']
e4s = pd.DataFrame(e4_rows, columns=['date', 'ret']).groupby('date')['ret'].mean()
daily['E4'] = e4s
daily['E1'] = daily['n_zt']
daily['E2'] = daily['max_streak']
daily['E3'] = daily['break_rate']
daily['E5'] = daily['E1'] / daily['E1'].shift(5) - 1
daily = daily.reindex(cal).ffill(limit=5)
print(f'情绪日表: {len(daily)} 日（E4覆盖 {daily.E4.notna().mean():.0%}）', flush=True)

# 事件挂情绪（T 日值）
ev = ev.merge(daily[['E4', 'E1', 'E2', 'E3', 'E5']], left_on='date',
              right_index=True, how='left')
ev = ev.dropna(subset=['E4'])
ev['q3'] = pd.qcut(ev['E4'], 3, labels=['低', '中', '高'])
ev.to_csv('ai-created/trial017p2_events.csv', index=False, encoding='utf-8-sig')

# ---------- 报告 ----------
def grp(df, tag, col='r1'):
    x = df[col]
    print(f'  {tag:<18} n={len(df):>6} | 均值 {x.mean():+.2%} 中位 {x.median():+.2%} '
          f'| 净 {x.mean()-COST:+.2%} | P25 {x.quantile(0.25):+.1%} '
          f'P75 {x.quantile(0.75):+.1%}', flush=True)

def daily_ttest(ev_a, ev_b, col='r1'):
    """日级聚合 Welch t：组内事件按日均值→日序列，组间检验（预注册统计纪律）"""
    da = ev_a.groupby('date')[col].mean()
    db = ev_b.groupby('date')[col].mean()
    t, p = stats.ttest_ind(da, db, equal_var=False)
    return len(da), len(db), t, p

print('\n== 主判据：E4 三分位 × 首板 r1（净=扣单边0.15%）==')
first = ev[ev.board == 1]
lo, mid, hi = first[first.q3 == '低'], first[first.q3 == '中'], first[first.q3 == '高']
for df, tag in ((lo, '低情绪'), (mid, '中情绪'), (hi, '高情绪')):
    grp(df, tag)
na, nb, t, p = daily_ttest(hi, lo)
print(f'  高vs低（日级聚合 n={na}/{nb} 日）: 差 {hi.r1.mean()-lo.r1.mean():+.2%} '
      f'| Welch t={t:.2f} p={p:.4f}', flush=True)

print('\n== E4 × 板档（r1 / r5）==')
for b in (1, 2, 3):
    for q in ('低', '中', '高'):
        grp(ev[(ev.board == b) & (ev.q3 == q)], f'{b}板·{q}')
print('  -- r5 --')
for b in (1, 2, 3):
    for q in ('低', '中', '高'):
        grp(ev[(ev.board == b) & (ev.q3 == q)], f'{b}板·{q}', col='r5')

print('\n== 辅助发现性指标三分位 × 首板 r1（不进入主判定）==')
for m in ('E1', 'E2', 'E3', 'E5'):
    ev[f'{m}_q'] = pd.qcut(ev[m], 3, labels=['低', '中', '高'])
    f1 = ev[ev.board == 1]
    for q in ('低', '中', '高'):
        grp(f1[f1[f'{m}_q'] == q], f'{m}·{q}')
    del ev[f'{m}_q']

print('\n== 稳健性（降级参考）：高情绪组两年代同号? ==')
for era in ('2015-2020', '2021-2026'):
    grp(hi[hi.era == era], f'高情绪·首板·{era}')

print('\n== 实战口径：年化金额（净均值×年频率，每万元满仓滚动理论上限）==')
for q, df in (('低', lo), ('中', mid), ('高', hi)):
    net = df.r1.mean() - COST
    freq = len(df) / 10.6
    print(f'  {q}情绪: {net:+.2%} × {freq:.0f} 次/年 = {net*freq:+.0%}')
amt = hi['amt']
print(f'  高组事件日成交额: 中位 {amt.median()/1e8:.1f}亿 P25 {amt.quantile(0.25)/1e8:.1f}亿（容量参考）')

# ---------- 主判据判定（预注册三条件） ----------
print('\n== P2 主判据判定 ==')
c1 = (hi.r1.mean() - lo.r1.mean()) > 0 and p < 0.05
c2 = (hi.r1.mean() - COST) > 0
c3 = (hi.r1.mean() - COST) * (len(hi) / 10.6) > 0
print(f'  1) 高vs低差>0且p<0.05: {"PASS" if c1 else "FAIL"}（差{hi.r1.mean()-lo.r1.mean():+.2%}, p={p:.4f}）')
print(f'  2) 高组净溢价>0:      {"PASS" if c2 else "FAIL"}（{hi.r1.mean()-COST:+.2%}）')
print(f'  3) 年化金额>0:        {"PASS" if c3 else "FAIL"}')
print(f'  ==> P2（E4情绪周期×首板）: {"PASS" if (c1 and c2 and c3) else "FAIL"}')

# ---------- 登记 ----------
from datetime import datetime
reg_p = 'results/data/trial_registry.json'
reg = json.load(open(reg_p, encoding='utf-8'))
reg['trials'].append({'label': '017-P2情绪周期×涨板溢价', 'ran_at': datetime.now().isoformat(timespec='seconds'),
                      'n_events': int(len(ev)), 'verdict': 'PASS' if (c1 and c2 and c3) else 'FAIL'})
reg['count'] = len(reg['trials'])
tmp = reg_p + '.tmp'
with open(tmp, 'w', encoding='utf-8') as f:
    json.dump(reg, f, ensure_ascii=False, indent=1)
os.replace(tmp, reg_p)
print(f'\n登记册: {reg["count"]} 条 | 明细: ai-created/trial017p2_events.csv')
