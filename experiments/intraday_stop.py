#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
盘中止损 Stage 1：边界研究（预注册见 docs/events研究方案.md 2026-10-19 节）

1. 拉取十年持有股票的 30 分钟线（JQ，2年大块，data_cache/jq30/）
2. 从 trade_log 重建平均成本 → 对每个(股票,持有日)判定：
   a组（10:00破且收盘仍破→现行也会止损）: 早出收益 = 收盘−10:00价
   b组（10:00破但收盘收回→鞭打）: 成本 = 10:00价−收盘
3. 净边界 = Σa − Σb（按股数加权为组合层面的近似）
判据：>+2pp/十年 → Stage 2；<+1pp → 关闭。
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

ART = 'results/data/portfolio_composite_v5_full_2017_2026_5因子生.json'
with open(ART, encoding='utf-8') as f:
    art = json.load(f)
held_days = {}      # sym -> set(dates)
for day, rows in art['daily_holdings'].items():
    for s, _v in rows:
        held_days.setdefault(s, set()).add(day)
syms = sorted(held_days)
print(f"持有股票 {len(syms)} 只 | 持有股票日 {sum(len(v) for v in held_days.values())}", flush=True)

# ── 数据拉取（30分钟线，2年大块；独立缓存目录） ──
CACHE = 'data_cache/jq30'
os.makedirs(CACHE, exist_ok=True)

from dotenv import load_dotenv
load_dotenv()
from jqdatasdk import auth, get_price
auth(os.getenv('JQ_USER'), os.getenv('JQ_PASSWORD'))

def to_jq(wind_sym):
    return wind_sym.replace('.SH', '.XSHG').replace('.SZ', '.XSHE')

PULL_START, PULL_END = '2016-12-01', '2026-10-15'
fetched = skipped = 0
for i, ws in enumerate(syms):
    js = to_jq(ws)
    path = f'{CACHE}/{ws.replace(".", "_")}_30m.parquet'
    if os.path.exists(path):
        continue
    frames = []
    cur = pd.Timestamp(PULL_START)
    end = pd.Timestamp(PULL_END)
    ok = True
    while cur <= end:
        seg_end = min(cur + pd.offsets.YearEnd(2), end)
        try:
            df = get_price(js, start_date=str(cur.date()), end_date=str(seg_end.date()),
                           frequency='30m', fields=['close'], skip_paused=False, fq=None)
        except Exception as e:
            print(f"  {ws} 拉取异常: {e}", flush=True)
            ok = False
            break
        if df is not None and len(df):
            frames.append(df)
        cur = seg_end + pd.offsets.Day(1)
    if not ok or not frames:
        skipped += 1
        continue
    out = pd.concat(frames)
    out.to_parquet(path)
    fetched += 1
    if (i + 1) % 50 == 0:
        print(f"  进度 {i+1}/{len(syms)}（新拉 {fetched}）", flush=True)
print(f"数据完成: 新拉 {fetched} | 跳过 {skipped}", flush=True)

# ── 成本重建（平均成本法，与账本同口径） ──
tl = art['trade_log']
cost, qty = {}, {}
cost_hist = []       # (date, sym, cost, qty)
for t in tl:         # [date, symbol, side, price, qty]
    d, s, side, px, q = t[0], t[1], t[2], float(t[3]), float(t[4])
    if side == 'BUY':
        c0, q0 = cost.get(s, 0.0), qty.get(s, 0.0)
        cost[s] = (c0 * q0 + px * q) / (q0 + q) if (q0 + q) > 0 else px
        qty[s] = q0 + q
    else:
        qty[s] = qty.get(s, 0.0) - q
        if qty[s] <= 1e-9:
            qty.pop(s, None); cost.pop(s, None)
    cost_hist.append((d, s, cost.get(s), qty.get(s, 0.0)))
# 每日收盘后的成本状态
import collections
cost_by_date = {}
for d, s, c, q in cost_hist:
    cost_by_date.setdefault(d, {})[s] = c

# ── 逐(股票,持有日)分类 ──
res_a, res_b = [], []
n_no_data = n_breach = 0
for ws, days in held_days.items():
    path = f'{CACHE}/{ws.replace(".", "_")}_30m.parquet'
    if not os.path.exists(path):
        n_no_data += len(days)
        continue
    bars = pd.read_parquet(path)
    if not len(bars):
        n_no_data += len(days); continue
    bars['d'] = bars.index.date
    # 30m bar 的收盘时点：10:00 bar=9:30-10:00；14:00 bar=13:30-14:00；
    # 日收盘=最后一根（14:30-15:00）的 close
    bars['hm'] = bars.index.strftime('%H:%M')
    p1000 = bars[bars.hm == '10:00'].set_index('d')['close']
    p1400 = bars[bars.hm == '14:00'].set_index('d')['close']
    pclose = bars.groupby('d')['close'].last()
    day_sorted = sorted(days)
    for day in day_sorted:
        dstr = str(day)
        # 当日前的成本（前一日状态，缺则再往前）
        c = None
        for prev in sorted(cost_by_date):
            if prev >= dstr:
                break
            c = cost_by_date[prev].get(ws, c)
        if not c:
            continue
        if day not in pclose.index or day not in p1000.index:
            continue
        c1000, cclose = float(p1000.loc[day]), float(pclose.loc[day])
        if pd.isna(c1000) or pd.isna(cclose):
            continue
        thr = c * 0.95
        if c1000 < thr:
            n_breach += 1
            if cclose < thr:      # a组：收盘仍破——现行也会在14:45止损
                res_a.append((ws, day, cclose - c1000))
            else:                  # b组：收盘收回——鞭打
                res_b.append((ws, day, c1000 - cclose))

a_total = sum(x[2] for x in res_a)
b_total = sum(x[2] for x in res_b)
# 组合层面近似：每事件影响 ≈ 差价/NAV（每股差价×股数未知→用每股差价的
# 相对值×单票权重1%近似：Σ(diff/price)×1%）
a_rel = sum(x[2] / max(abs(x[2]) + 1e-9, 1) for x in res_a)  # 占位，下方用正确口径
a_pct = sum(x[2] for x in res_a)
print(f"""
{'='*70}
Stage 1 边界结果（10:00 击穿 cost×0.95 的持有股票日）
{'='*70}
  击穿事件: {n_breach}（无数据跳过 {n_no_data} 持有日）
  a组（收盘仍破，早出有收益）: {len(res_a)} 次 | 每股差价合计 {a_pct:+,.0f} 元
  b组（收盘收回，鞭打成本）: {len(res_b)} 次 | 每股差价合计 {b_total:+,.0f} 元
  a组占比: {len(res_a)/max(n_breach,1):.0%} | b组占比: {len(res_b)/max(n_breach,1):.0%}""")

# 正确的组合层面口径：每股差价/当日该股价格 = 单票收益差；组合权重≈1%
per_event_ret = []
for ws, day, diff in res_a + [(w, d, -x) for w, d, x in res_b]:
    path = f'{CACHE}/{ws.replace(".", "_")}_30m.parquet'
    pc = pd.read_parquet(path)
    pc['d'] = pc.index.date
    cl = pc.groupby('d')['close'].last()
    if day in cl.index and cl.loc[day] and cl.loc[day] > 0:
        per_event_ret.append(diff / cl.loc[day] * 0.01)   # 1% 等权近似
net_pp = sum(per_event_ret) * 100
print(f"  组合净边界（1%等权近似）: {net_pp:+.2f}pp/十年")
v = ('STAGE2' if net_pp > 2.0 else ('CLOSE' if net_pp < 1.0 else 'GRAY'))
print(f"""
  判定（预注册：>+2pp→Stage2 | <+1pp→关闭）: {v}
  → {'值得做引擎改造全模拟' if v=='STAGE2' else
     ('关闭——盘中止损净边界不足' if v=='CLOSE' else '灰区——按预注册细看2018/2024子样本')}""")
pd.DataFrame(res_a, columns=['sym', 'day', 'a_diff']).to_csv(
    'ai-created/intraday_stop_group_a.csv', index=False)
pd.DataFrame(res_b, columns=['sym', 'day', 'b_cost']).to_csv(
    'ai-created/intraday_stop_group_b.csv', index=False)
