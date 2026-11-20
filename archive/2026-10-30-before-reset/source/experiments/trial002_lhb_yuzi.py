#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
试炼 002：游资龙虎榜——2a 跟随 vs 2b 反指（对立双假设）
预注册见 docs/研究日志/002_游资龙虎榜.md

事件定义（预注册，本脚本只实现不更改）：
  - 龙虎榜个股，营业部席位（非机构专用）合计净买入 > 0 且 ≥ 当日成交额×5%
  - 过滤：名称不含 ST/退；当日未涨停封死（涨跌幅 < 板块涨停比例-0.25pct）；
    仅沪深主板/创业板/科创板（北交所代码剔除）
  - 仅当日榜（剔除"连续/三日"类累计口径行，防 3 日榜金额混入当日过滤）
收益口径（预注册声明）：信号日收盘 → N 交易日后收盘，个股与行业同口径对称
  - 个股：后复权 close（cache adjclose）
  - 行业：PIT 行业等权日收益（月度快照，全缓存股票池成分）
  - 超额 = 个股 - 行业；全市场等权口径作诊断
判据：事件数≥100；5/20 日超额两期（2015-2020 / 2021-2026）同号；
  合并 Welch t≥2（20 日窗）。2a PASS=同号为正+t≥2；2b PASS=同号为负
  +t≤-2；否则双 FAIL。
诊断：截尾 1% 均值、按年分布、全市场等权口径。
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from datetime import date
import numpy as np
import pandas as pd

from data.layered import LayeredDataSource
from data.semantics import close_matrix
from data.industry_pit import industry_classification
from trading import rules

START_TS, END_TS = pd.Timestamp('2015-01-01'), pd.Timestamp('2026-10-15')
ERA_SPLIT = pd.Timestamp('2021-01-01')
WINDOWS = (5, 20)
MIN_NET_PCT = 0.05          # 营业部净买 ≥ 当日成交额×5%

# ── 1. 龙虎榜数据 ──
det = pd.read_parquet('data_cache/lhb/detail.parquet')
ins = pd.read_parquet('data_cache/lhb/inst.parquet')
print(f'原始: detail {len(det)} 行 | inst {len(ins)} 行', flush=True)

reason_col = next(c for c in det.columns if '解读' in c or '原因' in c)
multi = det[reason_col].astype(str).str.contains('连续|三日|三个交易日|3个交易日', na=False)
det = det[~multi]
det['上榜日'] = pd.to_datetime(det['上榜日'])
det = det.drop_duplicates(subset=['代码', '上榜日'], keep='first')

date_col_ins = next(c for c in ins.columns if '日' in c and '上榜' in c) if any(
    '上榜' in c for c in ins.columns) else next(c for c in ins.columns if '日' in c)
ins[date_col_ins] = pd.to_datetime(ins[date_col_ins])
ins = ins.drop_duplicates(subset=['代码', date_col_ins], keep='first')
ins = ins.rename(columns={date_col_ins: '上榜日'})

print(f'当日榜去重后: {len(det)} 行 | {det["上榜日"].min().date()} ~ {det["上榜日"].max().date()}', flush=True)

# ── 2. 事件定义 ──
m = det.merge(ins[['代码', '上榜日', '机构买入净额']], on=['代码', '上榜日'], how='left')
m['机构买入净额'] = m['机构买入净额'].fillna(0.0)
m['yys_net'] = m['龙虎榜净买额'] - m['机构买入净额']       # 营业部净买

def to_wind(code):
    code = str(code).zfill(6)
    if code[0] == '6':
        return code + '.SH'
    if code[0] in '03':
        return code + '.SZ'
    return None

m['sym'] = m['代码'].map(to_wind)
m = m.dropna(subset=['sym'])
m['name'] = m['名称'].astype(str)
m['pct'] = pd.to_numeric(m['涨跌幅'], errors='coerce')
m['mkt_amt'] = pd.to_numeric(m['市场总成交额'], errors='coerce')

ev = m[
    (m['yys_net'] > 0)
    & (m['mkt_amt'] > 0)
    & (m['yys_net'] >= m['mkt_amt'] * MIN_NET_PCT)
    & ~m['name'].str.contains('ST|退')
    & m['pct'].notna()
].copy()
ev['limit_pct'] = ev['sym'].map(lambda s: rules.price_limit_ratio(s) * 100)
ev = ev[ev['pct'] < ev['limit_pct'] - 0.25]
ev = ev[(ev['上榜日'] >= START_TS) & (ev['上榜日'] <= END_TS)]
print(f'事件（营业部净买≥当日成交5%，非ST，未封板）: {len(ev)} 只-日 '
      f'| {ev["上榜日"].min().date()} ~ {ev["上榜日"].max().date()}', flush=True)

# ── 3. 全缓存股票价格矩阵（直读 parquet，不触发补数）──
print('加载全缓存股票价格...', flush=True)
cache = {}
kline_dir = 'data_cache/kline'
files = sorted(os.listdir(kline_dir))
for i, fn in enumerate(files):
    if not fn.endswith('.parquet'):
        continue
    sym = fn[:-8].replace('_', '.')
    try:
        df = pd.read_parquet(os.path.join(kline_dir, fn))
        if 'adjclose' in df.columns and len(df):
            df['date'] = pd.to_datetime(df['date'])
            cache[sym] = df
    except Exception:
        continue
    if (i + 1) % 1000 == 0:
        print(f'  {i + 1}/{len(files)}', flush=True)
print(f'缓存股票: {len(cache)} 只', flush=True)

px = close_matrix(cache, basis='adjusted')
px = px[(px.index >= START_TS) & (px.index <= END_TS)]

# 事件股缺缓存剔除
have = ev['sym'].isin(px.columns)
print(f'事件股缺本地价格被剔除: {(~have).sum()} 只-日', flush=True)
ev = ev[have]

# ── 4. PIT 行业月度快照与行业日收益 ──
print('PIT 行业分类（月度）...', flush=True)
ds = LayeredDataSource()
months = sorted({(d.year, d.month) for d in px.index})
cls_by_month = {}
for ym in months:
    first = min(d for d in px.index if (d.year, d.month) == ym)
    syms = [s for s in px.columns]
    try:
        cls_by_month[ym] = industry_classification(ds, syms, first.date())
    except Exception as e:
        print(f'  ⚠️ {ym} 分类失败: {str(e)[:60]}', flush=True)
        cls_by_month[ym] = {}

ind_ret = {}
for ym, cls in cls_by_month.items():
    sub = px.loc[(px.index.year == ym[0]) & (px.index.month == ym[1])]
    if len(sub) < 5:
        continue
    groups = {}
    for s, ind in cls.items():
        if s in sub.columns and ind:
            groups.setdefault(ind, []).append(s)
    rets = sub.pct_change()
    ind_ret[ym] = pd.DataFrame({ind: rets[c].mean(axis=1) for ind, c in groups.items()})
ind_all = pd.concat([ind_ret[k] for k in sorted(ind_ret)], axis=0).sort_index()
ind_all = ind_all.fillna(0)
print(f'行业日收益矩阵: {ind_all.shape[0]} 日 × {ind_all.shape[1]} 行业', flush=True)

# 全市场等权日收益（诊断口径）
mkt_ret = px.pct_change().mean(axis=1)

# ── 5. 事件收益 ──
sym_bars = {s: cache[s].reset_index(drop=True) for s in ev['sym'].unique()}
date_pos = {s: {d: i for i, d in enumerate(sym_bars[s]['date'])}
            for s in sym_bars}

def stock_ret(sym, t, n):
    bars = sym_bars[sym]
    p = date_pos[sym].get(t)
    if p is None or p + n >= len(bars):
        return None, None
    c = bars['adjclose'].values
    if c[p] <= 0 or np.isnan(c[p]) or np.isnan(c[p + n]):
        return None, None
    return c[p + n] / c[p] - 1, bars['date'].values[p + n]

def series_ret(rets: pd.Series, t, n):
    """从t起n个交易日的累计收益（日收益序列）"""
    sub = rets.loc[t:]
    if len(sub) < n + 1:
        return None
    return float((1 + sub.iloc[1:n + 1]).prod() - 1)

rows = []
for _, r in ev.iterrows():
    t = r['上榜日']
    ym = (t.year, t.month)
    cls = cls_by_month.get(ym, {})
    ind = cls.get(r['sym'])
    for w in WINDOWS:
        sr, exit_d = stock_ret(r['sym'], t, w)
        if sr is None:
            continue
        ir = series_ret(ind_all[ind], t, w) if ind and ind in ind_all.columns else None
        mr = series_ret(mkt_ret, t, w)
        rows.append({'sym': r['sym'], 'date': t, 'window': w, 'era':
                     '2015-2020' if t < ERA_SPLIT else '2021-2026',
                     'stock_ret': sr, 'ind_ret': ir, 'mkt_ret': mr,
                     'excess_ind': (sr - ir) if ir is not None else None,
                     'excess_mkt': (sr - mr) if mr is not None else None})
df = pd.DataFrame(rows)
df.to_csv('ai-created/trial002_detail.csv', index=False, encoding='utf-8-sig')
print(f'事件×窗口样本: {len(df)}（事件数={len(ev)}）', flush=True)

# ── 6. 统计与判据 ──
def welch(a, b):
    a, b = pd.Series(a).dropna(), pd.Series(b).dropna()
    if len(a) < 2 or len(b) < 2:
        return np.nan
    sp = ((len(a) - 1) * a.var() + (len(b) - 1) * b.var()) / (len(a) + len(b) - 2)
    return (a.mean() - b.mean()) / np.sqrt(sp * (1 / len(a) + 1 / len(b))) if sp > 0 else 0.0

print('\n' + '=' * 72)
for w in WINDOWS:
    x = df[df.window == w]
    ex = x['excess_ind']
    print(f'\n{w}日（行业口径超额）: n={ex.notna().sum()}')
    print(f'  全样本均值 {ex.mean():+.3%} | 中位 {ex.median():+.3%} | '
          f'截尾1%均值 {ex.clip(ex.quantile(0.01), ex.quantile(0.99)).mean():+.3%}')
    for era in ('2015-2020', '2021-2026'):
        e_ = x[x.era == era]['excess_ind']
        print(f'  {era}: n={len(e_)} 均值 {e_.mean():+.3%}' if len(e_) else f'  {era}: 无事件')
    tw = welch(x['excess_ind'].dropna(), [0] * len(x))
    t_vs0 = x['excess_ind'].dropna()
    t_one = t_vs0.mean() / (t_vs0.std() / np.sqrt(len(t_vs0))) if len(t_vs0) > 2 else np.nan
    print(f'  Welch t(vs 0) = {tw:.2f} | 单样本 t = {t_one:.2f}')
    xm = x['excess_mkt'].dropna()
    if len(xm):
        print(f'  诊断·全市场口径: 均值 {xm.mean():+.3%} (n={len(xm)})')

# 判据（20日窗为主判据）
x20 = df[df.window == 20]['excess_ind'].dropna()
e1 = df[(df.window == 20) & (df.era == '2015-2020')]['excess_ind'].dropna()
e2 = df[(df.window == 20) & (df.era == '2021-2026')]['excess_ind'].dropna()
n_events = len(df[df.window == 20])
t20 = welch(x20, [0] * len(x20))
same_pos = len(e1) > 0 and len(e2) > 0 and e1.mean() > 0 and e2.mean() > 0
same_neg = len(e1) > 0 and len(e2) > 0 and e1.mean() < 0 and e2.mean() < 0
verdict = ('2a PASS' if (same_pos and t20 >= 2) else
           '2b PASS' if (same_neg and t20 <= -2) else '双 FAIL')
print(f'\n判据（20日窗）: n={n_events}(≥100 {n_events >= 100}) | '
      f'两期同号 {"正" if same_pos else ("负" if same_neg else "不同号/缺期")} | t={t20:.2f} → {verdict}')

# 按年分布
yr = df[df.window == 20].groupby(df[df.window == 20]['date'].dt.year)['excess_ind'].agg(['count', 'mean'])
print('\n按年（20日超额）:')
print(yr.round(4).to_string())

# ── 7. 登记册 ──
from datetime import datetime
reg_p = 'results/data/trial_registry.json'
reg = json.load(open(reg_p, encoding='utf-8'))
reg['trials'].append({
    'label': '002-游资龙虎榜(2a跟随/2b反指)', 'ran_at': datetime.now().isoformat(timespec='seconds'),
    'n_events': int(n_events),
    'excess20_mean': float(x20.mean()) if len(x20) else None,
    'era1_20d': float(e1.mean()) if len(e1) else None,
    'era2_20d': float(e2.mean()) if len(e2) else None,
    't_20d': float(t20) if not np.isnan(t20) else None,
    'verdict': verdict})
reg['count'] = len(reg['trials'])
tmp = reg_p + '.tmp'
with open(tmp, 'w', encoding='utf-8') as f:
    json.dump(reg, f, ensure_ascii=False, indent=1)
os.replace(tmp, reg_p)
print(f'\n登记册: {reg["count"]} 条 | 明细: ai-created/trial002_detail.csv')
