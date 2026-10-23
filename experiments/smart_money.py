#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Smart Money 研究 · 第1步：大宗交易机构行为信号的事件研究

从 ASHAREBLOCKTRADE 提取2024年以来（该表数据最完整时段）的大宗交易，
按买方类型分类（机构/营业部/北向），构建月度信号 → 测量前瞻收益。
同时提取股东户数变化（季频）和北向资金作为辅助信号。
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import date
import numpy as np
import pandas as pd

from data.layered import LayeredDataSource
from data.base.data_loader import DataLoader
from data.semantics import close_matrix
from data.industry_pit import industry_classification

START, END = date(2024, 1, 1), date(2026, 10, 9)
WINDOWS = (20, 60)

ds = LayeredDataSource()
cur = ds._db_src()._cursor()

# ── 1. 提取大宗交易 ──
print("提取大宗交易...", flush=True)
cur.execute("""
SELECT S_INFO_WINDCODE, TRADE_DT, S_BLOCK_PRICE, S_BLOCK_VOLUME,
       S_BLOCK_AMOUNT, S_BLOCK_BUYERNAME, S_BLOCK_SELLERNAME
FROM ASHAREBLOCKTRADE WHERE TRADE_DT >= :d0 AND TRADE_DT <= :d1
""", {'d0': '20230101', 'd1': END.strftime('%Y%m%d')})
rows = cur.fetchall()
bt = pd.DataFrame(rows, columns=['sym', 'dt', 'price', 'vol', 'amount',
                                  'buyer', 'seller'])
bt['dt'] = bt['dt'].astype(str)
print(f"大宗交易: {len(bt)} 笔 | {bt['dt'].min()} ~ {bt['dt'].max()}", flush=True)

# 买方分类
def classify_buyer(name):
    n = str(name or '')
    if '机构' in n:
        return 'inst'
    if '陆股通' in n or '沪股通' in n or '深股通' in n:
        return 'north'
    if '营业部' in n:
        return 'branch'
    return 'hq'

bt['btype'] = bt['buyer'].apply(classify_buyer)
print(f"买方分类: {bt['btype'].value_counts().to_dict()}", flush=True)
bt['dt_ts'] = pd.to_datetime(bt['dt'], format='%Y%m%d', errors='coerce')
bt['amount_wan'] = pd.to_numeric(bt['amount'], errors='coerce') / 1e4

# ── 2. 构建月度信号：机构净买入（20日窗口） ──
print("\n构建月度Smart Money信号...", flush=True)
# 每月末计算过去20日大宗净买入
bt['amount_signed'] = bt['amount_wan'] * np.where(bt['btype'] == 'inst', 1,
                                        np.where(bt['btype'] == 'branch', -1, 0))
# 机构净买 = 机构买入金额 - 营业部买入金额（营业部买=游资/散户接盘=负信号）
net_buy = bt.groupby(['sym', bt['dt_ts'].dt.to_period('M')]).agg(
    inst_buy=('amount_wan', lambda x: x[bt.loc[x.index, 'btype'] == 'inst'].sum()),
    branch_buy=('amount_wan', lambda x: x[bt.loc[x.index, 'btype'] == 'branch'].sum()),
    total=('amount_wan', 'sum'),
    n_trades=('amount_wan', 'count'),
).reset_index()
net_buy['smart_flow'] = net_buy['inst_buy'] - net_buy['branch_buy']

# ── 3. 提取股东户数变化 ──
print("提取股东户数...", flush=True)
cur.execute("""
SELECT S_INFO_WINDCODE, S_HOLDER_ENDDATE, S_HOLDER_NUM
FROM ASHAREHOLDERNUMBER WHERE S_HOLDER_ENDDATE >= '20230101'
""")
hn = pd.DataFrame(cur.fetchall(), columns=['sym', 'end_date', 'n_holders'])
hn['end_date'] = hn['end_date'].astype(str)
hn['n_holders'] = pd.to_numeric(hn['n_holders'], errors='coerce')
hn = hn.sort_values(['sym', 'end_date'])
hn['holder_chg'] = hn.groupby('sym')['n_holders'].pct_change()
hn['end_ts'] = pd.to_datetime(hn['end_date'], format='%Y%m%d', errors='coerce')
print(f"股东户数: {len(hn)} 条", flush=True)

# ── 4. 加载价格（全缓存） ──
print("加载全市场价格...", flush=True)
syms_all = sorted(set(bt['sym'].dropna()) | set(hn['sym'].dropna()))
loader = DataLoader.__new__(DataLoader)
loader.data_source_type = 'layered'
loader.data_source = ds
loader._cache = None
cache = loader.preload_klines(syms_all + ['000300.SH'], date(2023, 1, 1), END)._cache
px = close_matrix(cache, basis='adjusted')
print(f"价格矩阵: {px.shape}", flush=True)

# ── 5. PIT 行业分类 ──
print("PIT 行业分类...", flush=True)
all_syms = sorted(set(px.columns) & set(bt['sym'].dropna()))
month_keys = sorted(set((d.year, d.month) for d in px.index if d >= pd.Timestamp('2024-01-01')))
cls_by_month = {}
for ym in month_keys:
    first = min(d for d in px.index if (d.year, d.month) == ym)
    cls_by_month[ym] = industry_classification(ds, all_syms, first.date())

# 行业日收益
ind_ret = {}
for ym, cls in cls_by_month.items():
    cols = [s for s in cls if s in px.columns]
    sub = px[cols]
    sub = sub.loc[(sub.index.year == ym[0]) & (sub.index.month == ym[1])]
    if len(sub) < 5:
        continue
    groups = {}
    for s in cols:
        groups.setdefault(cls[s], []).append(s)
    rets = sub.pct_change()
    ind_ret[ym] = pd.DataFrame({ind: rets[c].mean(axis=1) for ind, c in groups.items()})

ind_ret_all = pd.concat([ind_ret[k] for k in sorted(ind_ret)], axis=0).sort_index().fillna(0)

# ── 6. 事件研究主循环 ──
print("\n事件研究...", flush=True)
dates = px.index
sig_dates = sorted(net_buy['dt_ts'].unique())  # 已是Period类型
rows_out = []
for m in sig_dates:
    # 信号日 = 每月第一个交易日
    try:
        entry = min(d for d in dates if d.to_period('M') == m)
    except ValueError:
        continue
    if entry < pd.Timestamp('2024-01-01'):
        continue
    # 过去20日大宗数据（entry往前找）
    lookback_start = entry - pd.Timedelta(days=35)
    mask = (bt['dt_ts'] >= lookback_start) & (bt['dt_ts'] < entry)
    recent = bt[mask]
    if recent.empty:
        continue

    # 机构净买/营业部净买 per stock
    flows = recent.groupby('sym').apply(
        lambda g: g.loc[g['btype'] == 'inst', 'amount_wan'].sum()
                - g.loc[g['btype'] == 'branch', 'amount_wan'].sum()
    ).dropna()

    if len(flows) < 10:
        continue

    cls = cls_by_month.get((entry.year, entry.month), {})
    ir = ind_ret.get((entry.year, entry.month), {})

    for sym in flows.index:
        if sym not in px.columns:
            continue
        s = px[sym].loc[entry:]
        if len(s) < WINDOWS[-1] + 1:
            continue
        ind = cls.get(sym)
        if ind is None or ind not in ind_ret_all.columns:
            continue
        irc = ind_ret_all[ind].loc[entry:]
        if len(irc) < WINDOWS[-1] + 1:
            continue
        sc = s / s.iloc[0]
        ic = (1 + irc.fillna(0)).cumprod()
        for w in WINDOWS:
            r_s = float(sc.iloc[w]) - 1
            r_i = float(ic.iloc[w]) - 1
            rows_out.append({
                'sym': sym, 'month': str(m), 'entry': entry,
                'flow': flows[sym], 'window': w,
                'excess': r_s - r_i,
                'era': '24-25' if entry < pd.Timestamp('2025-07-01') else '25-26',
            })

df = pd.DataFrame(rows_out)
df.to_csv('ai-created/smart_money_detail.csv', index=False)
print(f"有效事件×窗口: {len(df)}", flush=True)

# ── 7. 分组分析 ──
print(f"\n{'='*70}")
print(f"Smart Money 事件研究（机构净买入信号 → 行业内超额）")
print(f"{'='*70}")

for era in ('all', '24-25', '25-26'):
    dd = df if era == 'all' else df[df.era == era]
    if dd.empty:
        continue
    print(f"\n── {era} ──")
    for w in WINDOWS:
        x = dd[dd.window == w]
        if len(x) < 30:
            continue
        # 按flow分组：正（机构净买）vs 负（营业部主导）
        pos = x[x['flow'] > 0]['excess']
        neg = x[x['flow'] <= 0]['excess']
        if len(pos) < 10 or len(neg) < 10:
            continue
        t_pos = pos.mean() / (pos.std() / np.sqrt(len(pos))) if pos.std() > 0 else 0
        spread = pos.mean() - neg.mean()
        t_sp = spread / np.sqrt(pos.var()/len(pos) + neg.var()/len(neg)) if (pos.var() > 0 and neg.var() > 0) else 0
        print(f"  {w}日: 机构净买组({len(pos)}只) {pos.mean():+.2%}(t={t_pos:.1f})"
              f" | 营业部组({len(neg)}只) {neg.mean():+.2%}"
              f" | 差 {spread:+.2%}(t={t_sp:.1f})")

# 判定
print(f"\n{'='*70}")
print("注意：样本从2024年开始（大宗数据完整期），无法做17-21/22-26分期。")
print("如需完整分期验证，需补充2023年以前的大宗交易数据或使用JQData获取更早数据。")
print("当前时段（2024-2026）仅包含一个市场环境——即使信号通过也不能确认跨期稳健。")
