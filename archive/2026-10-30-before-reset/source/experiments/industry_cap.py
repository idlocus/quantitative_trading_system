#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
改进①单行业权重上限（改进清单第一项，2026-10-16）

设计：等权 Top100 下，行业权重上限按"选择贪心截断"实现——按分数降序
入选，行业满额即跳过，选满 100 只为止；被跳过者分数置 NaN。
两档：15%（每行业≤15只）/ 20%（≤20只）。PIT 万得 L2 行业。
对照：基线 A 口径（定稿重入协议）十年重放（应复现 +59.0%）。
判定框架：这是风控改进不是收益改进——报告"集中度下降幅度 vs 收益代价"，
行业集中度用（最大行业权重 / 前三行业合计权重）度量。
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import date

import numpy as np
import pandas as pd

from data.layered import LayeredDataSource
from data.industry_pit import industry_classification
from portfolio.daily_replay import DailyReplayBacktester, ReplayConfig
from trading import config as chain
from trading.strategy import build_strategy_score

START = date(2016, 6, 1)
END = date(2026, 10, 9)
SIGNAL_START = date(2017, 1, 1)
CAPS = (0.15, 0.20)

ds = LayeredDataSource()
pool = sorted(ds.list_symbols(as_of=SIGNAL_START))
print(f"池: {len(pool)} 只", flush=True)

print("构建基线分数（v2.1.0 唯一实现）…", flush=True)
score, cache, _mats = build_strategy_score(ds, pool, START, END)
print(f"分数矩阵: {score.shape}", flush=True)

# ── PIT 行业（月度查询，映射到全部分数日） ──
print("PIT 行业分类 …", flush=True)
score_dates = score.index
month_first, seen = {}, set()
for d in score_dates:
    m = (d.year, d.month)
    if m not in seen:
        seen.add(m)
        month_first[m] = d
cls_cache = {}
ind_rows = {}
for m, fd in month_first.items():
    cls_cache[m] = industry_classification(ds, list(score.columns), fd.date())
for d in score_dates:
    ind_rows[d] = cls_cache[(d.year, d.month)]
# dates × symbols 行业代码（orient='index'：外层键=行索引；此前用
# index= 显式传参导致符号错位全 NaN，cap 退化成全市场截断）
ind_mx = pd.DataFrame.from_dict(ind_rows, orient='index').reindex(score_dates)
cov = ind_mx.notna().mean().mean()
print(f"行业覆盖: {cov:.0%}", flush=True)
assert cov > 0.9, "行业覆盖异常——cap 实验不可信"


def cap_score(sc, cap: float):
    """贪心选择：按分数降序，行业满额跳过，选满 top_n；其余置 NaN。"""
    max_per = int(round(cap * chain.TOP_N))
    out = sc.copy()
    for dt in sc.index:
        row = sc.loc[dt].dropna()
        if len(row) <= chain.TOP_N:
            continue
        inds = ind_mx.loc[dt]
        picked, cnt = [], {}
        for sym in row.sort_values(ascending=False).index:
            ind = inds.get(sym)
            if pd.notna(ind) and cnt.get(ind, 0) >= max_per:
                continue          # 行业满额跳过；无行业分类（NaN）不设限
            picked.append(sym)
            cnt[ind] = cnt.get(ind, 0) + 1
            if len(picked) >= chain.TOP_N:
                break
        keep = pd.Series(False, index=sc.columns)
        keep[picked] = True
        out.loc[dt] = sc.loc[dt].where(keep)
    return out


# ── ST 矩阵（与 full_decade A 口径一致） ──
st_matrix = None
try:
    from data.st_provider import STProvider
    idx = pd.DatetimeIndex(pd.to_datetime(next(iter(cache.values()))['date']))
    st_matrix = STProvider(ds).st_matrix(idx, pool)
    print("ST矩阵就绪", flush=True)
except Exception as e:
    print(f"警告：ST矩阵不可用（{e}）", flush=True)


def run_replay(sc):
    cfg = ReplayConfig(top_n=chain.TOP_N, st_matrix=st_matrix,
                       use_risk_gate=True, st_data_required=st_matrix is not None)
    return DailyReplayBacktester(cache, cfg).run(
        sc, start_date=str(SIGNAL_START), end_date=str(END))


def concentration(res):
    """平均最大行业权重 / 前三行业合计（由 daily_holdings + PIT 行业算）"""
    dh = getattr(res, 'daily_holdings', {}) or {}
    maxws, top3s = [], []
    for d, holds in dh.items():
        if not holds:
            continue
        dt = pd.Timestamp(d)
        m = (dt.year, dt.month)
        cls = cls_cache.get(m, {})
        w = {}
        total = sum(v for _, v in holds)
        for sym, val in holds:
            ind = cls.get(sym, '未知')
            w[ind] = w.get(ind, 0) + val / max(total, 1)
        ws = sorted(w.values(), reverse=True)
        maxws.append(ws[0])
        top3s.append(sum(ws[:3]))
    return (float(np.mean(maxws)), float(np.mean(top3s))) if maxws else (np.nan, np.nan)


print("\n重放：基线 …", flush=True)
base = run_replay(score)
b_max, b_top3 = concentration(base)
print(f"  基线: {base.total_return:+.1%} 夏普{base.sharpe:.2f} 回撤{base.max_drawdown:.1%} "
      f"| 最大行业 {b_max:.0%} 前三 {b_top3:.0%}", flush=True)

results = {'baseline': (base, b_max, b_top3)}
for cap in CAPS:
    print(f"重放：行业上限 {cap:.0%} …", flush=True)
    sc = cap_score(score, cap)
    r = run_replay(sc)
    c_max, c_top3 = concentration(r)
    results[cap] = (r, c_max, c_top3)
    print(f"  上限{cap:.0%}: {r.total_return:+.1%} 夏普{r.sharpe:.2f} "
          f"回撤{r.max_drawdown:.1%} | 最大行业 {c_max:.0%} 前三 {c_top3:.0%}", flush=True)

print(f"""
{'=' * 72}
改进①判定表：行业权重上限（风控改进，代价-收益对照）
{'=' * 72}
  口径          十年收益   夏普   回撤    最大行业  前三行业  收益代价""")
b = results['baseline'][0]
print(f"  基线          {b.total_return:+6.1%}  {b.sharpe:5.2f}  {b.max_drawdown:6.1%}  "
      f"{b_max:7.0%}  {b_top3:7.0%}       —")
for cap in CAPS:
    r, c_max, c_top3 = results[cap]
    cost = (r.total_return - b.total_return) * 100
    print(f"  上限{cap:.0%}        {r.total_return:+6.1%}  {r.sharpe:5.2f}  {r.max_drawdown:6.1%}  "
          f"{c_max:7.0%}  {c_top3:7.0%}  {cost:+6.1f}pp")
print("""
判读（非单一判据，供决策）：
  - 若 20% 档集中度显著下降（最大行业≤20%）且代价 >-3pp → 推荐 20% 进五关
  - 若 15% 档代价 >-8pp 且回撤改善 → 15% 为备选
  - 集中度未降=上限未绑定（本就不超限）→ 无需改动""")
