#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
改进②组合构造 sweep（改进清单第二项，2026-10-16）

网格：TopN {100,150,200} × 调仓间隔 {20,40,60 交易日}（分数行下采样实现），
缓冲带固定 0.20；最优格再测带宽 {0.10, 0.20, 0.30}。
判定：寻找"高原"而非"尖峰"——相邻参数结果连续、无孤例最优（参数稳定哲学）。
指标：十年收益/夏普/回撤（A口径=定稿重入协议，含成本）。
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import date

import numpy as np
import pandas as pd

from data.layered import LayeredDataSource
from portfolio.daily_replay import DailyReplayBacktester, ReplayConfig
from trading import config as chain
from trading.strategy import build_strategy_score

START = date(2016, 6, 1)
END = date(2026, 10, 9)
SIGNAL_START = date(2017, 1, 1)
TOPNS = (100, 150, 200)
REBS = (20, 40, 60)

ds = LayeredDataSource()
pool = sorted(ds.list_symbols(as_of=SIGNAL_START))
print(f"池: {len(pool)} 只", flush=True)
score, cache, _ = build_strategy_score(ds, pool, START, END)
print(f"分数矩阵: {score.shape}", flush=True)

st_matrix = None
try:
    from data.st_provider import STProvider
    idx = pd.DatetimeIndex(pd.to_datetime(next(iter(cache.values()))['date']))
    st_matrix = STProvider(ds).st_matrix(idx, pool)
    print("ST矩阵就绪", flush=True)
except Exception as e:
    print(f"警告：ST不可用（{e}）", flush=True)


def run(sc, top_n, band=chain.BUFFER_BAND):
    cfg = ReplayConfig(top_n=top_n, st_matrix=st_matrix, band=band,
                       use_risk_gate=True, st_data_required=st_matrix is not None)
    return DailyReplayBacktester(cache, cfg).run(
        sc, start_date=str(SIGNAL_START), end_date=str(END))


# 交易日历对齐下采样：在价格日历上每 reb 个交易日保留一个分数行
# （分数索引非交易日粒度[3783行 vs 2516交易日]，iloc::N 切片语义混乱，
#  曾致 reb=20 目标列表 2021 年中后变空、5 年零买单的伪结果）
px_dates = pd.DatetimeIndex(sorted(
    pd.to_datetime(next(iter(cache.values()))['date']).unique()))

def downsample(sc, reb):
    keep_days = set(px_dates[::reb])
    # 保留分数行中 ≤ 各保留交易日的最后一行（等价于只在调仓日换目标）
    rows_keep = []
    for d in keep_days:
        pos = sc.index.searchsorted(d, side='right') - 1
        if pos >= 0:
            rows_keep.append(sc.index[pos])
    return sc.loc[sorted(set(rows_keep))]

rows = []
for reb in (1,) + REBS:              # reb=1 对照格（必须复现基线 +59.0%）
    sc = score if reb == 1 else downsample(score, reb)
    for top_n in TOPNS:
        r = run(sc, top_n)
        rows.append({'reb': reb, 'top_n': top_n, 'band': 0.20,
                     'ret': r.total_return, 'sharpe': r.sharpe,
                     'dd': r.max_drawdown})
        print(f"  调仓{reb}日 Top{top_n}: {r.total_return:+.1%} "
              f"夏普{r.sharpe:.2f} 回撤{r.max_drawdown:.1%}", flush=True)

df = pd.DataFrame(rows)
df = df[df.reb > 1] if (df.reb == 1).any() else df
best = df.loc[df.sharpe.idxmax()]
print(f"\n最优格（夏普）: 调仓{int(best.reb)}日 Top{int(best.top_n)} "
      f"{best.ret:+.1%}/{best.sharpe:.2f}——测带宽敏感性:", flush=True)
sc_best = downsample(score, int(best.reb))
band_rows = []
for band in (0.10, 0.20, 0.30):
    r = run(sc_best, int(best.top_n), band=band)
    band_rows.append({'band': band, 'ret': r.total_return, 'sharpe': r.sharpe,
                      'dd': r.max_drawdown})
    print(f"  带宽{band:.0%}: {r.total_return:+.1%} 夏普{r.sharpe:.2f} "
          f"回撤{r.max_drawdown:.1%}", flush=True)

print(f"""
{'=' * 72}
改进②矩阵（十年 A 口径，含成本）
{'=' * 72}""")
piv = df.pivot(index='reb', columns='top_n', values='sharpe')
print("夏普（行=调仓间隔日，列=TopN）:")
print(piv.round(2).to_string())
piv2 = df.pivot(index='reb', columns='top_n', values='ret')
print("\n收益:")
print((piv2 * 100).round(1).to_string())
print("\n带宽敏感性（最优格）:")
for b in band_rows:
    print(f"  {b['band']:.0%}: {b['ret']:+.1%} / {b['sharpe']:.2f} / {b['dd']:.1%}")
r_spread = df.ret.max() - df.ret.min()
print(f"\n网格收益极差: {r_spread * 100:.1f}pp（<15pp=高原形态，>30pp=尖峰风险）")
df.to_csv('ai-created/construction_sweep.csv', index=False, encoding='utf-8')
