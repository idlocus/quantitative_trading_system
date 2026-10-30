#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A4 稳定性测试（2026-10-21，预注册判据见 docs/策略变更窗口队列.md A4）

变体：120日rank动量 @ 权重{0.15, 0.20(已知+61.9%), 0.25, 0.30}
      + 回看{110, 130} @ 权重0.20
判据：C口径全部 ≥ 基线（本轮基线须复现≈+57.1%）→ A4 存活进窗口；
      任一 < 基线 → 关闭（多重比较噪音）。
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import date

import numpy as np
import pandas as pd

from data.layered import LayeredDataSource
from data.base.data_loader import DataLoader
from data.semantics import close_matrix
from data.fundamentals import WindFundamentalProvider, build_fundamental_matrix
from data.fundamental_pit import PITFinancialProvider
from portfolio.construction import composite_from_matrices
from portfolio.daily_replay import DailyReplayBacktester, ReplayConfig
from trading import config as chain
from trading.strategy import build_strategy_score

START, END = date(2016, 6, 1), date(2026, 10, 9)
SIGNAL_START = '2017-01-03'

ds = LayeredDataSource()
pool = sorted(ds.list_symbols(as_of=date(2017, 1, 3)))
assert len(pool) > 100

base_score, cache, _ = build_strategy_score(ds, pool, START, END)
assert base_score.shape[1] > 100
print(f"基线分数: {base_score.shape}", flush=True)

# ── 重建 value+quality（向量化管道，与 RPS 研究同款） ──
loader = DataLoader.__new__(DataLoader)
loader.data_source_type = 'layered'
loader.data_source = ds
loader._cache = None
loader_cache = loader.preload_klines(pool + ['000300.SH'], START, END)._cache

prov = WindFundamentalProvider(ds)
mats = {f: build_fundamental_matrix(prov, f, pool, START, END)
        for f in chain.PRODUCTION_FACTORS}
mats = {k: v for k, v in mats.items() if not v.empty}
value = composite_from_matrices(mats, directions=chain.PRODUCTION_FACTORS)
value = value[value.columns.intersection(loader_cache.keys())]
pit = PITFinancialProvider(ds)
roe = pit.get_factor_matrix('roe', pool, START, END)
lev = pit.get_factor_matrix('leverage', pool, START, END)
snv = value.copy()
for mx, thr, is_min in ((roe, 0.08, True), (lev, 65.0, False)):
    m = mx.reindex(index=snv.index, columns=snv.columns)
    snv = snv.mask(m < thr) if is_min else snv.mask(m > thr)
assert snv.notna().mean().mean() > 0.1


def zrow(mx):
    return mx.sub(mx.mean(axis=1), axis=0).div(mx.std(axis=1) + 1e-12, axis=0)


v_z = zrow(snv)
px = close_matrix(loader_cache, basis='adjusted')

st_matrix = None
try:
    from data.st_provider import STProvider
    idx = pd.DatetimeIndex(pd.to_datetime(next(iter(loader_cache.values()))['date']))
    st_matrix = STProvider(ds).st_matrix(idx, pool)
    print("ST矩阵就绪", flush=True)
except Exception as e:
    print(f"警告：ST不可用（{e}）", flush=True)

cfg = ReplayConfig(top_n=chain.TOP_N, st_matrix=st_matrix, band=chain.BUFFER_BAND,
                   use_risk_gate=False, st_data_required=False)


def run_c(sc):
    return DailyReplayBacktester(loader_cache, cfg).run(
        sc, start_date=SIGNAL_START, end_date=str(END))


def rps(lb):
    return (px / px.shift(lb) - 1).rank(axis=1, pct=True)


VARIANTS = [('基线', None, None)]
for w in (0.15, 0.25, 0.30):
    VARIANTS.append((f'w={w:.2f},lb=120', w, rps(120)))
for lb in (110, 130):
    VARIANTS.append((f'w=0.20,lb={lb}', 0.20, rps(lb)))

print(f"\n{'='*70}")
print(f"{'变体':<16} {'C收益':>8} {'夏普':>6} {'≥基线?':>7}")
print(f"{'='*70}")
res = {}
for name, w, rk in VARIANTS:
    if rk is None:
        r = run_c(base_score)
    else:
        al = rk.reindex_like(v_z).fillna(0.5)
        sc = zrow((1 - w) * v_z + w * zrow(al))
        r = run_c(sc)
    res[name] = r
    print(f"{name:<16} {r.total_return:>+8.1%} {r.sharpe:>6.2f}", flush=True)

b = res['基线'].total_return
print(f"\n{'='*70}")
print(f"基线 C = {b:+.1%}（须≈+57.1%）")
fails = [n for n, r in res.items() if n != '基线' and r.total_return < b]
if abs(b - 0.571) > 0.02:
    print(f"⚠️ 基线未复现（{b:+.1%}）——本轮结果全部存疑，需检查")
elif fails:
    print(f"❌ A4 关闭：{fails} 低于基线——多重比较噪音判定成立")
else:
    print("✅ A4 稳定性通过：全部邻域变体 ≥ 基线——进入变更窗口候选")
