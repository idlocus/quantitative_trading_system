#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
财报事件研究 P3：SUE 倾斜因子化（docs/events研究方案.md 预注册）

score2_z = score_z + coef × sue_z（调仓日 T 用 [T-60日, T-1] 内最新 SUE 的
月队列内 z；无事件=0 倾斜；coef ∈ {0.15, 0.25, 0.35} 三档稳定性）
对照：同一十年重放（A 口径=定稿重入协议），增量 >+2pp/十年（含成本）才集成。
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import date

import numpy as np
import pandas as pd

from data.layered import LayeredDataSource
from data.base.data_loader import DataLoader
from data.events_pit import sue_events
from portfolio.daily_replay import DailyReplayBacktester, ReplayConfig
from trading import config as chain
from trading.strategy import build_strategy_score
from research.artifacts import save_artifact, portfolio_payload

START = date(2016, 6, 1)
END = date(2026, 10, 9)
SIGNAL_START = date(2017, 1, 1)
COEFS = (0.15, 0.25, 0.35)
SUE_WINDOW_DAYS = 60

ds = LayeredDataSource()
pool = sorted(ds.list_symbols(as_of=SIGNAL_START))
print(f"池: {len(pool)} 只", flush=True)

# ── 基线分数（唯一实现） ──
print("构建基线分数（v2.1.0 唯一实现）…", flush=True)
score, cache, mats = build_strategy_score(ds, pool, START, END)
print(f"分数矩阵: {score.shape}", flush=True)

# ── SUE z 矩阵（对齐 score 日期索引；T-1 公告口径） ──
print("构建 SUE 倾斜矩阵 …", flush=True)
ev = sue_events(ds, date(2016, 10, 1), END)
ev = ev[ev.ann_date <= pd.Timestamp(END)]
print(f"SUE 事件: {len(ev)} 条", flush=True)

sue_z = pd.DataFrame(np.nan, index=score.index, columns=score.columns)
# 向量化：逐 symbol 用 searchsorted 解"每个分数日的回看窗口内最新事件"
# （旧逐日×逐股票 pandas 切片为 1280 万次调用，实测数小时不可用）
Ts = score.index.values
lo_cut = (score.index - pd.Timedelta(days=SUE_WINDOW_DAYS)).values
hi_cut = (score.index - pd.Timedelta(days=1)).values
raw = {}     # date_pos -> {sym: surprise}
for sym, g in ev.groupby('symbol'):
    gd = g.ann_date.values.astype('datetime64[ns]')
    gs = g.surprise.values
    i_hi = np.searchsorted(gd, hi_cut, side='right') - 1   # ≤T-1 的最后事件
    for k in np.nonzero(i_hi >= 0)[0]:
        j = i_hi[k]
        if gd[j] >= lo_cut[k]:                              # 且在 [T-60, T-1]
            raw.setdefault(k, {})[sym] = gs[j]
for k, vals in raw.items():
    if len(vals) >= 50:
        s = pd.Series(vals)
        z = (s - s.mean()) / (s.std() + 1e-12)
        sue_z.iloc[k] = pd.Series(z.values, index=list(vals.keys())).reindex(
            sue_z.columns).values

cov = sue_z.notna().sum(axis=1)
print(f"SUE 覆盖: 各分数日均 {int(cov.mean())} 只（{cov.min()}~{cov.max()}）", flush=True)

# 基线分数截面 z 化（与倾斜同量纲）
def zrows(mx):
    return mx.sub(mx.mean(axis=1), axis=0).div(mx.std(axis=1) + 1e-12, axis=0)

score_z = zrows(score)
sue_fill0 = sue_z.fillna(0.0)

# ── ST 矩阵（与 full_decade A 口径一致） ──
st_matrix = None
try:
    from data.st_provider import STProvider
    idx = pd.DatetimeIndex(pd.to_datetime(next(iter(cache.values()))['date']))
    st_matrix = STProvider(ds).st_matrix(idx, pool)
    print(f"ST矩阵就绪", flush=True)
except Exception as e:
    print(f"警告：ST矩阵不可用（{e}）", flush=True)


def run_replay(sc):
    cfg = ReplayConfig(top_n=chain.TOP_N, st_matrix=st_matrix,
                       use_risk_gate=True, st_data_required=st_matrix is not None)
    rp = DailyReplayBacktester(cache, cfg)
    r = rp.run(sc, start_date=str(SIGNAL_START), end_date=str(END))
    return r


# ── 基线与三档系数 ──
print("\n重放：基线 …", flush=True)
base = run_replay(score)
print(f"  基线(A口径): {base.total_return:+.1%} 夏普{base.sharpe:.2f} "
      f"回撤{base.max_drawdown:.1%}", flush=True)

results = {'baseline': base}
for c in COEFS:
    sc2 = score_z + c * sue_fill0
    print(f"重放：倾斜系数 {c} …", flush=True)
    r = run_replay(sc2)
    inc = (r.total_return - base.total_return) * 100
    print(f"  coef={c}: {r.total_return:+.1%} 夏普{r.sharpe:.2f} "
          f"回撤{r.max_drawdown:.1%} | 增量 {inc:+.1f}pp", flush=True)
    results[f'coef_{c}'] = r

incs = [(results[f'coef_{c}'].total_return - base.total_return) * 100 for c in COEFS]
c_pass = max(incs) > 2.0
c_stable = all(i > 0 for i in incs)
inc_str = " / ".join(f"{c}:{i:+.1f}pp" for c, i in zip(COEFS, incs))
print(f"""
{'=' * 70}
P3 预注册判定（十年重放增量，含成本；A口径=定稿重入协议）
{'=' * 70}
  基线: {base.total_return:+.1%}
  增量: {inc_str}
  最大增量 > +2pp: {'✅' if c_pass else '❌'}（{max(incs):+.1f}pp）
  三档全正: {'✅' if c_stable else '❌'}
  综合: {'✅ 通过——进入集成流程（R6+五关）' if (c_pass and c_stable) else
         '❌ 未达标——事件线按预注册关闭（H1b 需另行预注册）'}
{'=' * 70}""")

for c in COEFS:
    save_artifact('portfolio', f'event_overlay_{c}',
                  portfolio_payload(results[f'coef_{c}'],
                                    label=f'SUE倾斜{c} 十年（P3研究，未集成）'))
