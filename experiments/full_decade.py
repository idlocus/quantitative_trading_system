#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
完整十年回测验证（v2.1.0 唯一实现）—— 2017-2026 连续运行 + 三口径对照 + 收益归因 + 滚动窗口

R1 合规：分数构建经由 trading/strategy.py::build_strategy_score（唯一实现），
本脚本不出现任何因子列表/阈值/权重。与 mainchain_v5 完全同链
（build_strategy_score + ST矩阵 + 每日重放 + 生产闸门），仅时间窗扩展到全期。

三口径：
  A 生产口径    ：重入定稿语义（2026-10-15 两轮验证：清仓完成→立即以现金为
                  新锚满仓重入 + 365日历日>3次硬限=滚动熔断永久停机；
                  第一轮冷却/梯度设计被数据否决：-1.5%/-27.6% vs 立即重入
                  +59.0%/-24.9%）
  B 立即满仓重启：restart_after_liquidation=True（研究对照，无熔断计数；
                  A 与 B 一致=熔断未触发，A 更差=熔断触发）
  C 无闸门口径  ：use_risk_gate=False（隔离风控整体贡献）
对照：可交易等权 + 沪深300
诊断：生产因子月度 rank IC 分期（2017-2021 vs 2022-2026）
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import date
import numpy as np
import pandas as pd

from data.layered import LayeredDataSource
from portfolio.daily_replay import DailyReplayBacktester, ReplayConfig
from portfolio.backtester import PortfolioConfig
from portfolio.tradable_benchmark import tradable_equal_weight
from trading import config as chain
from trading.strategy import build_strategy_score
from research.artifacts import save_artifact, portfolio_payload

# ── 配置 ──
START = date(2016, 6, 1)       # warm-up：覆盖动量6个月回看
END = date(2026, 10, 9)
SIGNAL_START = date(2017, 1, 1)  # 统计起点（与 OOS/全市场两期一致）
ERA_SPLIT = pd.Timestamp('2022-01-01')  # IC 分期界（对齐 mainchain 两年代切分）

ds = LayeredDataSource()

# ── 股票池（PIT，与 mainchain OOS 同口径） ──
pool = sorted(ds.list_symbols(as_of=SIGNAL_START))
print(f"池: {len(pool)} 只 | 信号 {SIGNAL_START} ~ {END}", flush=True)

# ── 策略分数（唯一实现，日频矩阵；重放引擎自取 ≤T-1 最新行） ──
print("构建策略分数（v2.1.0 唯一实现）...", flush=True)
score, cache, mats = build_strategy_score(ds, pool, START, END)
print(f"分数矩阵: {score.shape[0]}日 × {score.shape[1]}只 | 因子: {sorted(mats)}", flush=True)

# ── ST 矩阵（与 mainchain 同链） ──
st_matrix = None
st_required = False
try:
    from data.st_provider import STProvider
    idx = pd.DatetimeIndex(pd.to_datetime(next(iter(cache.values()))['date']))
    st_matrix = STProvider(ds).st_matrix(idx, pool)
    print(f"ST矩阵就绪（{int(st_matrix.values.sum())} 标的日）", flush=True)
except Exception as e:
    st_required = True
    st_matrix = None
    print(f"警告：ST矩阵不可用（{e}），重放按生产安全口径禁买", flush=True)


def run_replay(label, gate_config=None, use_risk_gate=True):
    cfg = ReplayConfig(top_n=chain.TOP_N, st_matrix=st_matrix,
                       use_risk_gate=use_risk_gate, st_data_required=st_required,
                       gate_config=gate_config)
    rp = DailyReplayBacktester(cache, cfg)
    r = rp.run(score, start_date=str(SIGNAL_START), end_date=str(END))
    term = f"终态{r.terminal_date}" if r.terminal_date else "全程运行"
    print(f"  [{label}] {r.total_return:+.1%} 夏普{r.sharpe:.2f} 回撤{r.max_drawdown:.1%} "
          f"停机{r.n_halt_days}天 禁买{r.n_no_buy_days}天 | {term}", flush=True)
    return rp, r


# ── 三口径重放 ──
print("运行每日重放（A 生产 / B 自动重入 / C 无闸门）...", flush=True)
from live.risk_gate import GateConfig
rpA, rA = run_replay('A生产')
_, rB = run_replay('B重入', gate_config=GateConfig(restart_after_liquidation=True))
_, rC = run_replay('C无闸门', use_risk_gate=False)

# ── 可交易等权对照（与 mainchain --part=c 同链） ──
print("运行可交易等权对照...", flush=True)
ctrl_cfg = PortfolioConfig(top_n=chain.TOP_N, rebalance_days=20, st_matrix=st_matrix)
_, rEW, _ = tradable_equal_weight(cache, ctrl_cfg,
                                  start_date=str(SIGNAL_START), end_date=str(END))

# ── 市场基准 ──
bench_mx = rpA.close_mx
bench = '000300.SH' if '000300.SH' in bench_mx.columns else None
if bench:
    b = bench_mx[bench].loc[rA.equity_curve.index].ffill()
    bench_ret = float(b.iloc[-1] / b.iloc[0] - 1)
    bench_series = pd.Series(b.values / b.iloc[0], index=b.index)
else:
    bench_ret, bench_series = float('nan'), None

# ── 汇总净值曲线 ──
curves = {'A生产': rA.equity_curve, 'B重入': rB.equity_curve,
          'C无闸门': rC.equity_curve, '等权': rEW.equity_curve}
if bench_series is not None:
    curves['基准300'] = bench_series
nav = pd.DataFrame({k: v for k, v in curves.items()})
common = nav.dropna(how='all').index
nav = nav.loc[common]
nav = nav / nav.iloc[0]
nav.index.name = 'date'
os.makedirs('ai-created', exist_ok=True)
nav.to_csv('ai-created/nav_full_2017_2026.csv', encoding='utf-8')

# ═══════════════════════════════════════════════════
# 因子月度 rank IC 分期（诊断：因子逻辑是否状态依赖）
# ═════════════════════════════════════════════════
print("\n计算月度 rank IC（前复权前向收益，R2 声明 adjusted）...", flush=True)
from data.semantics import get_price_column

adj = {}
for sym, df in cache.items():
    if df is None or df.empty:
        continue
    c = get_price_column(df, basis='adjusted')
    adj[sym] = pd.Series(pd.to_numeric(df[c], errors='coerce').values,
                         index=pd.to_datetime(df['date']))
px = pd.DataFrame(adj)


def month_first(dates):
    """每月第一个信号日"""
    seen, out = set(), []
    for d in dates:
        m = (d.year, d.month)
        if m not in seen:
            seen.add(m)
            out.append(d)
    return pd.DatetimeIndex(out)


sig_m = month_first(sorted(score.index))
px_m = px.loc[px.index.intersection(sig_m)]
fwd = px_m.pct_change().shift(-1)   # 月度前向收益（信号日→下月信号日）

ic_summary = {}
for f, fm in mats.items():
    ics = {}
    for dt in fwd.index:
        if dt not in fm.index:
            continue
        r_row = fwd.loc[dt]
        f_row = fm.loc[dt].reindex(r_row.index)
        ok = f_row.notna() & r_row.notna()
        if ok.sum() < 100:
            continue
        ics[dt] = float(f_row[ok].corr(r_row[ok], method='spearman'))
    s = pd.Series(ics)
    if s.empty:
        continue
    e1 = s[s.index < ERA_SPLIT]
    e2 = s[s.index >= ERA_SPLIT]
    ic_summary[f] = {
        'n': int(s.size),
        'ic_2017_2021': float(e1.mean()) if len(e1) else None,
        'ic_2022_2026': float(e2.mean()) if len(e2) else None,
        'icir_2017_2021': float(e1.mean() / e1.std()) if len(e1) > 2 and e1.std() > 0 else None,
        'icir_2022_2026': float(e2.mean() / e2.std()) if len(e2) > 2 and e2.std() > 0 else None,
    }
    print(f"  {f}: 2017-21 IC={ic_summary[f]['ic_2017_2021']:+.3f} "
          f"ICIR={ic_summary[f]['icir_2017_2021']:+.2f} | "
          f"2022-26 IC={ic_summary[f]['ic_2022_2026']:+.3f} "
          f"ICIR={ic_summary[f]['icir_2022_2026']:+.2f}", flush=True)

# ── 保存工件（覆盖旧的无效工件） ──
extra = {
    'benchmark_series': {'dates': [str(d)[:10] for d in b.index],
     'values': [round(float(v / b.iloc[0] * rA.initial_capital), 2) for v in b.values]}
    } if bench else {}
extra['three_runs'] = {
    'A_生产': {'total_return': rA.total_return, 'sharpe': rA.sharpe,
               'terminal_date': str(rA.terminal_date) if rA.terminal_date else None,
               'active_period_return': rA.active_period_return,
               'idle_trading_days': rA.idle_trading_days},
    'B_自动重入': {'total_return': rB.total_return, 'sharpe': rB.sharpe,
                  'terminal_date': str(rB.terminal_date) if rB.terminal_date else None},
    'C_无闸门': {'total_return': rC.total_return, 'sharpe': rC.sharpe,
                'max_drawdown': rC.max_drawdown},
    '重入协议价值': rB.total_return - rA.total_return,
    '风控净贡献': rC.total_return - rA.total_return,
}
extra['rank_ic_by_era'] = ic_summary
save_artifact('portfolio', 'composite_v5_full_2017_2026_5因子生',
              portfolio_payload(rA, label='v2.1.0策略(唯一实现) 2017-2026全期',
                                extra=extra))
save_artifact('portfolio', 'equaltradable_v5_full_2017_2026_等权',
              portfolio_payload(rEW, label='可交易等权 2017-2026全期'))

# ═══════════════════════════════════════════════════
# 结果输出
# ═════════════════════════════════════════════════
print(f"""
{'='*72}
完整回测结果（{SIGNAL_START} ~ {END}，约9.8年，v2.1.0 唯一实现）
{'='*72}
┌──────────────────────────────────────────────────────────────────┐
│ 口径对比                                                         │
├──────────────────────────────────────────────────────────────────┤""")
for tag, r in [('A 生产（闸门+终态停机）', rA), ('B 自动重入', rB), ('C 无闸门', rC)]:
    term = f"终态{r.terminal_date}" if r.terminal_date else "全程运行"
    print(f"│ {tag:<18s} {r.total_return:+7.1%} 夏普{r.sharpe:6.2f} "
          f"回撤{r.max_drawdown:7.1%} | {term}")
print(f"│ 等权对照           {rEW.total_return:+7.1%} 夏普{rEW.sharpe:6.2f}")
print(f"│ 沪深300            {bench_ret:+7.1%}")
print(f"└──────────────────────────────────────────────────────────────────┘")

print(f"""
{'='*72}
收益拆解（对照用户验证清单第4条）
{'='*72}
  市场 beta（300 全期）:      {bench_ret:+.1%}
  A 选股超额（vs 可交易等权）: {rA.total_return - rEW.total_return:+.1%}
  A 选股超额（vs 300）:       {rA.total_return - bench_ret:+.1%}
  风控净贡献（C - A）:        {rC.total_return - rA.total_return:+.1%}
  重入协议价值（B - A）:      {rB.total_return - rA.total_return:+.1%}""")
if rA.terminal_date:
    cash = (1 + rA.total_return) / (1 + (rA.active_period_return or 0)) - 1
    print(f"  A 运行期收益:             {rA.active_period_return:+.1%}"
          f"（终态 {rA.terminal_date}，此后 {rA.idle_trading_days} 交易日持币"
          f" 收益 {cash:+.1%}）")
print(f"  A 交易成本:               {rA.total_cost / rA.initial_capital:.2%}")

# ── 年度收益拆解 ──
print(f"\n{'='*72}\n年度收益（A生产 / B重入 / C无闸门 / 等权 / 300）\n{'='*72}")
nav_y = nav.groupby(nav.index.year)
yl, yf = nav_y.last(), nav_y.first()
prev = None
print(f"{'年份':>6s} {'A生产':>7s} {'B重入':>7s} {'C无闸门':>7s} {'等权':>7s} {'300':>7s}")
for y in yl.index:
    base = 1.0 if prev is None else yl.loc[prev]
    row = yl.loc[y] / base - 1
    cells = ' '.join(f"{row.get(c, float('nan')):>7.1%}" for c in nav.columns)
    print(f"{y:>6d} {cells}")
    prev = y

# ── 滚动三年窗口 ──
print(f"\n{'='*72}\n滚动三年窗口（步长1年，A生产 / B重入 / 等权 / 300 与各自窗口起点比）\n{'='*72}")
idx = nav.index
cols = nav.columns.tolist()
print(f"{'窗口':>10s} " + ' '.join(f"{c:>8s}" for c in cols))
for sy in range(2017, 2025):
    mask = (idx.year >= sy) & (idx.year < sy + 3)
    if mask.sum() < 100:
        continue
    sub = nav[mask]
    rets = sub.iloc[-1] / sub.iloc[0] - 1
    print(f"{sy}-{sy+2:<4d} " + ' '.join(f"{rets[c]:>8.1%}" for c in cols))

print(f"\n净值数据: ai-created/nav_full_2017_2026.csv（5曲线）")
print(f"策略工件: results/data/portfolio_composite_v5_full_2017_2026_5因子生.json")
