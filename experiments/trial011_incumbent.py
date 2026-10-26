#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
试炼 011：现行价值/质量选股策略重验（新基线=PIT 滚动全市场宇宙首次上尺）
预注册见 docs/研究日志/011_现行价值质量重验.md

R1 合规：全部生产函数（composite_from_matrices / PITFinancialProvider /
trading.strategy._build_momentum_matrix / trading.config 参数），零私有实现。
宇宙：缓存全部沪深股票（主板+创业板+科创板；起点=上市或2016-06，
退市保留——覆盖扫描已验证为 PIT 滚动合格近似）；--bse 参数加入北交所
（消融臂）。判据：C口径两期年化均>0 + 全期夏普>0.4。
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from datetime import date
import numpy as np
import pandas as pd

from data.fundamentals import WindFundamentalProvider, build_fundamental_matrix
from data.fundamental_pit import PITFinancialProvider
from data.semantics import close_matrix
from data.st_provider import STProvider
from portfolio.construction import composite_from_matrices
from portfolio.daily_replay import DailyReplayBacktester, ReplayConfig
from trading import config as chain
from trading.strategy import _build_momentum_matrix

INCLUDE_BSE = '--bse' in sys.argv
USE_GATE = '--gate' in sys.argv          # A口径=生产闸门(15%熔断+立即重入+熔断器)
SOFT = HARD = None                       # 014: --soft 0.08 --hard 0.20 覆盖闸门阈值
for a in sys.argv:
    if a.startswith('--soft'): SOFT = float(a.split('=')[1])
    if a.startswith('--hard'): HARD = float(a.split('=')[1])
START, END = date(2016, 6, 1), date(2026, 10, 15)
SIGNAL_START = '2017-01-03'
ERA_SPLIT = '2021-12-31'

# ── 1. 宇宙与价格（直读 parquet，覆盖已扫描验证）──
cache = {}
for fn in os.listdir('data_cache/kline'):
    if not fn.endswith('.parquet'):
        continue
    code, suf = fn[:-8].split('_')
    if suf == 'SH' and code[0] == '6':
        sym = code + '.SH'
    elif suf == 'SZ' and code[0] in '03':
        sym = code + '.SZ'
    elif INCLUDE_BSE and suf == 'BJ':
        sym = code + '.BJ'
    else:
        continue
    try:
        df = pd.read_parquet(f'data_cache/kline/{fn}')
        if 'adjclose' in df.columns and len(df):
            df['date'] = pd.to_datetime(df['date'])
            df = df[(df['date'] >= pd.Timestamp(START)) & (df['date'] <= pd.Timestamp(END))]
            if len(df):
                cache[sym] = df
    except Exception:
        continue
# 基准必须入缓存：闸门 L2 检查依赖基准数据（缺失=每日HALT）；
# 指数无基本面→分数 NaN→不会入选，仅作闸门/参照用途
bfn = 'data_cache/kline/000300_SH.parquet'
if USE_GATE and os.path.exists(bfn):
    df = pd.read_parquet(bfn)
    df['date'] = pd.to_datetime(df['date'])
    df = df[(df['date'] >= pd.Timestamp(START)) & (df['date'] <= pd.Timestamp(END))]
    if len(df):
        cache['000300.SH'] = df

pool = sorted(k for k in cache if k != '000300.SH')
print(f'宇宙: {len(pool)} 只（{"含" if INCLUDE_BSE else "不含"}北交所）', flush=True)

px = close_matrix(cache, basis='adjusted')

# ── 2. 生产管道（R1：全部生产函数）──
from data.layered import LayeredDataSource
ds = LayeredDataSource()
prov = WindFundamentalProvider(ds)
mats = {}
for f in chain.PRODUCTION_FACTORS:
    fm = build_fundamental_matrix(prov, f, pool, START, END)
    if not fm.empty:
        mats[f] = fm
print(f'因子矩阵: {sorted(mats)}', flush=True)

pit = PITFinancialProvider(ds)
value = composite_from_matrices(mats, directions=chain.PRODUCTION_FACTORS)
value = value[value.columns.intersection(cache.keys())]
roe = pit.get_factor_matrix('roe', pool, START, END)
lev = pit.get_factor_matrix('leverage', pool, START, END)
snv = value.copy()
for mx_, thr, is_min in ((roe, chain.QUALITY_FILTER['roe_min'], True),
                         (lev, chain.QUALITY_FILTER['leverage_max'], False)):
    m = mx_.reindex(index=snv.index, columns=snv.columns)
    snv = snv.mask(m < thr) if is_min else snv.mask(m > thr)

def zrow(mx):
    return mx.sub(mx.mean(axis=1), axis=0).div(mx.std(axis=1) + 1e-12, axis=0)

v_z = zrow(snv)
mom = _build_momentum_matrix(cache, START, END)
# 生产口径（2026-10-26 勘误#4修正）：动量缺失保留 NaN（缺动量→不可入选），
# 与 trading.strategy._combine_value_momentum 一致；此前 fillna(0) 为变体
mom_z = zrow(mom.reindex_like(v_z))
score = zrow((1 - chain.MOMENTUM_WEIGHT) * v_z + chain.MOMENTUM_WEIGHT * mom_z)
print(f'分数矩阵: {score.shape} | 有效行均值 {score.notna().sum(axis=1).mean():.0f}', flush=True)

# 公司行动（勘误#1修正）：除权除息日对持仓入账分红现金与送转股数
from data.corporate_actions import load_corporate_actions
ca = load_corporate_actions(ds, pool, start='20160101', end='20261231')
print(f'公司行动事件: {len(ca)} 条（现金分红 {int((ca.cash_ps>0).sum())} / 送转 {int((ca.stk_ps>0).sum())}）', flush=True)

idx_dates = pd.DatetimeIndex(sorted({d for df in cache.values() for d in df['date']}))
st_matrix = None
try:
    st_matrix = STProvider(ds).st_matrix(idx_dates, pool)
except Exception as e:
    print(f'⚠️ ST矩阵失败: {str(e)[:80]}', flush=True)

gate_cfg = None
if USE_GATE:
    from live.risk_gate import GateConfig
    # 定稿重入协议：清仓→立即现金锚满仓重入；365日>REENTRY_MAX_LIQUIDATIONS次→永久停机
    kw = {'restart_after_liquidation': True}
    if SOFT is not None: kw['drawdown_soft'] = SOFT
    if HARD is not None: kw['drawdown_hard'] = HARD
    gate_cfg = GateConfig(**kw)
cfg = ReplayConfig(top_n=chain.TOP_N, st_matrix=st_matrix, band=chain.BUFFER_BAND,
                   use_risk_gate=USE_GATE, st_data_required=False, gate_config=gate_cfg)
print('重放中（含公司行动，逐日带宽调仓=生产语义）...', flush=True)
r = DailyReplayBacktester(cache, cfg, corporate_actions=ca).run(
    score, start_date=SIGNAL_START, end_date=str(END))
eq = r.equity_curve

# ── 3. 统计 ──
def ann_ret(seg):
    yrs = (seg.index[-1] - seg.index[0]).days / 365.25
    return float(seg.iloc[-1] / seg.iloc[0]) ** (1 / yrs) - 1 if yrs > 0.3 else np.nan

def mdd(seg):
    return float((seg / seg.cummax() - 1).min())

full_ann = ann_ret(eq)
e1 = eq[eq.index <= ERA_SPLIT]; e2 = eq[eq.index > ERA_SPLIT]
e1_ann, e2_ann = ann_ret(e1), ann_ret(e2)
yr_ret = eq.groupby(eq.index.year).last() / eq.groupby(eq.index.year).first() - 1
worst = float(yr_ret.min())

mode = ("A口径生产闸门" if USE_GATE else "C口径") + ("，含北交所" if INCLUDE_BSE else "")
print(f'\n{"="*72}')
print(f'011 现行价值/质量重验（{mode}）')
print(f'  全期: {r.total_return:+.1%} | 年化 {full_ann:+.2%} | 夏普 {r.sharpe:.2f} | 回撤 {mdd(eq):.1%}')
if USE_GATE:
    print(f'  闸门事件: 停机 {getattr(r, "n_halt_days", "?")} 天 | 禁买 '
          f'{getattr(r, "n_no_buy_days", "?")} 天 | 清仓 '
          f'{getattr(r, "n_liquidations", "?")} 次')
    apr = getattr(r, 'active_period_return', None)
    if apr is not None:
        print(f'  运行期收益（剔除停机期现金）: {apr:+.1%}')
print(f'  两期: 2017-21 {e1_ann:+.2%} | 2022-26 {e2_ann:+.2%}')
print(f'  最差单年: {worst:+.1%}（{yr_ret.idxmin()}）')
print(f'  分年度: ' + ' '.join(f'{y}:{v:+.0%}' for y, v in yr_ret.items()))

# 诊断：同宇宙全池等权（无选股能力）+ 指数参照
mret = px.pct_change().mean(axis=1)
mret_m = (1 + mret).groupby(mret.index.to_period('M')).prod() - 1
mret_m = mret_m[mret_m.index >= pd.Period('2017-01', 'M')]
def era_ann_monthly(x, a, b):
    seg = x[(x.index >= pd.Period(a, 'M')) & (x.index <= pd.Period(b, 'M'))]
    return float((1 + seg).prod() ** (12 / len(seg)) - 1) if len(seg) else np.nan
print(f'  诊断·全池等权: 全期 {float((1+mret_m).prod()**(12/len(mret_m))-1):+.2%} | '
      f'e1 {era_ann_monthly(mret_m, "2017-01", "2021-12"):+.2%} | '
      f'e2 {era_ann_monthly(mret_m, "2022-01", "2026-10"):+.2%}')
for sym, nm in (('000300.SH', '沪深300'), ('000905.SH', '中证500'), ('000852.SH', '中证1000')):
    b = pd.read_parquet(f'data_cache/kline/{sym.split(".")[0]}_SH.parquet')
    b['date'] = pd.to_datetime(b['date'])
    s = b.set_index('date')['close']
    s = s[(s.index >= SIGNAL_START) & (s.index <= str(END))]
    print(f'  参照·{nm}: 全期年化 {ann_ret(s):+.2%} | e1 {ann_ret(s[s.index<=ERA_SPLIT]):+.2%} | '
          f'e2 {ann_ret(s[s.index>ERA_SPLIT]):+.2%}')

pass_ = (e1_ann > 0 and e2_ann > 0 and r.sharpe > 0.4)
print(f'\n判据: 两期同号为正 {e1_ann>0 and e2_ann>0} | 夏普>0.4 {r.sharpe>0.4} → {"PASS" if pass_ else "FAIL"}')

# ── 4. 登记 ──
from datetime import datetime
reg_p = 'results/data/trial_registry.json'
reg = json.load(open(reg_p, encoding='utf-8'))
reg['trials'].append({'label': f'011-现行价值质量重验({"+北交所" if INCLUDE_BSE else "不含北交所"},C口径)',
                      'ran_at': datetime.now().isoformat(timespec='seconds'),
                      'full_ann': full_ann, 'sharpe': r.sharpe, 'mdd': mdd(eq),
                      'e1_ann': e1_ann, 'e2_ann': e2_ann, 'worst_year': worst,
                      'verdict': 'PASS' if pass_ else 'FAIL'})
reg['count'] = len(reg['trials'])
tmp = reg_p + '.tmp'
with open(tmp, 'w', encoding='utf-8') as f:
    json.dump(reg, f, ensure_ascii=False, indent=1)
os.replace(tmp, reg_p)
eq.to_csv(f'ai-created/trial011_equity_{"bse" if INCLUDE_BSE else "main"}.csv', encoding='utf-8-sig')
print(f'登记册: {reg["count"]} 条')
