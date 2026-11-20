#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
候选策略验证：质量过滤 + 价值合成 + 月度调仓 + 排名缓冲

假设登记：
  在流动性充足、财务质量较好的股票中寻找相对便宜的公司，
  通过较长持有期减少成本，检验是否存在扣费后的选股优势。

与当前5因子的区别：
  - 质量（ROE/现金流/杠杆）作为过滤条件，不是评分权重
  - EP/BP/SP 合成一个价值维度（避免三个相关指标占满权重）
  - 取消 Size=-1 小盘偏好（作为独立对照）
  - 月度调仓（信号月初更新）
  - 排名缓冲（入Top100/出Top150）

淘汰条件（预注册）：
  1. 毛收益（不扣费）必须为正——否则选股无效
  2. 必须跑赢同池等权对照——否则只是β不是α
  3. 成本翻倍后仍有正净收益——否则不够鲁棒
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import date, timedelta

import numpy as np
import pandas as pd

from data.layered import LayeredDataSource
from data.base.data_loader import DataLoader
from portfolio import composite_from_matrices
from portfolio.daily_replay import DailyReplayBacktester, ReplayConfig
from portfolio.tradable_benchmark import tradable_equal_weight
from trading import config as chain

BENCHES = ['000300.SH', '000905.SH', '000852.SH']
ERAS = [('fullmarket_2022_2026', date(2021, 11, 1), date(2026, 10, 9), date(2022, 1, 1))]


# ================= 因子构建 =================

def build_value_score(ds, pool, d0, d1):
    """价值合成：EP/BP/SP 各自标准化后等权合成一个维度"""
    from data.fundamentals import WindFundamentalProvider, build_fundamental_matrix
    prov = WindFundamentalProvider(ds)
    mats = {}
    for f in ['ep', 'bp', 'sp']:
        fm = build_fundamental_matrix(prov, f, pool, d0, d1)
        if not fm.empty:
            mats[f] = fm.loc[pd.Timestamp(d0):pd.Timestamp(d1)]
    # 等权合成（每个因子先做截面标准化，避免量纲差异）
    directions = {'ep': 1.0, 'bp': 1.0, 'sp': 1.0}
    return composite_from_matrices(mats, directions=directions, experimental=True)


def build_quality_filter(ds, pool, d0, d1):
    """
    质量过滤矩阵：True=通过质量筛选（可投资），False=不通过

    筛选条件（基于公告后可见的PIT数据）：
    - ROE > 8%（三年平均或最新，视数据可用性）
    - 经营现金流/总资产 > 0（正现金流）
    - 杠杆 < 行业中位数的1.5倍（暂用固定阈值60%负债率近似）
    """
    from data.fundamental_pit import PITFinancialProvider
    pit = PITFinancialProvider(ds)

    # ROE
    roe = pit.get_factor_matrix('roe', pool, d0, d1)
    if not roe.empty:
        roe = roe.loc[pd.Timestamp(d0):pd.Timestamp(d1)]
    # leverage（越低越好）
    lev = pit.get_factor_matrix('leverage', pool, d0, d1)
    if not lev.empty:
        lev = lev.loc[pd.Timestamp(d0):pd.Timestamp(d1)]

    # 构建布尔矩阵：所有条件同时满足
    dates = roe.index if not roe.empty else (lev.index if not lev.empty else pd.DatetimeIndex([]))
    if len(dates) == 0:
        return pd.DataFrame()

    symbols = sorted(set(roe.columns) | set(lev.columns))
    quality = pd.DataFrame(True, index=dates, columns=symbols)

    if not roe.empty:
        # ROE > 0.08（ PIT 因子值，低质量排除）
        quality = quality & (roe.reindex(index=dates, columns=symbols) > 0.08)
    if not lev.empty:
        # 杠杆（资产/权益倍数）< 3.0（约=负债率<67%，排除高杠杆）
        quality = quality & (lev.reindex(index=dates, columns=symbols) < 3.0)

    return quality


def apply_quality_filter(score_mx, quality_mx):
    """将质量过滤应用到评分矩阵：不通过筛选的标的分数设为NaN（不入选）"""
    if quality_mx.empty:
        return score_mx
    common_dates = score_mx.index.intersection(quality_mx.index)
    common_cols = score_mx.columns.intersection(quality_mx.columns)
    if len(common_dates) == 0 or len(common_cols) == 0:
        return score_mx
    score = score_mx.copy()
    q_aligned = quality_mx.reindex(index=common_dates, columns=common_cols)
    for dt in common_dates:
        fail = ~q_aligned.loc[dt].fillna(True)  # NaN=无数据→宽容放行
        if fail.any():
            score.loc[dt, fail[fail].index] = np.nan
    return score


# ================= 月度信号 =================

def monthly_signal(score_mx):
    """只在每月第一个交易日更新信号"""
    dates = score_mx.index
    monthly_dates = set()
    last_month = None
    for d in dates:
        m = (d.year, d.month)
        if m != last_month:
            monthly_dates.add(d)
            last_month = m
    return score_mx.loc[sorted(monthly_dates)]


# ================= 主实验 =================

def main():
    ds = LayeredDataSource()
    loader = DataLoader.__new__(DataLoader)
    loader.data_source_type = 'layered'
    loader.data_source = ds
    loader._cache = None

    era, d0, d1, ts = ERAS[0]
    print(f"===== {era} =====")
    pool = sorted(ds.list_symbols(as_of=ts))
    print(f"池 {len(pool)} 只", flush=True)
    cache = loader.preload_klines(pool + BENCHES, d0, d1)._cache

    # ---- 构建因子 ----
    print("构建价值合成分...", flush=True)
    value_score = build_value_score(ds, pool, d0, d1)
    value_score = value_score[value_score.columns.intersection(cache.keys())]
    print(f"价值分矩阵 {value_score.shape[0]}日 × {value_score.shape[1]}只", flush=True)

    print("构建质量过滤...", flush=True)
    quality = build_quality_filter(ds, pool, d0, d1)
    print(f"质量矩阵 {quality.shape[0]}日 × {quality.shape[1]}只", flush=True)

    # ---- 实验组 ----
    experiments = []

    # E1: 纯价值（无质量过滤，无小盘偏好）— 检验价值因子本身
    experiments.append(('E1', '纯价值合成（无小盘/无质量）',
                        value_score, chain.BUFFER_BAND, True, None))

    # E2: 质量过滤 + 价值（核心候选）
    value_quality = apply_quality_filter(value_score, quality)
    experiments.append(('E2', '质量过滤+价值合成（核心候选）',
                        value_quality, chain.BUFFER_BAND, True, None))

    # E3: E2 + 宽缓冲带（入100出150近似）
    experiments.append(('E3', '质量过滤+价值+宽缓冲50%',
                        value_quality, 0.50, True, None))

    # E4: E2 + 无止损（检验止损在质量策略中的作用）
    experiments.append(('E4', '质量过滤+价值+无止损',
                        value_quality, chain.BUFFER_BAND, True, 1.0))

    # E5: 当前5因子对照（含小盘偏好+杠杆）
    print("构建当前5因子对照分...", flush=True)
    from experiments.loss_attribution import build_scores as build_5f
    score_5f = build_5f(ds, pool, d0, d1)
    score_5f = score_5f[score_5f.columns.intersection(cache.keys())]
    experiments.append(('E5', '当前5因子对照（含小盘偏好）',
                        score_5f, chain.BUFFER_BAND, True, None))

    # ---- 运行实验 ----
    results = {}
    for name, desc, score, band, gate, sl in experiments:
        print(f"\n{'='*60}")
        print(f"实验 {name}: {desc}")
        print(f"{'='*60}", flush=True)

        # 月度信号
        score_m = monthly_signal(score)

        cfg = ReplayConfig(
            top_n=chain.TOP_N,
            band=band,
            use_risk_gate=gate,
            min_listing_days=chain.MIN_LISTING_DAYS,
            min_daily_turnover=chain.MIN_DAILY_TURNOVER,
        )
        if sl is not None and gate:
            from live.risk_gate import GateConfig
            cfg.gate_config = GateConfig(stop_loss_pct=sl)

        rp = DailyReplayBacktester(cache, cfg)
        r = rp.run(score_m, start_date=str(ts), end_date=str(d1))
        results[name] = r

        cost = r.total_cost / r.initial_capital
        gross = r.total_return + cost
        print(f"  净收益: {r.total_return:+.2%} | 成本: {cost:.2%} | 毛收益: {gross:+.2%}")
        print(f"  夏普: {r.sharpe:.2f} | 回撤: {r.max_drawdown:.2%} | 止损: {r.n_stop_sells}笔")
        if r.terminal_date:
            print(f"  ⚠ 终态: {r.terminal_date} (运行期{r.active_period_return:+.1%})")

    # ---- 对照：同池等权 ----
    print(f"\n{'='*60}")
    print("可交易等权对照")
    print(f"{'='*60}", flush=True)
    from portfolio.backtester import PortfolioConfig
    ctrl_cfg = PortfolioConfig(top_n=chain.TOP_N, rebalance_days=20)
    _, r_ew, _ = tradable_equal_weight(cache, ctrl_cfg, start_date=str(ts), end_date=str(d1))
    ew_cost = r_ew.total_cost / r_ew.initial_capital
    ew_gross = r_ew.total_return + ew_cost
    print(f"  净收益: {r_ew.total_return:+.2%} | 成本: {ew_cost:.2%} | 毛收益: {ew_gross:+.2%}")

    # ---- 汇总表 ----
    print(f"\n{'='*60}")
    print("候选策略汇总")
    print(f"{'='*60}")
    print(f"{'实验':>4s} {'描述':>24s} {'净收益':>7s} {'毛收益':>7s} "
          f"{'夏普':>5s} {'回撤':>6s} {'止损':>4s} {'终态':s}")

    for name, desc, _, _, _, _ in experiments:
        r = results[name]
        cost = r.total_cost / r.initial_capital
        gross = r.total_return + cost
        term = r.terminal_date[:7] if r.terminal_date else '无'
        print(f"  {name}  {desc:>24s} {r.total_return:>6.1%} {gross:>6.1%} "
              f"{r.sharpe:>5.2f} {r.max_drawdown:>5.1%} {r.n_stop_sells:>4d}  {term}")
    print(f"  EW   {'同池等权对照':>24s} {r_ew.total_return:>6.1%} {ew_gross:>6.1%} "
          f"{r_ew.sharpe:>5.2f} {r_ew.max_drawdown:>5.1%} {'—':>4s}  —")

    # ---- 淘汰判定 ----
    print(f"\n{'='*60}")
    print("淘汰判定（预注册条件）")
    print(f"{'='*60}")
    for name, desc, _, _, _, _ in experiments:
        r = results[name]
        cost = r.total_cost / r.initial_capital
        gross = r.total_return + cost
        # 条件1：毛收益为正
        c1 = '✅' if gross > 0 else '❌'
        # 条件2：跑赢等权
        c2 = '✅' if r.total_return > r_ew.total_return else '❌'
        # 条件3：成本翻倍后仍正
        c3 = '✅' if (gross - 2 * cost) > 0 else '❌'
        verdict = '通过' if (c1 == '✅' and c2 == '✅') else '淘汰'
        print(f"  {name}: 毛收益{c1} 跑赢等权{c2} 成本鲁棒{c3} → {verdict}")


if __name__ == '__main__':
    main()
