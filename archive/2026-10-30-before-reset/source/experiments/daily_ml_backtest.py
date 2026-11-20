#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
日频 ML 策略回测：LightGBM + 44量价因子 vs 等权对照
滚动训练（每月重训），日度调仓，真实成本
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datetime import date
import numpy as np
import pandas as pd

from data.layered import LayeredDataSource
from data.base.data_loader import DataLoader
from data.daily_factors import build_all_factors
from research.ml_model import DailyMLModel
from portfolio.backtester import PortfolioBacktester, PortfolioConfig
from research.artifacts import save_artifact, portfolio_payload
from utils.logger import get_logger

logger = get_logger(__name__)

BENCH = '000300.SH'
POOL_SIZE = 500
TOP_N = 50
TRAIN_MONTHS = 12
PRED_HORIZON = 5
RETRAIN_FREQ_DAYS = 20  # 每20个交易日重训一次
START, END = date(2024, 1, 1), date(2026, 9, 24)


def build_matrices(cache):
    """从缓存构建 OHLCV+Amount 矩阵"""
    highs, lows, opens, vols, closes = {}, {}, {}, {}, {}
    for sym, df in cache.items():
        if sym == BENCH or df is None or df.empty:
            continue
        df2 = df.set_index('date')
        closes[sym] = pd.to_numeric(df2['close'], errors='coerce')
        highs[sym] = pd.to_numeric(df2['high'], errors='coerce')
        lows[sym] = pd.to_numeric(df2['low'], errors='coerce')
        opens[sym] = pd.to_numeric(df2['open'], errors='coerce')
        vols[sym] = pd.to_numeric(df2['volume'], errors='coerce')

    idx = None
    for d in closes.values():
        idx = d.index if idx is None else idx.union(d.index)

    def to_mx(d):
        return pd.DataFrame(d).reindex(idx).sort_index()

    close, high, low, open_ = to_mx(closes), to_mx(highs), to_mx(lows), to_mx(opens)
    volume = to_mx(vols)
    amount = close * volume
    return close, high, low, open_, volume, amount


def main():
    ds = LayeredDataSource()
    loader = DataLoader.__new__(DataLoader)
    loader.data_source_type = 'layered'; loader.data_source = ds; loader._cache = None

    print(f"加载 {POOL_SIZE} 只全市场数据（{START} ~ {END}）...", flush=True)
    pool = sorted(ds.list_symbols(as_of=START))[:POOL_SIZE]
    cache = loader.preload_klines(pool + [BENCH], START, END)._cache
    bench = cache[BENCH].set_index('date')['close'] if BENCH in cache else None

    print("构建价格矩阵...", flush=True)
    close, high, low, open_, volume, amount = build_matrices(cache)
    print(f"矩阵: {close.shape[0]} 天 × {close.shape[1]} 只", flush=True)

    print("构建 44 个量价因子...", flush=True)
    factors = build_all_factors(close, high, low, open_, volume, amount, bench)
    print(f"因子数: {len(factors)}", flush=True)

    # ── 滚动 ML 回测 ──
    print("\n滚动 ML 回测...", flush=True)
    ml = DailyMLModel(train_months=TRAIN_MONTHS, pred_horizon=PRED_HORIZON, top_n=TOP_N)
    all_dates = close.index
    score_dict = {}
    last_train_idx = -RETRAIN_FREQ_DAYS  # 强制第一次就训练

    for i, d in enumerate(all_dates):
        if i < TRAIN_MONTHS * 21:  # 前 TRAIN_MONTHS 个月是训练期，不产生信号
            continue
        if i - last_train_idx >= RETRAIN_FREQ_DAYS:
            # 重训模型
            if ml.train(factors, close, d):
                last_train_idx = i
                logger.info(f"[{d.date()}] model retrained")

        # 每日预测
        pred = ml.predict(factors, d)
        if pred is not None and len(pred) >= TOP_N:
            # 取 TopN，转换为分数矩阵格式（越高越好）
            top = pred.head(TOP_N)
            score_dict[d] = top

    if not score_dict:
        print("无预测结果")
        return

    # 构建 score matrix
    score_mx = pd.DataFrame(score_dict).T
    score_mx = score_mx[sorted(score_mx.columns)]
    # 前向填充到日频（非预测日沿用最近一次预测）
    daily_idx = all_dates[all_dates >= score_mx.index[0]]
    score_mx = score_mx.reindex(daily_idx).ffill()
    print(f"预测覆盖: {score_mx.shape[0]} 天 × {score_mx.shape[1]} 只", flush=True)

    # ── 回测 ──
    bt_cache = {sym: cache[sym] for sym in score_mx.columns if sym in cache}
    bt_cache[BENCH] = cache[BENCH]

    combos = [
        ('ML日频', score_mx, TOP_N),
        ('等权对照', pd.DataFrame(1.0, index=score_mx.index, columns=score_mx.columns), min(500, len(score_mx.columns))),
    ]

    for label, score, topn in combos:
        bt = PortfolioBacktester(bt_cache, PortfolioConfig(
            top_n=topn, rebalance_days=5, slippage=0.001,
            benchmark_symbol=BENCH))
        r = bt.run(score)
        b = bt.close_mx[BENCH].loc[r.equity_curve.index].ffill()
        extra = {'benchmark_series': {'dates': [str(x)[:10] for x in b.index],
                'values': [round(float(v/b.iloc[0]*r.initial_capital),2) for v in b.values]}}
        save_artifact('portfolio', f'daily_ml_{label[:2]}',
                      portfolio_payload(r, label=label, extra=extra))
        print(f"  {label}: {r.total_return:+.1%} 夏普{r.sharpe:.2f} "
              f"回撤{r.max_drawdown:.1%} 换手{r.avg_turnover:.0%}/次", flush=True)

    # ── 特征重要性 ──
    if ml.model is not None:
        imp = ml.feature_importance()
        print("\n特征重要性 Top10:")
        for name, v in imp.head(10).items():
            print(f"  {name}: {v}")


if __name__ == '__main__':
    main()
