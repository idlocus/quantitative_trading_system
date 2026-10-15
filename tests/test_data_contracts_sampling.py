#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据契约测试层 —— 真实缓存抽样（评审 §2.5c，2026-10-15）

与合成数据的接口测试不同，本层对 data_cache/ 真实价格缓存抽样验证三类
"口径真值"（历史 4 起数据事故全部发生在此层）：

1. 复权比值：adjclose/close 的形态必须符合 Wind 后复权特征
   （≥1、随分红送转递增、末端=最大值、分段常数）。
   首轮实测即发现标签错误：文档曾写"前复权"，实测为后复权
   （收益计算等价，水平值语义不同）——这正是本测试层存在的意义。
2. 单位比值：amount/volume ≈ close（VWAP 量级）——历史事故：volume
   手/股错配（×100）与 amount 千元/元错配（×1000）在此现形。
3. 覆盖率：日期递增、无未来日期、价格列 0 NaN、行数充足。

缓存缺失时整文件跳过（研究环境无缓存时不阻断 CI）。
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CACHE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     'data_cache', 'kline')

pytestmark = pytest.mark.skipif(
    not os.path.isdir(CACHE) or not os.listdir(CACHE),
    reason="本地价格缓存不可用（data_cache/kline）——跳过真实数据契约抽检")


def _sample_files(n=30):
    files = sorted(os.listdir(CACHE))
    rng = np.random.RandomState(20261015)
    idx = rng.choice(len(files), size=min(n, len(files)), replace=False)
    return [files[i] for i in idx]


class TestAdjustedRatioContract:
    """复权比值契约：adjclose = Wind 后复权（S_DQ_ADJCLOSE）"""

    def test_ratio_shape_is_backward_adjusted(self):
        """后复权特征：比值>0、台阶式变化（除权日才跳）、末端=最大"""
        for f in _sample_files():
            df = pd.read_parquet(os.path.join(CACHE, f))
            if 'adjclose' not in df.columns or len(df) < 60:
                continue
            ratio = df['adjclose'] / df['close']
            assert (ratio > 0).all(), f"{f}: 复权比值非正"
            # 台阶式：相邻日变化仅出现在除权日。用相对阈值 0.5%——
            # adjclose 只有 2 位小数，低价股（~6元）的舍入噪声约 ±0.1%
            # 相对比值，绝对阈值会误报（600979 实测 528 次假台阶）
            steps = (ratio.pct_change().abs() > 0.005).sum()
            assert steps <= max(20, len(df) // 50), \
                f"{f}: 复权比值 {steps} 次台阶（行数{len(df)}）——" \
                f"疑似复权列被污染（应仅在除权日跳变）"
            # 后复权判别式：比值 ≥1（前复权锚定最新则 ≤1）。Wind 后复权因子
            # 自上市日起随分红送转累积，只增不减
            assert (ratio >= 1 - 1e-6).all(), \
                f"{f}: 复权比值 <1——形态不符后复权（若迁移为前复权请同步" \
                f"更新 semantics.get_price_column 文档与本测试）"
            # 末端≈最大（容差 0.5%：adjclose 2位小数在低价股上引入 ±0.1%
            # 舍入噪声，601111 实测末端偏差 0.07% 属噪声而非形态异常）
            assert abs(ratio.iloc[-1] - ratio.max()) / ratio.max() < 0.005, \
                f"{f}: 末端复权比值显著低于最大值——复权锚点异常"

    def test_raw_close_level_sane(self):
        """不复权收盘价水平合理（0.1~20000 元；下界容纳真实仙股如 600766
        历史最低 0.34 元——2026-10-15 抽样实测校准），与复权列分离"""
        for f in _sample_files():
            df = pd.read_parquet(os.path.join(CACHE, f))
            c = df['close'].dropna()
            if len(c) == 0:
                continue
            assert 0.1 <= c.min() and c.max() <= 20000, \
                f"{f}: 不复权价越界 [{c.min():.2f}, {c.max():.2f}]——" \
                f"疑似复权值混入 close 列（2026-10 第3轮事故形态）"


class TestUnitRatioContract:
    """单位比值契约：amount/volume ≈ close（VWAP 量级）"""

    def test_vwap_matches_close_within_bounds(self):
        for f in _sample_files():
            df = pd.read_parquet(os.path.join(CACHE, f))
            need = {'amount', 'volume', 'close'}
            if not need.issubset(df.columns):
                continue
            d = df.dropna(subset=['amount', 'volume', 'close'])
            d = d[(d['volume'] > 0) & (d['close'] > 0)]
            if len(d) < 60:
                continue
            vwap = d['amount'] / d['volume']
            rel = (vwap / d['close']).between(0.5, 2.0)
            assert rel.mean() > 0.98, \
                f"{f}: VWAP/close 仅{rel.mean():.0%}在[0.5,2]内——" \
                f"疑似 volume 手/股或 amount 千元/元单位错配（×100 形态）"


class TestCoverageContract:
    """覆盖率契约：日期有序无未来、价格列无 NaN"""

    def test_dates_ordered_and_no_future(self):
        import datetime as _dt
        today = pd.Timestamp(_dt.date.today())
        for f in _sample_files():
            df = pd.read_parquet(os.path.join(CACHE, f))
            dt = pd.to_datetime(df['date'])
            assert dt.is_monotonic_increasing, f"{f}: 日期非递增"
            assert dt.max() <= today + pd.Timedelta(days=1), \
                f"{f}: 存在未来日期 {dt.max()}——疑似数据源污染"

    def test_price_columns_no_nan(self):
        for f in _sample_files():
            df = pd.read_parquet(os.path.join(CACHE, f))
            for col in ('close', 'volume'):
                if col in df.columns:
                    assert df[col].notna().all(), \
                        f"{f}: {col} 存在 NaN——缓存迁移承诺 0 NaN（评审 §1.1）"
