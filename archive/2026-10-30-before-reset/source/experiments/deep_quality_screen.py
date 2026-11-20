#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
深度质量因子跨年代筛选：4个新因子 × 两年代 × {原始, 中性化}
验收标准（Phase B 教训）：中性化残差 IC 在两个年代同号且 |IR|>0.10
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datetime import date
import pandas as pd

from data.layered import LayeredDataSource
from data.base.data_loader import DataLoader
from data.fundamentals import WindFundamentalProvider
from data.fundamental_pit import PITFinancialProvider, PIT_FINANCIAL_FACTORS
from research.factor_eval import FactorEvaluator
from research.risk_model import style_exposures, neutralize

NEW_FACTORS = ['salecash', 'asset_g', 'ocf_to_rev', 'leverage']
ERAS = [('2017-2021', date(2016,11,1), date(2021,12,31), date(2017,1,1)),
        ('2022-2026', date(2021,11,1), date(2026,9,24), date(2022,1,1))]


def main():
    ds = LayeredDataSource()
    loader = DataLoader.__new__(DataLoader); loader.data_source_type='layered'
    loader.data_source = ds; loader._cache=None
    pit = PITFinancialProvider(ds)
    results = {}

    for era, d0, d1, ts in ERAS:
        print(f"===== {era} =====", flush=True)
        pool = sorted(ds.list_symbols(as_of=ts))[:2000]
        cache = loader.preload_klines(pool, d0, d1)._cache
        ev = FactorEvaluator(cache)
        mv = WindFundamentalProvider(ds).get_field_matrix('mkt_cap_ard', pool, d0, d1)
        expo = style_exposures(ev.close_mx, mv)
        results[era] = {}
        for f in NEW_FACTORS:
            mx = pit.get_factor_matrix(f, pool, d0, d1)
            if mx.empty:
                print(f"  {f}: 空矩阵"); continue
            r_raw = ev.evaluate_matrix(mx, name=f, period=60)
            mx_n = neutralize(mx, expo)
            r_neu = ev.evaluate_matrix(mx_n, name=f+'_n', period=60)
            if r_raw is None or r_neu is None:
                print(f"  {f}: 数据不足"); continue
            _, direction, _ = PIT_FINANCIAL_FACTORS[f]
            # 方向校正后的有效IC（方向×原始IC）
            results[era][f] = {
                'raw': direction * r_raw.rankic_mean,
                'raw_ir': direction * r_raw.rankic_ir,
                'neu': direction * r_neu.rankic_mean,
                'neu_ir': direction * r_neu.rankic_ir,
            }
            print(f"  {f:10s}: 原始IC={results[era][f]['raw']:+.4f}(IR{results[era][f]['raw_ir']:+.2f}) "
                  f"中性IC={results[era][f]['neu']:+.4f}(IR{results[era][f]['neu_ir']:+.2f})", flush=True)

    # 跨年代验收表
    print("\n===== 跨年代稳定性验收（中性化残差，方向校正后）=====")
    print(f"{'因子':<12}{'2017-21中性IC':>14}{'2022-26中性IC':>14}{'同号?':>7}{'双段IR>0.1?':>12}")
    for f in NEW_FACTORS:
        a = results.get('2017-2021', {}).get(f)
        b = results.get('2022-2026', {}).get(f)
        if not (a and b):
            continue
        same = a['neu'] * b['neu'] > 0
        both_ir = (abs(a['neu_ir']) > 0.10) and (abs(b['neu_ir']) > 0.10)
        print(f"{f:<12}{a['neu']:>+14.4f}{b['neu']:>+14.4f}{'✓' if same else '✗':>7}{'✓' if same and both_ir else '✗':>12}")


if __name__ == '__main__':
    main()
