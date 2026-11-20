#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""试验C+D：vol targeting 替换 + 亏损股排除叠加 V2（预注册见 FACTOR_PREREG
风控替换+排除试验节）。3 runs：V2×vol_target / V2×guard×亏损排除 /
V2×vol_target×亏损排除。基线=V2×guard(+0.04%/月，已有)。"""
import sys, os
if __name__ == '__main__':
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from pathlib import Path

import pandas as pd

from data.preparation import atomic_json
from research import technical_signals as ts
from research.composite_study import PORTFOLIO_PROTOCOL, portfolio_layer
from research.factor_screen2 import PROTOCOL, load_ohlc_panel

DEST = Path('ai-created/research/risk_exclusion2_2026-11-09')


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    panel, base, evidence = load_ohlc_panel(PROTOCOL)
    price = base['price']
    eligible = (base['usable']
                & price.notna().rolling(PROTOCOL['minimum_history_sessions']).sum()
                .eq(PROTOCOL['minimum_history_sessions'])
                & base['amount'].rolling(PROTOCOL['liquidity_window'], min_periods=PROTOCOL['liquidity_window'])
                .median().ge(PROTOCOL['minimum_median_daily_amount_local_currency']))
    signals = ts.build_all_signals(panel)
    d16 = signals['D16_rel_atr']
    pb = pd.read_parquet('data_cache/fundamental/pb_lf.parquet')
    pe = pd.read_parquet('data_cache/fundamental/pe_ttm.parquet')
    bp = (1.0 / pb.where(pb > 0)).reindex(index=price.index, columns=price.columns)
    pe_a = pe.reindex(index=price.index, columns=price.columns)
    loss = pe_a.isna()                  # 试验D'排除：pe缺失=亏损/覆盖不足代理
    v2 = (d16.rank(axis=1, pct=True) + bp.rank(axis=1, pct=True)) / 2.0
    composite = {
        'P1_v2_pure': v2,
        'P2_v2_pure_excl': v2.where(~loss),
        'P3_v2_guard_excl': v2.where(~loss),
    }
    atomic_json(DEST / 'inputs.json', evidence)
    print(f"亏损排除日均剔除 {int((loss & eligible).sum(axis=1).mean())} 只", flush=True)

    protocol = dict(PORTFOLIO_PROTOCOL, study='risk_exclusion_phase2',
                    risk_variants=['vol_target_pure', 'budget_guard'])
    reports = portfolio_layer(composite, eligible, d16, base, evidence,
                              protocol, dest=DEST)
    # portfolio_layer 会跑 signals×variants 全积——T1 只需 vol_target，
    # T2 只需 guard，T3 只需 vol_target；全跑 6 个也全报告（无挑选）
    print('\n== 全部 run（N=50 月调仓 25bp）==')
    verdict = {}
    for name, full in reports.items():
        r = full['metrics']
        va = full['splits']['validation']['geometric_monthly_return']
        verdict[name] = {'monthly': r['geometric_monthly_return'],
                         'validation_monthly': va,
                         'max_dd': r['max_drawdown'],
                         'qualifies': full.get('qualifies_research_candidate'),
                         'accounting_valid': full.get('accounting_valid')}
        print(f"{name:<38} 月收益 {r['geometric_monthly_return']:+.2%} | "
              f"验证月 {va:+.2%} | maxDD {abs(r['max_drawdown']):.1%} | "
              f"qualifies={full.get('qualifies_research_candidate')} | "
              f"账务={full.get('accounting_valid')}")
    atomic_json(DEST / 'verdict.json', verdict)

    winners = [k for k, v in verdict.items()
               if v['monthly'] > 0.0008 and v['qualifies']]
    from research.factor_screen2 import register
    register(f"试验C'+D' 风控/排除: {winners or '无达标'}", label='risk_exclusion_phase2')
    print(f"\n判据（>2×基线0.08%且qualifies）: {winners or 'FAIL'}")
    print(f"证据: {DEST}")


if __name__ == '__main__':
    main()
