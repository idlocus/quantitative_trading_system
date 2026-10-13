#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
K线缓存单位迁移（一次性）—— 修复历史缓存中 volume 单位混杂（DB行=手 vs 客户端行=股）

方法（精确比对，无近似）：
  对每只股票，用 DB ASHAREEODPRICES 的 S_DQ_VOLUME 逐日比对缓存行：
    ratio = 缓存volume / DB volume
    ratio≈1   → 该行来自DB（手）→ volume ×100
    ratio≈100 → 该行已是股     → 不变
  同时从 DB 回填 amount 列（S_DQ_AMOUNT 千元 ×1000 → 元）。
  指数（DB无行的日期）不参与成交额过滤，跳过判定。

用法: python scripts/migrate_kline_units.py [--dry-run] [--backup]
"""
import argparse
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from data.layered import LayeredDataSource

CACHE_DIR = Path('data_cache/kline')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry-run', action='store_true', help='只统计不写回')
    ap.add_argument('--backup', action='store_true', help='迁移前备份目录')
    args = ap.parse_args()

    ds = LayeredDataSource()
    db = ds._db_src()
    if not db:
        print("DB 不可用，无法迁移")
        sys.exit(1)

    files = sorted(CACHE_DIR.glob('*.parquet'))
    print(f"缓存文件 {len(files)} 个")
    if args.backup and not args.dry_run:
        bak = CACHE_DIR.parent / 'kline_backup_units'
        if not bak.exists():
            shutil.copytree(CACHE_DIR, bak)
            print(f"已备份 → {bak}")

    stats = {'scaled': 0, 'already_shares': 0, 'no_db_ref': 0, 'errors': 0}
    cur = db._cursor()
    for i, f in enumerate(files):
        sym = f.stem.replace('_', '.')
        try:
            df = pd.read_parquet(f)
            if df is None or df.empty or 'volume' not in df.columns:
                continue
            dts = df['date'].astype(str).str.replace('-', '')
            cur.execute(
                "SELECT TRADE_DT, S_DQ_VOLUME, S_DQ_AMOUNT FROM ASHAREEODPRICES "
                "WHERE S_INFO_WINDCODE = :1 AND TRADE_DT BETWEEN :2 AND :3",
                {'1': sym, '2': dts.min(), '3': dts.max()})
            ref = pd.DataFrame(cur.fetchall(), columns=['dt', 'db_vol', 'db_amt'])
            if ref.empty:
                stats['no_db_ref'] += 1
                continue
            ref['dt'] = ref['dt'].astype(str)
            ref = ref.drop_duplicates('dt', keep='last')   # DB版本重复行防merge膨胀
            m = df.assign(dt=dts.values).merge(ref, on='dt', how='left')
            db_vol = pd.to_numeric(m['db_vol'], errors='coerce')
            cache_vol = pd.to_numeric(m['volume'], errors='coerce')
            ratio = cache_vol / db_vol

            hand_rows = ((ratio > 0.5) & (ratio < 1.5)).values     # 缓存=手（DB原值）
            n_hand = int(hand_rows.sum())

            changed = False
            if hand_rows.any():
                df['volume'] = cache_vol.mask(pd.Series(hand_rows, index=cache_vol.index),
                                               cache_vol * 100).values
                changed = True
            # amount 回填（千元×1000→元），等长向量化（DB缺该日→保持原值/NaN）
            amt_full = (pd.to_numeric(m['db_amt'], errors='coerce') * 1000).values
            has_amt = ~pd.isna(amt_full) & (amt_full > 0)
            if 'amount' not in df.columns:
                df['amount'] = float('nan')
            keep = (df['amount'].notna() & (df['amount'] != 0)).values | ~has_amt
            df['amount'] = df['amount'].where(keep, pd.Series(amt_full).values)

            # 顺手剔除历史残缺行（只有close、OHLC/volume全NaN的污染bar）
            core = [c for c in ('open', 'high', 'low', 'close', 'volume') if c in df.columns]
            before = len(df)
            df = df.dropna(subset=core)
            if len(df) < before:
                changed = True

            if changed:
                stats['scaled'] += 1 if n_hand else 0
                stats['already_shares'] += 0 if n_hand else 1
                if not args.dry_run:
                    df.to_parquet(f, index=False)
            if (i + 1) % 200 == 0:
                print(f"  {i + 1}/{len(files)} ...")
        except Exception as e:
            stats['errors'] += 1
            print(f"  {sym}: {e}")
    cur.close()
    print(f"结果: DB手单位修复 {stats['scaled']} 只 | 已是股 {stats['already_shares']} 只 | "
          f"无DB参照(跳过) {stats['no_db_ref']} 只 | 错误 {stats['errors']} 只"
          + ("（dry-run 未写回）" if args.dry_run else ""))


if __name__ == '__main__':
    main()
