#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
K线缓存全量迁移 v2：修复价格基准断裂

旧格式：close = S_DQ_ADJCLOSE（前复权，错——实盘应用不复权）
新格式：close = S_DQ_CLOSE（不复权真实市场价）+ adjclose = S_DQ_ADJCLOSE

方法：对每个缓存文件，从DB批量查询 S_DQ_CLOSE 和 S_DQ_ADJCLOSE，
替换 close 列为不复权价，新增 adjclose 列。
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from pathlib import Path

CACHE_DIR = Path('data_cache/kline')
BATCH = 200


def migrate_batch(db, files):
    """批量迁移一组缓存文件"""
    cur = db._cursor()
    migrated, errors = 0, 0
    for f in files:
        sym = f.stem.replace('_', '.')
        try:
            df = pd.read_parquet(f)
            if df is None or df.empty:
                continue
            if 'adjclose' in df.columns:
                continue  # 已是新格式

            # 查DB获取正确的不复权价和前复权价（Oracle IN子句限1000，分批查）
            dts = df['date'].dt.strftime('%Y%m%d').tolist()
            ref = {}
            for ci in range(0, len(dts), 900):
                chunk = dts[ci:ci + 900]
                placeholders = ','.join(f':d{i}' for i in range(len(chunk)))
                binds = {f'd{i}': d for i, d in enumerate(chunk)}
                cur.execute(f"""
                    SELECT TRADE_DT, CAST(S_DQ_CLOSE AS FLOAT), CAST(S_DQ_ADJCLOSE AS FLOAT)
                    FROM AShareEODPrices
                    WHERE S_INFO_WINDCODE = :sym AND TRADE_DT IN ({placeholders})
                """, {**binds, 'sym': sym})
                for r in cur.fetchall():
                    ref[r[0]] = (r[1], r[2])

            if not ref:
                # DB无此股（可能已退市/指数），用close作adjclose退化
                df['adjclose'] = df['close']
                df.to_parquet(f, index=False)
                migrated += 1
                continue

            # 替换：close→不复权，adjclose→前复权
            new_close, new_adj = [], []
            for _, row in df.iterrows():
                dt = row['date'].strftime('%Y%m%d')
                if dt in ref:
                    c, ac = ref[dt]
                    new_close.append(c if c else row['close'])
                    new_adj.append(ac if ac else row['close'])
                else:
                    new_close.append(row['close'])
                    new_adj.append(row['close'])
            df['close'] = new_close
            df['adjclose'] = new_adj
            df.to_parquet(f, index=False)
            migrated += 1
        except Exception as e:
            errors += 1
            if errors <= 3:
                print(f"  ERROR {sym}: {e}")
    cur.close()
    return migrated, errors


def main():
    from data.layered import LayeredDataSource
    ds = LayeredDataSource()
    db = ds._db_src()
    if not db:
        print("DB不可用")
        return

    files = sorted(CACHE_DIR.glob('*.parquet'))
    old_format = [f for f in files if 'adjclose' not in pd.read_parquet(f).columns]
    print(f"总缓存 {len(files)} | 旧格式 {len(old_format)} | 新格式 {len(files) - len(old_format)}")

    if not old_format:
        print("全部已迁移")
        return

    total_migrated, total_errors = 0, 0
    t0 = time.time()
    for i in range(0, len(old_format), BATCH):
        batch = old_format[i:i + BATCH]
        m, e = migrate_batch(db, batch)
        total_migrated += m
        total_errors += e
        done = min(i + BATCH, len(old_format))
        print(f"  {done}/{len(old_format)} | 迁移{total_migrated} 错误{total_errors} "
              f"| {time.time() - t0:.0f}s", flush=True)

    # 验证
    sample = old_format[:10]
    ok = sum(1 for f in sample if 'adjclose' in pd.read_parquet(f).columns)
    print(f"\n完成：迁移{total_migrated} 错误{total_errors} | 验证 {ok}/10 有adjclose")


if __name__ == '__main__':
    main()
