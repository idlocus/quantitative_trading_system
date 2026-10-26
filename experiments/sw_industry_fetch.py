#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
申万一级行业数据抓取（试炼 003 数据前置；不接触收益数据之外的加工）

来源：akshare 申万宏源官方
  - index_hist_sw(code, 'day')：行业指数日线（31 只，逐只抓）
  - index_analysis_daily_sw('一级行业', start, end)：日频估值（PE/PB/股息率，
    按年分块 2012-2026）
落盘：data_cache/sw_industry/{price_{code}.parquet, valuation.parquet}
      + manifest.json（原子写，断点续传）
"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import pandas as pd
import akshare as ak

OUT = 'data_cache/sw_industry'
os.makedirs(OUT, exist_ok=True)
MANIFEST = f'{OUT}/manifest.json'
YEARS = range(2012, 2027)

def save_manifest(m):
    tmp = MANIFEST + '.tmp'
    json.dump(m, open(tmp, 'w', encoding='utf-8'))
    os.replace(tmp, MANIFEST)

man = json.load(open(MANIFEST, encoding='utf-8')) if os.path.exists(MANIFEST) else {'prices': [], 'val_years': []}

# ── 1. 行业清单 ──
info = ak.sw_index_first_info()
codes = info['行业代码'].str.split('.').str[0].tolist()
names = dict(zip(info['行业代码'].str.split('.').str[0], info['行业名称']))
print(f'一级行业 {len(codes)} 个', flush=True)
json.dump({'codes': codes, 'names': names}, open(f'{OUT}/industries.json', 'w', encoding='utf-8'), ensure_ascii=False)

# ── 2. 指数日线 ──
for code in codes:
    if code in man['prices']:
        continue
    for attempt in range(3):
        try:
            df = ak.index_hist_sw(symbol=code, period='day')
            df.to_parquet(f'{OUT}/price_{code}.tmp.parquet')
            os.replace(f'{OUT}/price_{code}.tmp.parquet', f'{OUT}/price_{code}.parquet')
            man['prices'].append(code)
            save_manifest(man)
            print(f'{code} {names[code]}: {len(df)} 行 {df["日期"].min()} ~ {df["日期"].max()}', flush=True)
            break
        except Exception as e:
            if attempt == 2:
                print(f'{code}: ⚠️ 三次失败 {str(e)[:80]}', flush=True)
            time.sleep(2 * (attempt + 1))
    time.sleep(0.5)

# ── 3. 日频估值（按年分块）──
parts = []
for y in YEARS:
    tag = str(y)
    if tag in man['val_years']:
        continue
    for attempt in range(3):
        try:
            df = ak.index_analysis_daily_sw(symbol='一级行业',
                                            start_date=f'{y}0101', end_date=f'{y}1231')
            if len(df):
                parts.append(df)
            man['val_years'].append(tag)
            save_manifest(man)
            print(f'{y}: {len(df)} 行', flush=True)
            break
        except Exception as e:
            if attempt == 2:
                print(f'{y}: ⚠️ 三次失败 {str(e)[:80]}', flush=True)
            else:
                time.sleep(2 * (attempt + 1))
    time.sleep(0.5)

if os.path.exists(f'{OUT}/valuation.parquet'):
    parts.append(pd.read_parquet(f'{OUT}/valuation.parquet'))
if parts:
    val = pd.concat(parts, ignore_index=True).drop_duplicates(
        subset=['指数代码', '发布日期'], keep='last')
    val.to_parquet(f'{OUT}/valuation.tmp.parquet')
    os.replace(f'{OUT}/valuation.tmp.parquet', f'{OUT}/valuation.parquet')
    print(f'\n估值合并: {len(val)} 行 | {val["发布日期"].min()} ~ {val["发布日期"].max()}', flush=True)
    n_ind = val['指数代码'].nunique()
    print(f'覆盖行业数: {n_ind} | 各行业日数: '
          f'{val.groupby("指数代码").size().describe()[["min","max"]].to_dict()}', flush=True)
