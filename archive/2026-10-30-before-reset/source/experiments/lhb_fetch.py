#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
龙虎榜数据抓取（试炼 002 数据前置；不接触收益数据）

来源：akshare 东财接口
  - stock_lhb_detail_em：每日龙虎榜个股明细（净买额/买入额/市场总成交额/涨跌幅…）
  - stock_lhb_jgmmtj_em：每日机构席位买卖统计（用于分离营业部 vs 机构）
区间：2015-01-01 ~ 2026-10-15，按月分块、断点续传（manifest 记录已完成月份）
落盘：data_cache/lhb/{detail,inst}.parquet + manifest.json（原子写）
"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import pandas as pd
import akshare as ak

START, END = '20150101', '20261015'
OUT = 'data_cache/lhb'
os.makedirs(OUT, exist_ok=True)
MANIFEST = f'{OUT}/manifest.json'

def load_manifest():
    if os.path.exists(MANIFEST):
        return json.load(open(MANIFEST, encoding='utf-8'))
    return {'done_months': []}

def months():
    out, y, m = [], 2015, 1
    while (y, m) <= (2026, 10):
        out.append((y, m))
        m += 1
        if m == 13:
            y, m = y + 1, 1
    return out

def month_range(y, m):
    import calendar
    last = calendar.monthrange(y, m)[1]
    end = min(f'{y}{m:02d}{last:02d}', END)
    return f'{y}{m:02d}01', end

def fetch_retry(fn, **kw):
    for i in range(3):
        try:
            return fn(**kw)
        except Exception as e:
            if i == 2:
                print(f'  ⚠️ 三次失败: {e}', flush=True)
                return None
            time.sleep(2 * (i + 1))

man = load_manifest()
d_all = pd.read_parquet(f'{OUT}/detail.parquet') if os.path.exists(f'{OUT}/detail.parquet') else None
i_all = pd.read_parquet(f'{OUT}/inst.parquet') if os.path.exists(f'{OUT}/inst.parquet') else None

for y, m in months():
    tag = f'{y}-{m:02d}'
    if tag in man['done_months']:
        continue
    s, e = month_range(y, m)
    d = fetch_retry(ak.stock_lhb_detail_em, start_date=s, end_date=e)
    time.sleep(0.4)
    i = fetch_retry(ak.stock_lhb_jgmmtj_em, start_date=s, end_date=e)
    time.sleep(0.4)
    if d is None and i is None:
        continue
    if d is not None and len(d):
        d_all = d if d_all is None else pd.concat([d_all, d], ignore_index=True)
    if i is not None and len(i):
        i_all = i if i_all is None else pd.concat([i_all, i], ignore_index=True)
    man['done_months'].append(tag)
    # 原子写
    if d_all is not None:
        d_all.to_parquet(f'{OUT}/detail.tmp.parquet')
        os.replace(f'{OUT}/detail.tmp.parquet', f'{OUT}/detail.parquet')
    if i_all is not None:
        i_all.to_parquet(f'{OUT}/inst.tmp.parquet')
        os.replace(f'{OUT}/inst.tmp.parquet', f'{OUT}/inst.parquet')
    tmp = MANIFEST + '.tmp'
    json.dump(man, open(tmp, 'w', encoding='utf-8'))
    os.replace(tmp, MANIFEST)
    print(f'{tag}: detail {0 if d is None else len(d)} 行, inst {0 if i is None else len(i)} 行 '
          f'| 累计 {0 if d_all is None else len(d_all)}/{0 if i_all is None else len(i_all)}', flush=True)

print('\n== 完成 ==')
if d_all is not None:
    print('detail:', len(d_all), '行 | 上榜日范围:', d_all['上榜日'].min(), '~', d_all['上榜日'].max())
    print('detail 全列:', d_all.columns.tolist())
if i_all is not None:
    print('inst:', len(i_all), '行')
    print('inst 全列:', i_all.columns.tolist())
