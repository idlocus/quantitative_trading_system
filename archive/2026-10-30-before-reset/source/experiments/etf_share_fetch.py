#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ETF 场内份额抓取（试炼 004 数据前置；不接触收益数据）

来源：上交所日频 ETF 份额（akshare fund_etf_scale_sse(date)，已验证
支持 2015-01 起任意日期，单位=份）
区间：2015-01-01 ~ 2026-10-15，逐交易日抓取（节假日跳过——接口对非
交易日返回空），断点续传（manifest 记录已完成日）
落盘：data_cache/etf_share/sse.parquet + manifest.json（原子写）
"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import pandas as pd
import akshare as ak

POOL = ['510300', '510310', '510330', '510050', '510500', '512100']
OUT = 'data_cache/etf_share'
os.makedirs(OUT, exist_ok=True)
MANIFEST = f'{OUT}/manifest.json'
PARQUET = f'{OUT}/sse.parquet'

# 交易日历：000300 本地缓存
idx = pd.read_parquet('data_cache/kline/000300_SH.parquet')
days = sorted(pd.to_datetime(idx['date']).dt.strftime('%Y%m%d'))
days = [d for d in days if '20150101' <= d <= '20261015']

man = json.load(open(MANIFEST, encoding='utf-8')) if os.path.exists(MANIFEST) else {'done': []}
done = set(man['done'])
rows = []
if os.path.exists(PARQUET):
    old = pd.read_parquet(PARQUET)
    rows = old.to_dict('records')

todo = [d for d in days if d not in done]
print(f'交易日 {len(days)} | 已完成 {len(done)} | 待抓 {len(todo)}', flush=True)

buf = []
for i, d in enumerate(todo):
    for attempt in range(3):
        try:
            df = ak.fund_etf_scale_sse(date=d)
            if df is not None and len(df):
                code_col = next(c for c in df.columns if '代码' in c)
                hit = df[df[code_col].astype(str).isin(POOL)]
                for _, r in hit.iterrows():
                    share_col = next(c for c in df.columns if '份额' in c)
                    rows.append({'date': d, 'code': str(r[code_col]),
                                 'share': float(r[share_col])})
            buf.append(d)
            break
        except Exception as e:
            if attempt == 2:
                print(f'  ⚠️ {d} 三次失败: {str(e)[:80]}', flush=True)
            else:
                time.sleep(2 * (attempt + 1))
    time.sleep(0.25)
    if (i + 1) % 100 == 0:
        man['done'] = list(set(man['done']) | set(buf)); buf = []
        pd.DataFrame(rows).to_parquet(PARQUET + '.tmp')
        os.replace(PARQUET + '.tmp', PARQUET)
        tmp = MANIFEST + '.tmp'
        json.dump(man, open(tmp, 'w', encoding='utf-8'))
        os.replace(tmp, MANIFEST)
        print(f'  {i + 1}/{len(todo)} | 累计 {len(rows)} 行', flush=True)

if buf:
    man['done'] = list(set(man['done']) | set(buf))
pd.DataFrame(rows).to_parquet(PARQUET + '.tmp')
os.replace(PARQUET + '.tmp', PARQUET)
tmp = MANIFEST + '.tmp'
json.dump(man, open(tmp, 'w', encoding='utf-8'))
os.replace(tmp, MANIFEST)

out = pd.DataFrame(rows)
print(f'\n== 完成 == {len(out)} 行')
if len(out):
    out['date'] = pd.to_datetime(out['date'], format='%Y%m%d')
    for c in POOL:
        s = out[out.code == c].sort_values('date')
        if len(s):
            print(f'{c}: {s["date"].min().date()} ~ {s["date"].max().date()} '
                  f'({len(s)} 日) | 份额 {s["share"].iloc[0]/1e8:.0f}亿 → {s["share"].iloc[-1]/1e8:.0f}亿')
        else:
            print(f'{c}: 无数据（上市晚或接口无此日）')
