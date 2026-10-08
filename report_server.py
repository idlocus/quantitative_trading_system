#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""分析报告 Flask 服务 — 股票 / 期货 / 指数通用"""

import sys, os
sys.path.insert(0, '.')

from flask import Flask, render_template_string, request
from datetime import date, timedelta
import pandas as pd
import numpy as np

app = Flask(__name__)

# ============================================================
# 数据与分析
# ============================================================

from data.wind_db import WindDBDataSource
from data.futures_db import FuturesDataSource
from data.fund_flow_provider import FundFlowProvider
from indicators.trend_indicators import DMI_ADX, SMA, MACD
from indicators.momentum_indicators import RSI
from indicators.fund_flow_indicators import InstitutionalFlowRatio, RetailFlowRatio

# 期货交易所后缀
_FUTURES_SUFFIXES = {'.CFE', '.SHF', '.DCE', '.CZC', '.INE'}


def get_data(symbol, days=252):
    """根据合约后缀自动选择数据源"""
    is_futures = any(symbol.endswith(s) for s in _FUTURES_SUFFIXES)

    if is_futures:
        ds = FuturesDataSource()
        end = date.today()
    else:
        ds = WindDBDataSource(
            oracle_dsn=os.getenv('ORACLE_DSN', ''),
            user=os.getenv('ORACLE_USER', ''),
            password=os.getenv('ORACLE_PASSWORD', '')
        )
        end = date(2026, 6, 2)

    start = end - timedelta(days=days)
    df = ds.get_kline(symbol, start, end)
    ds.close()

    # 统一列名：期货返回 o,h,l,c,v → 转为 open,high,low,close,volume
    col_map = {'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close',
               'v': 'volume', 'amt': 'amount', 'oi': 'open_interest'}
    df = df.rename(columns=col_map)
    return df.set_index('date').sort_index(), is_futures


def calc_indicators(df, symbol=None, is_futures=False):
    close = df['close']
    high = df['high']
    low = df['low']
    vol = df['volume']

    dmi = DMI_ADX(df, period=14)
    adx = dmi['adx'].iloc[-1]
    plus_di = dmi['plus_di'].iloc[-1]
    minus_di = dmi['minus_di'].iloc[-1]

    rsi14 = RSI(df, period=14).iloc[-1]
    rsi6 = RSI(df, period=6).iloc[-1]

    macd = MACD(df)
    macd_hist = macd['histogram'].iloc[-1]
    macd_line = macd['macd'].iloc[-1]
    macd_signal = macd['signal'].iloc[-1]

    ma5 = SMA(df, 5).iloc[-1]
    ma20 = SMA(df, 20).iloc[-1]
    ma60 = SMA(df, 60).iloc[-1]
    ma5_s = SMA(df, 5)
    ma20_s = SMA(df, 20)
    ma60_s = SMA(df, 60)

    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(14).mean().iloc[-1]

    rets = {
        5: (close.iloc[-1] / close.iloc[-6] - 1) * 100 if len(close) > 5 else 0,
        20: (close.iloc[-1] / close.iloc[-21] - 1) * 100 if len(close) > 20 else 0,
        60: (close.iloc[-1] / close.iloc[-61] - 1) * 100 if len(close) > 60 else 0,
        120: (close.iloc[-1] / close.iloc[-121] - 1) * 100 if len(close) > 120 else 0,
    }

    slopes = {}
    for p in [5, 20, 60]:
        ma = SMA(df, p)
        slopes[p] = (ma.iloc[-1] / ma.iloc[-11] - 1) * 100 if len(ma) > 10 else 0

    dmi_cross = 'golden' if (plus_di > minus_di and dmi['plus_di'].iloc[-2] <= dmi['minus_di'].iloc[-2]) else \
                'death' if (plus_di < minus_di and dmi['plus_di'].iloc[-2] >= dmi['minus_di'].iloc[-2]) else 'none'

    monthly = close.resample('M').last()
    monthly_ret = monthly.pct_change().dropna() * 100

    chart_data = {
        'dates': [str(d.date()) for d in close.index],
        'close': close.values.tolist(),
        'ma5': ma5_s.values.tolist(),
        'ma20': ma20_s.values.tolist(),
        'ma60': ma60_s.values.tolist(),
        'volume': (vol / 1e8).values.tolist() if not vol.isna().all() else [0] * len(vol),
        'rsi': RSI(df, 14).values.tolist(),
        'adx': dmi['adx'].values.tolist(),
        'plus_di': dmi['plus_di'].values.tolist(),
        'minus_di': dmi['minus_di'].values.tolist(),
        'macd_hist': macd['histogram'].values.tolist(),
        'monthly_ret': [{'date': str(monthly_ret.index[i].date()), 'ret': round(monthly_ret.iloc[i], 2)}
                        for i in range(len(monthly_ret))],
    }

    result = {
        'close_now': close.iloc[-1],
        'date_now': close.index[-1],
        'adx': adx,
        'plus_di': plus_di,
        'minus_di': minus_di,
        'rsi14': rsi14,
        'rsi6': rsi6,
        'macd_hist': macd_hist,
        'macd_line': macd_line,
        'macd_signal': macd_signal,
        'ma5': ma5,
        'ma20': ma20,
        'ma60': ma60,
        'ma_slopes': slopes,
        'rets': rets,
        'atr': atr,
        'atr_pct': atr / close.iloc[-1] * 100 if close.iloc[-1] != 0 else 0,
        'hist_high': high.max(),
        'hist_low': low.min(),
        'hist_high_date': str(high.idxmax().date()),
        'hist_low_date': str(low.idxmin().date()),
        'recent_high': high.iloc[-60:].max(),
        'recent_low': low.iloc[-60:].min(),
        'dmi_cross': dmi_cross,
        'vol_ma20_ratio': vol.iloc[-1] / vol.iloc[-20:].mean() if len(vol) >= 20 and not vol.iloc[-20:].isna().all() else 0,
        'chart_data': chart_data,
        'symbol': symbol or df.index.name or 'UNKNOWN',
        'is_futures': is_futures,
    }

    # 资金流向（仅股票有）
    if not is_futures:
        result['fund_flow'] = _get_fund_flow(df, symbol)
    else:
        result['fund_flow'] = None

    return result


def _get_fund_flow(df, symbol):
    """获取资金流向信号"""
    try:
        provider = FundFlowProvider()
        end = df.index[-1].date() if hasattr(df.index[-1], 'date') else df.index[-1]
        start = df.index[0].date() if hasattr(df.index[0], 'date') else df.index[0]
        ff = provider.get_fund_flow(symbol, start, end)
        if ff is None or ff.empty:
            return None
        ff_df = ff.copy()
        if 'date' not in ff_df.columns:
            return None
        ff_df['date'] = pd.to_datetime(ff_df['date'], errors='coerce')
        merged = df.copy()
        merged = merged.reset_index()
        if hasattr(merged.columns[0], 'lower'):
            merged.columns = [c.lower() if isinstance(c, str) else c for c in merged.columns]
        merged = merged.merge(ff_df, on='date', how='left', suffixes=('', '_ff'))
        for col in ['主力净流入', '散户净流入']:
            if col in merged.columns:
                merged[col] = merged[col].fillna(0)

        if '主力净流入' not in merged.columns or '散户净流入' not in merged.columns:
            return None

        signal = provider._compute_fund_flow_signal(merged)
        if signal:
            return signal
        return None
    except Exception:
        return None


# ============================================================
# HTML 模板
# ============================================================

TEMPLATE = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{{ symbol }} 分析报告</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: -apple-system, 'Microsoft YaHei', sans-serif; background: #0f1117; color: #e0e0e0; font-size: 14px; }
.container { max-width: 1200px; margin: 0 auto; padding: 20px; }

/* Header */
.header { display: flex; justify-content: space-between; align-items: center; padding: 20px 0; border-bottom: 1px solid #2a2a3a; margin-bottom: 24px; }
.header h1 { font-size: 24px; color: #fff; font-weight: 600; }
.header h1 span { color: #4fc3f7; }
.header .subtitle { color: #888; font-size: 13px; margin-top: 4px; }
.symbol-input { display: flex; gap: 8px; }
.symbol-input input { background: #1a1d2e; border: 1px solid #333; color: #fff; padding: 8px 14px; border-radius: 6px; font-size: 14px; width: 200px; }
.symbol-input button { background: #4fc3f7; border: none; color: #0f1117; padding: 8px 18px; border-radius: 6px; cursor: pointer; font-weight: 600; }

/* Tag for futures */
.asset-tag { display: inline-block; font-size: 10px; padding: 2px 8px; border-radius: 4px; margin-left: 8px; vertical-align: middle; }
.asset-tag.futures { background: #7b1fa2; color: #fff; }
.asset-tag.stock { background: #1565c0; color: #fff; }

/* Price card */
.price-card { background: linear-gradient(135deg, #1a1d2e 0%, #0f1117 100%); border-radius: 16px; padding: 28px; margin-bottom: 24px; border: 1px solid #2a2a3a; }
.price-row { display: flex; gap: 32px; flex-wrap: wrap; }
.price-main { flex: 1; min-width: 200px; }
.price-main .price { font-size: 48px; font-weight: 700; color: #fff; line-height: 1; }
.price-main .change { font-size: 18px; margin-top: 8px; }
.change.up { color: #ef5350; }
.change.dn { color: #26a69a; }
.change.neutral { color: #888; }
.price-meta { display: flex; gap: 40px; margin-top: 20px; flex-wrap: wrap; }
.price-meta .item { display: flex; flex-direction: column; gap: 2px; }
.price-meta .label { font-size: 11px; color: #666; text-transform: uppercase; letter-spacing: 0.5px; }
.price-meta .value { font-size: 16px; color: #ccc; font-weight: 500; }
.price-meta .value.highlight { color: #4fc3f7; }

/* Grid layout - 4 columns for compact cards, responsive */
.grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-bottom: 16px; }
@media (max-width: 900px) { .grid { grid-template-columns: repeat(2, 1fr); } }

/* Signal cards - compact */
.signal-card { background: #1a1d2e; border-radius: 10px; padding: 14px; border: 1px solid #2a2a3a; }
.signal-card h3 { font-size: 11px; color: #666; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 10px; border-bottom: 1px solid #2a2a3a; padding-bottom: 8px; }
.signal-row { display: flex; justify-content: space-between; align-items: center; padding: 5px 0; border-bottom: 1px solid #1e2030; }
.signal-row:last-child { border-bottom: none; }
.signal-row .name { color: #888; font-size: 11px; }
.signal-row .val { font-size: 13px; font-weight: 600; }
.val-green { color: #26a69a; }
.val-red { color: #ef5350; }
.val-yellow { color: #ffca28; }
.val-blue { color: #4fc3f7; }

/* Chart container */
.chart-container { background: #1a1d2e; border-radius: 12px; padding: 20px; border: 1px solid #2a2a3a; margin-bottom: 24px; }
.chart-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px; border-bottom: 1px solid #2a2a3a; padding-bottom: 12px; }
.chart-title { font-size: 14px; font-weight: 600; color: #ccc; }
.chart-tabs { display: flex; gap: 4px; }
.chart-tabs button { background: #252a3d; border: none; color: #888; padding: 5px 12px; border-radius: 4px; cursor: pointer; font-size: 12px; }
.chart-tabs button.active { background: #4fc3f7; color: #0f1117; font-weight: 600; }
.chart-wrap { position: relative; height: 360px; }
.chart-wrap canvas { position: absolute; top: 0; left: 0; width: 100%; height: 100%; }

/* Monthly bars */
.monthly-chart { display: flex; align-items: flex-end; gap: 3px; height: 80px; padding: 8px 0; }
.bar { flex: 1; border-radius: 2px 2px 0 0; min-width: 8px; }
.bar.up { background: #ef5350; }
.bar.dn { background: #26a69a; }

/* Advice card */
.advice-card { background: linear-gradient(135deg, #1a2e1a 0%, #1a1d2e 100%); border-radius: 12px; padding: 24px; border: 1px solid #2a4a2a; margin-bottom: 24px; }
.advice-card.buy { background: linear-gradient(135deg, #1a2e1a 0%, #1a1d2e 100%); border-color: #2a4a2a; }
.advice-card.sell { background: linear-gradient(135deg, #2e1a1a 0%, #1a1d2e 100%); border-color: #4a2a2a; }
.advice-card.neutral { background: linear-gradient(135deg, #2a2a1a 0%, #1a1d2e 100%); border-color: #4a4a2a; }
.advice-badge { display: inline-block; padding: 4px 12px; border-radius: 20px; font-size: 13px; font-weight: 600; margin-bottom: 12px; }
.advice-badge.buy { background: #26a69a; color: #0f1117; }
.advice-badge.sell { background: #ef5350; color: #fff; }
.advice-badge.neutral { background: #ffca28; color: #0f1117; }
.advice-title { font-size: 18px; font-weight: 600; color: #fff; margin-bottom: 8px; }
.advice-body { color: #aaa; font-size: 13px; line-height: 1.7; }
.advice-body strong { color: #4fc3f7; }
.advice-body ul { margin: 8px 0 0 20px; }
.advice-body li { margin: 4px 0; }

/* Support/Resistance table */
.sr-table { width: 100%; border-collapse: collapse; margin-top: 12px; }
.sr-table td { padding: 8px 12px; border-bottom: 1px solid #222; font-size: 13px; }
.sr-table td:first-child { color: #666; width: 80px; }
.sr-table .resistance { color: #ef5350; font-weight: 600; }
.sr-table .support { color: #26a69a; font-weight: 600; }
.sr-table .current { color: #fff; font-weight: 600; background: #252a3d; border-radius: 4px; }

/* Footer */
.footer { text-align: center; color: #444; font-size: 11px; padding: 30px 0; border-top: 1px solid #1a1d2e; margin-top: 30px; }

canvas { -webkit-tap-highlight-color: transparent; }
</style>
</head>
<body>
<div class="container">
  <div class="header">
    <div>
      <h1>
        <span>{{ symbol }}</span>
        <span class="asset-tag {% if data.is_futures %}futures{% else %}stock{% endif %}">
          {% if data.is_futures %}期货{% else %}股票{% endif %}
        </span>
        分析报告
      </h1>
      <div class="subtitle">{{ date_now }} &nbsp;·&nbsp; 数据来源: Wind Oracle</div>
    </div>
    <form class="symbol-input" method="get">
      <input name="symbol" value="{{ symbol }}" placeholder="股票代码或期货合约，如 300433.SZ / IF00.CFE">
      <button type="submit">查询</button>
    </form>
  </div>

  <!-- 价格卡片 -->
  <div class="price-card">
    <div class="price-row">
      <div class="price-main">
        <div class="price">{{ "%.2f"|format(data.close_now) }}</div>
        <div class="change {% if data.rets[5] > 0 %}up{% elif data.rets[5] < 0 %}dn{% else %}neutral{% endif %}">
          {% if data.rets[5] > 0 %}▲{% elif data.rets[5] < 0 %}▼{% else %}―{% endif %}
          {{ "%.2f"|format(data.rets[5]) }}% (近5日)
        </div>
      </div>
      <div class="price-meta">
        <div class="item">
          <span class="label">历史高点</span>
          <span class="value highlight">{{ "%.2f"|format(data.hist_high) }}</span>
          <span class="label" style="font-size:10px">{{ data.hist_high_date }}</span>
        </div>
        <div class="item">
          <span class="label">历史低点</span>
          <span class="value">{{ "%.2f"|format(data.hist_low) }}</span>
          <span class="label" style="font-size:10px">{{ data.hist_low_date }}</span>
        </div>
        <div class="item">
          <span class="label">距高点</span>
          <span class="value {% if (data.close_now/data.hist_high*100) > 90 %}val-red{% endif %}">{{ "%.1f"|format(data.close_now/data.hist_high*100) }}%</span>
        </div>
        <div class="item">
          <span class="label">ATR(14)</span>
          <span class="value">{{ "%.2f"|format(data.atr) }}</span>
          <span class="label" style="font-size:10px">日波幅 {{ "%.1f"|format(data.atr_pct) }}%</span>
        </div>
      </div>
    </div>
  </div>

  <!-- 信号仪表盘 -->
  <div class="grid">
    <!-- 趋势信号 -->
    <div class="signal-card">
      <h3>🎯 趋势信号</h3>
      <div class="signal-row">
        <span class="name">ADX (趋势强度)</span>
        <span class="val {% if data.adx > 30 %}val-green{% elif data.adx < 20 %}val-red{% else %}val-yellow{% endif %}">{{ "%.1f"|format(data.adx) }}</span>
      </div>
      <div class="signal-row">
        <span class="name">+DI / -DI</span>
        <span class="val {% if data.plus_di > data.minus_di %}val-green{% else %}val-red{% endif %}">{{ "%.1f"|format(data.plus_di) }} / {{ "%.1f"|format(data.minus_di) }}</span>
      </div>
      <div class="signal-row">
        <span class="name">DMI交叉</span>
        <span class="val {% if data.dmi_cross == 'golden' %}val-green{% elif data.dmi_cross == 'death' %}val-red{% else %}val-yellow{% endif %}">
          {% if data.dmi_cross == 'golden' %}🟢 金叉{% elif data.dmi_cross == 'death' %}🔴 死叉{% else %}⚪ 无叉{% endif %}
        </span>
      </div>
      <div class="signal-row">
        <span class="name">均线状态</span>
        <span class="val {% if data.ma5 > data.ma20 > data.ma60 %}val-green{% elif data.ma5 < data.ma20 < data.ma60 %}val-red{% else %}val-yellow{% endif %}">
          {% if data.ma5 > data.ma20 > data.ma60 %}🟢 多头排列{% elif data.ma5 < data.ma20 < data.ma60 %}🔴 空头排列{% else %}🟡 混乱{% endif %}
        </span>
      </div>
    </div>

    <!-- 动量信号 -->
    <div class="signal-card">
      <h3>⚡ 动量信号</h3>
      <div class="signal-row">
        <span class="name">RSI(14)</span>
        <span class="val {% if data.rsi14 > 70 %}val-red{% elif data.rsi14 < 30 %}val-green{% else %}val-yellow{% endif %}">{{ "%.1f"|format(data.rsi14) }}</span>
      </div>
      <div class="signal-row">
        <span class="name">RSI(6)</span>
        <span class="val {% if data.rsi6 > 75 %}val-red{% elif data.rsi6 < 40 %}val-green{% else %}val-yellow{% endif %}">{{ "%.1f"|format(data.rsi6) }}</span>
      </div>
      <div class="signal-row">
        <span class="name">MACD柱体</span>
        <span class="val {% if data.macd_hist > 0 %}val-green{% else %}val-red{% endif %}">{{ "%+.3f"|format(data.macd_hist) }}</span>
      </div>
      <div class="signal-row">
        <span class="name">成交量比MA20</span>
        <span class="val {% if data.vol_ma20_ratio > 1.5 %}val-green{% elif data.vol_ma20_ratio < 0.7 %}val-red{% else %}val-yellow{% endif %}">{{ "%.2f"|format(data.vol_ma20_ratio) }}x</span>
      </div>
    </div>

    <!-- 均线系统 -->
    <div class="signal-card">
      <h3>📐 均线系统</h3>
      <div class="signal-row">
        <span class="name">MA5</span>
        <span class="val val-blue">{{ "%.2f"|format(data.ma5) }}</span>
      </div>
      <div class="signal-row">
        <span class="name">MA20</span>
        <span class="val val-blue">{{ "%.2f"|format(data.ma20) }}</span>
      </div>
      <div class="signal-row">
        <span class="name">MA60</span>
        <span class="val val-blue">{{ "%.2f"|format(data.ma60) }}</span>
      </div>
      <div class="signal-row">
        <span class="name">MA斜率(10日)</span>
        <span class="val">
          <span style="{% if data.ma_slopes[5] > 2 %}color:#26a69a{% elif data.ma_slopes[5] < -2 %}color:#ef5350{% endif %}">MA5 {{ "%+.1f"|format(data.ma_slopes[5]) }}%</span>
          /
          <span style="{% if data.ma_slopes[20] > 2 %}color:#26a69a{% elif data.ma_slopes[20] < -2 %}color:#ef5350{% endif %}">MA20 {{ "%+.1f"|format(data.ma_slopes[20]) }}%</span>
        </span>
      </div>
    </div>

    <!-- 收益一览 -->
    <div class="signal-card">
      <h3>📈 区间收益</h3>
      <div class="signal-row">
        <span class="name">近5日</span>
        <span class="val {% if data.rets[5] > 3 %}val-green{% elif data.rets[5] < -3 %}val-red{% else %}val-yellow{% endif %}">{{ "%+.1f"|format(data.rets[5]) }}%</span>
      </div>
      <div class="signal-row">
        <span class="name">近20日</span>
        <span class="val {% if data.rets[20] > 10 %}val-green{% elif data.rets[20] < -10 %}val-red{% else %}val-yellow{% endif %}">{{ "%+.1f"|format(data.rets[20]) }}%</span>
      </div>
      <div class="signal-row">
        <span class="name">近60日</span>
        <span class="val {% if data.rets[60] > 15 %}val-green{% elif data.rets[60] < -15 %}val-red{% else %}val-yellow{% endif %}">{{ "%+.1f"|format(data.rets[60]) }}%</span>
      </div>
      <div class="signal-row">
        <span class="name">近120日</span>
        <span class="val {% if data.rets[120] > 20 %}val-green{% elif data.rets[120] < -20 %}val-red{% else %}val-yellow{% endif %}">{{ "%+.1f"|format(data.rets[120]) }}%</span>
      </div>
    </div>
  </div>

  <!-- 资金流向信号（仅股票） -->
  {% if not data.is_futures %}
    {% if data.fund_flow %}
  <div class="chart-container">
    <div class="chart-header">
      <span class="chart-title">💰 资金流向信号</span>
    </div>
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:16px">
      <div class="signal-card">
        <h3>🏦 机构资金（5日汇总）</h3>
        <div class="signal-row">
          <span class="name">机构净流入</span>
          <span class="val {% if data.fund_flow.inst_net_5d > 0 %}val-green{% else %}val-red{% endif %}">{{ "%+.1f"|format(data.fund_flow.inst_net_5d/1e4) }} 万</span>
        </div>
        <div class="signal-row">
          <span class="name">机构占总流入</span>
          <span class="val {% if data.fund_flow.inst_pct > 50 %}val-green{% elif data.fund_flow.inst_pct < 20 %}val-red{% else %}val-yellow{% endif %}">{{ "%.1f"|format(data.fund_flow.inst_pct) }}%</span>
        </div>
      </div>
      <div class="signal-card">
        <h3>🎰 游资资金（5日汇总）</h3>
        <div class="signal-row">
          <span class="name">游资净流入</span>
          <span class="val {% if data.fund_flow.retail_net_5d < 0 %}val-green{% else %}val-red{% endif %}">{{ "%+.1f"|format(data.fund_flow.retail_net_5d/1e4) }} 万</span>
        </div>
        <div class="signal-row">
          <span class="name">游资占总流入</span>
          <span class="val {% if data.fund_flow.retail_pct > 50 %}val-red{% elif data.fund_flow.retail_pct < 20 %}val-green{% else %}val-yellow{% endif %}">{{ "%.1f"|format(data.fund_flow.retail_pct) }}%</span>
        </div>
      </div>
    </div>
    <div style="text-align:center;padding:16px;background:#1a1d2e;border-radius:10px;border:1px solid {% if data.fund_flow.signal_color == 'green' %}#26a69a{% elif data.fund_flow.signal_color == 'red' %}#ef5350{% else %}#ffca28{% endif %}">
      <div style="font-size:32px;margin-bottom:8px">{{ data.fund_flow.signal_icon }}</div>
      <div style="font-size:18px;font-weight:600;color:#fff;margin-bottom:4px">{{ data.fund_flow.verdict }}</div>
      <div style="font-size:12px;color:#666">
        {% if data.fund_flow.signal == 'institutional_accumulation' %}机构持续买入，上升趋势健康
        {% elif data.fund_flow.signal == 'institutional_buy' %}机构买入但上方有阻力
        {% elif data.fund_flow.signal == 'retail_hot' %}游资主导炒作，短期行情波动大
        {% elif data.fund_flow.signal == 'retail_active' %}游资活跃，谨慎追高
        {% elif data.fund_flow.signal == 'inst_takes_retail' %}机构吸筹，游资离场，上升趋势确立
        {% elif data.fund_flow.signal == 'inst_distributes' %}机构派发，游资接盘，短期顶部
        {% elif data.fund_flow.signal == 'neutral' %}资金平衡，无明确方向
        {% elif data.fund_flow.signal == 'funds_outflow' %}资金全面外逃
        {% else %}待观察{% endif %}
      </div>
    </div>
  </div>
    {% else %}
  <div class="chart-container">
    <div class="chart-header">
      <span class="chart-title">💰 资金流向</span>
    </div>
    <div style="color:#666;font-size:13px;text-align:center;padding:20px">资金流向数据暂时不可用</div>
  </div>
    {% endif %}
  {% else %}
  <div class="chart-container">
    <div class="chart-header">
      <span class="chart-title">📊 期货持仓与成交</span>
    </div>
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px">
      <div class="signal-card">
        <h3>📈 期货指标</h3>
        <div class="signal-row">
          <span class="name">持仓量(OI)</span>
          <span class="val val-blue">{{ "%.0f"|format(data.chart_data.volume[-1] if data.chart_data.volume else 0) }}</span>
        </div>
        <div class="signal-row">
          <span class="name">ATR(14)</span>
          <span class="val">{{ "%.2f"|format(data.atr) }} ({{ "%.1f"|format(data.atr_pct) }}%)</span>
        </div>
      </div>
      <div class="signal-card">
        <h3>📐 均线信号</h3>
        <div class="signal-row">
          <span class="name">MA5 vs MA20</span>
          <span class="val {% if data.ma5 > data.ma20 %}val-green{% else %}val-red{% endif %}">
            MA5 {{ "%.2f"|format(data.ma5) }} / MA20 {{ "%.2f"|format(data.ma20) }}
          </span>
        </div>
        <div class="signal-row">
          <span class="name">趋势方向</span>
          <span class="val {% if data.adx > 25 %}val-green{% else %}val-yellow{% endif %}">
            {% if data.adx > 40 %}强趋势{% elif data.adx > 25 %}趋势确认{% else %}震荡{% endif %}
          </span>
        </div>
      </div>
    </div>
  </div>
  {% endif %}

  <!-- 图表区 -->
  <div class="chart-container">
    <div class="chart-header">
      <span class="chart-title">📊 价格走势与指标</span>
      <div class="chart-tabs">
        <button class="active" onclick="showPanel('price')">价格</button>
        <button onclick="showPanel('rsi')">RSI</button>
        <button onclick="showPanel('adx')">ADX</button>
        <button onclick="showPanel('monthly')">月度</button>
      </div>
    </div>
    <div class="chart-wrap" id="chart-price" style="display:block">
      <canvas id="priceCanvas"></canvas>
    </div>
    <div class="chart-wrap" id="chart-rsi" style="display:none">
      <canvas id="rsiCanvas"></canvas>
    </div>
    <div class="chart-wrap" id="chart-adx" style="display:none">
      <canvas id="adxCanvas"></canvas>
    </div>
    <div class="chart-wrap" id="chart-monthly" style="display:none">
      <canvas id="monthlyCanvas"></canvas>
    </div>
  </div>

  <!-- 支撑阻力 -->
  <div class="chart-container">
    <div class="chart-header">
      <span class="chart-title">🔐 支撑与阻力</span>
    </div>
    <table class="sr-table">
      <tr>
        <td>阻力2</td>
        <td class="resistance">{{ "%.1f"|format(data.hist_high * 1.10) }} (+10%)</td>
      </tr>
      <tr>
        <td>阻力1</td>
        <td class="resistance">{{ "%.1f"|format(data.hist_high) }} (历史高点)</td>
      </tr>
      <tr>
        <td class="current">当前价</td>
        <td class="current">{{ "%.2f"|format(data.close_now) }}</td>
      </tr>
      <tr>
        <td>支撑1</td>
        <td class="support">{{ "%.1f"|format(data.close_now - data.atr * 1) }} (ATR 1x, {{ "%.1f"|format(data.atr_pct) }}%)</td>
      </tr>
      <tr>
        <td>支撑2</td>
        <td class="support">{{ "%.1f"|format(data.close_now - data.atr * 1.5) }} (ATR 1.5x)</td>
      </tr>
      <tr>
        <td>支撑3</td>
        <td class="support">{{ "%.1f"|format(data.recent_low) }} (近60日低点)</td>
      </tr>
    </table>
  </div>

  <!-- 综合建议 -->
  <div class="advice-card {{ advice_class }}">
    <div class="advice-badge {{ advice_class }}">{{ advice_badge }}</div>
    <div class="advice-title">{{ advice_title }}</div>
    <div class="advice-body">
      {{ advice_body | safe }}
    </div>
  </div>

  <div class="footer">
    报告自动生成 · 数据来源: Wind Oracle · {{ date_now }}
  </div>
</div>

<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<script>
Chart.defaults.color = '#888';
Chart.defaults.borderColor = '#2a2a3a';
Chart.defaults.font.family = "-apple-system, 'Microsoft YaHei', sans-serif";

const raw = {{ chart_data | tojson }};

new Chart(document.getElementById('priceCanvas'), {
  type: 'line',
  data: {
    labels: raw.dates,
    datasets: [
      { label: '收盘价', data: raw.close, borderColor: '#4fc3f7', borderWidth: 2, fill: false, pointRadius: 0, tension: 0.1 },
      { label: 'MA5',   data: raw.ma5,   borderColor: '#ff9800', borderWidth: 1.2, fill: false, pointRadius: 0, tension: 0.1 },
      { label: 'MA20',  data: raw.ma20,  borderColor: '#4caf50', borderWidth: 1.2, fill: false, pointRadius: 0, tension: 0.1 },
      { label: 'MA60',  data: raw.ma60,  borderColor: '#e91e63', borderWidth: 1.2, fill: false, pointRadius: 0, tension: 0.1 },
    ]
  },
  options: {
    responsive: true, maintainAspectRatio: false,
    interaction: { mode: 'index', intersect: false },
    plugins: { legend: { position: 'top' }, tooltip: { callbacks: { label: ctx => ctx.dataset.label + ': ' + ctx.parsed.y.toFixed(2) } } },
    scales: { x: { display: true, ticks: { maxTicksLimit: 8 }, grid: { color: '#1a1d2e' } }, y: { display: true, grid: { color: '#1a1d2e' } } }
  }
});

new Chart(document.getElementById('rsiCanvas'), {
  type: 'line',
  data: { labels: raw.dates, datasets: [{ label: 'RSI(14)', data: raw.rsi, borderColor: '#9c27b0', borderWidth: 1.5, fill: false, pointRadius: 0 }] },
  options: {
    responsive: true, maintainAspectRatio: false,
    plugins: { legend: { display: false } },
    scales: { x: { display: true, ticks: { maxTicksLimit: 8 }, grid: { color: '#1a1d2e' } }, y: { display: true, min: 0, max: 100, grid: { color: '#1a1d2e' } } }
  }
});

new Chart(document.getElementById('adxCanvas'), {
  type: 'line',
  data: {
    labels: raw.dates,
    datasets: [
      { label: 'ADX',    data: raw.adx,      borderColor: '#3f51b5', borderWidth: 1.5, fill: false, pointRadius: 0 },
      { label: '+DI',    data: raw.plus_di,   borderColor: '#e91e63', borderWidth: 1, fill: false, pointRadius: 0 },
      { label: '-DI',    data: raw.minus_di,  borderColor: '#4caf50', borderWidth: 1, fill: false, pointRadius: 0 },
    ]
  },
  options: {
    responsive: true, maintainAspectRatio: false,
    interaction: { mode: 'index', intersect: false },
    plugins: { legend: { position: 'top' } },
    scales: { x: { display: true, ticks: { maxTicksLimit: 8 }, grid: { color: '#1a1d2e' } }, y: { display: true, grid: { color: '#1a1d2e' } } }
  }
});

new Chart(document.getElementById('monthlyCanvas'), {
  type: 'bar',
  data: {
    labels: raw.monthly_ret.map(m => m.date),
    datasets: [{ label: '月度收益', data: raw.monthly_ret.map(m => m.ret), backgroundColor: raw.monthly_ret.map(m => m.ret >= 0 ? '#ef5350' : '#26a69a'), borderRadius: 2 }]
  },
  options: {
    responsive: true, maintainAspectRatio: false,
    plugins: { legend: { display: false } },
    scales: { x: { display: true, ticks: { maxTicksLimit: 12 }, grid: { color: '#1a1d2e' } }, y: { display: true, grid: { color: '#1a1d2e' } } }
  }
});

function showPanel(name) {
  document.querySelectorAll('.chart-wrap').forEach(el => el.style.display = 'none');
  document.getElementById('chart-' + name).style.display = 'block';
  document.querySelectorAll('.chart-tabs button').forEach(btn => btn.classList.remove('active'));
  event.target.classList.add('active');
}
</script>
</body>
</html>
"""


# ============================================================
# 建议逻辑
# ============================================================

def get_advice(data):
    adx = data['adx']
    rsi14 = data['rsi14']
    ma5 = data['ma5']
    ma20 = data['ma20']
    close = data['close_now']
    atr = data['atr']
    atr_pct = data['atr_pct']
    hist_high = data['hist_high']
    ff = data.get('fund_flow')

    if adx > 40 and rsi14 < 70 and ma5 > ma20:
        body = (
            f"ADX={adx:.0f} 极强趋势，均线多头排列。<br>"
            f"建议在 <strong>{close - atr*0.5:.0f} ~ {close:.0f}</strong> 区间买入<br>"
            f"止损: <strong>{close - atr*1.5:.0f}</strong> (ATR 1.5x = {atr_pct*1.5:.1f}%回撤)<br>"
            f"目标: <strong>{hist_high:.0f}</strong><br>"
            f"<ul><li>RSI={rsi14:.0f}，未超买，可继续持有</li>"
            f"<li>成交量比={data['vol_ma20_ratio']:.2f}x，筹码稳定</li>"
        )
        if ff:
            if ff.get('inst_pct', 0) > 50:
                body += f"<li>机构资金强势买入（占比{ff['inst_pct']:.1f}%）</li>"
        body += "</ul>"
        return 'buy', '✅ 积极买入', '极强上升趋势', body

    elif adx > 30 and ma5 > ma20:
        body = (
            f"ADX={adx:.0f} 确认趋势，上方临近阻力。<br>"
            f"建议分批买入，首批50%仓位<br>"
            f"止损: <strong>{close - atr*1.5:.0f}</strong><br>"
            f"目标: <strong>{close * 1.10:.0f}</strong> (+10%)"
        )
        return 'buy', '✅ 可以考虑买入', '趋势确认', body

    elif rsi14 > 70:
        body = (
            f"RSI={rsi14:.0f} 短期超买，注意回调风险。<br>"
            f"建议等RSI回落至50-60再买入。<br>"
            f"若坚持买入，止损放在 <strong>{close - atr*2:.0f}</strong>"
        )
        return 'sell', '⚠️ 谨慎追高', '短期超买', body

    elif adx < 20:
        body = (
            f"ADX={adx:.0f} 市场震荡，无明确趋势。<br>"
            f"建议等ADX&gt;25趋势确认后再介入。"
        )
        return 'neutral', '⏸️ 观望为主', '市场震荡', body

    else:
        body = (
            f"ADX={adx:.0f} 趋势不明确，建议等待更明确信号。"
        )
        return 'neutral', '⚖️ 中性观望', '趋势不明确', body


# ============================================================
# Flask 路由
# ============================================================

@app.route('/')
def index():
    symbol = request.args.get('symbol', '300433.SZ').strip()
    try:
        df, is_futures = get_data(symbol, days=252)
        data = calc_indicators(df, symbol, is_futures)
    except Exception as e:
        return f'<html><body style="background:#0f1117;color:#ef5350;font-family:monospace;padding:40px"><h2>Error: {e}</h2><p>请检查代码是否正确（股票如 300433.SZ，期货如 IF00.CFE）</p><a href="/" style="color:#4fc3f7">返回重试</a></body></html>'

    advice_class, advice_badge, advice_title, advice_body = get_advice(data)

    return render_template_string(TEMPLATE,
        symbol=symbol,
        data=data,
        date_now=data['date_now'].strftime('%Y-%m-%d'),
        advice_class=advice_class,
        advice_badge=advice_badge,
        advice_title=advice_title,
        advice_body=advice_body,
        chart_data=data['chart_data'],
    )


if __name__ == '__main__':
    print('=' * 50)
    print('  分析报告服务启动中...')
    print('  股票示例: 300433.SZ / 600519.SH')
    print('  期货示例: IF00.CFE / IC00.CFE / AG.SHF')
    print('  访问 http://127.0.0.1:5555')
    print('=' * 50)
    app.run(host='0.0.0.0', port=5555, debug=True)