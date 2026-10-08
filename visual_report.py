#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""股票可视化分析报告生成器"""

import sys, os
sys.path.insert(0, '.')

from data.wind_db import WindDBDataSource
from datetime import date, timedelta
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from matplotlib.gridspec import GridSpec
import io
import base64

from indicators.trend_indicators import DMI_ADX, SMA, MACD
from indicators.momentum_indicators import RSI


# 中文字体设置
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'KaiTi']
plt.rcParams['axes.unicode_minus'] = False


def get_stock_data(symbol, start, end):
    db = WindDBDataSource(
            oracle_dsn=os.getenv('ORACLE_DSN', ''),
            user=os.getenv('ORACLE_USER', ''),
            password=os.getenv('ORACLE_PASSWORD', '')
        )
    df = db.get_kline(symbol, start, end)
    db.close()
    return df.set_index('date').sort_index()


def analyze_stock(symbol, days=252):
    """计算股票各项指标"""
    end = date(2026, 6, 2)
    start = end - timedelta(days=days)
    df = get_stock_data(symbol, start, end)
    close = df['close']
    high = df['high']
    low = df['low']
    vol = df['volume']

    # 指标计算
    dmi = DMI_ADX(df, period=14)
    adx = dmi['adx'].iloc[-1]
    plus_di = dmi['plus_di'].iloc[-1]
    minus_di = dmi['minus_di'].iloc[-1]

    rsi14 = RSI(df, period=14).iloc[-1]
    rsi6 = RSI(df, period=6).iloc[-1]

    macd = MACD(df)
    macd_hist = macd['histogram'].iloc[-1]

    ma5 = SMA(df, 5).iloc[-1]
    ma20 = SMA(df, 20).iloc[-1]
    ma60 = SMA(df, 60).iloc[-1]

    ma5_series = SMA(df, 5)
    ma20_series = SMA(df, 20)
    ma60_series = SMA(df, 60)

    # 收益
    rets = {n: (close.iloc[-1] / close.iloc[-n-1] - 1) * 100 if len(close) > n else 0 for n in [5, 20, 60, 120]}

    # 波动
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(14).mean().iloc[-1]

    # 高低点
    recent_high_60 = high.iloc[-60:].max()
    recent_low_60 = low.iloc[-60:].min()
    hist_high = high.max()
    hist_low = low.min()

    # 均线斜率
    ma_slopes = {}
    for p in [5, 20, 60]:
        ma = SMA(df, p)
        ma_slopes[p] = (ma.iloc[-1] / ma.iloc[-11] - 1) * 100 if len(ma) > 10 else 0

    # DMI交叉
    if plus_di > minus_di and dmi['plus_di'].iloc[-2] <= dmi['minus_di'].iloc[-2]:
        dmi_cross = 'golden'
    elif plus_di < minus_di and dmi['plus_di'].iloc[-2] >= dmi['minus_di'].iloc[-2]:
        dmi_cross = 'death'
    else:
        dmi_cross = 'none'

    return {
        'symbol': symbol,
        'close': close,
        'high': high,
        'low': low,
        'vol': vol,
        'ma5_series': ma5_series,
        'ma20_series': ma20_series,
        'ma60_series': ma60_series,
        'dmi_series': dmi,
        'rsi_series': RSI(df, period=14),
        'macd_series': macd,
        'close_now': close.iloc[-1],
        'date_now': close.index[-1],
        'adx': adx,
        'plus_di': plus_di,
        'minus_di': minus_di,
        'rsi14': rsi14,
        'rsi6': rsi6,
        'macd_hist': macd_hist,
        'ma5': ma5,
        'ma20': ma20,
        'ma60': ma60,
        'rets': rets,
        'atr': atr,
        'atr_pct': atr / close.iloc[-1] * 100,
        'recent_high_60': recent_high_60,
        'recent_low_60': recent_low_60,
        'hist_high': hist_high,
        'hist_low': hist_low,
        'hist_high_date': high.idxmax(),
        'hist_low_date': low.idxmin(),
        'ma_slopes': ma_slopes,
        'dmi_cross': dmi_cross,
        'vol_ma20': vol.iloc[-20:].mean(),
        'vol_now': vol.iloc[-1],
    }


def gauge_chart(value, max_val=100, width=120, height=40, label='', color='#2196F3'):
    """生成仪表盘风格的ASCII条形图"""
    filled = int(value / max_val * width)
    bar = '█' * filled + '░' * (width - filled)
    return f'{label}: [{bar}] {value:.1f}'


def spark_line(series, points=20):
    """生成迷你Sparkline"""
    vals = series.iloc[-points:].values
    min_v, max_v = vals.min(), vals.max()
    if max_v == min_v:
        return '─' * points
    norm = (vals - min_v) / (max_v - min_v) * 8
    chars = ' ▁▂▃▄▅▆▇█'
    return ''.join(chars[int(v)] for v in norm)


def plot_kline_with_indicators(data, save_path):
    """绘制K线+指标图"""
    close = data['close']
    high = data['high']
    low = data['low']
    vol = data['vol']
    ma5 = data['ma5_series']
    ma20 = data['ma20_series']
    ma60 = data['ma60_series']
    dmi = data['dmi_series']
    rsi = data['rsi_series']
    macd = data['macd_series']

    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(4, 1, height_ratios=[3, 1, 1, 1], hspace=0.3)

    ax_price = fig.add_subplot(gs[0])
    ax_vol = fig.add_subplot(gs[1], sharex=ax_price)
    ax_rsi = fig.add_subplot(gs[2], sharex=ax_price)
    ax_adx = fig.add_subplot(gs[3], sharex=ax_price)

    # K线
    x = range(len(close))
    ax_price.fill_between(x, close, close.iloc[0], alpha=0.1, color='gray')
    ax_price.plot(x, close, color='#555', linewidth=1, alpha=0.5)
    ax_price.plot(x, close, color='#1e88e5', linewidth=1.5)

    # 均线
    ax_price.plot(x, ma5, color='#ff9800', linewidth=1, label='MA5', alpha=0.9)
    ax_price.plot(x, ma20, color='#4caf50', linewidth=1.2, label='MA20', alpha=0.9)
    ax_price.plot(x, ma60, color='#e91e63', linewidth=1.2, label='MA60', alpha=0.9)

    # 成交量
    colors = ['#ef5350' if close.iloc[i] < close.iloc[i-1] else '#26a69a'
              for i in range(1, len(close))]
    colors = ['#26a69a'] + colors
    ax_vol.bar(x, vol, color=colors, alpha=0.7, width=0.8)
    ax_vol.axhline(y=vol.iloc[-20:].mean(), color='orange', linewidth=1, linestyle='--', alpha=0.7)

    # RSI
    ax_rsi.fill_between(x, rsi, 70, where=(rsi >= 70), color='#ef5350', alpha=0.3)
    ax_rsi.fill_between(x, rsi, 30, where=(rsi <= 30), color='#26a69a', alpha=0.3)
    ax_rsi.plot(x, rsi, color='#9c27b0', linewidth=1)
    ax_rsi.axhline(y=70, color='#ef5350', linewidth=0.8, linestyle='--', alpha=0.7)
    ax_rsi.axhline(y=30, color='#26a69a', linewidth=0.8, linestyle='--', alpha=0.7)
    ax_rsi.axhline(y=50, color='gray', linewidth=0.5, linestyle=':', alpha=0.5)
    ax_rsi.set_ylim(0, 100)

    # ADX + DI
    ax_adx.fill_between(x, dmi['adx'], color='#3f51b5', alpha=0.3)
    ax_adx.plot(x, dmi['adx'], color='#3f51b5', linewidth=1, label='ADX')
    ax_adx.plot(x, dmi['plus_di'], color='#e91e63', linewidth=1, label='+DI')
    ax_adx.plot(x, dmi['minus_di'], color='#4caf50', linewidth=1, label='-DI')
    ax_adx.axhline(y=30, color='gray', linewidth=0.8, linestyle='--', alpha=0.5)
    ax_adx.axhline(y=20, color='gray', linewidth=0.8, linestyle=':', alpha=0.5)

    # 标注当前值
    last = len(close) - 1
    for ax, series, color, val_fmt in [
        (ax_price, close, '#1e88e5', '{:.2f}'),
        (ax_price, ma5, '#ff9800', '{:.2f}'),
        (ax_price, ma20, '#4caf50', '{:.2f}'),
        (ax_price, ma60, '#e91e63', '{:.2f}'),
    ]:
        val = val_fmt.format(series.iloc[-1])
        ax.annotate(val, xy=(last, series.iloc[-1]),
                   fontsize=8, color=color,
                   xytext=(5, 0), textcoords='offset points')

    # 底部标签
    for ax in [ax_price, ax_vol, ax_rsi, ax_adx]:
        ax.set_xlim(0, len(close))
        ax.tick_params(labelbottom=False, labelsize=8)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    ax_price.legend(loc='upper left', fontsize=8)
    ax_rsi.legend(loc='upper left', fontsize=8)
    ax_adx.legend(loc='upper left', fontsize=8)
    ax_vol.tick_params(labelbottom=False)

    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    return save_path


def plot_monthly_returns(data, save_path):
    """月度收益柱状图"""
    close = data['close']
    monthly = close.resample('M').last()
    monthly_ret = monthly.pct_change().dropna() * 100

    fig, ax = plt.subplots(figsize=(10, 4))
    colors = ['#26a69a' if v >= 0 else '#ef5350' for v in monthly_ret.values]
    bars = ax.bar(range(len(monthly_ret)), monthly_ret.values, color=colors, width=0.7)
    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.set_title('月度收益率 (%)', fontsize=12, fontweight='bold')
    ax.set_xticks(range(0, len(monthly_ret), 3))
    ax.set_xticklabels([monthly_ret.index[i].strftime('%Y-%m') for i in range(0, len(monthly_ret), 3)],
                      rotation=45, fontsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    return save_path


def plot_risk_reward(data, save_path):
    """风险收益比图"""
    close = data['close']
    recent_high = data['recent_high_60']
    recent_low = data['recent_low_60']
    hist_high = data['hist_high']
    hist_low = data['hist_low']
    close_now = data['close_now']
    atr = data['atr']

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)

    # 背景色块
    ax.axvspan(0, 30, alpha=0.1, color='#ef5350')
    ax.axvspan(70, 100, alpha=0.1, color='#26a69a')
    ax.axhspan(0, 30, alpha=0.1, color='#26a69a')
    ax.axhspan(70, 100, alpha=0.1, color='#ef5350')

    # 当前位置
    pos_x = (close_now - recent_low) / (recent_high - recent_low) * 100 if recent_high != recent_low else 50
    ax.scatter([pos_x], [data['rsi14']], s=300, color='#1e88e5', zorder=5, edgecolors='white', linewidth=2)
    ax.annotate(f'{close_now:.1f}\n({pos_x:.0f}%)', xy=(pos_x, data['rsi14']), fontsize=9,
               xytext=(pos_x + 5, data['rsi14'] + 5))

    # ATR止损线
    stop_loss = close_now - 1.5 * atr
    ax.axhline(y=stop_loss, color='#ff9800', linewidth=1.5, linestyle='--', label=f'止损位: {stop_loss:.1f}')

    ax.set_xlabel('价格位置 (近60日低点 → 近60日高点)', fontsize=10)
    ax.set_ylabel('RSI', fontsize=10)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xticklabels(['低点', '25%', '中点', '75%', '高点'])
    ax.set_yticks([0, 30, 50, 70, 100])
    ax.set_yticklabels(['0', '30\n超卖', '50', '70\n超买', '100'])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    return save_path


def generate_visual_report(symbol, days=252):
    """生成完整可视化报告"""
    results_dir = r'D:\python_work\results'
    os.makedirs(results_dir, exist_ok=True)

    data = analyze_stock(symbol, days)

    print(f'\n{"="*60}')
    print(f'  {symbol} 可视化分析报告')
    print(f'{"="*60}\n')

    # 图表1: K线+指标
    chart_path = os.path.join(results_dir, f'{symbol}_chart.png')
    plot_kline_with_indicators(data, chart_path)
    print(f'[图表1] K线+指标图 → {chart_path}')

    # 图表2: 月度收益
    monthly_path = os.path.join(results_dir, f'{symbol}_monthly.png')
    plot_monthly_returns(data, monthly_path)
    print(f'[图表2] 月度收益图 → {monthly_path}')

    # 图表3: 风险收益
    risk_path = os.path.join(results_dir, f'{symbol}_risk.png')
    plot_risk_reward(data, risk_path)
    print(f'[图表3] 风险收益图 → {risk_path}')

    # 文本报告
    report = generate_text_report(data)
    report_path = os.path.join(results_dir, f'{symbol}_report.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f'[报告] Markdown → {report_path}')

    return chart_path, monthly_path, risk_path, report_path, data


def generate_text_report(data):
    """生成带图标的Markdown文本报告"""
    sym = data['symbol']
    close = data['close_now']
    adx = data['adx']
    rsi14 = data['rsi14']
    rsi6 = data['rsi6']
    macd_hist = data['macd_hist']
    ma5 = data['ma5']
    ma20 = data['ma20']
    ma60 = data['ma60']
    atr = data['atr']
    atr_pct = data['atr_pct']
    rets = data['rets']
    ma_slopes = data['ma_slopes']
    dmi_cross = data['dmi_cross']
    hist_high = data['hist_high']
    hist_low = data['hist_low']
    hist_high_date = data['hist_high_date']
    hist_low_date = data['hist_low_date']
    recent_high = data['recent_high_60']
    recent_low = data['recent_low_60']
    close_now = data['close_now']

    # 图标判断
    def badge(cond, true_txt, false_txt='⚪'):
        return true_txt if cond else false_txt

    trend_icon = badge(adx > 30, '🟢 强趋势', badge(adx < 20, '🔴 震荡', '🟡 中性'))
    rsi_icon = badge(rsi14 > 70, '🔴 超买', badge(rsi14 < 30, '🟢 超卖', '⚪ 中性'))
    ma_icon = badge(ma5 > ma20 > ma60, '🟢 多头排列', badge(ma5 < ma20 < ma60, '🔴 空头排列', '🟡 混乱'))
    macd_icon = badge(macd_hist > 0, '🟢 MACD多头', '🔴 MACD空头')
    dmi_icon = badge(dmi_cross == 'golden', '🟢 DMI金叉', badge(dmi_cross == 'death', '🔴 DMI死叉', '⚪ 无叉'))

    report = f"""# {sym} 个股分析报告

**生成日期**: {data['date_now'].strftime('%Y-%m-%d')}
**数据区间**: 近{data['close'].index[0].strftime('%Y-%m-%d')} 至 {data['date_now'].strftime('%Y-%m-%d')}

---

## 📊 价格与趋势

```
收盘价: {close:.2f}
历史高点: {hist_high:.2f} ({hist_high_date.strftime('%Y-%m-%d')})
历史低点: {hist_low:.2f} ({hist_low_date.strftime('%Y-%m-%d')})
当前位置: {(close/hist_high)*100:.1f}% (距高点)
```

### 📈 区间收益

| 周期 | 收益率 | 状态 |
|------|--------|------|
| 近5日 | {rets[5]:+.1f}% | {'🟢 强势' if rets[5] > 3 else '🔴 回调' if rets[5] < -3 else '⚪ 震荡'} |
| 近20日 | {rets[20]:+.1f}% | {'🟢 强势' if rets[20] > 10 else '🔴 回调' if rets[20] < -10 else '⚪ 震荡'} |
| 近60日 | {rets[60]:+.1f}% | {'🟢 强势' if rets[60] > 15 else '🔴 回调' if rets[60] < -15 else '⚪ 震荡'} |
| 近120日 | {rets[120]:+.1f}% | {'🟢 强势' if rets[120] > 20 else '🔴 回调' if rets[120] < -20 else '⚪ 震荡'} |

---

## 🎯 核心指标仪表盘

| 指标 | 当前值 | 参考区间 | 信号 |
|------|--------|----------|------|
| **ADX** | {adx:.1f} | >30强趋势 / <20震荡 | {trend_icon} |
| **+DI / -DI** | {data['plus_di']:.1f} / {data['minus_di']:.1f} | 差值越大趋势越强 | {'🟢 多方主导' if data['plus_di'] > data['minus_di'] else '🔴 空方主导'} |
| **RSI(14)** | {rsi14:.1f} | >70超买 / <30超卖 | {rsi_icon} |
| **RSI(6)** | {rsi6:.1f} | >75短期过热 | {'🔴 短期超买' if rsi6 > 75 else '🟢 正常' if rsi6 < 50 else '⚪ 中性'} |
| **MACD柱体** | {macd_hist:+.3f} | >0多头 / <0空头 | {macd_icon} |
| **均线状态** | MA5={ma5:.1f} / MA20={ma20:.1f} / MA60={ma60:.1f} | 价格vs均系 | {ma_icon} |

---

## 📉 均线系统

| 均线 | 当前值 | 10日斜率 | 方向 |
|------|--------|----------|------|
| MA5 | {ma5:.2f} | {ma_slopes[5]:+.1f}% | {'↗ 上升' if ma_slopes[5] > 2 else '↘ 下降' if ma_slopes[5] < -2 else '→ 横盘'} |
| MA20 | {ma20:.2f} | {ma_slopes[20]:+.1f}% | {'↗ 上升' if ma_slopes[20] > 2 else '↘ 下降' if ma_slopes[20] < -2 else '→ 横盘'} |
| MA60 | {ma60:.2f} | {ma_slopes[60]:+.1f}% | {'↗ 上升' if ma_slopes[60] > 2 else '↘ 下降' if ma_slopes[60] < -2 else '→ 横盘'} |

---

## 🔐 波动与风险

```
ATR(14): {atr:.2f} ({atr_pct:.1f}% of price)
日均波动: ±{atr/2:.1f}
近60日低点: {recent_low:.2f}  /  高点: {recent_high:.2f}
```

### 支撑与阻力

```
阻力位: {hist_high:.1f} (历史高点)
       {(close * 1.05):.1f} (+5%)
       {(close * 1.10):.1f} (+10%)

当前价: {close:.2f}

支撑1: {(close - atr*1):.1f} (-{atr_pct:.1f}%, ATR 1x)
支撑2: {(close - atr*1.5):.1f} (-{atr_pct*1.5:.1f}%, ATR 1.5x)
支撑3: {recent_low:.1f} (近60日低点)
```

---

## 📋 综合判断

**趋势**: {trend_icon.replace('🟢 强趋势','极强上升趋势') if adx > 40 else trend_icon.replace('🟡 中性','趋势不明') if adx < 25 else trend_icon}

**操作建议**:

"""

    # 生成具体建议
    if adx > 40 and rsi14 < 70 and ma5 > ma20:
        report += f"""✅ **积极买入**
- ADX={adx:.0f} 极强趋势，均线多头，机构在控盘
- 建议在 {close - atr*0.5:.0f}~{close:.0f} 区间买入
- 止损: {close - atr*1.5:.0f} (ATR 1.5x = {atr_pct*1.5:.1f}%回撤)
- 目标: {hist_high:.0f} (历史高点)

"""
    elif adx > 30 and ma5 > ma20:
        report += f"""✅ **可以考虑买入**
- ADX={adx:.0f} 确认趋势，但上方临近阻力
- 建议分批买入，首批50%仓位
- 止损: {close - atr*1.5:.0f}
- 目标: {close * 1.10:.0f} (+10%)

"""
    elif rsi14 > 70:
        report += f"""⚠️ **谨慎追高**
- RSI={rsi14:.0f} 短期超买，短期可能回调
- 建议等RSI回落至50-60再买入
- 若坚持买入，止损放在 {close - atr*2:.0f} (ATR 2x)

"""
    elif adx < 20:
        report += f"""⏸️ **观望为主**
- ADX={adx:.0f} 市场震荡，无明确趋势
- 建议等ADX>25趋势确认后再介入
- 如有持仓考虑高抛低吸

"""
    else:
        report += f"""⚖️ **中性观望**
- ADX={adx:.0f} 趋势不明确
- 建议等待更明确信号

"""

    report += f"""---

*报告自动生成 | 数据来源: Wind Oracle*
"""
    return report


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--symbol', default='300433.SZ')
    parser.add_argument('--days', type=int, default=252)
    args = parser.parse_args()

    generate_visual_report(args.symbol, args.days)