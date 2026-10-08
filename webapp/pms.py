#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PMS（组合管理系统）—— 持仓/交易/归因 一目了然

/api/pms/data 汇聚账本 + 最新缓存价格 + 当日归因；
/pms 单页应用（总览/持仓/交易/归因 四标签）。
"""

import json
import os
from datetime import date, timedelta

from flask import Blueprint, render_template, jsonify

from utils.logger import get_logger

logger = get_logger(__name__)

bp = Blueprint('pms', __name__, url_prefix='/pms')

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # webapp/ 的上级 = 项目根
LEDGER_PATH = os.path.join(ROOT, 'live', 'ledger.json')


def _latest_prices(symbols):
    """从本地 parquet 缓存取最新两日收盘价（快，无需 Wind 会话）"""
    import pandas as pd
    prices, prev = {}, {}
    kline_dir = os.path.join(ROOT, 'data_cache', 'kline')
    for sym in symbols:
        path = os.path.join(kline_dir, f"{sym.replace('.', '_')}.parquet")
        if not os.path.exists(path):
            continue
        try:
            df = pd.read_parquet(path)
            if df.empty:
                continue
            prices[sym] = float(df['close'].iloc[-1])
            if len(df) > 1:
                prev[sym] = float(df['close'].iloc[-2])
        except Exception:
            continue
    return prices, prev


@bp.route('/')
def index():
    return render_template('pms.html')


@bp.route('/api/symbol/<symbol>')
def api_symbol(symbol):
    """个股钻取：K线（近250日）+ 该股成交记录 + 当前持仓 + 成本线"""
    import pandas as pd

    def _kline_from_parquet(sym):
        path = os.path.join(ROOT, 'data_cache', 'kline', f"{sym.replace('.', '_')}.parquet")
        if not os.path.exists(path):
            return pd.DataFrame()
        try:
            return pd.read_parquet(path)
        except Exception:
            return pd.DataFrame()

    df = _kline_from_parquet(symbol)
    if df.empty:
        try:
            from dotenv import load_dotenv
            load_dotenv()
            from data.layered import LayeredDataSource
            from datetime import date, timedelta
            ds = LayeredDataSource()
            df = ds.preload_batch([symbol], date.today() - timedelta(days=450), date.today())\
                .get(symbol, pd.DataFrame())
        except Exception:
            df = pd.DataFrame()
    if df.empty:
        return jsonify({'error': f'无 {symbol} K线数据'}), 404

    df = df.tail(250)
    # MA5 / MA20
    ma5 = df['close'].rolling(5).mean()
    ma20 = df['close'].rolling(20).mean()

    try:
        with open(LEDGER_PATH, encoding='utf-8') as f:
            ledger = json.load(f)
    except FileNotFoundError:
        ledger = {'positions': {}, 'cost': {}, 'trades': []}

    return jsonify({
        'symbol': symbol,
        'kline': {
            'dates': [str(d)[:10] for d in df['date']],
            # ECharts candlestick: [open, close, low, high]
            'ohlc': [[round(float(o), 2), round(float(c), 2), round(float(l), 2), round(float(h), 2)]
                     for o, c, l, h in zip(df['open'], df['close'], df['low'], df['high'])],
            'volume': [float(v) for v in df['volume']],
            'ma5': [None if pd.isna(v) else round(float(v), 2) for v in ma5],
            'ma20': [None if pd.isna(v) else round(float(v), 2) for v in ma20],
        },
        'trades': [t for t in ledger.get('trades', []) if t['symbol'] == symbol],
        'position': {
            'qty': ledger.get('positions', {}).get(symbol, 0),
            'cost': ledger.get('cost', {}).get(symbol, 0),
        },
    })


@bp.route('/api/data')
def api_data():
    try:
        with open(LEDGER_PATH, encoding='utf-8') as f:
            ledger = json.load(f)
    except FileNotFoundError:
        return jsonify({'error': '账本不存在，先运行 python main.py daily --execute paper'}), 404

    positions = ledger.get('positions', {})
    trades = ledger.get('trades', [])
    nav_history = ledger.get('nav_history', [])

    prices, prev_prices = _latest_prices(list(positions.keys()))

    # 持仓明细（含当日归因）
    total_mv = sum(q * prices.get(s, 0.0) for s, q in positions.items())
    rows = []
    day_pnl = 0.0
    for s, q in positions.items():
        cost = ledger.get('cost', {}).get(s, 0.0)
        px = prices.get(s, 0.0)
        pp = prev_prices.get(s, px)
        day_chg = (px / pp - 1) if pp > 0 else 0.0
        contrib = q * (px - pp)
        day_pnl += contrib
        rows.append({
            'symbol': s, 'qty': q, 'cost': round(cost, 3), 'price': round(px, 3),
            'day_chg': round(day_chg, 4), 'pnl': round((px - cost) * q, 0),
            'weight': round(q * px / total_mv, 4) if total_mv > 0 else 0,
            'day_contrib': round(contrib, 0),
        })
    rows.sort(key=lambda r: -r['weight'])

    nav = nav_history[-1]['nav'] if nav_history else ledger.get('cash', 0)
    invested = sum(ledger.get('cost', {}).get(s, 0) * q for s, q in positions.items())

    return jsonify({
        'as_of': nav_history[-1]['date'] if nav_history else str(date.today()),
        'kpi': {
            'nav': nav, 'cash': round(ledger.get('cash', 0), 2),
            'invested': round(invested, 2),
            'total_pnl': round(nav - invested, 0),
            'day_pnl': round(day_pnl, 0),
            'n_positions': len(positions), 'n_trades': len(trades),
        },
        'positions': rows,
        'trades': list(reversed(trades)),   # 最新在前
        'nav_history': nav_history,
    })
