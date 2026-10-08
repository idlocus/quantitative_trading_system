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


@bp.route('/api/intraday/<symbol>/<day>')
def api_intraday(symbol, day):
    """分时数据（新浪1分钟线，免费源仅近几个交易日）"""
    import pandas as pd
    try:
        import akshare as ak
        pre = {'6': 'sh', '0': 'sz', '3': 'sz', '4': 'bj', '8': 'bj', '9': 'bj'}.get(symbol[0], 'sh')
        raw = ak.stock_zh_a_minute(symbol=f"{pre}{symbol.split('.')[0]}", period='1', adjust='')
    except Exception as e:
        return jsonify({'error': f'分时数据获取失败: {e}'}), 502

    raw['day'] = raw['day'].astype(str)
    sub = raw[raw['day'].str.startswith(day)]
    if sub.empty:
        return jsonify({'error': f'{day} 无分时数据（免费源仅保留近几个交易日）'}), 404

    price = sub['close'].astype(float)
    vol = sub['volume'].astype(float)
    # 均价线（累计成交量加权）与前收盘（用于红绿着色）
    cum_pv = (price * vol).cumsum() / vol.cumsum().replace(0, float('nan'))
    df_full = _kline_df(symbol)
    prev_close = None
    if df_full is not None and not df_full.empty:
        d = df_full[df_full['date'].astype(str).str[:10] < day]
        if not d.empty:
            prev_close = float(d['close'].iloc[-1])

    return jsonify({
        'symbol': symbol, 'day': day, 'prev_close': prev_close,
        'times': [t[11:16] for t in sub['day']],
        'price': [round(float(v), 3) for v in price],
        'avg': [None if pd.isna(v) else round(float(v), 3) for v in cum_pv],
        'volume': [float(v) for v in vol],
    })


def _kline_df(symbol):
    import pandas as pd
    path = os.path.join(ROOT, 'data_cache', 'kline', f"{symbol.replace('.', '_')}.parquet")
    if os.path.exists(path):
        try:
            return pd.read_parquet(path)
        except Exception:
            return None
    return None


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

    # 剔除残缺行（当日快照可能只有收盘价，NaN会破坏蜡烛图）
    df = df.dropna(subset=['open', 'high', 'low', 'close', 'volume'])
    df = df.tail(250)
    # MA 均线组（Wind 默认 5/10/20/30）
    mas = {f'ma{n}': df['close'].rolling(n).mean() for n in (5, 10, 20, 30)}

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
            **{k: [None if pd.isna(v) else round(float(v), 2) for v in series]
               for k, series in mas.items()},
        },
        'trades': [t for t in ledger.get('trades', []) if t['symbol'] == symbol],
        'position': {
            'qty': ledger.get('positions', {}).get(symbol, 0),
            'cost': ledger.get('cost', {}).get(symbol, 0),
        },
    })


@bp.route('/api/rationale/<symbol>')
def api_rationale(symbol):
    """推荐逻辑（独立异步端点：全市场矩阵计算较慢，不阻塞K线渲染）"""
    r = _rationale(symbol, None)
    return jsonify(r if r else {})


def _rationale(symbol, kline_df):
    """推荐逻辑（买入口径：首笔买入日；无成交则最新日）"""
    try:
        from dotenv import load_dotenv
        load_dotenv()
        from data.layered import LayeredDataSource
        from webapp.rationale import build_rationale
        trades = []
        try:
            with open(LEDGER_PATH, encoding='utf-8') as f:
                trades = json.load(f).get('trades', [])
        except FileNotFoundError:
            pass
        day = None
        for t in trades:
            if t['symbol'] == symbol and t['side'] == 'BUY':
                day = date.fromisoformat(t['date'][:10])
                break
        if day is None and kline_df is not None and not kline_df.empty:
            d = kline_df['date'].iloc[-1]
            day = d.date() if hasattr(d, 'date') else d
        if day is None:
            day = date.today()
        ds = LayeredDataSource()
        return build_rationale(ds, symbol, day)
    except Exception as e:
        logger.warning(f'rationale failed for {symbol}: {e}')
        return None


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
