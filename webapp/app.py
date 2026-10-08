#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Flask Web Application for Quantitative Trading System.

Routes only - all database logic is in database.py.
"""

from flask import Flask, render_template, jsonify, request, send_from_directory
import os

from .database import BacktestDB
from .results_view import bp as results_bp
from .dashboard import bp as dashboard_bp
from .pms import bp as pms_bp

app = Flask(__name__)
db = BacktestDB(os.path.join(os.path.dirname(__file__), 'backtest.db'))
app.register_blueprint(results_bp)
app.register_blueprint(dashboard_bp)
app.register_blueprint(pms_bp)


def _prewarm_rationale():
    """后台预热：当日全市场因子矩阵进缓存，用户首次点击个股逻辑即秒回"""
    import threading
    def _warm():
        try:
            from dotenv import load_dotenv
            load_dotenv()
            from data.layered import LayeredDataSource
            from webapp.rationale import _load_matrices
            _load_matrices(LayeredDataSource(), __import__('datetime').date.today())
        except Exception as e:
            print(f'[prewarm] skip: {e}')
    threading.Thread(target=_warm, daemon=True).start()


_prewarm_rationale()


@app.route('/')
def index():
    """首页 → PMS（组合管理）"""
    from flask import redirect
    return redirect('/pms')


@app.route('/backtests')
def backtests_page():
    """历史回测记录"""
    backtests = db.list_backtests(limit=20)
    return render_template('index.html', backtests=backtests)


@app.route('/backtest/<int:backtest_id>')
def backtest_detail(backtest_id):
    """Backtest detail page with K-line chart"""
    backtest = db.load_backtest(backtest_id)
    if not backtest:
        return "Backtest not found", 404
    trades = db.get_trades(backtest_id)
    return render_template('backtest_detail.html', backtest=backtest, trades=trades)


@app.route('/api/backtest/<int:backtest_id>/trades')
def api_backtest_trades(backtest_id):
    """Get trades for a backtest"""
    trades = db.get_trades(backtest_id)
    return jsonify(trades)


@app.route('/api/backtest/<int:backtest_id>/symbols')
def api_backtest_symbols(backtest_id):
    """Get unique symbols in a backtest"""
    symbols = db.get_symbols(backtest_id)
    return jsonify(symbols)


@app.route('/api/kline/<symbol>')
def api_kline(symbol):
    """Get K-line data for a symbol"""
    klines = db.load_kline(symbol)
    result = []
    for k in klines:
        result.append({
            'time': k['date'][:10] if isinstance(k['date'], str) else k['date'],
            'open': k['open'],
            'high': k['high'],
            'low': k['low'],
            'close': k['close'],
            'volume': k.get('volume', 0)
        })
    return jsonify(result)


@app.route('/api/backtest/<int:backtest_id>/signals/<symbol>')
def api_backtest_signals(backtest_id, symbol):
    """Get buy/sell signals for a symbol in a backtest"""
    signals = db.get_signals(backtest_id, symbol)
    return jsonify(signals)


@app.route('/api/backtest/<int:backtest_id>/attribution')
def api_backtest_attribution(backtest_id):
    """Get performance attribution by symbol"""
    result = db.get_attribution(backtest_id)
    return jsonify(result)


@app.route('/api/equity_curve/<int:backtest_id>')
def api_equity_curve(backtest_id):
    """Get equity curve for a backtest"""
    curve = db.get_equity_curve(backtest_id)
    return jsonify(curve)


@app.route('/save', methods=['POST'])
def save_backtest():
    """Save backtest results via HTTP"""
    data = request.json
    backtest_id = db.save_backtest(data)
    return jsonify({'id': backtest_id, 'status': 'saved'})


@app.route('/kline/<symbol>')
def serve_kline(symbol):
    """Serve K-line data file for a symbol"""
    return send_from_directory(app.config.get('KLINE_DIR', ''), f'{symbol}.json')


if __name__ == '__main__':
    debug = os.getenv('FLASK_DEBUG', 'false').lower() in ('1', 'true', 'yes')
    if not debug:
        from waitress import serve
        print('waitress serving on http://0.0.0.0:5000 (PMS: /pms)')
        serve(app, host='0.0.0.0', port=5000, threads=8)
        raise SystemExit(0)
    print("=" * 60)
    print("Quantitative Trading System - Web Server")
    print("=" * 60)
    print("Starting server at http://localhost:5000")
    print("Press Ctrl+C to stop")
    print("=" * 60)
    app.run(debug=debug, host='0.0.0.0', port=5000)