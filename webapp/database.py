#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Database access layer for backtest results and K-line data.
Separated from Flask routes to keep concerns clean.
"""

import sqlite3
import json
import os
from datetime import datetime, date as date_type
from typing import Optional, List, Dict, Any
import pandas as pd


class BacktestDB:
    """Database access layer for backtest persistence."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_schema()

    def _get_conn(self):
        """Get a database connection with row factory."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self):
        """Initialize database schema, creating tables if they don't exist."""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        conn = self._get_conn()
        cursor = conn.cursor()

        # Backtest results table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS backtests (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                created_at TEXT NOT NULL,
                start_date TEXT,
                end_date TEXT,
                initial_capital REAL,
                final_capital REAL,
                total_return REAL,
                annual_return REAL,
                sharpe_ratio REAL,
                max_drawdown REAL,
                win_rate REAL,
                trade_count INTEGER,
                config TEXT,
                equity_curve TEXT,
                benchmark_return REAL,
                benchmark_symbol TEXT,
                UNIQUE(name, created_at)
            )
        ''')

        # Add equity_curve column if not exists (existing DBs)
        try:
            cursor.execute('SELECT equity_curve FROM backtests LIMIT 1')
        except sqlite3.OperationalError:
            cursor.execute('ALTER TABLE backtests ADD COLUMN equity_curve TEXT')

        # Add benchmark columns if not exists (existing DBs)
        cursor.execute("PRAGMA table_info(backtests)")
        bt_columns = [col[1] for col in cursor.fetchall()]
        if 'benchmark_return' not in bt_columns:
            cursor.execute('ALTER TABLE backtests ADD COLUMN benchmark_return REAL')
        if 'benchmark_symbol' not in bt_columns:
            cursor.execute('ALTER TABLE backtests ADD COLUMN benchmark_symbol TEXT')

        # Trades table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                backtest_id INTEGER NOT NULL,
                symbol TEXT NOT NULL,
                action TEXT NOT NULL,
                trade_date TEXT NOT NULL,
                price REAL NOT NULL,
                quantity INTEGER NOT NULL,
                commission REAL,
                reason TEXT,
                signal_score INTEGER,
                pnl REAL,
                conditions TEXT,
                FOREIGN KEY (backtest_id) REFERENCES backtests(id)
            )
        ''')

        # Add conditions column if not exists
        cursor.execute("PRAGMA table_info(trades)")
        columns = [col[1] for col in cursor.fetchall()]
        if 'conditions' not in columns:
            cursor.execute("ALTER TABLE trades ADD COLUMN conditions TEXT")

        # K-line data table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS kline_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                date TEXT NOT NULL,
                open REAL NOT NULL,
                high REAL NOT NULL,
                low REAL NOT NULL,
                close REAL NOT NULL,
                volume REAL,
                UNIQUE(symbol, date)
            )
        ''')

        # Data load log table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS data_load_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                load_date TEXT NOT NULL,
                record_count INTEGER,
                start_date TEXT,
                end_date TEXT,
                source TEXT,
                backtest_id INTEGER
            )
        ''')

        cursor.execute('CREATE INDEX IF NOT EXISTS idx_load_symbol ON data_load_log(symbol)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_load_date ON data_load_log(load_date)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_trades_backtest ON trades(backtest_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_kline_symbol ON kline_data(symbol)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_kline_date ON kline_data(date)')

        conn.commit()
        conn.close()

    def save_backtest(self, data: Dict[str, Any]) -> int:
        """Save a backtest and its trades. Returns backtest_id."""
        conn = self._get_conn()
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO backtests (
                name, created_at, start_date, end_date, initial_capital, final_capital,
                total_return, annual_return, sharpe_ratio, max_drawdown, win_rate,
                trade_count, config, equity_curve, benchmark_return, benchmark_symbol
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            data['name'],
            datetime.now().isoformat(),
            data.get('start_date'),
            data.get('end_date'),
            data['initial_capital'],
            data['final_capital'],
            data['total_return'],
            data['annual_return'],
            data['sharpe_ratio'],
            data['max_drawdown'],
            data['win_rate'],
            data['trade_count'],
            json.dumps(data.get('config', {})),
            json.dumps(data.get('equity_curve', [])),
            data.get('benchmark_return'),
            data.get('benchmark_symbol', '')
        ))

        backtest_id = cursor.lastrowid

        # Insert trades
        for trade in data.get('trades', []):
            pnl = trade.get('pnl')
            conditions_json = json.dumps(trade.get('conditions', []))
            cursor.execute('''
                INSERT INTO trades (
                    backtest_id, symbol, action, trade_date, price, quantity,
                    commission, reason, signal_score, pnl, conditions
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                backtest_id,
                trade['symbol'],
                trade['action'],
                trade['date'],
                trade['price'],
                trade['quantity'],
                trade.get('commission', 0),
                trade.get('reason', ''),
                trade.get('signal_score', 0),
                pnl,
                conditions_json
            ))

        conn.commit()
        conn.close()
        return backtest_id

    def load_backtest(self, backtest_id: int) -> Optional[Dict]:
        """Load a backtest by ID."""
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM backtests WHERE id = ?', (backtest_id,))
        row = cursor.fetchone()
        conn.close()
        return dict(row) if row else None

    def list_backtests(self, limit: int = 20) -> List[Dict]:
        """List recent backtests."""
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute('''
            SELECT id, name, created_at, total_return, trade_count, sharpe_ratio
            FROM backtests
            ORDER BY created_at DESC
            LIMIT ?
        ''', (limit,))
        rows = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return rows

    def get_trades(self, backtest_id: int) -> List[Dict]:
        """Get all trades for a backtest."""
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute('''
            SELECT * FROM trades
            WHERE backtest_id = ?
            ORDER BY trade_date, id
        ''', (backtest_id,))
        rows = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return rows

    def get_symbols(self, backtest_id: int) -> List[str]:
        """Get unique symbols in a backtest."""
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute('''
            SELECT DISTINCT symbol FROM trades
            WHERE backtest_id = ?
            ORDER BY symbol
        ''', (backtest_id,))
        rows = [row['symbol'] for row in cursor.fetchall()]
        conn.close()
        return rows

    def get_equity_curve(self, backtest_id: int) -> List:
        """Get equity curve for a backtest."""
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute('SELECT equity_curve FROM backtests WHERE id = ?', (backtest_id,))
        row = cursor.fetchone()
        conn.close()
        if row and row['equity_curve']:
            return json.loads(row['equity_curve'])
        return []

    def get_attribution(self, backtest_id: int) -> List[Dict]:
        """Get performance attribution by symbol."""
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute('''
            SELECT
                symbol,
                SUM(CASE WHEN action = 'BUY' THEN 1 ELSE 0 END) as buy_count,
                SUM(CASE WHEN action = 'SELL' THEN 1 ELSE 0 END) as sell_count,
                SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as profit_count,
                SUM(CASE WHEN pnl < 0 THEN 1 ELSE 0 END) as loss_count,
                SUM(COALESCE(pnl, 0)) as total_pnl
            FROM trades
            WHERE backtest_id = ?
            GROUP BY symbol
            ORDER BY total_pnl DESC
        ''', (backtest_id,))
        rows = cursor.fetchall()
        conn.close()

        total_pnl = sum(row['total_pnl'] for row in rows) if rows else 1
        result = []
        for row in rows:
            contribution = (row['total_pnl'] / total_pnl * 100) if total_pnl != 0 else 0
            result.append({
                'symbol': row['symbol'],
                'buy_count': row['buy_count'],
                'sell_count': row['sell_count'],
                'profit_count': row['profit_count'],
                'loss_count': row['loss_count'],
                'total_pnl': row['total_pnl'],
                'profit_contribution': contribution
            })
        return result

    def get_signals(self, backtest_id: int, symbol: str) -> List[Dict]:
        """Get buy/sell signals for a symbol in a backtest."""
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute('''
            SELECT trade_date, action, price, reason, pnl, conditions
            FROM trades
            WHERE backtest_id = ? AND symbol = ?
            ORDER BY trade_date
        ''', (backtest_id, symbol))
        signals = []
        for row in cursor.fetchall():
            sig = dict(row)
            try:
                sig['conditions'] = json.loads(sig.get('conditions') or '[]')
            except:
                sig['conditions'] = []
            signals.append(sig)
        conn.close()
        return signals

    def save_kline(self, symbol: str, kline_df: pd.DataFrame, source: str = 'wind', backtest_id: int = None) -> int:
        """Save K-line data. Returns record count."""
        conn = self._get_conn()
        cursor = conn.cursor()
        record_count = 0
        start_date = None
        end_date = None

        for _, row in kline_df.iterrows():
            if pd.isna(row['open']) or pd.isna(row['close']):
                continue
            cursor.execute('''
                INSERT OR REPLACE INTO kline_data (symbol, date, open, high, low, close, volume)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                symbol,
                str(row['date'])[:10],
                float(row['open']),
                float(row['high']),
                float(row['low']),
                float(row['close']),
                float(row.get('volume', 0)) if pd.notna(row.get('volume')) else 0
            ))
            record_count += 1
            if start_date is None:
                start_date = str(row['date'])[:10]
            end_date = str(row['date'])[:10]

        # Log the load operation
        load_date = date_type.today().isoformat()
        cursor.execute('''
            INSERT INTO data_load_log (symbol, load_date, record_count, start_date, end_date, source, backtest_id)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (symbol, load_date, record_count, start_date, end_date, source, backtest_id))

        conn.commit()
        conn.close()
        return record_count

    def load_kline(self, symbol: str, start_date: str = None, end_date: str = None) -> List[Dict]:
        """Load K-line data from database."""
        conn = self._get_conn()
        cursor = conn.cursor()

        if start_date and end_date:
            cursor.execute('''
                SELECT date, open, high, low, close, volume
                FROM kline_data
                WHERE symbol = ? AND date >= ? AND date <= ?
                ORDER BY date
            ''', (symbol, str(start_date)[:10], str(end_date)[:10]))
        else:
            cursor.execute('''
                SELECT date, open, high, low, close, volume
                FROM kline_data
                WHERE symbol = ?
                ORDER BY date
            ''', (symbol,))

        rows = cursor.fetchall()
        conn.close()
        return [dict(row) for row in rows]

    def clear_backtest(self, backtest_id: int):
        """Delete a backtest and its trades."""
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute('DELETE FROM trades WHERE backtest_id = ?', (backtest_id,))
        cursor.execute('DELETE FROM backtests WHERE id = ?', (backtest_id,))
        conn.commit()
        conn.close()