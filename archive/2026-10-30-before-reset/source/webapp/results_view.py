#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""研究结果浏览器 Blueprint：每日交易报告（主）+ 研究档案（次）"""

import os
import re
from datetime import datetime

from flask import Blueprint, render_template, abort, send_from_directory
import markdown

bp = Blueprint('results', __name__, url_prefix='/results')

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(ROOT, 'results')
JOURNAL_DIR = os.path.join(ROOT, 'journal')

CATEGORIES = [
    ('daily_scan', '每日选股'),
    ('factor_screen', '因子批量筛选'),
    ('factor_eval', '因子评价'),
    ('fundamental_eval', '基本面因子'),
    ('portfolio_', '组合回测'),
    ('walkforward', '样本外验证'),
    ('fullmarket', '全市场验证'),
    ('oos_', '真样本外验证'),
    ('backtest_report', '策略回测'),
    ('rolling', '滚动回测'),
]


def _category(filename: str) -> str:
    for prefix, label in CATEGORIES:
        if filename.startswith(prefix):
            return label
    return '研究档案'


def _mtime(path) -> str:
    try:
        return datetime.fromtimestamp(os.path.getmtime(path)).strftime('%Y-%m-%d %H:%M')
    except OSError:
        return '-'


def _journal_summary(path) -> dict:
    """从交易日志 md 提取摘要：净值/累计盈亏/成交笔数/持仓数"""
    out = {'nav': None, 'total_pnl': None, 'trades': 0, 'positions': 0, 'mkt': ''}
    try:
        text = open(path, encoding='utf-8').read()
        m = re.search(r'账户净值 \*\*([\d,\.]+)\*\*（累计投入约 [\d,\.]+，累计盈亏 ([\+\-][\d,]+)）', text)
        if m:
            out['nav'] = m.group(1)
            out['total_pnl'] = m.group(2)
        # 成交行：| 成交 | 代码 | 买入/卖出 |（表格行含"买入"或"卖出"）
        out['trades'] = len(re.findall(r'\| (?:买入|卖出) \|', text))
        # 持仓数：持仓表行 "| 代码 |" 以6位数字开头的行
        out['positions'] = len(re.findall(r'\| \d{6}\.', text))
        m2 = re.search(r'沪深300 (?:昨收|今日) ([\+\-][\d\.]+%)', text)
        if m2:
            out['mkt'] = m2.group(1)
    except Exception:
        pass
    return out


def _journal_dates() -> list:
    """已有日志的日期列表（升序）"""
    if not os.path.isdir(JOURNAL_DIR):
        return []
    return sorted(f[:-3] for f in os.listdir(JOURNAL_DIR)
                  if re.fullmatch(r'\d{4}-\d{2}-\d{2}\.md', f))


@bp.route('/')
def index():
    """每日交易报告（主）+ 研究档案（次）"""
    journals = []
    for d in reversed(_journal_dates()):  # 最新在前
        path = os.path.join(JOURNAL_DIR, f'{d}.md')
        journals.append({'date': d, **_journal_summary(path),
                         'weekday': datetime.strptime(d, '%Y-%m-%d').strftime('%A')})

    reports = []
    if os.path.isdir(RESULTS_DIR):
        for fn in sorted(os.listdir(RESULTS_DIR), reverse=True):
            path = os.path.join(RESULTS_DIR, fn)
            if not os.path.isfile(path) or fn.startswith('.') or not fn.endswith(('.md', '.html')):
                continue
            reports.append({'name': fn, 'category': _category(fn), 'mtime': _mtime(path),
                            'is_html': fn.endswith('.html'), 'size_kb': os.path.getsize(path) // 1024})
    return render_template('results_index.html', journals=journals, reports=reports)


@bp.route('/view/<path:name>')
def view(name):
    """渲染 Markdown 或直接返回 HTML；交易日志附前后日导航"""
    if '/' in name or '..' in name:
        abort(404)
    for base_dir in (RESULTS_DIR, JOURNAL_DIR):
        path = os.path.join(base_dir, name)
        if os.path.isfile(path):
            break
    else:
        abort(404)

    if name.endswith('.html'):
        return send_from_directory(base_dir, name)

    with open(path, encoding='utf-8') as f:
        text = f.read()

    # 交易日志：前后日导航
    nav = None
    m = re.fullmatch(r'(\d{4}-\d{2}-\d{2})\.md', name)
    if m and base_dir == JOURNAL_DIR:
        dates = _journal_dates()
        i = dates.index(m.group(1))
        nav = {'prev': dates[i - 1] if i > 0 else None,
               'next': dates[i + 1] if i < len(dates) - 1 else None}

    lines = text.splitlines()
    out = []
    for ln in lines:
        if ln.lstrip().startswith('|') and out and out[-1].strip() and not out[-1].lstrip().startswith('|'):
            out.append('')
        out.append(ln)
    html = markdown.markdown('\n'.join(out), extensions=['tables', 'fenced_code'])
    return render_template('results_view.html', name=name, content=html, nav=nav)
