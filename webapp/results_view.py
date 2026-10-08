#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""研究结果浏览器 Blueprint：列出并渲染 results/ 下的 Markdown/HTML 报告"""

import os
import re
from datetime import datetime

from flask import Blueprint, render_template, abort, send_from_directory
import markdown

bp = Blueprint('results', __name__, url_prefix='/results')

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')

# 文件名前缀 → 分类标签
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
    return '其他'


def _mtime(path) -> str:
    try:
        return datetime.fromtimestamp(os.path.getmtime(path)).strftime('%Y-%m-%d %H:%M')
    except OSError:
        return '-'


@bp.route('/')
def index():
    """研究结果列表页"""
    items = []
    if os.path.isdir(RESULTS_DIR):
        for fn in sorted(os.listdir(RESULTS_DIR), reverse=True):
            path = os.path.join(RESULTS_DIR, fn)
            if not os.path.isfile(path) or fn.startswith('.'):
                continue
            if fn.endswith(('.md', '.html')):
                items.append({
                    'name': fn,
                    'category': _category(fn),
                    'mtime': _mtime(path),
                    'is_html': fn.endswith('.html'),
                    'size_kb': os.path.getsize(path) // 1024,
                })
    return render_template('results_index.html', items=items)


@bp.route('/view/<path:name>')
def view(name):
    """渲染 Markdown 或直接返回 HTML 报告"""
    if '/' in name or '..' in name:
        abort(404)
    path = os.path.join(RESULTS_DIR, name)
    if not os.path.isfile(path):
        abort(404)

    if name.endswith('.html'):
        return send_from_directory(RESULTS_DIR, name)

    with open(path, encoding='utf-8') as f:
        text = f.read()
    # 表格前补空行（python-markdown 要求表格块与上文之间有空行）
    lines, out = text.splitlines(), []
    for i, ln in enumerate(lines):
        if ln.lstrip().startswith('|') and out and out[-1].strip() and not out[-1].lstrip().startswith('|'):
            out.append('')
        out.append(ln)
    text = '\n'.join(out)
    # 表格扩展渲染 to_markdown 生成的管道表
    html = markdown.markdown(text, extensions=['tables', 'fenced_code'])
    return render_template('results_view.html', name=name, content=html)
