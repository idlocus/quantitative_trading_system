#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
策略回测页 —— 生产策略的证据链可视化 + 优缺点分析（数据来自结构化工件与研究档案）
"""

import glob
import json
import os

from flask import Blueprint, render_template, jsonify

bp = Blueprint('strategy', __name__, url_prefix='/strategy')

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                        'results', 'data')

ERAS = [
    ('oos_2017_2021', '样本外 2017-2021', 'composite'),
    ('fullmarket_2022_2026', '全市场 2022-2026', 'composite'),
]


def _load(pattern):
    files = glob.glob(os.path.join(DATA_DIR, pattern))
    if not files:
        return None
    with open(files[0], encoding='utf-8') as f:
        return json.load(f)


@bp.route('/api/process/<era>')
def api_process(era):
    """回测过程明细（按年代一次拉取，前端本地回放）"""
    d = _load(f'portfolio_process_{era}_5因子生产.json')
    if not d:
        return jsonify({'error': '过程工件不存在（先运行 experiments 重跑）'}), 404
    return jsonify({
        'dates': d['equity']['dates'],
        'equity': d['equity']['values'],
        'benchmark': (d.get('benchmark_series') or {}).get('values'),
        'daily_holdings': d.get('daily_holdings', {}),
        'daily_contrib': d.get('daily_contrib', {}),
        'trade_log': d.get('trade_log', []),
    })


@bp.route('/')
def index():
    eras = []
    for key, label, _ in ERAS:
        strat = _load(f'portfolio_composite_v5_{key}_5因子生产.json') \
            or _load(f'portfolio_composite_v4_{key}_5因子生产.json')
        equal = _load(f'portfolio_ablation_{key}_等权.json')
        if not strat:
            continue
        eras.append({
            'key': key, 'label': label,
            'strategy': strat,
            'equal_weight': equal,
            'n_years': len(strat.get('yearly', {})),
        })
    return render_template('strategy.html', eras=eras)
