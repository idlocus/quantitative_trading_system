#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""研究仪表盘 Blueprint：结构化工件的可视化（净值/基准/年度/IC 序列）"""

import json
import os
from datetime import datetime

from flask import Blueprint, render_template, abort

bp = Blueprint('dashboard', __name__, url_prefix='/research')

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        'results', 'data')

ERA_LABELS = [
    ('oos_2017_2021', '样本外 2017-2021'),
    ('fullmarket_2022_2026', '全市场 2022-2026'),
    ('enhanced_2017_2021', '300指增 2017-2021'),
    ('enhanced_2022_2026', '300指增 2022-2026'),
    ('ablation_oos_2017_2021', '因子贡献·样本外 2017-2021'),
    ('ablation_fullmarket_2022_2026', '因子贡献·全市场 2022-2026'),
    ('composite_v2_oos_2017_2021', '五因子复检·2017-2021'),
    ('composite_v2_fullmarket_2022_2026', '五因子复检·2022-2026'),
    ('phaseB_oos_2017_2021', 'PhaseB 优化器/对冲·2017-2021'),
    ('phaseB_fullmarket_2022_2026', 'PhaseB 优化器/对冲·2022-2026'),
    ('phaseB中性_oos_2017_2021', 'PhaseB 市场中性·2017-2021'),
    ('phaseB中性_fullmarket_2022_2026', 'PhaseB 市场中性·2022-2026'),
]


def _load_all():
    items = []
    if not os.path.isdir(DATA_DIR):
        return items
    for fn in sorted(os.listdir(DATA_DIR)):
        if not fn.endswith('.json'):
            continue
        try:
            with open(os.path.join(DATA_DIR, fn), encoding='utf-8') as f:
                d = json.load(f)
            d['_file'] = fn[:-5]
            d['_mtime'] = datetime.fromtimestamp(
                os.path.getmtime(os.path.join(DATA_DIR, fn))).strftime('%m-%d %H:%M')
            items.append(d)
        except Exception:
            continue
    return items


# 实验组的人话解释（为什么做这个实验、结论是什么）
ERA_NOTES = {
    'oos_2017_2021': '核心验证：策略在从未见过的 2017-2021 数据上的表现——跑赢等权对照 50pp，证明选股能力真实',
    'fullmarket_2022_2026': '核心验证：策略在研究期 2022-2026 的表现——跑赢等权对照 63pp',
    'enhanced_2017_2021': '被否决：限定在沪深300成分内做指增——蓝筹年代跑输基准，成分域限制破坏了多因子互补',
    'enhanced_2022_2026': '被否决：指增形态在小盘年代大幅跑赢但两段 IR 翻转（-0.56/+0.34），不稳定',
    'phaseB_oos_2017_2021': '收益分解：β对冲后残差为负——该年代的正收益全部来自大盘β，选股残差 -31pp',
    'phaseB_fullmarket_2022_2026': '收益分解：对冲后残差 +25pp——小盘年代的选股残差为正',
    'phaseB中性': '市场中性形态检验（等权/优化器 × 对冲）',
    'ablation_oos_2017_2021': '因子贡献消融：单因子皆弱（bp单用最差），合成才有效——多因子互补去噪是alpha来源',
    'ablation_fullmarket_2022_2026': '因子贡献消融（同左，2022-2026 段）',
    'composite_v2': '被否决：中性化ROE因子进组合后两段均稀释',
    'composite_v5': '4因子基线 vs 5因子（+低杠杆）：低杠杆两段均小幅改善，进入生产',
    'process': '回测过程明细（供策略页回放）',
}

@bp.route('/')
def index():
    items = _load_all()
    portfolios = [x for x in items if x.get('kind') == 'portfolio']
    factors = [x for x in items if x.get('kind') == 'factor']

    # 按 era 分组对比矩阵
    eras = []
    for prefix, label in ERA_LABELS:
        members = [x for x in portfolios if x['_file'].startswith(f'portfolio_{prefix}')]
        if not members:
            continue
        cols = {}
        for m in members:
            variant = m['_file'].split(f'portfolio_{prefix}_')[-1]
            cols[variant] = m
        eras.append({'prefix': prefix, 'label': label, 'cols': cols,
                     'note': ERA_NOTES.get(prefix, ''),
                     'benchmark_return': members[0].get('benchmark_return')})

    return render_template('dashboard_index.html', eras=eras, factors=factors)


@bp.route('/view/<kind>/<name>')
def view(kind, name):
    if kind not in ('portfolio', 'factor') or '/' in name or '..' in name:
        abort(404)
    path = os.path.join(DATA_DIR, f"{kind}_{name}.json")
    if not os.path.isfile(path):
        abort(404)
    with open(path, encoding='utf-8') as f:
        d = json.load(f)
    d['_name'] = name
    d['_kind'] = kind
    return render_template('dashboard_detail.html', d=d)
