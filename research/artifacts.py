#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
研究工件持久化 —— 把因子评价/组合回测的结构化结果存为 JSON
（供 webapp 仪表盘渲染曲线与图表，而非只读 Markdown 摘要）
"""

import json
from datetime import datetime
import os
from datetime import date as _date

import pandas as pd

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')
DATA_DIR = os.path.join(RESULTS_DIR, 'data')


def _jsonable(obj):
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, (pd.Timestamp, _date)):
        return str(obj)[:10]
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(x) for x in obj]
    return str(obj)


def _git_hash() -> str:
    """获取当前代码的 git commit hash（定位代码版本）"""
    import subprocess
    try:
        r = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'],
                          capture_output=True, text=True, timeout=5)
        return r.stdout.strip()
    except Exception:
        return 'unknown'


def save_artifact(kind: str, name: str, payload: dict) -> str:
    """
    保存研究工件 JSON（自动注入版本元数据）

    Args:
        kind: 'portfolio' | 'factor' | 'walkforward'
        name: 工件名（不含扩展名）
        payload: 可序列化字典
    Returns:
        保存路径
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    # 自动注入版本元数据（代码版本、生成时间）
    if '_meta' not in payload:
        payload['_meta'] = {
            'code_version': _git_hash(),
            'generated_at': datetime.now().isoformat(),
        }
    path = os.path.join(DATA_DIR, f"{kind}_{name}.json")
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(_jsonable(payload), f, ensure_ascii=False)
    return path


def portfolio_payload(result, label: str = '', extra: dict = None) -> dict:
    """组合回测结果 → 结构化工件（含净值/基准/年度序列）"""
    eq = result.equity_curve
    yearly = {}
    for y, g in eq.groupby(eq.index.year):
        if len(g) >= 10:
            yearly[str(y)] = float(g.iloc[-1] / g.iloc[0] - 1)
    payload = {
        'kind': 'portfolio',
        'label': label or result.config.top_n,
        'top_n': result.config.top_n,
        'rebalance_days': result.config.rebalance_days,
        'n_dates': result.n_dates,
        'n_rebalances': result.n_rebalances,
        'total_return': result.total_return,
        'annual_return': result.annual_return,
        'sharpe': result.sharpe,
        'max_drawdown': result.max_drawdown,
        'avg_turnover': result.avg_turnover,
        'total_cost_pct': result.total_cost / result.initial_capital,
        'benchmark_return': result.benchmark_return,
        'benchmark_suite': getattr(result, 'benchmark_suite', {}),
        'benchmark_symbol': result.config.benchmark_symbol,
        'equity': {'dates': [str(d)[:10] for d in eq.index],
                   'values': [round(float(v), 2) for v in eq.values]},
        'yearly': yearly,
        # 过程明细（回放用）：每日持仓市值、每日个股贡献、成交流水
        'daily_holdings': getattr(result, 'daily_holdings', {}),
        'daily_contrib': getattr(result, 'daily_contrib', {}),
        'trade_log': getattr(result, 'trade_log', []),
    }
    if extra:
        payload.update(extra)
    return payload


def factor_payload(result, horizon_note: str = '') -> dict:
    """因子评价结果 → 结构化工件（含 IC 序列与分位收益）"""
    ic = result.ic_series
    return {
        'kind': 'factor',
        'factor': result.name,
        'period': result.period,
        'n_dates': result.n_dates,
        'n_symbols': result.n_symbols,
        'ic_mean': result.ic_mean,
        'ic_ir': result.ic_ir,
        'rankic_mean': result.rankic_mean,
        'rankic_ir': result.rankic_ir,
        'ic_positive_rate': result.ic_positive_rate,
        'long_short_annual': result.long_short_annual,
        'turnover': result.turnover,
        'quantile_mean_ret': {str(k): v for k, v in result.quantile_mean_ret.items()},
        'ic_series': {'dates': [str(d)[:10] for d in ic.index],
                      'values': [round(float(v), 4) for v in ic.values]},
        'horizon_note': horizon_note,
    }
