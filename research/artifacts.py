#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
研究工件持久化 —— 把因子评价/组合回测的结构化结果存为 JSON
（供 webapp 仪表盘渲染曲线与图表，而非只读 Markdown 摘要）

版本元数据（_meta）：commit hash 只定位基线——dirty 工作区（未提交修改/未跟踪
文件）下相同 commit 会产出不同结果，因此 _meta 同时记录 git_dirty、工作区 diff
hash、未跟踪文件清单、关键依赖版本与负载内容 hash。正式工件（formal=True）在
dirty 状态下直接拒绝生成，防止不可复现的结果进入 results/。
"""

import hashlib
import json
from datetime import datetime
import os
from datetime import date as _date

import pandas as pd

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')
DATA_DIR = os.path.join(RESULTS_DIR, 'data')

# 正式工件记录的关键依赖（版本波动会改变数值结果）
_TRACKED_DEPS = ('pandas', 'numpy')


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


def _run_git(args):
    import subprocess
    try:
        r = subprocess.run(['git'] + args, capture_output=True, text=True,
                           timeout=10, cwd=os.path.dirname(os.path.abspath(__file__)))
        return r.stdout.strip() if r.returncode == 0 else ''
    except Exception:
        return ''


def _git_hash() -> str:
    """获取当前代码的 git commit hash（定位代码版本基线）"""
    h = _run_git(['rev-parse', '--short', 'HEAD'])
    return h or 'unknown'


def _git_dirty_state() -> dict:
    """工作区脏状态：未提交修改/未跟踪文件清单 + 全量 diff hash（同一 commit
    的不同工作区状态由 diff_hash 区分，结果可追溯到具体代码内容）"""
    status = _run_git(['status', '--porcelain'])
    if not status:
        return {'git_dirty': False, 'diff_hash': None, 'untracked': [],
                'modified': []}
    lines = [l for l in status.splitlines() if l.strip()]
    untracked = [l[3:] for l in lines if l.startswith('??')]
    modified = [l[3:] for l in lines if not l.startswith('??')]
    diff_parts = [_run_git(['diff', 'HEAD']), status]
    diff_hash = hashlib.sha256('\n'.join(diff_parts).encode('utf-8')).hexdigest()[:16] \
        if any(diff_parts) else None
    return {'git_dirty': True, 'diff_hash': diff_hash,
            'untracked': untracked[:50], 'modified': modified[:50]}


def _dep_versions() -> dict:
    """关键依赖版本（pandas/numpy 版本变化会改变数值结果）"""
    import importlib
    out = {}
    for dep in _TRACKED_DEPS:
        try:
            out[dep] = importlib.import_module(dep).__version__
        except Exception:
            out[dep] = 'unknown'
    out['python'] = os.sys.version.split()[0]
    return out


def _payload_hash(payload: dict) -> str:
    """负载内容 hash（区分同代码不同参数/配置的运行）"""
    body = {k: v for k, v in payload.items() if k != '_meta'}
    try:
        blob = json.dumps(_jsonable(body), sort_keys=True, default=str)
    except Exception:
        blob = str(sorted(body.keys()))
    return hashlib.sha256(blob.encode('utf-8')).hexdigest()[:16]


def build_meta(formal: bool = False) -> dict:
    """组装版本元数据；formal=True 且工作区脏时抛错（正式产物必须可复现）"""
    dirty = _git_dirty_state()
    meta = {
        'code_version': _git_hash(),
        'generated_at': datetime.now().isoformat(),
        **dirty,
        'deps': _dep_versions(),
    }
    if formal and dirty['git_dirty']:
        raise RuntimeError(
            "正式工件（formal=True）要求干净工作区——当前有未提交修改/未跟踪文件。"
            f"修改: {dirty['modified'][:5]}... 未跟踪: {dirty['untracked'][:5]}...。"
            "先提交代码再生成正式结果，或改用 formal=False（元数据仍记录 dirty 状态）")
    return meta


def save_artifact(kind: str, name: str, payload: dict, formal: bool = False) -> str:
    """
    保存研究工件 JSON（自动注入完整版本元数据+策略规格）

    Args:
        kind: 'portfolio' | 'factor' | 'walkforward'
        name: 工件名（不含扩展名）
        payload: 可序列化字典
        formal: True=正式工件（要求干净 git 工作区，dirty 时拒绝生成）；
                False=研究迭代（元数据仍记录 dirty/diff_hash/deps，可追溯）

    Returns:
        保存路径
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    # 注入策略规格（版本治理：每个工件可回答"哪个版本产生的"）
    from trading.spec import attach_spec_to_artifact
    payload = attach_spec_to_artifact(payload)
    # 注入数据契约指纹（评审 §2.7/#11：契约改动应使旧工件可比性显式失效——
    # 因子单位/值域/覆盖率定义变了，跨契约对比工件无意义）
    if 'contracts_hash' not in payload.get('_meta', {}):
        payload.setdefault('_meta', {})['contracts_hash'] = contracts_fingerprint()
    # 注入版本元数据（代码版本、dirty状态、diff hash、依赖版本、负载hash）
    if '_meta' not in payload or 'code_version' not in payload.get('_meta', {}):
        meta = build_meta(formal=formal)
        meta['payload_hash'] = _payload_hash(payload)
        payload['_meta'].update(meta)
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
        'risk_events': getattr(result, 'risk_events', {}),        # 生效风控事件计数
        'n_stop_sells': getattr(result, 'n_stop_sells', None),   # L4止损笔数（重放口径）
        'n_halt_days': getattr(result, 'n_halt_days', None),     # L2停机天数（重放口径）
        'n_no_buy_days': getattr(result, 'n_no_buy_days', None), # L3禁买天数（重放口径）
        'terminal_note': getattr(result, 'terminal_note', None), # 终态说明（空仓停机等）
        'terminal_date': getattr(result, 'terminal_date', None), # 最后有持仓日（None=全程运行）
        'active_period_return': getattr(result, 'active_period_return', None),  # 运行期收益
        'idle_trading_days': getattr(result, 'idle_trading_days', None),        # 停机期交易日数
        'first_liquidation_date': getattr(result, 'first_liquidation_date', None),
        'n_liquidation_events': getattr(result, 'n_liquidation_events', None),
        'restart_mode': getattr(result, 'restart_mode', None),
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


def contracts_fingerprint() -> str:
    """数据契约注册表指纹（FACTOR_CONTRACTS 内容哈希，前16位）。

    工件携带此指纹后，契约定义（单位/值域/覆盖率）变更可被检测：
    跨指纹对比工件 = 跨口径对比，结果无意义时应显式重跑。"""
    from data.contracts import FACTOR_CONTRACTS
    import hashlib
    items = sorted(
        (name, c.unit, str(c.expected_range), c.min_coverage)
        for name, c in FACTOR_CONTRACTS.items())
    blob = json.dumps(items, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(blob.encode('utf-8')).hexdigest()[:16]
