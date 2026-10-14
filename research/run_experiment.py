#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一实验运行入口 —— 所有正式实验必须经由这里

用法:
  python -m research.run_experiment --experiment mainchain_v5
  python -m research.run_experiment --experiment loss_attribution
  python -m research.run_experiment --experiment quality_value
  python -m research.run_experiment --experiment momentum_overlay
  python -m research.run_experiment --experiment param_stability
  python -m research.run_experiment --list

每次运行自动记录（experiments/results/registry.jsonl）：
  experiment_id, strategy_version, code_commit, timestamp, status, result_summary
"""
import json
import os
import subprocess
import sys
import uuid
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

REGISTRY_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'experiments', 'results', 'registry.jsonl')

EXPERIMENTS = {
    'mainchain_v5': {
        'description': '主链v5重放回测（严格生产语义）',
        'module': 'experiments.mainchain_v5',
        'args': ['--era={era}'] if '--era' in sys.argv else [],
    },
    'loss_attribution': {
        'description': '亏损归因实验（A/B/C/D）',
        'module': 'experiments.loss_attribution',
    },
    'quality_value': {
        'description': '质量价值候选策略（E1-E5）',
        'module': 'experiments.quality_value_candidate',
    },
    'momentum_overlay': {
        'description': '趋势维度实验（M0-M4）',
        'module': 'experiments.momentum_overlay',
    },
    'param_stability': {
        'description': '参数稳定性检查（11组）',
        'module': 'experiments.param_stability',
    },
    'etf_trend': {
        'description': 'ETF多资产趋势',
        'module': 'experiments.etf_trend',
    },
    'combined_strategy': {
        'description': '组合策略（趋势×选股）',
        'module': 'experiments.combined_strategy',
    },
}


def register(experiment: str, status: str, summary: str = ''):
    """写入实验登记（JSONL append，含失败实验）"""
    entry = {
        'experiment_id': f"{experiment}_{uuid.uuid4().hex[:8]}",
        'experiment': experiment,
        'strategy_version': _get_strategy_version(),
        'code_commit': _get_git_commit(),
        'timestamp': datetime.now().isoformat(),
        'status': status,
        'result_summary': summary[:200],
    }
    os.makedirs(os.path.dirname(REGISTRY_FILE), exist_ok=True)
    with open(REGISTRY_FILE, 'a', encoding='utf-8') as f:
        f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    print(f"[Registry] {experiment} → {status}")
    return entry


def _get_strategy_version() -> str:
    try:
        from trading.spec import STRATEGY_VERSION
        return STRATEGY_VERSION
    except Exception:
        return 'unknown'


def _get_git_commit() -> str:
    import subprocess
    try:
        r = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'],
                          capture_output=True, text=True, timeout=5,
                          cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        return r.stdout.strip()
    except Exception:
        return 'unknown'


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Unified experiment runner')
    parser.add_argument('--experiment', default=None,
                        help='实验名（见 --list）')
    parser.add_argument('--list', action='store_true', help='列出可用实验')
    parser.add_argument('--args', default='', help='传递给实验脚本的参数')
    opts, extra = parser.parse_known_args()

    if opts.list or not opts.experiment:
        print("可用实验:")
        for name, info in EXPERIMENTS.items():
            print(f"  {name:24s} {info['description']}")
        if not opts.experiment:
            print("\n用法: python -m research.run_experiment --experiment <name>")
        return

    exp = opts.experiment
    if exp not in EXPERIMENTS:
        print(f"未知实验: {exp}（--list 查看）")
        sys.exit(1)

    info = EXPERIMENTS[exp]
    module = info['module']
    print(f"运行实验: {exp} ({info['description']})")
    print(f"策略版本: {_get_strategy_version()} | 代码: {_get_git_commit()}")
    print(f"命令: python {module}.py {opts.args} {' '.join(extra)}")

    # 运行实验脚本
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cmd = [sys.executable, os.path.join(project_root, module.replace('.', os.sep) + '.py')]
    if opts.args:
        cmd.extend(opts.args.split())
    cmd.extend(extra)

    register(exp, 'started')
    try:
        result = subprocess.run(cmd, cwd=project_root, timeout=3600)
        if result.returncode == 0:
            register(exp, 'completed', f'exit=0')
        else:
            register(exp, 'failed', f'exit={result.returncode}')
            sys.exit(result.returncode)
    except subprocess.TimeoutExpired:
        register(exp, 'timeout', 'exceeded 3600s')
        sys.exit(1)
    except Exception as e:
        register(exp, 'error', str(e))
        sys.exit(1)


if __name__ == '__main__':
    main()
