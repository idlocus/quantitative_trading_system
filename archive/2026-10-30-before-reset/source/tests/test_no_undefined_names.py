#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""未定义名门槛 —— 主链模块 pyflakes F821 扫描（2026-10-15）

背景：paper 验证期首跑抓到 signals.py 的 NameError（'pit' 未定义）——
依赖重接线时误删定义，而 404 项测试全绿：合成数据路径在覆盖率闸门早退，
覆盖不到真数据才走的分支。测试的形状与 bug 的形状不匹配（评审根因3.3）。
pyflakes 的 F821（undefined name）与运行路径无关，恰好补上这个盲区。
"""
import os
import subprocess
import sys

import pytest

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MAIN_CHAIN_DIRS = ['live', 'trading', 'data', 'portfolio', 'monitoring', 'scripts']
MAIN_CHAIN_FILES = ['run_daily.py', 'run_live.py', 'run_etf.py',
                    'run_daily_replay.py']


def _pyflakes_targets():
    targets = []
    for d in MAIN_CHAIN_DIRS:
        dirpath = os.path.join(PROJECT, d)
        if not os.path.isdir(dirpath):
            continue
        for root, _dirs, files in os.walk(dirpath):
            if '__pycache__' in root:
                continue
            targets.extend(os.path.join(root, f) for f in files
                           if f.endswith('.py'))
    targets.extend(os.path.join(PROJECT, f) for f in MAIN_CHAIN_FILES
                   if os.path.isfile(os.path.join(PROJECT, f)))
    return targets


def test_no_undefined_names_in_main_chain():
    """主链代码不得有未定义名（F821）——测试路径覆盖不到的分支由静态扫描兜底。

    忽略未用导入等风格警告（范围收窄，只拦会运行时崩溃的 NameError 类）。"""
    pytest.importorskip('pyflakes')
    targets = _pyflakes_targets()
    assert targets, "未找到主链文件——检查路径"

    r = subprocess.run([sys.executable, '-m', 'pyflakes', *targets],
                       capture_output=True, text=True, encoding='utf-8',
                       errors='replace', timeout=300)
    undefined = [ln for ln in (r.stdout or '').splitlines()
                 if 'undefined name' in ln]
    assert not undefined, (
        f"主链存在未定义名（会在真数据分支运行时崩溃）：\n"
        + "\n".join(undefined[:20]))
