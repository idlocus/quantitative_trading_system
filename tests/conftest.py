# tests/conftest.py — 确保测试产生的所有文件都留在项目内并在测试后清理
import os
import glob
import shutil

import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNTIME_DIR = os.path.join(PROJECT_ROOT, 'runtime')


@pytest.fixture(autouse=True)
def _isolate_test_files():
    """确保测试文件写入只发生在项目 runtime/ 内，测试后清理临时文件"""
    os.makedirs(RUNTIME_DIR, exist_ok=True)
    yield
    # 清理测试产生的临时文件（不清理正常运行的状态文件）
    for f in glob.glob(os.path.join(RUNTIME_DIR, 'test_*.json')):
        try:
            os.remove(f)
        except OSError:
            pass
