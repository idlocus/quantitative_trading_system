#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""启动器生成器测试：模板要素/CRLF+ASCII/自检敏感度（2026-10-30）"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'deploy'))
import build_launchers as bl  # noqa: E402


class TestRenderBat:
    def test_crlf_and_ascii(self):
        body = bl.render_bat(Path('C:/x/uv.exe'), 'scripts/paper_day.py')
        assert b'\r\n' in body and b'\n' not in body.replace(b'\r\n', b'')
        body.decode('ascii')   # 非 ASCII 会在此抛 UnicodeDecodeError

    def test_template_essentials(self, tmp_path, monkeypatch):
        monkeypatch.setattr(bl, 'PROJECT_ROOT', tmp_path)
        uv = Path('C:/x/uv.exe')
        body = bl.render_bat(uv, 'scripts/paper_day.py')
        text = body.decode('ascii')
        assert f'cd /d "{tmp_path}"' in text          # 定位项目根
        assert f'"{uv}" run python scripts/paper_day.py' in text  # uv 自愈入口
        assert 'exit /b %ERRORLEVEL%' in text          # 退出码透传（schtasks 可见）

    def test_all_targets_exist(self):
        """登记表里的目标脚本必须在仓库中（防改名后生成器静默产出坏 bat）"""
        for _bat, script_rel, _clock in bl.LAUNCHERS:
            assert (Path(__file__).resolve().parent.parent / script_rel).is_file(), \
                f"{script_rel} 不存在——LAUNCHERS 登记过期"


class TestCheck:
    def _run_check(self, tmp_path, monkeypatch, mutate=None):
        monkeypatch.setattr(bl, 'PROJECT_ROOT', tmp_path)
        (tmp_path / 'scripts').mkdir(parents=True)
        fake_uv = tmp_path / 'uv.exe'
        fake_uv.write_bytes(b'')   # build 要求 uv 路径真实存在
        for _bat, script_rel, _clock in bl.LAUNCHERS:
            (tmp_path / script_rel).write_text('# stub')
        bl.build(uv_path=fake_uv)
        if mutate:
            mutate(tmp_path)
        return bl.check(uv_path=fake_uv)

    def test_check_passes_after_build(self, tmp_path, monkeypatch):
        assert self._run_check(tmp_path, monkeypatch) == 0

    def test_check_detects_hand_edit(self, tmp_path, monkeypatch):
        def touch(root):
            p = root / bl.LAUNCHERS[0][0]
            p.write_bytes(p.read_bytes() + b'\r\nrem hand edit\r\n')
        assert self._run_check(tmp_path, monkeypatch, touch) == 1, \
            "手改过的 bat 必须被自检发现"

    def test_check_detects_missing_bat(self, tmp_path, monkeypatch):
        def remove(root):
            (root / bl.LAUNCHERS[0][0]).unlink()
        assert self._run_check(tmp_path, monkeypatch, remove) == 1
