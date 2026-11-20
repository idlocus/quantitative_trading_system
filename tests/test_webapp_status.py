#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""webapp 信号展示行为测试（不入网，_z8_frames 打桩）。

覆盖第四轮复核两项：
1. 压缩但方向为零（收盘恰等于均线）→ 不持仓、方向显示 —，绝不出现 1/0；
2. 六品种共同停更 → meta.stale=True（页面据此显示停更横幅）。
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import pytest

webapp = pytest.importorskip('webapp.app')
cs = pytest.importorskip('research.cta_signal')


def _frame(days=280, vol_pct=0.1, compressed=True, direction=1.0,
           end='2026-11-17'):
    idx = pd.bdate_range(end=end, periods=days)
    n = len(idx)
    return pd.DataFrame({
        'close': np.linspace(100, 120, n), 'open': np.linspace(100, 120, n),
        'ret': np.zeros(n), 'vol20': np.zeros(n),
        'vol_pct': np.full(n, vol_pct), 'ma': np.linspace(100, 120, n),
        'clarity': np.zeros(n), 'compressed': np.full(n, compressed),
        'direction': np.full(n, direction)}, index=idx)


@pytest.fixture
def patched_frames(monkeypatch):
    def _install(frames):
        monkeypatch.setattr(webapp, '_z8_frames', lambda: ({}, frames))
    return _install


class TestZeroDirection:
    def test_compressed_zero_direction_not_held(self, patched_frames):
        """压缩但收盘=均线（方向0）：排名排除、不持仓、无 1/0。"""
        patched_frames({
            'AU0': _frame(direction=0.0),          # 压缩但无方向
            'CU0': _frame(direction=1.0),          # 正常持仓
        })
        status = webapp.z8_status()
        au = next(s for s in status if s['sym'] == 'AU0')
        cu = next(s for s in status if s['sym'] == 'CU0')
        assert au['direction'] == '—', '方向为零不得显示为空'
        assert au['position'] == '空仓', '方向为零的压缩品种不得持仓'
        assert '1/0' not in au['position'] and '1/0' not in cu['position']
        assert cu['position'].startswith('持仓 多 1/')


class TestStaleMeta:
    def test_common_lag_flags_stale(self, patched_frames):
        """六品种共同滞后10天 → meta.stale=True（完整性检查可过但停更须提示）。"""
        patched_frames({s: _frame(end='2026-11-03') for s in webapp.SYMS})
        meta = webapp.z8_meta()
        assert meta['stale'] is True
        assert meta['expected_date'] > meta['last_date']


class TestIndexRoute:
    """首页最终 HTML 测试（复核方以实际路由复现过停更仍显示实时信号）。

    审计 2026-11-19 四-h：冻结"预期已收盘交易日"，不再依赖真实系统时钟
    （换日期运行可能出现无关失败）。"""

    EXPECTED = '2026-11-18'

    @pytest.fixture
    def client_and_patch(self, monkeypatch, patched_frames):
        monkeypatch.setattr(webapp, 'load_config', lambda: {
            'g1': {'enabled': False}, 'z8': {'enabled': True}})
        monkeypatch.setattr(webapp, '_expected_last_closed',
                            lambda: __import__('datetime').date
                            .fromisoformat(self.EXPECTED))
        webapp.app.config['TESTING'] = True
        return webapp.app.test_client(), patched_frames

    def test_stale_shows_snapshot_banner(self, client_and_patch):
        """六品种共同停更 → 首页必须显示历史快照标记与两个日期。"""
        client, install = client_and_patch
        install({s: _frame(end='2026-11-03') for s in webapp.SYMS})
        html = client.get('/').get_data(as_text=True)
        assert '历史快照' in html
        assert '2026-11-03' in html and self.EXPECTED in html  # 数据日与预期日
        assert '实时信号' not in html, '停更时不得再称实时信号'

    def test_incomplete_shows_missing(self, client_and_patch):
        """一品种空表 + 五品种有效 → 首页显示覆盖不完整横幅，不报错。"""
        client, install = client_and_patch
        good = _frame(end='2026-11-17')
        frames = {s: good.copy() for s in webapp.SYMS}
        frames['AU0'] = good.iloc[0:0]  # 空表
        install(frames)
        resp = client.get('/')
        assert resp.status_code == 200
        html = resp.get_data(as_text=True)
        assert '覆盖不完整' in html


class TestAudit20261119Repairs:
    """审计 2026-11-19 二-3/二-4/二-5/二-7：目标组合口径、刷新原子性、
    杠杆缺失停发新增指令、健康页与 CLI 同一核验实现。"""

    def test_verify_archive_uses_cli_semantics(self):
        """网页核验必须与 CLI 同一实现：归档完好 → source_verified=True。"""
        result = webapp._verify_archive()
        assert result is not None, '归档目录存在时应返回核验结果'
        assert result['source_verified'] is True
        assert result['source_files'] >= 300

    def test_detail_summary_counts_target_book_not_compressed(self, monkeypatch):
        """六品种全压缩但方向为零 → 目标组合为空，汇总必须显示空仓。"""
        monkeypatch.setattr(webapp, '_z8_frames', lambda: (
            {}, {s: _frame(direction=0.0) for s in webapp.SYMS}))
        monkeypatch.setattr(webapp, 'z8_trade_summary', lambda: [])
        monkeypatch.setattr(webapp, '_expected_last_closed', lambda: pd.Timestamp('2026-11-17').date())
        monkeypatch.setattr(webapp, 'z8_meta', lambda: {'ok': True, 'stale': False})
        webapp.app.config['TESTING'] = True
        html = webapp.app.test_client().get('/strategy/z8').get_data(as_text=True)
        assert '6 品种各 1/6' not in html, '零方向压缩品种不得计入目标持仓'
        assert '空仓' in html

    def test_refresh_failure_keeps_previous_cache(self, monkeypatch, tmp_path):
        """刷新拉取失败 → 不删除旧缓存，如实返回 kept_previous。"""
        monkeypatch.setattr(webapp, 'CACHE', tmp_path)
        import pickle
        import sys
        from types import SimpleNamespace
        frame = pd.DataFrame(1., index=pd.bdate_range('2020-01-01', periods=cs.MIN_BARS),
                             columns=webapp.REQUIRED_COLS)
        payload = pickle.dumps({sym: frame for sym in webapp.SYMS})
        (tmp_path / 'daily_data.pkl').write_bytes(payload)
        monkeypatch.setitem(sys.modules, 'akshare', SimpleNamespace())
        webapp.app.config['TESTING'] = True
        resp = webapp.app.test_client().get('/api/refresh').get_json()
        assert resp == {'status': 'kept_previous', 'previous_cache_kept': True}
        assert (tmp_path / 'daily_data.pkl').read_bytes() == payload
        (tmp_path / 'daily_data.pkl').write_bytes(b'invalid-cache')
        resp = webapp.app.test_client().get('/api/refresh').get_json()
        assert resp == {'status': 'unavailable', 'previous_cache_kept': False}

    def test_frames_valid_rejects_incomplete(self):
        assert webapp._frames_valid(None) is False
        assert webapp._frames_valid({}) is False
        assert webapp._frames_valid({'AU0': None}) is False
        good = {s: pd.DataFrame(1., columns=webapp.REQUIRED_COLS,
                                index=pd.bdate_range('2020-01-01', periods=cs.MIN_BARS))
                for s in webapp.SYMS}
        assert webapp._frames_valid(good) is True
        bad = dict(good, CU0=pd.DataFrame(
            {'开盘价': [], '收盘价': []}, index=pd.DatetimeIndex([])))
        assert webapp._frames_valid(bad) is False

    def test_render_actions_without_leverage_stops_new_risk(self):
        """杠杆未知 → 只保留平仓类指令，开/翻/调仓停发并给出警示。"""
        actions = [{'sym': 'AU0', 'name': '黄金', 'action': '开多',
                    'weight': 1.0, 'from': 0.0},
                   {'sym': 'CU0', 'name': '沪铜', 'action': '平仓',
                    'weight': 0.0, 'from': -0.5},
                   {'sym': 'RB0', 'name': '螺纹钢', 'action': '翻空',
                    'weight': -0.4, 'from': 0.3}]
        lines = cs.render_actions(actions, lev_held=None, lev_next=None,
                                  yesterday_known=True)
        text = '\n'.join(lines)
        assert '平仓' in text
        assert '开多' not in text and '翻空' not in text
        assert '杠杆计算异常' in text

    def test_daily_symbol_table_zero_direction_shows_dash(self):
        """日报明细：方向为零显示 —，不得写"空"。"""
        import scripts.run_cta_daily as daily
        f = _frame(direction=0.0)
        lines = []
        daily._append_symbol_table(lines, {'AU0': f}, {'AU0': ('黄金', 'AU')})
        row = [ln for ln in lines if ln.startswith('| 黄金')][0]
        assert row.rstrip().endswith('| — |'), f'零方向必须显示 —: {row}'
