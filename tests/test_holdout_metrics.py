#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""样本外指标口径守护（审计 2026-11-19 一-1/一-2）。

G1/P3 的月度指标必须由 research/portfolio_metrics.performance() 计算
（首月基准=上年最后交易日净值、自然月数、峰值含 initial 跨期延续），
换手必须是金额口径（先裁样本外日期、逐成交日金额/当日收盘净值、按
自然月平均），成交笔数单独报告。本测试用已保存的实验证据复算并锁定
2026-11-19 更正值，防止口径回退（旧口径：本月首日收盘为基准 +
交易日数/21 估算月数 + 笔数冒充换手）。
"""
import sys, os
import json
import hashlib
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path

import pandas as pd
import pytest

from research.portfolio_metrics import performance

from scripts.run_g1_holdout import DEST as G1_RELEASE
from scripts.run_holdout import DEST as P3_RELEASE
ROOT = Path(__file__).resolve().parents[1]
G1_DIR = ROOT / G1_RELEASE
P3_DIR = ROOT / P3_RELEASE
HOLDOUT_START = pd.Timestamp('2024-01-01')


def _segment_metrics(nav_csv: Path, holdout_start=HOLDOUT_START):
    rows = pd.read_csv(nav_csv)
    rows['date'] = pd.to_datetime(rows['date'])
    pre = rows[rows['date'] < holdout_start]
    seg = rows[rows['date'] >= holdout_start]
    perf = performance(seg.set_index('date')['nav'],
                       initial=float(pre['nav'].iloc[-1]))
    return perf, rows, seg


class TestG1CorrectedMetrics:
    NAME = 'G1_daily_budget_guard_25bps'

    def test_monthly_metrics_use_prior_year_base(self):
        perf, _, _ = _segment_metrics(G1_DIR / f'{self.NAME}_nav.csv')
        assert perf['months'] == 34, '月数=自然月数，不得用交易日数/21 估算'
        reported = json.loads((G1_DIR / 'verdict.json').read_text(encoding='utf8'))
        for metric, key in [('geometric_monthly_return', 'monthly'), ('max_drawdown', 'max_dd'),
                            ('worst_calendar_month', 'worst_month'), ('worst_daily_month_to_date', 'worst_mtd')]:
            assert perf[metric] == pytest.approx(reported[key])

    def test_turnover_is_amount_based_and_holdout_clipped(self):
        _, rows, _ = _segment_metrics(G1_DIR / f'{self.NAME}_nav.csv')
        tr = pd.read_csv(G1_DIR / f'{self.NAME}_trades.csv')
        tr['date'] = pd.to_datetime(tr['date'])
        tr_seg = tr[tr['date'] >= HOLDOUT_START].copy()
        tr_seg['amount'] = tr_seg['price'] * tr_seg['quantity']
        nav_by_date = rows.set_index('date')['nav']
        amt = tr_seg.groupby('date')['amount'].sum()
        turnover = float((amt / nav_by_date.reindex(amt.index)).sum()) / 34
        # 旧口径（笔数/21日/50股）≈4.3-4.4x，与金额换手完全不可比
        verdict = json.loads((G1_DIR / 'verdict.json').read_text(encoding='utf8'))
        assert turnover == pytest.approx(verdict['turnover_double'])
        assert int((tr['date'] < HOLDOUT_START).sum()) == verdict['trades_before_holdout']
        assert len(tr_seg) == verdict['trades_in_holdout']
        assert rows.position_count.max() <= 50

    def test_valuation_inputs_pinned(self):
        evidence = json.loads((G1_DIR / 'inputs.json').read_text(encoding='utf8'))
        assert all(len(evidence[key]) == 64 for key in ('pe_ttm_sha256', 'pb_lf_sha256'))


def test_g1_spec_matches_current_release():
    verdict = json.loads((G1_DIR / 'verdict.json').read_text(encoding='utf8'))
    spec = (ROOT / 'docs/DAILY_STRATEGY_SPEC.md').read_text(encoding='utf8')
    link = f'../{G1_RELEASE.as_posix()}/verdict.json'
    section = next(line for line in spec.splitlines() if link in line).replace('−', '-')
    assert f"{verdict['monthly']:+.2%}" in section
    assert f"{verdict['max_dd']:.2%}" in section
    assert f"{verdict['turnover_double']:.3f}" in section


def test_g1_registry_matches_current_release():
    from research.overfitting import TRIALS_PATH
    name = 'holdout_manifest.json'
    digest = hashlib.sha256((G1_DIR / name).read_bytes()).hexdigest()
    registry = json.loads(Path(TRIALS_PATH).read_text(encoding='utf8'))
    matches = [r for r in registry['trials'] if r.get('release_sha256') == digest]
    assert len(matches) == 1
    row = matches[0]
    assert row['release_manifest'] == (G1_RELEASE / name).as_posix()
    verdict = json.loads((G1_DIR / 'verdict.json').read_text(encoding='utf8'))
    for key in ('monthly', 'max_dd', 'turnover_double', 'months', 'trades_in_holdout', 'PASSED'):
        assert row['metrics'][key] == verdict[key]


def test_register_published_g1_is_idempotent_and_rejects_changed_artifacts(tmp_path, monkeypatch):
    from research import overfitting
    from scripts import run_g1_holdout as g1
    dest = tmp_path / 'release'
    registry = tmp_path / 'registry.json'
    monkeypatch.setattr(g1, 'DEST', dest)
    monkeypatch.setattr(overfitting, 'TRIALS_PATH', str(registry))
    def fake_compute(stage):
        verdict = dict(monthly=.01, max_dd=-.05, turnover_double=.4, months=3,
                       trades_in_holdout=42, PASSED=True)
        (stage / 'verdict.json').write_text(json.dumps(verdict), encoding='utf8')
        (stage / 'nav.csv').write_bytes(b'sample')
        return verdict
    monkeypatch.setattr(g1, '_compute', fake_compute)
    g1.main()
    before = registry.read_bytes()
    monkeypatch.setattr(g1, '_compute', lambda _: pytest.fail('Unexpected recomputation'))
    g1.main(published_only=True)
    assert json.loads(before)['count'] == 1 and registry.read_bytes() == before
    (dest / 'nav.csv').write_bytes(b'changed')
    with pytest.raises(ValueError, match='Published artifact changed'):
        g1.main(published_only=True)
    (dest / 'nav.csv').unlink()
    with pytest.raises(ValueError, match='Incomplete publication'):
        g1.main(published_only=True)
    assert registry.read_bytes() == before


class TestP3CorrectedMetrics:
    def test_p3_monthly_corrected(self):
        perf, _, _ = _segment_metrics(P3_DIR / 'HOLDOUT_P3_budget_guard_25bps_nav.csv')
        assert perf['months'] == 34
        verdict = json.loads((P3_DIR / 'verdict.json').read_text(encoding='utf8'))['HOLDOUT_P3_budget_guard_25bps']
        assert perf['geometric_monthly_return'] == pytest.approx(verdict['monthly'])
        assert perf['max_drawdown'] == pytest.approx(verdict['max_dd'])
