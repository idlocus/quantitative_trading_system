from datetime import date

import pandas as pd
import pytest

from data.preparation import atomic_json
from data.wind_preparation import save_frame
from data.year_release import reconcile_empty_patches, stage_year


@pytest.mark.parametrize('patch_valid', [True, False])
def test_staging_preserves_sources_and_marks_basis(tmp_path, patch_valid):
    cache = tmp_path / 'cache'
    output = tmp_path / 'prepared'
    (cache / 'kline').mkdir(parents=True)
    original = pd.DataFrame({'date': pd.to_datetime(['2020-01-02', '2020-01-03']),
        'open': [10., 100.], 'high': [11., 110.], 'low': [9., 90.], 'close': [10., 10.],
        'volume': [100., 100.], 'amount': [1000., 1000.], 'adjclose': [10., 100.]})
    source = cache / 'kline' / '600000_SH.parquet'
    original.to_parquet(source)
    before = source.read_bytes()
    patch = original.iloc[1:].copy()
    patch[['open', 'high', 'low']] = [[10., 11., 9.]]
    patch = patch.drop(columns='adjclose').assign(adjfactor=2.)
    root = output / 'repair' / '2020'
    record = save_frame(patch, root / 'patches' / 'sample.parquet',
                        {'code': '600000.SH', 'validation': {'status': 'validated' if patch_valid else 'invalid'}})
    atomic_json(root / 'patch_manifest.json', {'files': {'sample.parquet': record}})
    result = stage_year(cache, output, 2020, date(2020, 1, 3))
    assert result['status_counts'] == {'validated' if patch_valid else 'invalid': 1}
    assert result['research_ready'] is False
    staged = pd.read_parquet(output / 'yearly' / '2020' / source.name)
    assert staged.open.tolist() == [10., 10. if patch_valid else 100.]
    assert staged.row_source.tolist() == ['retained_cache_unverified_basis', 'wind_raw_patch' if patch_valid else 'retained_cache_unverified_basis']
    assert result['rejected_patches'] == ([] if patch_valid else ['sample.parquet'])
    assert source.read_bytes() == before


def test_empty_quote_reconciliation_requires_delist_evidence(tmp_path):
    lifecycle = pd.DataFrame({'wind_code': ['600000.SH'], 'DELIST_DATE': [date(2020, 1, 3)]})
    record = save_frame(lifecycle, tmp_path / 'lifecycle' / 'chunk.parquet', {'method': 'test'})
    atomic_json(tmp_path / 'lifecycle' / 'manifest.json', {'files': {'chunk.parquet': record}})
    records = {}
    root = tmp_path / 'repair' / '2020'
    for day in ('2020-01-02', '2020-01-03'):
        frame = pd.DataFrame({'open': [float('nan')], 'high': [float('nan')],
                              'low': [float('nan')], 'close': [10.],
                              'volume': [float('nan')], 'amount': [float('nan')]})
        name = day + '.parquet'
        records[name] = save_frame(frame, root / 'patches' / name,
            {'code': '600000.SH', 'start': day, 'end': day, 'validation': {'status': 'invalid'}})
    atomic_json(root / 'patch_manifest.json', {'files': records})
    result = reconcile_empty_patches(tmp_path, 2020)
    assert result['counts'] == {'requires_market_status_review': 1, 'no_full_bar_on_delist_effective_date': 1}
    assert result['research_ready'] is False


def test_staging_flags_patches_without_cache_file(tmp_path):
    """审计 2026-11-19 四-e：只有补丁、没有原缓存文件的证券必须显式列出
    （此前静默跳过，年度输出缺该证券无据可查）。"""
    cache = tmp_path / 'cache'
    output = tmp_path / 'prepared'
    (cache / 'kline').mkdir(parents=True)
    frame = pd.DataFrame({'date': pd.to_datetime(['2020-01-02']),
        'open': [10.], 'high': [11.], 'low': [9.], 'close': [10.],
        'volume': [100.], 'amount': [1000.], 'adjclose': [10.]})
    # 原缓存只有 600000；补丁覆盖 600000 + 000001（后者无原缓存）
    frame.to_parquet(cache / 'kline' / '600000_SH.parquet')
    patch = frame.copy().drop(columns='adjclose').assign(adjfactor=2.)
    root = output / 'repair' / '2020'
    records = {}
    for code in ('600000.SH', '000001.SZ'):
        stem = code.replace('.', '_')
        records[f'{stem}.parquet'] = save_frame(
            patch.copy(), root / 'patches' / f'{stem}.parquet',
            {'code': code, 'validation': {'status': 'validated'}})
    atomic_json(root / 'patch_manifest.json', {'files': records})
    result = stage_year(cache, output, 2020, date(2020, 1, 3))
    orphans = result['patches_without_cache_file']
    assert [o['code'] for o in orphans] == ['000001.SZ']
    assert orphans[0]['reason'] == 'no_original_cache_file'
    assert any(f.startswith('000001') for f in orphans[0]['patch_files'])


def test_staging_rejects_upper_manifest_hash_divergence(tmp_path):
    """审计 2026-11-19 四-e：补丁与 sidecar 一同更新、上层清单哈希未同步
    → 必须拒绝（三方比对）。"""
    cache = tmp_path / 'cache'
    output = tmp_path / 'prepared'
    (cache / 'kline').mkdir(parents=True)
    frame = pd.DataFrame({'date': pd.to_datetime(['2020-01-02']),
        'open': [10.], 'high': [11.], 'low': [9.], 'close': [10.],
        'volume': [100.], 'amount': [1000.], 'adjclose': [10.]})
    frame.to_parquet(cache / 'kline' / '600000_SH.parquet')
    patch = frame.copy().drop(columns='adjclose').assign(adjfactor=2.)
    root = output / 'repair' / '2020'
    record = save_frame(patch, root / 'patches' / 'sample.parquet',
                        {'code': '600000.SH', 'validation': {'status': 'validated'}})
    record['sha256'] = '0' * 64      # 上层清单哈希被不同步地留在旧值
    atomic_json(root / 'patch_manifest.json', {'files': {'sample.parquet': record}})
    with pytest.raises(ValueError, match='不一致'):
        stage_year(cache, output, 2020, date(2020, 1, 3))


class TestZ9ChunkDownloadHelpers:
    """审计 2026-11-19 四-d：分块落盘 + 完整性验收 + 缺块不写成品。"""

    def test_chunk_valid_checks_span_and_tail(self, tmp_path, monkeypatch):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            'z9dl', 'scripts/run_z9_batch_download.py')
        z9 = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(z9)
        import pickle
        good = pd.DataFrame({'close': [1.0] * 200, 'volume': [1] * 200},
                            index=pd.date_range('2025-01-02', periods=200, freq='5min'))
        p = tmp_path / 'c.pkl'
        pickle.dump(good, open(p, 'wb'))
        assert not z9.chunk_valid(p, '2025-01-01', '2025-03-31')
        days = pd.bdate_range('2025-01-01', '2025-03-31')
        monkeypatch.setattr(z9, 'expected_sessions', lambda start, end: days)
        idx = pd.DatetimeIndex([d + pd.Timedelta(minutes=m) for d in days
                                for m in range(545, 901, 5)])
        full = pd.DataFrame({'close': 1., 'volume': 1}, index=idx)
        full.to_pickle(p)
        assert z9.chunk_valid(p, '2025-01-01', '2025-03-31')
        full.drop(idx[len(idx) // 2:len(idx) // 2 + 100]).to_pickle(p)
        assert not z9.chunk_valid(p, '2025-01-01', '2025-03-31')
        # 截断块（末bar远离季末）→ 无效
        trunc = pd.DataFrame({'close': [1.0] * 200, 'volume': [1] * 200},
                             index=pd.date_range('2025-01-02 09:00', periods=200, freq='5min'))
        pickle.dump(trunc, open(p, 'wb'))
        assert z9.chunk_valid(p, '2025-01-01', '2025-03-31') is False
        # 越界块 → 无效
        outside = pd.DataFrame({'close': [1.0] * 200, 'volume': [1] * 200},
                               index=pd.date_range('2024-12-02 09:00', periods=200, freq='5min'))
        pickle.dump(outside, open(p, 'wb'))
        assert z9.chunk_valid(p, '2025-01-01', '2025-03-31') is False


def test_research_scripts_import_without_side_effects(tmp_path, monkeypatch):
    """审计 2026-11-19 四-h：研究脚本 import 不得触发数据读取/建目录/实验。"""
    import importlib
    monkeypatch.chdir(tmp_path)          # 任何意外的相对路径写盘都会落到临时目录
    for name in ('scripts.run_p2_tailport', 'scripts.run_p_intraday', 'scripts.run_q_pairs',
                 'scripts.run_z8_full_stop', 'scripts.run_t_pull_and_test', 'scripts.run_z7_signal_matrix'):
        importlib.import_module(name)
    assert not (tmp_path / 'ai-created').exists(), 'import 不得创建输出目录'
