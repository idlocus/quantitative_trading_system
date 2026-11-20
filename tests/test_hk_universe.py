from datetime import date

import pandas as pd
import pytest

from data.hk_universe import build_static_universe, research_symbols
from data.preparation import atomic_json
from data.wind_preparation import save_frame


def setup_sources(output, policy_path):
    master = pd.DataFrame({'S_INFO_WINDCODE': ['00001.HK', '00002.HK', '00003.HK'],
        'SECURITYTYPE': ['ORD', 'ORD', 'WAR'], 'IS_HKSC': [1, 1, 1],
        'S_INFO_LISTDATE': ['20100101', '20200102', '20100101'],
        'S_INFO_DELISTDATE': [None, None, None]})
    record = save_frame(master, output / 'oracle/reference/HKSHAREDESCRIPTION.parquet', {'source': 'test'})
    atomic_json(output / 'oracle/hk_current_candidates/manifest.json', {'candidate_count': 2,
        'files': {'sample': {'rows': 2, 'request': {'candidate_snapshot_sha256': record['sha256']}}}})
    atomic_json(policy_path, {'hk_universe_mode': 'current_connect_snapshot',
        'historical_connect_membership_required': False, 'scope': 'research_only',
        'user_confirmed_on': '2026-11-03'})
    return master


def test_static_sample_keeps_listing_boundary_and_historical_fact_separate(tmp_path):
    policy = tmp_path / 'policy.json'
    setup_sources(tmp_path, policy)
    result = build_static_universe(tmp_path, policy)
    assert result['symbols'] == 2
    assert result['historical_eligibility_verified'] is False
    assert result['historical_eligibility_blocking'] is False
    frame = pd.read_parquet(tmp_path / 'hk_research_universe/current_snapshot.parquet')
    assert research_symbols(frame, date(2019, 1, 1)) == ('00001.HK',)
    assert research_symbols(frame, date(2020, 1, 2)) == ('00001.HK', '00002.HK')


def test_delist_boundary_and_unknown_listing_are_not_waived():
    frame = pd.DataFrame({'symbol': ['00001.HK'], 'list_date': pd.to_datetime(['2020-01-02']),
                          'delist_date': pd.to_datetime(['2020-01-06'])})
    assert research_symbols(frame, date(2020, 1, 1)) == ()
    assert research_symbols(frame, date(2020, 1, 3)) == ('00001.HK',)
    assert research_symbols(frame, date(2020, 1, 6)) == ()
    frame['list_date'] = pd.NaT
    with pytest.raises(ValueError):
        research_symbols(frame, date(2020, 1, 3))


def test_policy_and_snapshot_mismatch_fail_closed(tmp_path):
    policy = tmp_path / 'policy.json'
    master = setup_sources(tmp_path, policy)
    atomic_json(policy, {'hk_universe_mode': 'strict'})
    with pytest.raises(ValueError, match='explicit research policy'):
        build_static_universe(tmp_path, policy)
    setup_sources(tmp_path, policy)
    build_static_universe(tmp_path, policy)
    master.loc[0, 'IS_HKSC'] = 0
    save_frame(master, tmp_path / 'oracle/reference/HKSHAREDESCRIPTION.parquet', {'source': 'test'})
    with pytest.raises(ValueError, match='cannot switch'):
        build_static_universe(tmp_path, policy)


def test_summary_stops_treating_accepted_history_assumption_as_blocker(tmp_path):
    from data.prepared import prepared_summary
    policy = tmp_path / 'policy.json'
    setup_sources(tmp_path, policy)
    build_static_universe(tmp_path, policy)
    result = prepared_summary(tmp_path)
    assert result['hk_research_universe']['assumption_accepted']
    assert not result['hk_research_universe']['historical_eligibility_blocking']
    assert 'Historical southbound buy/sell-only qualification' not in result['remaining']
    assert not result['research_ready']
