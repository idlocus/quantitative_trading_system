from application.cli import main
from application.workspace import Workspace
from data.preparation import atomic_json
from data.prepared import prepared_summary


def test_summary_never_promotes_current_hk_flags_to_history(tmp_path):
    atomic_json(tmp_path / 'oracle/hk_current_candidates/manifest.json',
                {'files': {'sample': {'rows': 5}}, 'candidate_count': 1})
    result = prepared_summary(tmp_path)
    assert result['hk_current_candidates']['rows'] == 5
    assert result['hk_current_candidates']['historical_eligibility_verified'] is False
    assert result['research_ready'] is False


def test_empty_workspace_summary_is_offline(tmp_path, capsys):
    assert main(['data', 'prepared-summary'], Workspace(tmp_path)) == 0
    assert 'research_ready' in capsys.readouterr().out
    assert not (tmp_path / '.env').exists()


def test_past_failure_does_not_override_a_completed_run(tmp_path):
    atomic_json(tmp_path / 'connect_daily/failure.json', {'past_failure': True})
    atomic_json(tmp_path / 'connect_daily/manifest.json', {'files': {}, 'last_attempt_status': 'completed'})
    assert not prepared_summary(tmp_path)['connect_provider_failure']
