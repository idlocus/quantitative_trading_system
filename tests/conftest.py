"""Shared pytest configuration. Tests use per-test tmp_path fixtures."""
import pytest


@pytest.fixture(autouse=True)
def isolate_trading_controls(tmp_path, monkeypatch):
    from scripts import kill_switch, trading_switch
    monkeypatch.setattr(kill_switch, 'KILL_FILE', str(tmp_path / 'KILL'))
    monkeypatch.setattr(trading_switch, 'RUNTIME_DIR', str(tmp_path))
    monkeypatch.setattr(trading_switch, 'STATE_FILE', str(tmp_path / 'switch.json'))
    monkeypatch.setenv('QMT_PATH', '')
    monkeypatch.setenv('PYTHON_DOTENV_DISABLED', '1')
