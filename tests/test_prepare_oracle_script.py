from contextlib import contextmanager
import json
import sys
from types import SimpleNamespace

import data.oracle_preparation as preparation
from scripts.prepare_oracle_data import main


def test_script_never_prints_connection_exception_details(monkeypatch, capsys):
    @contextmanager
    def connect(settings, driver):
        raise RuntimeError('sensitive-connection-details')
        yield
    monkeypatch.setattr(sys, 'argv', ['prepare_oracle_data.py', '--job', 'probe'])
    monkeypatch.setitem(sys.modules, 'oracledb', SimpleNamespace())
    monkeypatch.setattr(preparation, 'connection_settings', lambda *args: {})
    monkeypatch.setattr(preparation, 'connect_readonly', connect)
    assert main() == 2
    output = capsys.readouterr().out
    assert 'sensitive-connection-details' not in output
    assert json.loads(output)['error_type'] == 'RuntimeError'
