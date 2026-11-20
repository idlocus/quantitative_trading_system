import json
import sys
from types import SimpleNamespace

from scripts.prepare_wind_data import main


def test_connection_failure_is_structured(monkeypatch, capsys):
    monkeypatch.setattr(sys, 'argv', ['prepare_wind_data.py', '--job', 'lifecycle'])
    monkeypatch.setitem(sys.modules, 'WindPy', SimpleNamespace(w=SimpleNamespace(
        start=lambda **kwargs: SimpleNamespace(ErrorCode=-1))))
    assert main() == 2
    assert json.loads(capsys.readouterr().out)['error_code'] == -1


def test_sdk_exception_does_not_expose_exception_text(monkeypatch, capsys):
    def fail(**kwargs):
        raise RuntimeError('private-provider-details')
    monkeypatch.setattr(sys, 'argv', ['prepare_wind_data.py', '--job', 'lifecycle'])
    monkeypatch.setitem(sys.modules, 'WindPy', SimpleNamespace(w=SimpleNamespace(start=fail)))
    assert main() == 2
    captured = capsys.readouterr().out
    assert 'private-provider-details' not in captured
    assert json.loads(captured)['error_type'] == 'RuntimeError'
