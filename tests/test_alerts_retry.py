#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""告警 webhook 重试与死信（评审 §2.8/#12）"""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from monitoring import alerts


class TestWebhookRetry:

    def test_retry_succeeds_on_third_attempt(self, monkeypatch, tmp_path):
        """前两次失败第三次成功 → 不写死信"""
        monkeypatch.setattr(alerts, 'MONITORING_DIR', str(tmp_path))
        monkeypatch.setattr(alerts.time, 'sleep', lambda s: None)
        calls = {'n': 0}

        def flaky(url, event):
            calls['n'] += 1
            if calls['n'] < 3:
                raise IOError('网络抖动')
        monkeypatch.setattr(alerts, '_send_webhook', flaky)
        ok = alerts._send_webhook_with_retry('http://x', {'code': 'T'})
        assert ok is True and calls['n'] == 3
        assert not (tmp_path / 'webhook_dead.jsonl').exists()

    def test_all_failures_write_dead_letter(self, monkeypatch, tmp_path):
        """3次全败 → 死信落盘可人工重放"""
        monkeypatch.setattr(alerts, 'MONITORING_DIR', str(tmp_path))
        monkeypatch.setattr(alerts.time, 'sleep', lambda s: None)

        def always_fail(url, event):
            raise IOError('断网')
        monkeypatch.setattr(alerts, '_send_webhook', always_fail)
        ok = alerts._send_webhook_with_retry(
            'http://x', {'code': 'P0_TEST', 'level': 'P0', 'message': 'm'})
        assert ok is False
        dead = tmp_path / 'webhook_dead.jsonl'
        assert dead.exists()
        rec = json.loads(dead.read_text(encoding='utf-8').strip())
        assert rec['code'] == 'P0_TEST' and 'retry_error' in rec
