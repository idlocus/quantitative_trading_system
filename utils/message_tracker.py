#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
消息发送追踪器
避免重复发送相同的消息
"""

import os
import json
import hashlib
from datetime import datetime


class MessageTracker:
    """消息发送追踪器"""

    def __init__(self, tracker_file='analysis_reports/.message_tracker.json'):
        self.tracker_file = tracker_file
        self.tracker_dir = os.path.dirname(tracker_file)
        os.makedirs(self.tracker_dir, exist_ok=True)

    def _load_tracker(self):
        """加载追踪记录"""
        if os.path.exists(self.tracker_file):
            try:
                with open(self.tracker_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                return {}
        return {}

    def _save_tracker(self, tracker):
        """保存追踪记录"""
        with open(self.tracker_file, 'w', encoding='utf-8') as f:
            json.dump(tracker, f, ensure_ascii=False, indent=2)

    def _hash_message(self, message):
        """生成消息hash"""
        # 只取消息的前500字符和日期来生成hash，避免完全相同的消息
        msg_part = message[:500] + datetime.now().strftime('%Y%m%d')
        return hashlib.md5(msg_part.encode('utf-8')).hexdigest()

    def is_sent_today(self, message_type):
        """检查今天是否已发送过此类消息"""
        tracker = self._load_tracker()
        today = datetime.now().strftime('%Y%m%d')

        if message_type in tracker:
            last_sent = tracker[message_type].get('date', '')
            if last_sent == today:
                return True
        return False

    def should_send(self, message_type, message):
        """判断是否应该发送消息"""
        tracker = self._load_tracker()
        today = datetime.now().strftime('%Y%m%d')
        msg_hash = self._hash_message(message)

        # 检查今天是否已发送
        if message_type in tracker:
            last_sent = tracker[message_type].get('date', '')
            last_hash = tracker[message_type].get('hash', '')

            if last_sent == today and last_hash == msg_hash:
                print(f"[跳过] {message_type} 今日已发送，内容未变化")
                return False

        # 记录本次发送
        tracker[message_type] = {
            'date': today,
            'hash': msg_hash,
            'time': datetime.now().strftime('%H:%M:%S')
        }
        self._save_tracker(tracker)
        return True

    def mark_sent(self, message_type, message):
        """标记消息已发送"""
        tracker = self._load_tracker()
        today = datetime.now().strftime('%Y%m%d')
        msg_hash = self._hash_message(message)

        tracker[message_type] = {
            'date': today,
            'hash': msg_hash,
            'time': datetime.now().strftime('%H:%M:%S')
        }
        self._save_tracker(tracker)


# 全局实例
_tracker = MessageTracker()


def should_send_message(message_type, message):
    """判断是否应该发送消息"""
    return _tracker.should_send(message_type, message)


def mark_message_sent(message_type, message):
    """标记消息已发送"""
    _tracker.mark_sent(message_type, message)
