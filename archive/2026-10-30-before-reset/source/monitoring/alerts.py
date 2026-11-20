#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一告警模块 —— 去重/冷却/多通道

通道：
  1. 本地 alerts.jsonl（审计记录，始终写入）
  2. Webhook（飞书/企业微信，配置后自动发送）
  3. 邮件（备用）

去重：同 code+symbol 在 cooldown_seconds 内只发一次
"""
import json
import os
import time
from datetime import datetime
from typing import Optional
import logging

logger = logging.getLogger(__name__)

MONITORING_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'runtime')
ALERTS_FILE = os.path.join(MONITORING_DIR, 'alerts.jsonl')

# 冷却表 {alert_key: last_sent_timestamp}
_cooldown = {}

# 预定义告警级别
P0 = 'P0'  # 阻断交易，需立即处理
P1 = 'P1'  # 影响功能，需当日处理
P2 = 'P2'  # 提醒，不影响交易

# 冷却时间（秒）
COOLDOWN = {P0: 300, P1: 1800, P2: 3600}


def load_webhook_config() -> Optional[str]:
    """从 .env 或环境变量读取 webhook URL"""
    url = os.getenv('ALERT_WEBHOOK_URL', '')
    if not url:
        # 尝试从 .env 读取
        env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env')
        if os.path.isfile(env_path):
            with open(env_path, encoding='utf-8') as f:
                for line in f:
                    if line.startswith('ALERT_WEBHOOK_URL='):
                        url = line.split('=', 1)[1].strip()
                        break
    return url or None


def alert(code: str, level: str, message: str, **context):
    """
    发送告警（去重+多通道）。

    Args:
        code: 告警代码（如 SIGNAL_COVERAGE_COLLAPSE）
        level: P0/P1/P2
        message: 人可读消息
        **context: 附加上下文
    """
    symbol = context.get('symbol', '')
    key = f"{code}:{symbol}"
    now = time.time()
    cooldown_sec = COOLDOWN.get(level, 1800)

    # 去重：冷却期内跳过
    if key in _cooldown and now - _cooldown[key] < cooldown_sec:
        return

    _cooldown[key] = now

    event = {
        'timestamp': datetime.now().isoformat(),
        'code': code,
        'level': level,
        'message': message,
        **context
    }

    # 1. 本地审计（始终写入）
    os.makedirs(MONITORING_DIR, exist_ok=True)
    with open(ALERTS_FILE, 'a', encoding='utf-8') as f:
        f.write(json.dumps(event, ensure_ascii=False) + '\n')

    # 2. 日志
    log_fn = logger.critical if level == P0 else (logger.warning if level == P1 else logger.info)
    log_fn(f"[Alert:{level}] {code}: {message}")

    # 3. Webhook（如果配置；失败重试+死信落盘，告警不丢——评审 §2.8/#12）
    webhook_url = load_webhook_config()
    if webhook_url:
        _send_webhook_with_retry(webhook_url, event)


def _send_webhook_with_retry(url: str, event: dict, max_retries: int = 3):
    """webhook 发送：3次重试（2s/4s 退避），全败写死信文件（可人工重放）。"""
    import time as _t
    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            _send_webhook(url, event)
            return True
        except Exception as e:
            last_err = e
            logger.warning(f"[Alert] webhook 第{attempt}/{max_retries}次发送失败: {e}")
            if attempt < max_retries:
                _t.sleep(min(2 ** attempt, 8))   # 2s → 4s 退避
    try:
        os.makedirs(MONITORING_DIR, exist_ok=True)
        with open(os.path.join(MONITORING_DIR, 'webhook_dead.jsonl'), 'a',
                  encoding='utf-8') as f:
            f.write(json.dumps({'retry_error': str(last_err), **event},
                               ensure_ascii=False) + '\n')
        logger.error(f"[Alert] webhook {max_retries}次全败，已写死信 "
                     f"(monitoring/webhook_dead.jsonl): {last_err}")
    except Exception:
        logger.critical(f"[Alert] webhook 全败且死信写入失败: {last_err}")
    return False


def _send_webhook(url: str, event: dict):
    """发送webhook（飞书/企业微信格式兼容）"""
    import urllib.request
    import urllib.error

    # 飞书格式（企业微信也兼容）
    emoji = {'P0': '🔴', 'P1': '🟡', 'P2': '🔵'}
    body = json.dumps({
        'msg_type': 'text',
        'content': {
            'text': f"{emoji.get(event['level'], '⚪')} [{event['level']}] {event['code']}\n"
                   f"{event['message']}\n"
                   f"时间: {event['timestamp'][:19]}"
        }
    }).encode('utf-8')

    req = urllib.request.Request(
        url, data=body,
        headers={'Content-Type': 'application/json'})
    urllib.request.urlopen(req, timeout=5)


# ================= 预定义告警事件 =================

def alert_signal_collapse(n_targets: int, min_required: int):
    alert('SIGNAL_COVERAGE_COLLAPSE', P0,
          f"目标持仓仅{n_targets}只（下限{min_required}只），数据链路异常",
          n_targets=n_targets, min_required=min_required)


def alert_factor_coverage(factor: str, coverage: float, threshold: float):
    alert('FACTOR_COVERAGE_LOW', P0 if coverage < 0.1 else P1,
          f"因子'{factor}'覆盖率{coverage:.0%}低于{threshold:.0%}",
          factor=factor, coverage=coverage)


def alert_reconciliation_failed(n_diffs: int, date: str):
    alert('RECONCILIATION_FAILED', P0,
          f"对账不一致：{n_diffs}项差异 @ {date}",
          n_diffs=n_diffs, date=date)


def alert_qmt_order_failed(symbol: str, reason: str):
    alert('QMT_ORDER_FAILED', P0,
          f"QMT下单失败 {symbol}: {reason}",
          symbol=symbol)


def alert_qmt_query_failed(order_id: str):
    alert('QMT_QUERY_FAILED', P0,
          f"QMT查询失败，在途订单状态未知: {order_id}",
          order_id=order_id)


def alert_switch_state_changed(from_state: str, to_state: str, reason: str):
    alert('SWITCH_STATE_CHANGED', P0 if to_state == 'HALT_ALL' else P1,
          f"交易控制: {from_state} → {to_state} ({reason})",
          from_state=from_state, to_state=to_state)


def alert_data_source_degraded(source: str, detail: str):
    alert('DATA_SOURCE_DEGRADED', P1,
          f"数据源降级: {source} ({detail})",
          source=source)


def alert_nav_anomaly(nav_change: float):
    alert('NAV_ANOMALY', P0 if abs(nav_change) > 0.1 else P1,
          f"净值异常变动{nav_change:+.1%}",
          nav_change=nav_change)


def alert_daily_task_not_run(date: str):
    alert('DAILY_TASK_NOT_RUN', P1,
          f"当日任务未运行: {date}", date=date)
