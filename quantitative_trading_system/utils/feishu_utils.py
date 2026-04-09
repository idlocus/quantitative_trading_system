#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
飞书Webhook发送工具
"""

import requests


FEISHU_WEBHOOK = "https://open.feishu.cn/open-apis/bot/v2/hook/c1daaf4c-bc18-4a96-b0c6-812fedf81bff"


def send_feishu_message(message: str) -> bool:
    """
    发送消息到飞书

    Args:
        message: 消息内容

    Returns:
        bool: 是否发送成功
    """
    try:
        payload = {
            "msg_type": "text",
            "content": {
                "text": message
            }
        }
        response = requests.post(FEISHU_WEBHOOK, json=payload, timeout=10)
        result = response.json()

        if result.get("code") == 0 or result.get("StatusCode") == 0:
            return True
        else:
            print(f"飞书发送失败: {result}")
            return False

    except Exception as e:
        print(f"飞书发送异常: {e}")
        return False


def send_feishu_card_message(title: str, content: str) -> bool:
    """
    发送卡片消息到飞书

    Args:
        title: 卡片标题
        content: 卡片内容

    Returns:
        bool: 是否发送成功
    """
    try:
        payload = {
            "msg_type": "interactive",
            "card": {
                "config": {
                    "wide_screen_mode": True
                },
                "header": {
                    "title": {
                        "tag": "plain_text",
                        "content": title
                    },
                    "template": "blue"
                },
                "elements": [
                    {
                        "tag": "div",
                        "text": {
                            "tag": "lark_md",
                            "content": content
                        }
                    }
                ]
            }
        }
        response = requests.post(FEISHU_WEBHOOK, json=payload, timeout=10)
        result = response.json()

        if result.get("code") == 0 or result.get("StatusCode") == 0:
            return True
        else:
            print(f"飞书卡片发送失败: {result}")
            return False

    except Exception as e:
        print(f"飞书卡片发送异常: {e}")
        return False


if __name__ == "__main__":
    # 测试发送
    test_msg = "这是一条测试消息"
    print(f"发送测试消息到飞书...")
    success = send_feishu_message(test_msg)
    print(f"发送{'成功' if success else '失败'}")
