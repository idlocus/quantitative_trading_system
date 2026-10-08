#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Wind MCP CLI 封装

通过 node scripts/cli.mjs 调用 Wind MCP 服务，执行自然语言选股等预筛选操作。
"""

import subprocess
import json
import os
from typing import List, Optional

from utils.logger import get_logger

logger = get_logger(__name__)

# Wind MCP skill 目录（全局安装路径）
_WIND_MCP_SKILL_DIR = os.path.expanduser("~/.agents/skills/wind-mcp-skill")
_CLI_SCRIPT = "scripts/cli.mjs"


class WindMCPClient:
    """
    Wind MCP CLI 封装类。

    用于自然语言预筛选（选股条件生成、宏观数据查询等）。
    数据源通过 CLI 调用万得金融数据服务。
    """

    def __init__(self, skill_dir: str = None):
        self.skill_dir = skill_dir or _WIND_MCP_SKILL_DIR
        self._cli = os.path.join(self.skill_dir, _CLI_SCRIPT)

    def search_stocks(self, question: str) -> List[str]:
        """
        自然语言选股，返回股票代码列表。

        Args:
            question: 自然语言描述的选股条件，如 "沪深市值超500亿连续5日上涨"

        Returns:
            股票代码列表，如 ["600519.SH", "000858.SZ"]
        """
        params = {"question": question.replace(" ", "")}
        result = self._call("stock_data", "search_stocks", params)
        if result is None:
            return []
        return result

    def get_index_pe_hist(self, index_code: str) -> Optional[dict]:
        """获取指数 PE/PB 历史分位（Wind MCP 实现）"""
        params = {"windcode": index_code}
        result = self._call("index_data", "get_index_history", params)
        return result

    def _call(self, server_type: str, tool_name: str, params: dict) -> Optional[any]:
        """
        执行一次 Wind MCP CLI 调用。

        Args:
            server_type: server_type 名称
            tool_name: 工具名称
            params: JSON 可序列化的参数字典

        Returns:
            解析后的结果（dict/list），失败返回 None
        """
        import shlex

        params_json = json.dumps(params, ensure_ascii=False)
        cmd = f'cd {self.skill_dir} && node {self._cli} call {server_type} {tool_name} {params_json}'

        try:
            logger.debug(f"Wind MCP call: {cmd[:120]}")
            output = subprocess.check_output(
                cmd, shell=True, stderr=subprocess.STDOUT, timeout=30
            )
            output_str = output.decode("utf-8", errors="replace").strip()

            # 解析 stdout 中的 MCP result
            return self._parse_mcp_result(output_str)

        except subprocess.CalledProcessError as e:
            output = e.output.decode("utf-8", errors="replace")
            logger.warning(f"Wind MCP CLI error: {output[:200]}")
            return None
        except subprocess.TimeoutExpired:
            logger.warning(f"Wind MCP call timeout: {server_type}.{tool_name}")
            return None
        except Exception as e:
            logger.warning(f"Wind MCP call failed: {e}")
            return None

    def _parse_mcp_result(self, raw: str) -> Optional[any]:
        """
        解析 Wind MCP CLI stdout。

        Wind MCP 成功时 stdout 是一个 MCP JSON result 块。
        尝试从 raw text 中提取 JSON。
        """
        # 如果是纯 JSON，直接解析
        try:
            return json.loads(raw)
        except Exception:
            pass

        # 尝试提取 ```json ... ``` 包裹的内容
        import re
        m = re.search(r"```json\s*(.*?)\s*```", raw, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(1))
            except Exception:
                pass

        # 尝试提取 "content":[{"text":"..."}]" 包裹的文本
        m = re.search(r'"text"\s*:\s*"(\[.*?\]|"[^"]*")', raw, re.DOTALL)
        if m:
            inner = m.group(1)
            if inner.startswith("["):
                try:
                    return json.loads(inner)
                except Exception:
                    pass
            elif inner.startswith('"'):
                try:
                    return json.loads(inner)
                except Exception:
                    pass

        logger.debug(f"Cannot parse MCP result: {raw[:100]}")
        return None