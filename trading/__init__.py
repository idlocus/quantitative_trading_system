#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
唯一主链 —— 交易语义共享层

主链（回测与实盘共用，单一事实来源）：
    数据快照 → PIT因子 → 信号 → 目标组合 → 订单 → 成交模拟/真实成交 → 账本 → 净值与归因

共用组件（本包）：
    config.py  主链参数（因子/TopN/缓冲带/过滤/费用/约束），回测与实盘的默认值全部取自这里
    rules.py   交易规则（板块涨跌停比例与判定、整手、费用率计算）

接入方：
    - portfolio/backtester.py（回测执行器）
    - live/signals.py + live/broker.py（生产执行器：Paper/QMT 同路径）

差异只允许存在于：
    - 成交执行器：回测=向量化模拟（含市场冲击），实盘=PaperBroker/QMTBroker
    - 风控层：回测=risk_manager（研究约束），实盘=risk_gate（L2-L4 强制执行）——阈值口径一致

旧链定位（不在主链上，属研究遗留）：
    backtesting/engine.py + strategy/ + selection/ —— 个股止损型策略引擎，
    与生产因子链是两个产品形态，勿相互引用语义。
"""

from . import config, rules

__all__ = ['config', 'rules']
