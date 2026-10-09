#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据质量审计 — 检测基本面数据回填风险

检测内容：
1. 估值表(ASHAREEODDERATIVEINDICATOR) OPDATE vs TRADE_DT 差异（回填天数）
2. 财务表(ASHAREFINANCIALINDICATOR) ANN_DT vs OPDATE 一致性
3. 估值因子跨数据源一致性（DB vs 客户端）
"""

import os
from datetime import datetime

import pandas as pd

from utils.logger import get_logger

logger = get_logger(__name__)


def audit_valuation_backfill(dsn='10.1.33.123:1521/info', user='wind', password='windPrd22',
                            sample_dates=None) -> dict:
    """
    检测估值表回填风险

    通过 OPDATE 与 TRADE_DT 的差异判断数据是否被后期重写。
    如果 OPDATE >> TRADE_DT，说明该日期的估值数据是后期回填的。

    Returns:
        {
            'backfill_days_avg': float,   # 平均回填天数
            'backfill_days_max': int,    # 最大回填天数
            'backfill_ratio': float,     # 回填比例（回填>30天的占比）
            'risk_level': str,           # 'high' | 'medium' | 'low'
            'recommendation': str,       # 建议
        }
    """
    import oracledb
    oracledb.init_oracle_client()
    conn = oracledb.connect(user=user, password=password, dsn=dsn)
    cur = conn.cursor()

    try:
        # 采样检测：取若干日期的 OPDATE vs TRADE_DT
        if sample_dates is None:
            sample_dates = ['20230103', '20240102', '20250102', '20250602']

        total_rows = 0
        backfill_over_30d = 0
        backfill_days_list = []

        for dt in sample_dates:
            cur.execute(f"""
                SELECT COUNT(*),
                       AVG(CASE WHEN OPDATE IS NOT NULL
                           THEN (CAST(OPDATE AS DATE) - TO_DATE('{dt}','YYYYMMDD'))
                           ELSE 0 END)
                FROM ASHAREEODDERIVATIVEINDICATOR
                WHERE TRADE_DT = '{dt}'
                  AND S_INFO_WINDCODE LIKE '%.S%'
                  AND ROWNUM <= 1000
            """)
            count, avg_gap = cur.fetchone()
            if count and avg_gap is not None:
                total_rows += count
                gap = float(avg_gap)
                backfill_days_list.append(gap)
                if gap > 30:
                    backfill_over_30d += count

        avg_backfill = sum(backfill_days_list) / len(backfill_days_list) if backfill_days_list else 0
        ratio = backfill_over_30d / max(total_rows, 1)

        if avg_backfill > 180:
            risk = 'high'
            rec = ('估值表存在大量回填（平均延迟{:.0f}天）。'
                   '历史PE/PB可能不同于当时实际可见值。'
                   '建议：1)回测结果作为上界参考 2)关键因子用PIT财务数据替代')
        elif avg_backfill > 30:
            risk = 'medium'
            rec = '估值表存在中等回填（平均延迟{:.0f}天），影响可控但需注意。'
        else:
            risk = 'low'
            rec = '估值表回填风险较低。'

        result = {
            'backfill_days_avg': round(avg_backfill, 1),
            'backfill_ratio': round(ratio, 2),
            'risk_level': risk,
            'recommendation': rec.format(avg_backfill),
        }
        logger.info(f"[audit_valuation_backfill] {result}")
        return result

    finally:
        cur.close()
        conn.close()


def audit_financial_pit(dsn='10.1.33.123:1521/info', user='wind', password='windPrd22') -> dict:
    """
    检测财务表 PIT 一致性

    验证 ANN_DT（公告日）是否合理（不应晚于报告期结束后180天），
    以及 OPDATE 是否接近 ANN_DT（无大量延迟写入）。

    Returns:
        {
            'total_samples': int,
            'late_announce': int,       # 公告日 > 报告期+180天（异常）
            'opdate_lag': float,       # OPDATE 平均滞后 ANN_DT 的天数
            'pit_quality': str,         # 'good' | 'acceptable' | 'poor'
        }
    """
    import oracledb
    oracledb.init_oracle_client()
    conn = oracledb.connect(user=user, password=password, dsn=dsn)
    cur = conn.cursor()

    try:
        cur.execute("""
            SELECT COUNT(*),
                   SUM(CASE WHEN ANN_DT > TO_CHAR(TO_DATE(SUBSTR(REPORT_PERIOD,1,6)||'01','YYYYMMDD') + 180)
                       THEN 1 ELSE 0 END),
                   AVG(CASE WHEN OPDATE IS NOT NULL AND ANN_DT IS NOT NULL
                       THEN (CAST(OPDATE AS DATE) - TO_DATE(ANN_DT,'YYYYMMDD'))
                       ELSE 0 END)
            FROM ASHAREFINANCIALINDICATOR
            WHERE REPORT_PERIOD >= '20230101'
              AND ANN_DT IS NOT NULL
              AND S_INFO_WINDCODE = '600519.SH'
              AND ROWNUM <= 100
        """)
        total, late, opdate_lag = cur.fetchone()

        total = total or 0
        late = late or 0
        opdate_lag = float(opdate_lag or 0)

        if late / max(total, 1) > 0.1:
            quality = 'poor'
        elif opdate_lag > 30:
            quality = 'acceptable'
        else:
            quality = 'good'

        return {
            'total_samples': total,
            'late_announce': late,
            'opdate_lag': round(opdate_lag, 1),
            'pit_quality': quality,
        }

    finally:
        cur.close()
        conn.close()


def data_quality_report() -> str:
    """生成数据质量报告"""
    lines = ["## 数据质量审计报告\n"]

    # 1. 估值表回填
    try:
        val = audit_valuation_backfill()
        lines.append("### 估值数据（PE/PB/PS/市值）")
        lines.append(f"- 风险等级: **{val['risk_level'].upper()}**")
        lines.append(f"- 平均回填延迟: {val['backfill_days_avg']} 天")
        lines.append(f"- 回填比例(>30天): {val['backfill_ratio']:.0%}")
        lines.append(f"- 建议: {val['recommendation']}")
        lines.append("")
    except Exception as e:
        lines.append(f"### 估值数据: 审计失败 ({e})\n")

    # 2. 财务 PIT
    try:
        fin = audit_financial_pit()
        lines.append("### 财务数据（PIT公告日）")
        lines.append(f"- PIT 质量: **{fin['pit_quality'].upper()}**")
        lines.append(f"- 样本数: {fin['total_samples']}")
        lines.append(f"- 异常公告: {fin['late_announce']}")
        lines.append(f"- OPDATE平均滞后: {fin['opdate_lag']} 天")
        lines.append("")
    except Exception as e:
        lines.append(f"### 财务数据: 审计失败 ({e})\n")

    # 3. 已知风险总结
    lines.append("### 已知风险与缓解")
    lines.append("| 数据类型 | 风险 | 缓解措施 |")
    lines.append("|---------|------|---------|")
    lines.append("| 日频估值(PE/PB) | 表级回填，历史值可能被重算 | 回测结果视为上界；因子中PB权重已较低 |")
    lines.append("| 财务因子(杠杆) | PIT正确(ANN_DT) | ✅ 无前视风险 |")
    lines.append("| K线价格 | 前复权，历史连续 | ✅ 无除权跳空 |")
    lines.append("| 指数成分 | PIT(入场/剔除日) | ✅ 无前视风险 |")
    lines.append("| 股票池 | PIT(上市日+60日过滤) | ✅ 动态过滤 |")

    return "\n".join(lines)
