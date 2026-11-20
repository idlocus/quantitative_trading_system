#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
回测报告生成器
"""

import json
from typing import List

from .engine import BacktestResult, Trade


class BacktestReporter:
    """回测报告生成器"""

    def __init__(self, template_dir: str = None):
        self.template_dir = template_dir or "backtesting/templates"

    def generate_markdown(self, result: BacktestResult) -> str:
        """
        生成Markdown格式报告

        Args:
            result: 回测结果

        Returns:
            Markdown格式报告字符串
        """
        lines = []

        # 标题
        lines.append("# 回测报告\n")

        # 数据局限性说明（诚实性声明）
        lines.append("> **数据局限性**：akshare 免费数据不含已退市股票（幸存者偏差，"
                     "收益可能被高估）；股票池为当前市场列表，历史时点成分无法完全还原。"
                     "涨跌停不可成交与交易税费（印花税/过户费）已在引擎中建模。\n")

        # 性能汇总表
        lines.append("## 性能汇总\n")
        lines.append("| 指标 | 值 |")
        lines.append("| --- | --- |")
        lines.append(f"| 总收益率 | {result.total_return * 100:.2f}% |")
        if result.benchmark_return is not None:
            lines.append(f"| 基准收益（{result.benchmark_symbol} 买入持有） | {result.benchmark_return * 100:.2f}% |")
            lines.append(f"| 超额收益 | {(result.total_return - result.benchmark_return) * 100:+.2f}% |")
        lines.append(f"| 年化收益率 | {result.annual_return * 100:.2f}% |")
        lines.append(f"| 夏普比率 | {result.sharpe_ratio:.2f} |")
        lines.append(f"| 最大回撤 | {result.max_drawdown * 100:.2f}% |")
        lines.append(f"| 胜率 | {result.win_rate * 100:.2f}% |")
        lines.append(f"| 交易次数 | {result.trade_count} |")
        lines.append(f"| 平均持仓天数 | {result.avg_holding_days:.1f} |")
        lines.append(f"| 初始资金 | {result.initial_capital:,.2f} |")
        lines.append(f"| 最终资金 | {result.final_capital:,.2f} |")
        lines.append("")

        # 交易记录表
        lines.append("## 交易记录\n")
        if result.trades:
            lines.append("| 序号 | 日期 | 股票 | 操作 | 价格 | 数量 | 手续费 | 触发原因 | 信号评分 | 阶段 |")
            lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
            for trade in result.trades:
                lines.append(
                    f"| {trade.id} | {trade.date} | {trade.symbol} | {trade.action} | "
                    f"{trade.price:.2f} | {trade.quantity} | {trade.commission:.2f} | "
                    f"{trade.reason} | {trade.signal_score} | {trade.phase} |"
                )
            lines.append("")
            # 列出每笔交易的指标状态详情
            lines.append("### 交易指标详情\n")
            for trade in result.trades:
                lines.append(f"**Trade #{trade.id} ({trade.action}) - {trade.symbol} on {trade.date}**")
                lines.append(f"- 价格: {trade.price:.2f}, 数量: {trade.quantity}")
                lines.append(f"- 手续费: {trade.commission:.2f}, 触发原因: {trade.reason}")
                lines.append(f"- 信号评分: {trade.signal_score}, 阶段: {trade.phase}")
                if trade.conditions:
                    lines.append("- 指标状态:")
                    for cond in trade.conditions:
                        lines.append(f"  - {cond}")
                lines.append("")
        else:
            lines.append("*暂无交易记录*\n")

        return "\n".join(lines)

    def generate_html(self, result: BacktestResult) -> str:
        """
        生成HTML可视化报告

        Args:
            result: 回测结果

        Returns:
            HTML格式报告字符串
        """
        equity_json = self._get_equity_curve_json(result)

        html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>回测报告</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif; background: #f5f5f5; padding: 20px; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        h1 {{ color: #333; margin-bottom: 20px; font-size: 28px; }}
        h2 {{ color: #444; margin: 20px 0 10px; font-size: 20px; border-bottom: 2px solid #007bff; padding-bottom: 5px; }}
        .metrics-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin-bottom: 20px; }}
        .metric-card {{ background: white; border-radius: 8px; padding: 15px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .metric-label {{ color: #666; font-size: 12px; margin-bottom: 5px; text-transform: uppercase; }}
        .metric-value {{ color: #333; font-size: 24px; font-weight: bold; }}
        .positive {{ color: #28a745; }}
        .negative {{ color: #dc3545; }}
        .chart-container {{ background: white; border-radius: 8px; padding: 20px; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .table-container {{ background: white; border-radius: 8px; padding: 20px; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); overflow-x: auto; }}
        table {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
        th, td {{ padding: 10px 12px; text-align: left; border-bottom: 1px solid #eee; }}
        th {{ background: #f8f9fa; font-weight: 600; color: #333; position: sticky; top: 0; }}
        tr:hover {{ background: #f8f9fa; }}
        .buy {{ color: #28a745; font-weight: bold; }}
        .sell {{ color: #dc3545; font-weight: bold; }}
        .conditions-list {{ font-size: 12px; color: #666; max-width: 200px; }}
        .conditions-list li {{ margin: 2px 0; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>回测报告</h1>

        <h2>性能指标</h2>
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-label">总收益率</div>
                <div class="metric-value {'positive' if result.total_return >= 0 else 'negative'}">{result.total_return * 100:.2f}%</div>
            </div>
"""
        if result.benchmark_return is not None:
            excess = result.total_return - result.benchmark_return
            html += f"""
            <div class="metric-card">
                <div class="metric-label">基准收益（{result.benchmark_symbol}）</div>
                <div class="metric-value {'positive' if result.benchmark_return >= 0 else 'negative'}">{result.benchmark_return * 100:.2f}%</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">超额收益</div>
                <div class="metric-value {'positive' if excess >= 0 else 'negative'}">{excess * 100:+.2f}%</div>
            </div>
"""
        html += f"""
            <div class="metric-card">
                <div class="metric-label">年化收益率</div>
                <div class="metric-value {'positive' if result.annual_return >= 0 else 'negative'}">{result.annual_return * 100:.2f}%</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">夏普比率</div>
                <div class="metric-value">{result.sharpe_ratio:.2f}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">最大回撤</div>
                <div class="metric-value negative">{result.max_drawdown * 100:.2f}%</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">胜率</div>
                <div class="metric-value">{result.win_rate * 100:.2f}%</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">交易次数</div>
                <div class="metric-value">{result.trade_count}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">平均持仓天数</div>
                <div class="metric-value">{result.avg_holding_days:.1f}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">初始资金</div>
                <div class="metric-value">{result.initial_capital:,.2f}</div>
            </div>
        </div>

        <h2>权益曲线</h2>
        <div class="chart-container">
            <canvas id="equityChart" height="100"></canvas>
        </div>

        <h2>交易记录</h2>
        <div class="table-container">
            <table>
                <thead>
                    <tr>
                        <th>序号</th>
                        <th>日期</th>
                        <th>股票</th>
                        <th>操作</th>
                        <th>价格</th>
                        <th>数量</th>
                        <th>手续费</th>
                        <th>触发原因</th>
                        <th>信号评分</th>
                        <th>阶段</th>
                    </tr>
                </thead>
                <tbody>
"""

        for trade in result.trades:
            action_class = 'buy' if trade.action == 'BUY' else 'sell'
            conditions_str = self._format_conditions_html(trade.conditions)
            html += f"""
                    <tr>
                        <td>{trade.id}</td>
                        <td>{trade.date}</td>
                        <td>{trade.symbol}</td>
                        <td class="{action_class}">{trade.action}</td>
                        <td>{trade.price:.2f}</td>
                        <td>{trade.quantity}</td>
                        <td>{trade.commission:.2f}</td>
                        <td>{trade.reason}</td>
                        <td>{trade.signal_score}</td>
                        <td>{trade.phase}</td>
                    </tr>
"""

        html += """
                </tbody>
            </table>
        </div>

        <h2>触发条件</h2>
        <div class="table-container">
            <table>
                <thead>
                    <tr>
                        <th>序号</th>
                        <th>日期</th>
                        <th>股票</th>
                        <th>操作</th>
                        <th>触发条件</th>
                    </tr>
                </thead>
                <tbody>
"""
        for trade in result.trades:
            action_class = 'buy' if trade.action == 'BUY' else 'sell'
            conditions_str = self._format_conditions(trade.conditions)
            html += f"""
                    <tr>
                        <td>{trade.id}</td>
                        <td>{trade.date}</td>
                        <td>{trade.symbol}</td>
                        <td class="{action_class}">{trade.action}</td>
                        <td>{conditions_str}</td>
                    </tr>
"""

        html += """
                </tbody>
            </table>
        </div>
    </div>

    <script>
        const equityData = """ + equity_json + """;
        const ctx = document.getElementById('equityChart').getContext('2d');
        new Chart(ctx, {
            type: 'line',
            data: {
                labels: equityData.labels,
                datasets: [{
                    label: '权益曲线',
                    data: equityData.values,
                    borderColor: '#007bff',
                    backgroundColor: 'rgba(0, 123, 255, 0.1)',
                    fill: true,
                    tension: 0.1,
                    pointRadius: 0
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: false }
                },
                scales: {
                    x: { grid: { display: false } },
                    y: { grid: { color: '#eee' } }
                }
            }
        });
    </script>
</body>
</html>
"""
        return html

    def generate_daily_monitor(self, engine, latest_date: str) -> str:
        """
        生成每日监控报告（Markdown格式）

        Args:
            engine: 回测引擎（含最新市场环境状态）
            latest_date: 最新交易日字符串 YYYYMMDD

        Returns:
            Markdown格式每日监控报告
        """
        lines = []
        lines.append(f"# 每日监控报告 - {latest_date}\n")

        # 市场环境信号
        env = getattr(engine, '_current_market_env', None)
        if env is not None:
            lines.append("## 市场环境\n")
            lines.append("| 指标 | 值 | 信号 |")
            lines.append("| --- | --- | --- |")
            lines.append(f"| 市场综合分 | {env.market_score} | 仓位系数 {env.position_factor:.2f} |")
            lines.append(f"| 市场宽度 | {env.breadth_signal} | {'看多' if env.breadth_signal == 'expanding' else '看空'} |")
            lines.append(f"| 估值水平 | {env.valuation_signal} | {'便宜' if env.valuation_signal == 'cheap' else '贵'} |")
            lines.append(f"| 宏观状态 | {env.macro_signal} | {'扩张' if env.macro_signal == 'expanding' else '收缩'} |")
            lines.append(f"| 先行指标 | {env.leading_signal} | {'扩张' if env.leading_signal == 'expanding' else '收缩'} |")
            lines.append(f"| 季节性 | {env.seasonal_signal} | {'有利' if env.seasonal_signal == 'favorable' else '不利'} |")
            lines.append(f"| 季节性因子 | {getattr(env, 'seasonal_factor', 1.0):.3f} | - |")
            lines.append("")

        # 当前持仓
        lines.append("## 当前持仓\n")
        if engine.positions:
            lines.append("| 股票 | 数量 | 入场价 | 现价 | 浮动盈亏 | 浮动收益率 |")
            lines.append("| --- | --- | --- | --- | --- | --- |")
            last_date = getattr(engine, '_last_trading_date', None)
            for symbol, pos in engine.positions.items():
                current = 0.0
                if last_date is not None and getattr(engine, '_data', None) is not None:
                    try:
                        current = engine._data.get_close(symbol, last_date)
                    except Exception:
                        current = 0.0
                if current > 0:
                    pnl = (current - pos.entry_price) * pos.quantity
                    pnl_pct = (current - pos.entry_price) / pos.entry_price * 100
                    lines.append(
                        f"| {symbol} | {pos.quantity} | {pos.entry_price:.2f} | {current:.2f} | {pnl:+,.2f} | {pnl_pct:+.2f}% |"
                    )
                else:
                    lines.append(
                        f"| {symbol} | {pos.quantity} | {pos.entry_price:.2f} | - | - | - |"
                    )
            lines.append("")
        else:
            lines.append("*当前无持仓*\n")

        # 今日交易
        today_trades = [t for t in engine.trades if t.date.strftime('%Y%m%d') == latest_date]
        lines.append(f"## 今日交易（{latest_date}）\n")
        if today_trades:
            lines.append("| 股票 | 操作 | 价格 | 数量 | 手续费 | 触发原因 | 信号评分 |")
            lines.append("| --- | --- | --- | --- | --- | --- | --- |")
            for t in today_trades:
                lines.append(
                    f"| {t.symbol} | {t.action} | {t.price:.2f} | {t.quantity} | "
                    f"{t.commission:.2f} | {t.reason} | {t.signal_score} |"
                )
            lines.append("")
        else:
            lines.append("*今日无交易*\n")

        # 账户概览
        total_equity = engine.equity_curve[-1] if engine.equity_curve else getattr(engine, 'cash', 0)
        pos_value = sum(pos.entry_price * pos.quantity for pos in engine.positions.values())
        lines.append("## 账户概览\n")
        lines.append(f"- **当日权益**: {total_equity:,.2f}")
        lines.append(f"- **持仓市值**: {pos_value:,.2f}")
        lines.append(f"- **现金**: {engine.cash:,.2f}")
        lines.append(f"- **持仓数量**: {len(engine.positions)}/{engine.config.max_positions}")
        lines.append("")

        return "\n".join(lines)

    def _format_conditions(self, conditions: List[dict]) -> str:
        """格式化条件列表为字符串"""
        if not conditions:
            return "-"
        parts = []
        for c in conditions:
            if c.get('type') == 'sector_filter':
                parts.append(f"[板块]{c.get('industry_name', '')}: {c.get('indicator', '')}")
            else:
                parts.append(f"{c.get('indicator', '')}{c.get('operator', '')}{c.get('threshold', '')}")
        return "; ".join(parts)

    def _format_conditions_html(self, conditions: List[dict]) -> str:
        """格式化条件列表为HTML"""
        if not conditions:
            return "-"
        html = "<ul>"
        for c in conditions:
            if c.get('type') == 'sector_filter':
                html += f"<li>[板块] {c.get('industry_name', '')}: {c.get('indicator', '')} = {c.get('value', '')}</li>"
            else:
                html += f"<li>{c.get('indicator', '')} {c.get('operator', '')} {c.get('threshold', '')} = {c.get('value', '')}</li>"
        return html + "</ul>"

    def _get_equity_curve_json(self, result: BacktestResult) -> str:
        """获取权益曲线JSON数据（以交易日为标签）"""
        values = result.equity_curve.tolist() if hasattr(result.equity_curve, 'tolist') else list(result.equity_curve)
        index = result.equity_curve.index if hasattr(result.equity_curve, 'index') else range(len(values))
        labels = [str(d) for d in index]
        return json.dumps({"labels": labels, "values": values})
