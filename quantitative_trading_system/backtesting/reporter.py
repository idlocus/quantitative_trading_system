#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
回测报告生成器
"""

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

        # 性能汇总表
        lines.append("## 性能汇总\n")
        lines.append("| 指标 | 值 |")
        lines.append("| --- | --- |")
        lines.append(f"| 总收益率 | {result.total_return * 100:.2f}% |")
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
            lines.append("| 序号 | 日期 | 股票 | 操作 | 价格 | 数量 | 手续费 | 触发原因 | 信号评分 |")
            lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
            for trade in result.trades:
                conditions_str = self._format_conditions(trade.conditions)
                lines.append(
                    f"| {trade.id} | {trade.date} | {trade.symbol} | {trade.action} | "
                    f"{trade.price:.2f} | {trade.quantity} | {trade.commission:.2f} | "
                    f"{trade.reason} | {trade.signal_score} |"
                )
            lines.append("")
            # 列出每笔交易的指标状态详情
            lines.append("### 交易指标详情\n")
            buy_trades = [t for t in result.trades if t.action == "BUY"]
            sell_trades = [t for t in result.trades if t.action == "SELL"]
            for trade in result.trades:
                lines.append(f"**Trade #{trade.id} ({trade.action}) - {trade.symbol} on {trade.date}**")
                lines.append(f"- 价格: {trade.price:.2f}, 数量: {trade.quantity}")
                lines.append(f"- 手续费: {trade.commission:.2f}, 触发原因: {trade.reason}")
                lines.append(f"- 信号评分: {trade.signal_score}")
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
                    legend: {
                        display: false
                    }
                },
                scales: {
                    x: {
                        grid: { display: false }
                    },
                    y: {
                        grid: { color: '#eee' }
                    }
                }
            }
        });
    </script>
</body>
</html>
"""
        return html

    def _format_conditions(self, conditions: List[dict]) -> str:
        """格式化条件列表为字符串"""
        if not conditions:
            return "-"
        return "; ".join(str(c) for c in conditions)

    def _format_conditions_html(self, conditions: List[dict]) -> str:
        """格式化条件列表为HTML"""
        if not conditions:
            return "-"
        return "<ul>" + "".join(f"<li>{c}</li>" for c in conditions) + "</ul>"

    def _get_equity_curve_json(self, result: BacktestResult) -> str:
        """获取权益曲线JSON数据"""
        values = result.equity_curve.tolist() if hasattr(result.equity_curve, 'tolist') else list(result.equity_curve)
        labels = [str(i) for i in range(len(values))]
        import json
        return json.dumps({"labels": labels, "values": values})