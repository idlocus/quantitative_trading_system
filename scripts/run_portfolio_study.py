"""Run registered stock portfolio baselines without accounts or network access."""
import json
from pathlib import Path
import sys

if __name__ == '__main__':
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research.portfolio_study import run_portfolio_study


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    protocol = json.loads((root / 'research/portfolio_protocol.json').read_text(encoding='utf-8'))
    signals = json.loads((root / 'research/baseline_protocol.json').read_text(encoding='utf-8'))
    result = run_portfolio_study(root / 'data_cache/preparation',
        root / 'ai-created/research/first_a_portfolio_2026-11-03_v3', protocol, signals)
    for name, report in result['runs'].items():
        metrics = report['metrics']
        print(json.dumps({'run': name, 'geometric_monthly_return': metrics['geometric_monthly_return'],
            'max_drawdown': metrics['max_drawdown'], 'max_drawdown_amount': metrics['max_drawdown_amount'],
            'worst_daily_month_to_date': metrics['worst_daily_month_to_date'],
            'accounting_issues': report['accounting_issues'],
            'passes_observed_research_risk_limits': report['passes_observed_research_risk_limits']}, allow_nan=False), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
