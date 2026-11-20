"""Run the first stock signal study offline, leaving the holdout untouched."""
import argparse
import json
from pathlib import Path
import sys

if __name__ == '__main__':
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.preparation import atomic_json
from research.signal_screen import screen_market


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--market', choices=['a', 'hk', 'all'], default='all')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    protocol = json.loads((root / 'research/baseline_protocol.json').read_text(encoding='utf-8'))
    destination = root / 'ai-created/research/first_stock_signal_screen_2026-11-03_v2'
    registered = destination / 'protocol.json'
    if registered.exists() and json.loads(registered.read_text(encoding='utf-8')) != protocol:
        raise ValueError('Registered study protocol cannot change in place')
    atomic_json(registered, protocol)
    markets = ['a', 'hk'] if args.market == 'all' else [args.market]
    for market in markets:
        print(json.dumps({'status': 'running', 'market': market}), flush=True)
        result = screen_market(root / 'data_cache/preparation', destination, market, protocol)
        print(json.dumps(result, allow_nan=False), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
