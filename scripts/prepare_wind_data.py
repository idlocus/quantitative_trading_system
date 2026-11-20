"""Explicit read-only data preparation through an existing Wind terminal."""
import argparse
import contextlib
import io
import json
from datetime import date
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job', choices=['lifecycle', 'repair-year', 'market-status', 'connect-daily'], required=True)
    parser.add_argument('--start', type=date.fromisoformat)
    parser.add_argument('--end', type=date.fromisoformat)
    parser.add_argument('--year', type=int)
    parser.add_argument('--cutoff', type=date.fromisoformat)
    parser.add_argument('--max-values', type=int, default=250_000)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    from data.wind_preparation import prepare_lifecycle, prepare_market_status
    from data.year_repair import build_year_plan, fetch_year_patches
    if args.job == 'repair-year' and (args.year is None or args.cutoff is None):
        parser.error('repair-year requires --year and --cutoff')
    if args.job == 'market-status' and args.year is None:
        parser.error('market-status requires --year')
    if args.job == 'connect-daily' and (args.start is None or args.end is None):
        parser.error('connect-daily requires --start and --end')

    try:
        with contextlib.redirect_stdout(io.StringIO()):
            from WindPy import w
            connection = w.start(waitTime=10)
            if connection.ErrorCode != 0:
                result = {'status': 'connection_error', 'error_code': connection.ErrorCode}
            elif args.job == 'lifecycle':
                result = prepare_lifecycle(w, root / 'data_cache', root / 'data_cache' / 'preparation')
            elif args.job == 'market-status':
                result = prepare_market_status(w, root / 'data_cache' / 'preparation', args.year)
            elif args.job == 'connect-daily':
                from data.connect_preparation import download_connect_daily
                result = download_connect_daily(w, root / 'data_cache' / 'preparation', args.start, args.end)
            else:
                output = root / 'data_cache' / 'preparation'
                plan = build_year_plan(root / 'data_cache', output, args.year, args.cutoff)
                result = fetch_year_patches(w, output, plan, args.max_values)
    except Exception as exc:
        result = {'status': 'error', 'error_type': type(exc).__name__}
    print(json.dumps(result, ensure_ascii=False, default=str))
    return 2 if result.get('errors') or result.get('status') in ('connection_error', 'blocked', 'error') else 0


if __name__ == '__main__':
    raise SystemExit(main())
