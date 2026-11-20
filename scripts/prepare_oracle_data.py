"""Prepare research data through the authorized existing Oracle connection."""
import argparse
import json
import os
from pathlib import Path
import sys
from datetime import date

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job', choices=['probe', 'references', 'prices', 'coverage', 'supporting', 'hk-candidates', 'financials'], required=True)
    parser.add_argument('--years', nargs='+', type=int)
    parser.add_argument('--cutoff', type=date.fromisoformat)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    from data.oracle_preparation import connection_settings, connect_readonly, download_a_prices, download_hk_candidates, inspect_coverage, prepare_references, prepare_supporting, probe_database, safe_error, download_financials
    if args.job in ('prices', 'hk-candidates', 'supporting', 'financials') and (not args.years or args.cutoff is None):
        parser.error('This job requires --years and --cutoff')
    try:
        import oracledb
        settings = connection_settings(root / '.env', os.environ)
        with connect_readonly(settings, oracledb) as reader:
            output = root / 'data_cache' / 'preparation'
            if args.job == 'references':
                result = {'status': 'connected', 'read_only': True, **prepare_references(reader, output)}
            elif args.job == 'prices':
                result = {'status': 'downloaded', 'read_only': True,
                          **download_a_prices(reader, output, args.years, args.cutoff)}
            elif args.job == 'coverage':
                result = {'status': 'connected', 'read_only': True, **inspect_coverage(reader, output)}
            elif args.job == 'supporting':
                result = {'status': 'downloaded', 'read_only': True, **prepare_supporting(reader, output, min(args.years), args.cutoff)}
            elif args.job == 'financials':
                result = {'status': 'downloaded', 'read_only': True,
                          **download_financials(reader, output, args.cutoff.isoformat())}
            elif args.job == 'hk-candidates':
                result = {'status': 'downloaded', 'read_only': True, **download_hk_candidates(reader, output, args.years, args.cutoff)}
            else:
                result = probe_database(reader, output)
                result = {'status': result['status'], 'read_only': True,
                          'candidate_tables': len(result['candidate_tables']),
                          'available_schemas': {name: len(columns) for name, columns in result['schemas'].items()},
                          'table_errors': result['table_errors']}
    except Exception as exc:
        result = {'status': 'error', **safe_error(exc)}
    print(json.dumps(result, ensure_ascii=True))
    return 0 if result['status'] in ('connected', 'downloaded') and not result.get('errors') else 2


if __name__ == '__main__':
    raise SystemExit(main())
