"""Offline workspace commands; retired trading commands explicitly refuse to run."""
import argparse
from dataclasses import asdict
from datetime import date
import json
from typing import Sequence

from .workspace import Workspace


def retired_entrypoint(name: str) -> int:
    print(json.dumps({'status': 'disabled', 'entrypoint': name,
                      'reason': 'All previous strategies and scheduled trading jobs are archived.',
                      'next': 'Use python main.py status or python main.py data summary.'}))
    return 2


def main(argv: Sequence[str] | None = None, workspace: Workspace | None = None) -> int:
    workspace = workspace or Workspace.current()
    parser = argparse.ArgumentParser(description='Strategy-free quantitative research workspace')
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('status', help='Show workspace state without opening accounts or credentials')
    commands.add_parser('strategies', help='List explicitly registered strategies')
    archive = commands.add_parser('archive', help='Show or verify the source snapshot')
    archive.add_argument('--verify', action='store_true')
    run = commands.add_parser('run', help='Check an explicit strategy selection; no automatic execution')
    run.add_argument('--strategy', required=True)
    data = commands.add_parser('data', help='Inspect retained local caches without downloading')
    actions = data.add_subparsers(dest='action', required=True)
    actions.add_parser('summary')
    actions.add_parser('audit', help='Audit local metadata without downloads or cache writes')
    actions.add_parser('prepare-local', help='Validate retained daily bars with resumable checkpoints')
    actions.add_parser('prepared-summary', help='Inspect prepared datasets without credentials or provider connections')
    actions.add_parser('prepare-hk-universe', help='Freeze the explicitly authorized static-current research universe')
    annual = actions.add_parser('audit-years', help='Write yearly cache diagnostics using downloaded calendars')
    annual.add_argument('--start', type=date.fromisoformat, required=True)
    annual.add_argument('--end', type=date.fromisoformat, required=True)
    plan = actions.add_parser('repair-plan', help='Plan a yearly download without contacting a provider')
    plan.add_argument('--year', type=int, required=True)
    plan.add_argument('--cutoff', type=date.fromisoformat, required=True)
    stage = actions.add_parser('stage-year', help='Build yearly staging files from verified downloaded patches')
    stage.add_argument('--year', type=int, required=True)
    stage.add_argument('--cutoff', type=date.fromisoformat, required=True)
    listing = actions.add_parser('list')
    listing.add_argument('--category')
    listing.add_argument('--limit', type=int, default=20)
    show = actions.add_parser('show')
    show.add_argument('symbol')
    show.add_argument('--start', type=date.fromisoformat)
    show.add_argument('--end', type=date.fromisoformat)
    show.add_argument('--limit', type=int, default=10)
    # Existing task definitions remain harmless even if they call main.py.
    for name in ('daily', 'am', 'pm', 'scan', 'dashboard', 'report'):
        commands.add_parser(name, help='Archived command; execution is disabled')
    args, extras = parser.parse_known_args(argv)
    if args.command in ('daily', 'am', 'pm', 'scan', 'dashboard', 'report'):
        return retired_entrypoint(args.command)
    if extras:
        parser.error(f'Unrecognized arguments: {" ".join(extras)}')
    from strategies import registry
    try:
        if args.command == 'status':
            result = {'mode': 'unconfigured', 'active_strategies': list(registry.names()),
                      'automatic_trading': False, 'live_execution': False,
                      'cache_exists': workspace.cache.is_dir(),
                      'archive_exists': (workspace.archive / 'manifest.json').is_file()}
        elif args.command == 'strategies':
            result = {'strategies': list(registry.names()), 'default': None}
        elif args.command == 'archive':
            if args.verify:
                from .archive import verify_snapshot
                result = verify_snapshot(workspace)
                print(json.dumps(result, ensure_ascii=False, indent=2))
                return 0 if result['source_verified'] and result['retained_files_unchanged'] else 2
            result = {'path': str(workspace.archive),
                      'manifest_exists': (workspace.archive / 'manifest.json').is_file()}
        elif args.command == 'run':
            registry.get(args.strategy)
            raise RuntimeError('Execution requires an explicit simulation request; no live runner is configured')
        else:
            from data.catalog import LocalDataCatalog
            catalog = LocalDataCatalog(workspace.cache)
            if args.action == 'summary':
                result = catalog.summary()
            elif args.action == 'audit':
                from data.readiness import audit_local_readiness
                result = audit_local_readiness(workspace.cache)
            elif args.action == 'prepare-local':
                from data.preparation import prepare_local
                result = prepare_local(workspace.cache, workspace.cache / 'preparation')
            elif args.action == 'prepared-summary':
                from data.prepared import prepared_summary
                result = prepared_summary(workspace.cache / 'preparation')
            elif args.action == 'prepare-hk-universe':
                from data.hk_universe import build_static_universe
                result = build_static_universe(workspace.cache / 'preparation', workspace.root / 'research/data_policy.json')
            elif args.action == 'audit-years':
                from data.annual_audit import audit_years
                result = audit_years(workspace.cache, workspace.cache / 'preparation', args.start, args.end)
            elif args.action == 'repair-plan':
                from data.year_repair import build_year_plan
                result = build_year_plan(workspace.cache, workspace.cache / 'preparation', args.year, args.cutoff)
                result = {key: value for key, value in result.items() if key != 'intervals'}
            elif args.action == 'stage-year':
                from data.year_release import stage_year
                result = stage_year(workspace.cache, workspace.cache / 'preparation', args.year, args.cutoff)
            elif args.action == 'list':
                if args.limit < 1:
                    raise ValueError('Limit must be positive')
                result = [asdict(item) for item in catalog.datasets(args.category)[:args.limit]]
            else:
                if args.limit < 1:
                    raise ValueError('Limit must be positive')
                frame = catalog.kline(args.symbol, args.start, args.end)
                result = {'symbol': args.symbol, 'rows': len(frame),
                          'columns': list(frame.columns),
                          'preview': json.loads(frame.tail(args.limit).to_json(orient='records', date_format='iso'))}
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (LookupError, ValueError, RuntimeError, OSError) as exc:
        print(json.dumps({'status': 'error', 'message': str(exc)}, ensure_ascii=False))
        return 2
