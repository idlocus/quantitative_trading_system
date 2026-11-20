"""Offline summary of independently prepared datasets; no credential or SDK loading."""
import json
from pathlib import Path


def prepared_summary(output: Path) -> dict:
    def read(relative):
        path = output / relative
        return json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}

    a = read('oracle/normalized_a/manifest.json')
    hk = read('oracle/hk_current_candidates/manifest.json')
    supporting = read('oracle/supporting/manifest.json')
    connect = read('connect_daily/manifest.json')
    universe = read('hk_research_universe/manifest.json')
    static_accepted = (universe.get('mode') == 'current_connect_snapshot' and
                       universe.get('scope') == 'research_only' and universe.get('assumption_accepted') is True)
    remaining = ['HK corporate actions and settlement FX', 'Historical execution rules and fees',
                 'Lifecycle and calendar coverage acceptance']
    if not static_accepted:
        remaining.insert(0, 'Historical southbound buy/sell-only qualification')
    return {'research_ready': False,
            'a_share': {'months': len(a.get('files', {})), 'years': a.get('years', {})},
            'hk_current_candidates': {'months': len(hk.get('files', {})),
                'rows': sum(item['rows'] for item in hk.get('files', {}).values()),
                'candidate_count': hk.get('candidate_count'), 'historical_eligibility_verified': False},
            'hk_research_universe': {'mode': universe.get('mode', 'unconfigured'),
                'symbols': universe.get('symbols'), 'assumption_accepted': static_accepted,
                'historical_eligibility_verified': False, 'historical_eligibility_blocking': not static_accepted,
                'limitations': universe.get('policy', {}).get('limitations', [])},
            'supporting_rows': {name: item.get('rows') for name, item in supporting.get('files', {}).items()},
            'connect_dated_snapshots': len(connect.get('files', {})),
            'connect_provider_failure': (connect.get('last_attempt_status') == 'blocked' or
                ('last_attempt_status' not in connect and (output / 'connect_daily/failure.json').exists())),
            'remaining': remaining}
