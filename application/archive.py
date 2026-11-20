"""Read-only verification of the pre-reset source snapshot and retained data."""
import hashlib
import json
from pathlib import Path

from .workspace import Workspace


def verify_snapshot(workspace: Workspace) -> dict:
    archive = workspace.archive.resolve()
    source = archive / 'source'
    manifest = json.loads((archive / 'manifest.json').read_text(encoding='utf-8'))
    failures = []
    for entry in manifest['source']:
        path = (source / entry['path']).resolve()
        if not path.is_relative_to(source) or not path.is_file():
            failures.append(f'Missing or invalid snapshot path: {entry["path"]}')
        elif hashlib.sha256(path.read_bytes()).hexdigest() != entry['sha256']:
            failures.append(f'Snapshot content mismatch: {entry["path"]}')
    protected_changes = []
    root = workspace.root.resolve()
    for entry in manifest['protected']:
        path = (root / entry['path']).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            protected_changes.append(f'Missing retained file: {entry["path"]}')
            continue
        if 'min_trials' in entry:
            # 登记册是设计上会追加的运行日志（DSR 按累计尝试数 deflate）：
            # 保护语义=存在且计数单调不减（防丢失/回滚），而非字节不变。
            # 2026-11-06 第一次研究写入（第128条）暴露此矛盾后修正。
            try:
                count = json.loads(path.read_text(encoding='utf-8'))['count']
            except Exception:
                protected_changes.append(f'Registry unreadable: {entry["path"]}')
                continue
            if count < entry['min_trials']:
                protected_changes.append(
                    f'Registry shrank: count {count} < baseline {entry["min_trials"]}')
            continue
        stat = path.stat()
        if stat.st_size != entry['size'] or stat.st_mtime_ns != entry['mtime_ns']:
            protected_changes.append(f'Retained file changed: {entry["path"]}')
    return {'source_verified': not failures, 'source_files': len(manifest['source']),
            'source_failures': failures, 'retained_files': len(manifest['protected']),
            'retained_files_unchanged': not protected_changes,
            'retained_file_changes': protected_changes,
            'retained_verification': 'file presence, byte length and modification time; contents not opened'}
