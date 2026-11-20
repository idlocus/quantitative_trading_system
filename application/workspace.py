"""Explicit workspace paths without environment loading or directory creation."""
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Workspace:
    root: Path

    @classmethod
    def current(cls):
        return cls(Path(__file__).resolve().parents[1])

    @property
    def cache(self) -> Path:
        return self.root / 'data_cache'

    @property
    def archive(self) -> Path:
        return self.root / 'archive' / '2026-10-30-before-reset'
