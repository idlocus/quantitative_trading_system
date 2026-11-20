"""Retired entry point; the original source is preserved in archive/."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from application.cli import retired_entrypoint

if __name__ == '__main__':
    raise SystemExit(retired_entrypoint('scripts/paper_day.py'))
