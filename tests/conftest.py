"""Shared setup: put src/ on the import path, and give tests the committed
version of a processed CSV (git HEAD) to compare a freshly rebuilt one against."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"
sys.path.insert(0, str(ROOT / "src"))


def committed_lines(rel: str) -> int | None:
    """Line count of the previous version of `rel` - the no-collapse baseline:
    git HEAD's copy for committed files, else the last published build's copy
    (release asset, extracted into $PREV_BUILD_DIR by the refresh) for the
    gitignored build outputs. None if neither exists."""
    try:
        out = subprocess.run(["git", "show", f"HEAD:{rel}"], cwd=ROOT, capture_output=True, check=True)
        return out.stdout.count(b"\n")
    except (OSError, subprocess.CalledProcessError):
        pass
    prev = os.environ.get("PREV_BUILD_DIR")
    if prev and (Path(prev) / rel).exists():
        return (Path(prev) / rel).read_bytes().count(b"\n")
    return None


@pytest.fixture
def proc() -> Path:
    return PROC
