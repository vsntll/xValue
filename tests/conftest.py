"""Shared setup: put src/ on the import path, and give tests the committed
version of a processed CSV (git HEAD) to compare a freshly rebuilt one against."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"
sys.path.insert(0, str(ROOT / "src"))


def committed_lines(rel: str) -> int | None:
    """Line count of `rel` as committed at git HEAD, or None if it isn't tracked
    there (a new file, or no git) - the no-collapse baseline."""
    try:
        out = subprocess.run(["git", "show", f"HEAD:{rel}"], cwd=ROOT, capture_output=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    return out.stdout.count(b"\n")


@pytest.fixture
def proc() -> Path:
    return PROC
