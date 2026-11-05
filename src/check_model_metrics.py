"""Regression gate for the refresh: compare the scores the models just got
(data/processed/model_metrics.json, written by train_value_model.py and
train_outcome_model.py) against the last committed ones (git HEAD), and exit
non-zero if either model got materially worse - so the workflow stops before
committing a worse model and a site built on it.

Both test sets are fixed seasons (2024-25 + 2025-26), so a retrain on a few
more days of data should move the scores by thousandths; the tolerances are
several times that. A run that legitimately changes a model (new features, a
bigger holdout) and trips the gate: check the numbers, then commit the new
model_metrics.json from a local run - the next refresh compares against that.

Run:  py -3.11 src/check_model_metrics.py
      py -3.11 src/check_model_metrics.py --baseline path/to/old_metrics.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import model_metrics  # noqa: E402

VALUE_R2_DROP = 0.02      # R²(log) may fall at most this much
LOG_LOSS_RISE = 0.01      # 1X2 test log-loss may rise at most this much


def _committed() -> dict | None:
    rel = model_metrics.PATH.relative_to(ROOT).as_posix()
    try:
        out = subprocess.run(["git", "show", f"HEAD:{rel}"], cwd=ROOT, capture_output=True,
                             check=True, text=True, encoding="utf-8")
    except (OSError, subprocess.CalledProcessError):
        return None
    return json.loads(out.stdout)


def compare(old: dict, new: dict) -> list[str]:
    """Human-readable failures (empty = pass); prints every comparison."""
    fails = []
    checks = [("value R²(log)", ("value", "r2_log"), -VALUE_R2_DROP),
              ("outcome pure log-loss", ("outcome", "pure", "log_loss"), LOG_LOSS_RISE),
              ("outcome hybrid log-loss", ("outcome", "hybrid", "log_loss"), LOG_LOSS_RISE)]
    for label, path, tol in checks:
        a, b = old, new
        for k in path:
            a = a.get(k) if isinstance(a, dict) else None
            b = b.get(k) if isinstance(b, dict) else None
        if a is None or b is None:
            print(f"  {label:26} skipped (missing in {'baseline' if a is None else 'new run'})")
            continue
        worse = b < a + tol if tol < 0 else b > a + tol
        print(f"  {label:26} {a:.4f} -> {b:.4f}  ({b - a:+.4f}, limit {tol:+.3f})  {'FAIL' if worse else 'ok'}")
        if worse:
            fails.append(f"{label} {a:.4f} -> {b:.4f}")
    return fails


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--baseline", type=Path, help="compare against this file instead of git HEAD's")
    args = ap.parse_args()
    new = model_metrics.load()
    if not new:
        raise SystemExit(f"{model_metrics.PATH} missing - the training steps didn't write scores")
    old = json.loads(args.baseline.read_text(encoding="utf-8")) if args.baseline else _committed()
    if not old:
        print("no committed model_metrics.json to compare against - first run, gate passes")
        return
    print("model scores vs the last committed run:")
    fails = compare(old, new)
    if fails:
        raise SystemExit("models got worse past tolerance - not committing:\n  " + "\n  ".join(fails))
    print("gate passed")


if __name__ == "__main__":
    main()
