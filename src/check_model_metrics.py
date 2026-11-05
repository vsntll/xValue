"""Regression gate for the refresh: compare the scores the models just got
(data/processed/model_metrics.json, written by train_value_model.py and
train_outcome_model.py) against the last committed ones (git HEAD), and exit
non-zero if either model got materially worse - so the workflow stops before
committing a worse model and a site built on it.

Both test sets are fixed seasons (2024-25 + 2025-26), so a retrain on a few
more days of data should move the scores by thousandths; the tolerances are
several times that.

It also checks the same tolerances against accepted_metrics.json - scores
pinned by hand - so a model can't be walked down a little per run (each step
inside tolerance of the last, the total far outside it). A deliberate model
change (new features, a bigger holdout) that trips either check: verify the
numbers, run `--accept`, and commit model_metrics.json + accepted_metrics.json.

Run:  py -3.11 src/check_model_metrics.py
      py -3.11 src/check_model_metrics.py --baseline path/to/old_metrics.json
      py -3.11 src/check_model_metrics.py --accept      # pin the current scores
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
# hand-pinned scores of the model as last deliberately accepted - only
# `--accept` (run by you, committed with the model change) ever rewrites it
ACCEPTED = ROOT / "accepted_metrics.json"


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
    ap.add_argument("--accept", action="store_true",
                    help="pin the current model_metrics.json as accepted_metrics.json (a deliberate model change)")
    args = ap.parse_args()
    new = model_metrics.load()
    if not new:
        raise SystemExit(f"{model_metrics.PATH} missing - the training steps didn't write scores")
    if args.accept:
        ACCEPTED.write_text(json.dumps(new, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"pinned the current scores as {ACCEPTED.name} - commit it with the model change")
        return
    fails = []
    old = json.loads(args.baseline.read_text(encoding="utf-8")) if args.baseline else _committed()
    if old:
        print("model scores vs the last committed run:")
        fails += compare(old, new)
    else:
        print("no committed model_metrics.json to compare against - skipping the run-to-run check")
    # the run-to-run check alone can be walked down (0.009 worse every run
    # passes forever); the pinned baseline caps the total drift
    if ACCEPTED.exists():
        print(f"model scores vs the accepted baseline ({ACCEPTED.name}):")
        fails += [f + " (vs accepted)" for f in compare(json.loads(ACCEPTED.read_text(encoding="utf-8")), new)]
    else:
        print(f"no {ACCEPTED.name} - run with --accept to pin one")
    if fails:
        raise SystemExit("models got worse past tolerance - not committing:\n  " + "\n  ".join(fails))
    print("gate passed")


if __name__ == "__main__":
    main()
