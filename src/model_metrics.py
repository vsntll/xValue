"""data/processed/model_metrics.json - the test-set scores of the latest trained
models, written by train_value_model.py and train_outcome_model.py.

It is the single source for every headline number in README.md and
docs/models.md (src/update_doc_metrics.py writes them in), and the baseline the
refresh's regression gate compares a new training run against
(src/check_model_metrics.py) - so the docs can't drift from the models and a
worse model can't be committed silently.
"""

from __future__ import annotations

import json
from pathlib import Path

PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "model_metrics.json"


def load(path: Path = PATH) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def update(section: str, values: dict, path: Path = PATH) -> None:
    """Merge `values` into `section` (keys not in `values` keep their value -
    e.g. a non-hybrid outcome run leaves the hybrid score alone)."""
    data = load(path)
    data.setdefault(section, {}).update(values)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def r(x: float, nd: int = 4) -> float:
    return round(float(x), nd)