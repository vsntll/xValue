"""Write the model scores in data/processed/model_metrics.json into README.md
and docs/models.md, so the docs always state what the latest trained models
actually scored (they used to be hand-edited and drifted apart).

Each generated passage sits between `<!-- metrics:NAME -->` and
`<!-- /metrics:NAME -->` markers; everything outside the markers is left alone.
Run after train_value_model.py + train_outcome_model.py (the refresh does).

Run:  py -3.11 src/update_doc_metrics.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import model_metrics  # noqa: E402


def _pct(x: float) -> str:
    return f"{x:.0%}"


def _eur_m(x: float) -> str:
    return f"€{x / 1e6:.1f}M"


def _gap(m: dict, name: str) -> str:
    """Share of the base-rate -> bookmaker log-loss gap a model closes."""
    o = m["outcome"]
    base, book = o["base_rate"]["log_loss"], o["bookmaker"]["log_loss"]
    return _pct((base - o[name]["log_loss"]) / (base - book))


def blocks(m: dict) -> dict[str, str]:
    v, o = m["value"], m["outcome"]
    seg = v["segments"]
    test = f"{v['test_seasons'][0][:4]}-{v['test_seasons'][-1][-2:]}"
    pure, hyb, book = o["pure"], o.get("hybrid"), o["bookmaker"]
    hyb_txt = f" / **{hyb['log_loss']:.3f}** (with market odds)" if hyb else ""
    rows = [("base rate", "base_rate"), ("logreg", "logreg"), ("hgb", "hgb"),
            ("poisson-Skellam", "poisson"), ("poisson + xG", "poisson_xg"),
            ("blend {poisson, logreg, hgb} (**pure**)", "pure"),
            ("**hybrid (+ market opening odds)**", "hybrid"), ("Bet365 closing", "bookmaker")]
    table = "\n".join(f"  | {label} | {o[k]['acc']:.3f} | {o[k]['log_loss']:.3f} |"
                      for label, k in rows if k in o)
    return {
        "readme-headline": (
            "- **Value regression** — predicts a player's market value from his season + value\n"
            "  history. Stacked trees (direct + change-from-last-value blend, capped at\n"
            f"  Transfermarkt's €220M ceiling), R²(log) **{v['r2_log']:.2f}**, MAE **{_eur_m(v['mae_eur'])}**, "
            f"within-2×\n  **{_pct(v['within_2x'])}** on a {test} big-5 holdout — "
            f"**{seg['with_prev']['r2_log']:.2f}** with a prior-season value, "
            f"**{_pct(seg['elite']['medape'])}**\n"
            "  median error in the €100M+ band. Outfield + keepers; scores every season\n"
            "  including the current one.\n"
            "- **Outcome classifier** — pre-match home-win / draw / away-win. Poisson-Skellam\n"
            f"  on Elo + form + xG, log-loss **{pure['log_loss']:.3f}** (pure){hyb_txt},\n"
            f"  vs Bet365 closing {book['log_loss']:.3f}."),
        "models-summary": (
            f"**Current results** (temporal holdout {' + '.join(v['test_seasons'])}, all **big-5** "
            "leagues; generated from `data/processed/model_metrics.json`):\n\n"
            "| | value model | outcome (pure) | outcome (hybrid) |\n"
            "| --- | --- | --- | --- |\n"
            f"| metric | R²(log) **{v['r2_log']:.2f}**, MAE **{_eur_m(v['mae_eur'])}**, "
            f"within-2x **{_pct(v['within_2x'])}**, medAPE {_pct(v['medape'])} | "
            f"log-loss **{pure['log_loss']:.3f}**, acc **{pure['acc']:.3f}** | "
            + (f"log-loss **{hyb['log_loss']:.3f}**, acc **{hyb['acc']:.3f}** |" if hyb else "— |") + "\n"
            "| v1 was | 0.70 / €9.1M / 68% | 1.014 / 0.511 | — |\n"
            f"| reference | — | Bet365 closing {book['log_loss']:.3f} / {book['acc']:.3f} | "
            "(uses market opening odds as a feature) |"),
        "models-value-split": (
            "The value model splits cleanly by whether a prior-season value exists:\n"
            f"**R²(log) {seg['with_prev']['r2_log']:.2f}** for the ~{_pct(seg['with_prev']['share'])} "
            f"that have one, **{seg['cold_start']['r2_log']:.2f}** for cold-start arrivals with\n"
            "none (promoted-club squads, Eredivisie/Primeira/Championship signings, academy\n"
            "graduates, and Serie A / Ligue 1 players whose only prior value predates the\n"
            f"2015 mirror). Keepers score **{seg['goalkeepers']['r2_log']:.2f}**."),
        "models-elite": (
            "`medAPE` for the **€100M+ band is "
            f"{_pct(seg['elite']['medape'])}** (was ~30-40% before the cap - the trees squashed\n"
            f"the tail toward the mean), though the ~{seg['elite']['n']}-row band sits slightly "
            f"*under* the\nlisted values (median pred/listed {seg['elite']['pred_over_listed']:.2f}). "
            f"Mid-range (€3-40M) predictions carry a\n**median pred/listed of "
            f"{seg['mid']['pred_over_listed']:.2f}** but individual scatter stays "
            f"~{_pct(seg['mid']['medape'])} —"),
        "models-value-result": (
            f"- **Result**: R2(log) **{v['r2_log']:.2f}**, MAE **{_eur_m(v['mae_eur'])}**, "
            f"medAPE **{_pct(v['medape'])}**, within-2x\n"
            f"  **{_pct(v['within_2x'])}** ({seg['with_prev']['r2_log']:.2f} with a prior value, "
            f"{seg['cold_start']['r2_log']:.2f} cold-start on its own stack -\n"
            f"  medAPE {_pct(seg['cold_start']['medape'])} - "
            f"{seg['goalkeepers']['r2_log']:.2f} keepers). €100M+ band medAPE "
            f"**{_pct(seg['elite']['medape'])}**, median pred/listed "
            f"**{seg['elite']['pred_over_listed']:.2f}**."),
        "models-outcome-table": (
            f"- **Result** (log-loss / accuracy, test {' + '.join(o['test_seasons'])}, "
            f"n={o['n_test']:,}):\n\n"
            "  | model | acc | log-loss |\n  | --- | --- | --- |\n" + table + "\n\n"
            f"  The pure model closes ~{_gap(m, 'pure')} of the base-rate → bookmaker gap on "
            "log-loss" + (f"; the\n  hybrid, which adds the market's *opening* line as a feature, "
                          f"closes ~{_gap(m, 'hybrid')} of\n  it" if hyb else "") +
            " - the last sliver is information the market has and we don't\n"
            "  (confirmed lineups, injuries, sharp money)."),
    }


DOCS = {"readme-headline": "README.md", "models-summary": "docs/models.md",
        "models-value-split": "docs/models.md", "models-elite": "docs/models.md",
        "models-value-result": "docs/models.md", "models-outcome-table": "docs/models.md"}


def main() -> None:
    m = model_metrics.load()
    if not {"value", "outcome"} <= m.keys():
        raise SystemExit(f"{model_metrics.PATH} lacks value/outcome scores - train both models first")
    new = blocks(m)
    for doc in sorted(set(DOCS.values())):
        p = ROOT / doc
        s = p.read_text(encoding="utf-8")
        for name in (n for n, d in DOCS.items() if d == doc):
            pat = re.compile(rf"(<!-- metrics:{name} -->\n).*?(\n<!-- /metrics:{name} -->)", re.S)
            if not pat.search(s):
                raise SystemExit(f"{doc}: missing <!-- metrics:{name} --> markers")
            s = pat.sub(lambda mo: mo.group(1) + new[name] + mo.group(2), s)
        p.write_text(s, encoding="utf-8")
        print(f"updated {doc}")


if __name__ == "__main__":
    main()
