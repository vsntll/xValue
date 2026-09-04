"""xValue dashboard - the thing you'd actually check every week.

Reads the CSVs/JSON the pipeline already writes to data/processed/ - it never
recomputes anything, so it's always in sync with whatever the last pipeline
run produced, and it's the fastest way to notice the pipeline broke (a page
that's empty or a season stuck at last week's date is a bug report on its own).

Three tabs:
  - Predictions vs results: every 2026-27 match the outcome model has scored,
    next to the real result as it lands, plus a running scoreboard.
  - Value leaderboard: biggest predicted-vs-listed gaps from the value model,
    both directions, and a greedy "most undervalued XI" / "most overpriced XI".
  - Source health: the latest src/health_check.py report.

Run:  py -3.11 -m streamlit run src/dashboard.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"

st.set_page_config(page_title="xValue dashboard", page_icon="⚽", layout="wide")


@st.cache_data(ttl=300)
def _read_csv(name: str) -> pd.DataFrame | None:
    p = PROC / name
    if not p.exists():
        return None
    return pd.read_csv(p, low_memory=False)


@st.cache_data(ttl=300)
def _read_json(name: str) -> dict | None:
    p = PROC / name
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def _missing(name: str, script: str) -> None:
    st.warning(f"**{name}** not found yet. Run `py -3.11 {script}` (see `run_pipeline.md`).")


LABELS3 = np.array(["A", "D", "H"])


# ---------------------------------------------------------------------------
# Tab 1 - predictions vs results
# ---------------------------------------------------------------------------

def tab_predictions() -> None:
    df = _read_csv("outcome_model_predictions_all.csv")
    if df is None:
        _missing("outcome_model_predictions_all.csv", "src/train_outcome_model.py")
        return

    live = df[df["split"] == "live"].copy()
    if live.empty:
        st.info("No 2026-27 matches scored yet - they show up here once played and the "
                "pipeline's rebuilt (weekly, or after a manual refresh).")
        return
    live["Date"] = pd.to_datetime(live["Date"])
    live = live.sort_values("Date", ascending=False)

    pred_idx = live[["p_away", "p_draw", "p_home"]].to_numpy().argmax(1)
    live["predicted"] = LABELS3[pred_idx]
    live["hit"] = live["predicted"] == live["FTR"]
    ou_actual = np.where(live["FTHG"] + live["FTAG"] > 2.5, "over", "under")
    live["ou_hit"] = np.where(live["p_over25"] >= 0.5, "over", "under") == ou_actual
    btts_actual = (live["FTHG"] > 0) & (live["FTAG"] > 0)
    live["btts_hit"] = (live["p_btts_yes"] >= 0.5) == btts_actual

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Matches scored (2026-27)", len(live))
    c2.metric("1X2 accuracy so far", f"{live['hit'].mean():.0%}")
    c3.metric("O/U 2.5 accuracy", f"{live['ou_hit'].mean():.0%}")
    c4.metric("BTTS accuracy", f"{live['btts_hit'].mean():.0%}")

    leagues = ["All"] + sorted(live["comp"].dropna().unique().tolist())
    league = st.selectbox("League", leagues)
    view = live if league == "All" else live[live["comp"] == league]

    show = view[["Date", "comp", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR",
                "p_home", "p_draw", "p_away", "predicted", "hit",
                "p_over25", "p_btts_yes"]].copy()
    show.columns = ["Date", "League", "Home", "Away", "HG", "AG", "Result",
                    "P(home)", "P(draw)", "P(away)", "Model pick", "Hit",
                    "P(over 2.5)", "P(BTTS)"]
    for c in ["P(home)", "P(draw)", "P(away)", "P(over 2.5)", "P(BTTS)"]:
        show[c] = (show[c] * 100).round(0).astype(int).astype(str) + "%"
    show["Hit"] = show["Hit"].map({True: "✓", False: "✗"})
    show["Date"] = show["Date"].dt.strftime("%Y-%m-%d")
    st.dataframe(show, width='stretch', hide_index=True, height=520)

    # running accuracy over the season, to eyeball whether it's improving/decaying
    roll = live.sort_values("Date").assign(cum_acc=lambda d: d["hit"].expanding().mean())
    fig = px.line(roll, x="Date", y="cum_acc", title="Running 1X2 accuracy this season",
                 labels={"cum_acc": "cumulative accuracy"})
    fig.update_yaxes(range=[0, 1], tickformat=".0%")
    st.plotly_chart(fig, width='stretch')

    bench = live_split_benchmark(df)
    if bench is not None:
        st.caption(bench)


def live_split_benchmark(df: pd.DataFrame) -> str | None:
    test = df[df["split"] == "test"]
    if test.empty:
        return None
    from sklearn.metrics import log_loss
    p = test[["p_away", "p_draw", "p_home"]].to_numpy()
    p = p / p.sum(axis=1, keepdims=True)  # rounding to 4dp on disk can leave sums a hair off 1.0
    pred = LABELS3[p.argmax(1)]
    acc = (pred == test["FTR"]).mean()
    ll = log_loss(test["FTR"], p, labels=["A", "D", "H"])
    return (f"For reference, this same model on its proper 2024-26 holdout (not this season, "
            f"so no peeking): accuracy {acc:.1%}, log-loss {ll:.3f}.")


# ---------------------------------------------------------------------------
# Tab 2 - value leaderboard
# ---------------------------------------------------------------------------

FORMATION = [("GK", 1), ("DF", 4), ("MF", 3), ("FW", 3)]
MIN_MIN_XI = 300  # need a real sample before trusting a ratio for the XI picker


def _with_minutes(cur: pd.DataFrame, season: str) -> tuple[pd.DataFrame, str]:
    """Join in season minutes (not in value_model_predictions.csv) and filter to
    a real sample, so the XI isn't a fringe player with a nominal listed value."""
    stats = _read_csv("fbref_player_season_stats.csv")
    if stats is None:
        return cur, " - unfiltered by minutes, fbref_player_season_stats.csv not found"
    m = stats[stats["season"] == season][["src_league", "Player", "Squad", "standard__Playing Time_Min"]]
    m = m.rename(columns={"standard__Playing Time_Min": "min"}).drop_duplicates(
        subset=["src_league", "Player", "Squad"])
    j = cur.merge(m, on=["src_league", "Player", "Squad"], how="left")
    filtered = j[j["min"].fillna(0) >= MIN_MIN_XI]
    if filtered.empty:  # e.g. very early in a season - fall back to unfiltered
        return cur, " - no one has 300+ minutes yet this season, unfiltered"
    return filtered, f", {MIN_MIN_XI}+ minutes"


def _pick_xi(pool: pd.DataFrame, ascending: bool) -> pd.DataFrame:
    rows = []
    for pos, n in FORMATION:
        cand = pool[pool["pos"] == pos].sort_values("ratio", ascending=ascending)
        rows.append(cand.head(n))
    return pd.concat(rows, ignore_index=True) if rows else pool.head(0)


def tab_value() -> None:
    v = _read_csv("value_model_predictions.csv")
    if v is None:
        _missing("value_model_predictions.csv", "src/train_value_model.py")
        return

    seasons = sorted(v["season"].unique())
    season = st.selectbox("Season", seasons, index=len(seasons) - 1)
    cur = v[(v["season"] == season) & (v["value_imputed"] == 0)].copy()
    if cur.empty:
        st.info(f"No real (non-imputed) values for {season}.")
        return

    c1, c2 = st.columns(2)
    c1.metric("Players valued", len(cur))
    c2.metric("Median predicted / listed", f"{cur['ratio'].median():.2f}x")

    st.subheader("Biggest gaps: model vs listed value")
    lcol, rcol = st.columns(2)
    with lcol:
        st.markdown("**Model says undervalued** (predicted >> listed)")
        bargains = cur.sort_values("ratio", ascending=False).head(15)
        st.dataframe(_fmt_value_table(bargains), width='stretch', hide_index=True)
    with rcol:
        st.markdown("**Model says overpriced** (predicted << listed)")
        rich = cur.sort_values("ratio", ascending=True).head(15)
        st.dataframe(_fmt_value_table(rich), width='stretch', hide_index=True)

    pool, min_note = _with_minutes(cur, season)
    st.subheader(f"Most under/overvalued XI (4-3-3{min_note})")
    xi_u, xi_o = _pick_xi(pool, ascending=False), _pick_xi(pool, ascending=True)
    xcol1, xcol2 = st.columns(2)
    with xcol1:
        st.markdown("**Undervalued XI**")
        st.dataframe(_fmt_value_table(xi_u), width='stretch', hide_index=True)
    with xcol2:
        st.markdown("**Overpriced XI**")
        st.dataframe(_fmt_value_table(xi_o), width='stretch', hide_index=True)
    st.caption("XI is a greedy pick: highest/lowest predicted-vs-listed ratio per position, "
              "1 GK + 4 DF + 3 MF + 3 FW - a fun readout of the model, not a scouting report.")

    fig = px.histogram(cur, x="ratio", nbins=40, title="Predicted / listed value ratio - whole squad")
    fig.add_vline(x=1.0, line_dash="dash")
    st.plotly_chart(fig, width='stretch')


def _fmt_value_table(d: pd.DataFrame) -> pd.DataFrame:
    t = d[["Player", "Squad", "pos", "market_value_eur", "predicted_eur", "ratio"]].copy()
    t.columns = ["Player", "Squad", "Pos", "Listed", "Predicted", "Ratio"]
    t["Listed"] = (t["Listed"] / 1e6).round(1).astype(str) + "M"
    t["Predicted"] = (t["Predicted"] / 1e6).round(1).astype(str) + "M"
    t["Ratio"] = t["Ratio"].round(2)
    return t


# ---------------------------------------------------------------------------
# Tab 3 - source health
# ---------------------------------------------------------------------------

def tab_health() -> None:
    h = _read_json("health_check.json")
    if h is None:
        _missing("health_check.json", "src/health_check.py")
        return
    st.caption(f"Last checked: {h['checked_at']}  (`.github/workflows/health-check.yml` runs this daily)")
    rows = []
    for name, r in h["results"].items():
        icon = {"ok": "✅", "blocked (known)": "\U0001f7e1", "skipped": "⚪"}.get(r["status"], "\U0001f534")
        rows.append({"": icon, "Source": name, "Status": r["status"], "Detail": r["detail"],
                    "Seconds": r.get("seconds", "")})
    st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True)
    bad = [r for r in h["results"].values() if r["status"] in ("DEGRADED", "ERROR")]
    if bad:
        st.error(f"{len(bad)} source(s) need attention right now.")
    else:
        st.success("Every source is healthy, or blocked exactly the way it's always been.")


def main() -> None:
    st.title("⚽ xValue")
    st.caption("Live model predictions vs results, the value model's biggest calls, "
              "and whether the data pipeline is actually still working.")
    t1, t2, t3 = st.tabs(["Predictions vs results", "Value leaderboard", "Source health"])
    with t1:
        tab_predictions()
    with t2:
        tab_value()
    with t3:
        tab_health()


if __name__ == "__main__":
    main()
