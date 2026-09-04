"""Export a self-contained JSON payload for the player/odds website.

Combines:
  - current-season (2026-27) + last-season (2025-26) FBref stats per EPL player
  - the value model's predicted market value (2025-26, `value_model_predictions.csv`)
  - a simple full-season pace projection from current per-90 rates
  - next-fixture win/draw/loss odds per team, from a Dixon-Coles model
    (`src/dixon_coles.py`) fit on all competitions through today
  - per-player anytime goal / assist odds for that next fixture, from a
    share-of-team-expected-goals split (player npxG90 * minutes-share, normalised
    within the matchday squad) fed through a Poisson

Run: py -3.11 src/export_site_data.py   (or plain python3 - no nodriver needed)
Output: site/data.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from dixon_coles import fit, match_probs  # noqa: E402
from live.schema import normalize_team  # noqa: E402

PROC = ROOT / "data" / "processed"
OUT = ROOT / "site" / "data.json"
LEAGUE = "Premier League"
SRC_LEAGUE = "ENG1"
MIN_MIN_CURRENT = 45   # min minutes this season to trust current-season rates
MIN_MIN_PROJECT = 180  # min minutes to publish a pace projection / prop odds


def _num(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    return round(float(x), 3)


def load_players() -> pd.DataFrame:
    # source CSV is actually latin-1 (accented names mojibake under the utf-8 default)
    df = pd.read_csv(PROC / "fbref_player_season_stats.csv", low_memory=False, encoding="latin-1")
    df = df[df["src_league"] == SRC_LEAGUE].copy()
    keep = {
        "season": "season", "Player": "player", "Squad": "squad", "Pos": "pos", "Age": "age",
        "standard__Playing Time_MP": "mp", "standard__Playing Time_Starts": "starts",
        "standard__Playing Time_Min": "min", "playing_time__Playing Time_Min%": "min_pct",
        "standard__Performance_Gls": "gls", "standard__Performance_Ast": "ast",
        "standard__Performance_G-PK": "npg", "standard__Performance_CrdY": "cy",
        "standard__Performance_CrdR": "cr",
        "standard__Per 90 Minutes_Gls": "gls90", "standard__Per 90 Minutes_Ast": "ast90",
        "shooting__Standard_Sh": "shots", "shooting__Standard_SoT": "sot",
        "standard__xG_Expected": "xg", "standard__npxG_Expected": "npxg",
        "standard__xAG_Expected": "xag",
        "standard__xG_Per": "xg90", "standard__npxG_Per": "npxg90",
        "standard__xAG_Per": "xag90",
        "market_value_eur": "market_value_eur",
        # FBref's own xG/xAG-Per columns are gated (empty) for every EPL season in this
        # dataset - fall back to Understat season totals, converted to per-90 below.
        "understat__np_xg": "us_npxg", "understat__xa": "us_xa", "understat__xg": "us_xg",
    }
    df = df[list(keep)].rename(columns=keep)
    for c in ["mp", "starts", "min", "min_pct", "gls", "ast", "npg", "cy", "cr", "shots",
              "sot", "xg", "npxg", "xag", "gls90", "ast90", "xg90", "npxg90", "xag90",
              "age", "market_value_eur", "us_npxg", "us_xa", "us_xg"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    nineties = (df["min"] / 90.0).replace(0, np.nan)
    df["npxg90"] = df["npxg90"].where(df["npxg90"].notna(), df["us_npxg"] / nineties)
    df["xag90"] = df["xag90"].where(df["xag90"].notna(), df["us_xa"] / nineties)
    df["xg90"] = df["xg90"].where(df["xg90"].notna(), df["us_xg"] / nineties)
    df["npxg"] = df["npxg"].where(df["npxg"].notna(), df["us_npxg"])
    df["xag"] = df["xag"].where(df["xag"].notna(), df["us_xa"])
    df["xg"] = df["xg"].where(df["xg"].notna(), df["us_xg"])
    df["team_key"] = df["squad"].map(normalize_team)
    return df


def load_value_predictions() -> pd.DataFrame:
    v = pd.read_csv(PROC / "value_model_predictions.csv", encoding="latin-1")
    v = v[(v["src_league"] == SRC_LEAGUE) & (v["season"] == "2025-26")].copy()
    v["team_key"] = v["Squad"].map(normalize_team)
    return v[["Player", "team_key", "predicted_eur", "market_value_eur", "ratio"]].rename(
        columns={"Player": "player", "market_value_eur": "listed_value_eur"})


SHRINK_MIN = 400  # minutes of current-season data at which blended rate is ~50/50 cur/prior


def _blend_rate(cur_min, cur_rate, prior_rate):
    """Shrink a per-90 rate toward last season's until enough current-season
    minutes have accumulated. Falls back to whichever season has data."""
    cur_min = cur_min if pd.notna(cur_min) else 0.0
    has_cur = pd.notna(cur_rate)
    has_prior = pd.notna(prior_rate)
    if not has_cur and not has_prior:
        return 0.0
    if not has_prior:
        return float(cur_rate)
    if not has_cur:
        return float(prior_rate)
    w = cur_min / (cur_min + SHRINK_MIN)
    return float(w * cur_rate + (1 - w) * prior_rate)


def build_players_payload() -> tuple[list[dict], dict[str, str]]:
    df = load_players()
    vpred = load_value_predictions()

    cur = df[df["season"] == "2026-27"]
    prev = df[df["season"] == "2025-26"].set_index(["player", "team_key"])

    out = []
    team_display = {}
    blended_rows = []
    for _, r in cur.iterrows():
        if pd.isna(r["min"]) or r["min"] < MIN_MIN_CURRENT:
            continue
        team_display[r["team_key"]] = r["squad"]
        key = (r["player"], r["team_key"])
        p_row = prev.loc[key] if key in prev.index else None
        blended_rows.append({
            "player": r["player"], "team_key": r["team_key"], "pos": r["pos"],
            "min_pct": r["min_pct"],
            "npxg90": _blend_rate(r["min"], r["npxg90"], p_row["npxg90"] if p_row is not None else np.nan),
            "xag90": _blend_rate(r["min"], r["xag90"], p_row["xag90"] if p_row is not None else np.nan),
        })

        rec = {
            "player": r["player"], "squad": r["squad"], "team_key": r["team_key"],
            "pos": r["pos"], "age": _num(r["age"]),
            "current": {
                "mp": _num(r["mp"]), "starts": _num(r["starts"]), "min": _num(r["min"]),
                "gls": _num(r["gls"]), "ast": _num(r["ast"]), "npg": _num(r["npg"]),
                "shots": _num(r["shots"]), "sot": _num(r["sot"]),
                "xg": _num(r["xg"]), "npxg": _num(r["npxg"]), "xag": _num(r["xag"]),
                "gls90": _num(r["gls90"]), "ast90": _num(r["ast90"]),
                "npxg90": _num(r["npxg90"]), "xag90": _num(r["xag90"]),
            },
            "last_season": None,
            "projected_38": None,
            "value": None,
        }
        if p_row is not None:
            rec["last_season"] = {
                "min": _num(p_row["min"]), "gls": _num(p_row["gls"]), "ast": _num(p_row["ast"]),
                "npxg": _num(p_row["npxg"]), "xag": _num(p_row["xag"]),
            }
        if r["min"] >= MIN_MIN_PROJECT and pd.notna(r["gls90"]):
            mins_per_mp = r["min"] / r["mp"] if r["mp"] else 90.0
            proj_matches = 38.0
            proj_min = min(90.0, mins_per_mp) * proj_matches * (r["min_pct"] / 100.0 if pd.notna(r["min_pct"]) else 1.0)
            factor = proj_min / 90.0
            rec["projected_38"] = {
                "note": "pace projection: current-season per-90 rate x projected minutes over 38 games, not a trained model",
                "proj_min": _num(proj_min),
                "gls": _num(r["gls90"] * factor),
                "ast": _num(r["ast90"] * factor),
                "npxg": _num((r["npxg90"] or 0) * factor) if pd.notna(r["npxg90"]) else None,
            }
        vrow = vpred[(vpred["player"] == r["player"]) & (vpred["team_key"] == r["team_key"])]
        if len(vrow):
            vr = vrow.iloc[0]
            rec["value"] = {
                "listed_value_eur": _num(vr["listed_value_eur"]),
                "predicted_eur": _num(vr["predicted_eur"]),
                "ratio": _num(vr["ratio"]),
                "as_of_season": "2025-26",
            }
        out.append(rec)
    return out, team_display, pd.DataFrame(blended_rows)


def load_upcoming_fixtures(asof: pd.Timestamp) -> pd.DataFrame:
    live = pd.read_csv(PROC / "live_matches_2026-27.csv")
    live = live[live["league"] == LEAGUE].copy()
    live["Date"] = pd.to_datetime(live["Date"], errors="coerce")
    scheduled = {"SCHEDULED", "TIMED"}
    live = live[live["status"].isin(scheduled) & live["Date"].notna() & (live["Date"] >= asof)]
    live["h"] = live["HomeTeam"].map(normalize_team)
    live["a"] = live["AwayTeam"].map(normalize_team)
    live = live.sort_values("Date")
    return live


def next_fixture_per_team(live: pd.DataFrame) -> pd.DataFrame:
    """First (date-sorted) row touching each team. A row is kept if either
    side hasn't had its next fixture found yet - both sides of that row are
    then marked found, since for two teams meeting each other this one row
    *is* both teams' next match."""
    rows = []
    seen = set()
    for _, r in live.iterrows():
        if r["h"] not in seen or r["a"] not in seen:
            rows.append(r)
        seen.add(r["h"])
        seen.add(r["a"])
    return pd.DataFrame(rows).drop_duplicates(subset=["h", "a", "Date"])


def prep_dc_matches() -> pd.DataFrame:
    m = pd.read_csv(PROC / "matches_all.csv")
    m["Date"] = pd.to_datetime(m["Date"], errors="coerce")
    m = m.dropna(subset=["Date", "FTHG", "FTAG"])
    m["h"] = m["HomeTeam"].map(normalize_team)
    m["a"] = m["AwayTeam"].map(normalize_team)
    warm = ROOT / "data" / "raw" / "football_data" / "_elo_warmup.csv"
    if warm.exists():
        w = pd.read_csv(warm)
        w["Date"] = pd.to_datetime(w["Date"], dayfirst=True, errors="coerce")
        w["h"] = w["HomeTeam"].map(normalize_team)
        w["a"] = w["AwayTeam"].map(normalize_team)
        m = pd.concat([w.dropna(subset=["Date"]), m], ignore_index=True)
    return m.sort_values("Date")


def expected_goals(model, home: str, away: str) -> tuple[float, float]:
    if model is None or home not in model["idx"] or away not in model["idx"]:
        return (1.3, 1.1)
    h, a = model["idx"][home], model["idx"][away]
    lh = float(np.exp(model["att"][h] + model["dfn"][a] + model["home"]))
    la = float(np.exp(model["att"][a] + model["dfn"][h]))
    return lh, la


def player_props_for_fixture(blended_df: pd.DataFrame, team_key: str, lam_team: float) -> list[dict]:
    squad = blended_df[blended_df["team_key"] == team_key].copy()
    if squad.empty or lam_team <= 0:
        return []
    med = squad["min_pct"].median()
    squad["minute_frac"] = (squad["min_pct"].fillna(med if pd.notna(med) else 50.0) / 100.0).clip(0.05, 1.0)
    squad["atk_weight"] = squad["npxg90"].fillna(0).clip(lower=0) * squad["minute_frac"]
    squad["asg_weight"] = squad["xag90"].fillna(0).clip(lower=0) * squad["minute_frac"]
    atk_total = squad["atk_weight"].sum()
    asg_total = squad["asg_weight"].sum()
    props = []
    for _, r in squad.iterrows():
        g_share = r["atk_weight"] / atk_total if atk_total > 0 else 0
        a_share = r["asg_weight"] / asg_total if asg_total > 0 else 0
        lam_g = lam_team * g_share
        lam_a = lam_team * a_share
        props.append({
            "player": r["player"], "pos": r["pos"],
            "p_anytime_goal": _num(1 - np.exp(-lam_g)),
            "p_anytime_assist": _num(1 - np.exp(-lam_a)),
            "exp_goals": _num(lam_g), "exp_assists": _num(lam_a),
        })
    props.sort(key=lambda p: p["p_anytime_goal"] or 0, reverse=True)
    return props


def main() -> None:
    players, team_display, blended_df = build_players_payload()

    dc_matches = prep_dc_matches()
    asof = dc_matches["Date"].max() + pd.Timedelta(days=1)
    model = fit(dc_matches, asof)

    live = load_upcoming_fixtures(asof)
    nxt = next_fixture_per_team(live)

    fixtures = []
    for _, r in nxt.iterrows():
        lh, la = expected_goals(model, r["h"], r["a"])
        pA, pD, pH = match_probs(model, r["h"], r["a"])
        fx = {
            "date": r["Date"].strftime("%Y-%m-%d %H:%M"),
            "home": r["HomeTeam"], "away": r["AwayTeam"],
            "home_key": r["h"], "away_key": r["a"],
            "p_home_win": _num(pH), "p_draw": _num(pD), "p_away_win": _num(pA),
            "exp_goals_home": _num(lh), "exp_goals_away": _num(la),
            "home_props": player_props_for_fixture(blended_df, r["h"], lh),
            "away_props": player_props_for_fixture(blended_df, r["a"], la),
        }
        fixtures.append(fx)

    payload = {
        "generated_asof": asof.strftime("%Y-%m-%d"),
        "league": LEAGUE,
        "players": players,
        "fixtures": fixtures,
        "notes": {
            "current_stats": "2026-27 FBref season-to-date stats (min 45 minutes played).",
            "last_season": "2025-26 full-season stats for the same player, where available.",
            "projected_38": "Simple pace projection: current per-90 rate x projected minutes over a 38-game season. Not a trained model.",
            "value": "Predicted market value from the trained value-regression model (2025-26 season, HGB, R2(log) 0.82) vs listed market value.",
            "match_odds": "Win/draw/loss odds from a Dixon-Coles attack/defence model fit on all competitions through the date above.",
            "player_props": "Anytime goal/assist odds: team's Dixon-Coles expected goals split across the matchday squad by each player's (non-penalty xG90 or xA90, shrunk toward last season's rate early in the current season) x season minutes-share, then Poisson P(>=1).",
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=None, separators=(",", ":")), encoding="utf-8")
    print(f"wrote {OUT}  ({len(players)} players, {len(fixtures)} fixtures)  size={OUT.stat().st_size/1024:.0f} KB")


if __name__ == "__main__":
    main()
