"""Export a self-contained JSON payload for the player/odds website.

Covers the Premier League, La Liga and Bundesliga. Combines:
  - current-season (2026-27) + last-season (2025-26) FBref stats per player
  - the value model's predicted market value (`value_model_predictions.csv`,
    current season preferred, falls back to last season)
  - a simple full-season pace projection from current per-90 rates
  - next-fixture win/draw/loss odds per team, from a Dixon-Coles model
    (`src/dixon_coles.py`) fit on all competitions through today
  - per-player anytime goal / assist odds for that next fixture, from a
    share-of-team-expected-goals split (player npxG90 * minutes-share, normalised
    within the matchday squad) fed through a Poisson
  - per-team pages: roster (from `players`), last 8 results (any competition),
    full league tables per season (computed from results), and cup finals -
    each inferred as the last-dated match of that season/competition
  - a projected final table (current points + expected points from each team's
    remaining fixtures, via the same Dixon-Coles model)
  - a value-model leaderboard (biggest predicted-vs-listed gaps, both ways)
  - one fully worked example (real fixture + real player) for the Methodology tab

Run: py -3.11 src/export_site_data.py   (or plain python3 - no nodriver needed)
Output: site/data.json, then spliced into site/template.html to produce the
committed, self-contained site/index.html - both written by this one run.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import ftfy
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from dixon_coles import fit, match_probs  # noqa: E402
from live.schema import normalize_team  # noqa: E402

PROC = ROOT / "data" / "processed"
OUT = ROOT / "site" / "data.json"
TEMPLATE = ROOT / "site" / "template.html"
INDEX = ROOT / "site" / "index.html"
PLACEHOLDER = "__SITE_DATA_JSON__"
LEAGUES = {"ENG1": "Premier League", "ESP1": "La Liga", "GER1": "Bundesliga"}
MIN_MIN_CURRENT = 45   # min minutes this season to trust current-season rates
MIN_MIN_PROJECT = 180  # min minutes to publish a pace projection / prop odds


def _num(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    return round(float(x), 3)


def _fix_names(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Files are read as utf-8 (the correct encoding for this pipeline as of the
    parse_fbref_player_stats.py rewrite). This is a defense-in-depth pass, not the
    primary fix: ftfy.fix_text() is a no-op on already-correct text (verified - 0
    rows changed across every processed CSV) but repairs the rare still-mangled
    row (e.g. "HÃ¥vard Nordtveit" -> "Håvard Nordtveit") if one slips through from
    an older cached file or a future regression."""
    for c in cols:
        if c in df.columns:
            df[c] = df[c].map(lambda s: ftfy.fix_text(s) if isinstance(s, str) else s)
    return df


def load_players(src_league: str) -> pd.DataFrame:
    # utf-8 (the correct encoding here - see the module docstring); _fix_names()
    # is the defense-in-depth pass for any row that slipped through mis-encoded
    df = pd.read_csv(PROC / "fbref_player_season_stats.csv", low_memory=False, encoding="utf-8")
    df = df[df["src_league"] == src_league].copy()
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
    df = _fix_names(df, ["player", "squad"])
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


def load_value_predictions(src_league: str) -> pd.DataFrame:
    v = pd.read_csv(PROC / "value_model_predictions.csv", encoding="utf-8")
    v = v[(v["src_league"] == src_league) & v["season"].isin(["2025-26", "2026-27"])].copy()
    v = _fix_names(v, ["Player", "Squad"])  # must match load_players()'s names for the join in build_players_payload
    v["team_key"] = v["Squad"].map(normalize_team)
    # prefer the current season's prediction over last season's, per player+team
    v = v.sort_values("season").drop_duplicates(subset=["Player", "team_key"], keep="last")
    return v[["Player", "team_key", "season", "predicted_eur", "market_value_eur", "ratio"]].rename(
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


def build_players_payload(src_league: str, league_name: str) -> tuple[list[dict], dict[str, str], pd.DataFrame]:
    df = load_players(src_league)
    vpred = load_value_predictions(src_league)

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

        # current-season Age is unpopulated upstream for every 2026-27 row - fall
        # back to last season's age + 1 rather than showing a blank
        age = r["age"]
        if pd.isna(age) and p_row is not None and pd.notna(p_row.get("age")):
            age = p_row["age"] + 1
        rec = {
            "player": r["player"], "squad": r["squad"], "team_key": r["team_key"],
            "league": league_name,
            "pos": r["pos"], "age": _num(age),
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
                "as_of_season": vr["season"],
            }
        out.append(rec)
    return out, team_display, pd.DataFrame(blended_rows)


def load_upcoming_fixtures(asof: pd.Timestamp, league_name: str) -> pd.DataFrame:
    live = pd.read_csv(PROC / "live_matches_2026-27.csv", encoding="utf-8")
    live = live[live["league"] == league_name].copy()
    live = _fix_names(live, ["HomeTeam", "AwayTeam"])
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
    m = pd.read_csv(PROC / "matches_all.csv", encoding="utf-8")
    m = _fix_names(m, ["HomeTeam", "AwayTeam"])  # a mis-decoded name can normalize to the wrong team_key
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


# ---------------------------------------------------------------------------
# team pages: rosters (reuses `players`), recent results, league tables, cup finals
# ---------------------------------------------------------------------------

RECENT_N = 8   # matches shown per team's recent-results list
# matches_all.csv mixes two source labels for the same European competitions
# across overlapping seasons - normalise before grouping by competition.
COMP_ALIASES = {
    "Champions Lg": "Champions League",
    "Europa Lg": "Europa League",
    "Conf Lg": "Europa Conference League",
}
CUP_TYPES = {"domestic_cup", "european", "league_cup", "super_cup"}
COMPLETE_SEASONS = {"2020-21", "2021-22", "2022-23", "2023-24", "2024-25", "2025-26"}  # excludes in-progress 2026-27


def load_all_matches() -> pd.DataFrame:
    m = pd.read_csv(PROC / "matches_all.csv", encoding="utf-8")
    m = _fix_names(m, ["HomeTeam", "AwayTeam", "comp"])
    m["Date"] = pd.to_datetime(m["Date"], errors="coerce")
    m = m.dropna(subset=["Date", "FTHG", "FTAG"])
    m["h"] = m["HomeTeam"].map(normalize_team)
    m["a"] = m["AwayTeam"].map(normalize_team)
    m["comp"] = m["comp"].replace(COMP_ALIASES)
    return m.sort_values("Date")


def build_teams_list() -> list[dict]:
    sq = pd.read_csv(PROC / "squad_season_features.csv", encoding="utf-8")
    sq = _fix_names(sq, ["team"])
    # recompute team_key from the now-fixed name rather than trust the CSV's own
    # team_key column - that one was normalized upstream from the raw (possibly
    # still-mojibake) name, and would then disagree with every team_key computed
    # in this script from matches_all.csv / fbref_player_season_stats.csv
    sq["team_key"] = sq["team"].map(normalize_team)
    sq = sq[sq["src_league"].isin(LEAGUES)].copy()
    sq["league"] = sq["src_league"].map(LEAGUES)
    latest = sq.sort_values("season").groupby(["team_key", "league"], as_index=False).last()
    return [
        {"team_key": r["team_key"], "name": r["team"], "league": r["league"]}
        for _, r in latest.iterrows()
    ]


def build_recent_matches(m: pd.DataFrame) -> dict[str, list[dict]]:
    recent: dict[str, list[dict]] = {}
    for _, r in m.sort_values("Date", ascending=False).iterrows():
        legs = (
            (r["h"], r["a"], r["HomeTeam"], r["AwayTeam"], r["FTHG"], r["FTAG"],
             r.get("HS"), r.get("AS"), r.get("HPoss"), r.get("APoss"), r.get("HxG"), r.get("AxG"), True),
            (r["a"], r["h"], r["AwayTeam"], r["HomeTeam"], r["FTAG"], r["FTHG"],
             r.get("AS"), r.get("HS"), r.get("APoss"), r.get("HPoss"), r.get("AxG"), r.get("HxG"), False),
        )
        for team_key, opp_key, _team_name, opp_name, gf, ga, sf, sa, pf, pa, xgf, xga, is_home in legs:
            lst = recent.setdefault(team_key, [])
            if len(lst) >= RECENT_N:
                continue
            result = "W" if gf > ga else "L" if gf < ga else "D"
            lst.append({
                "date": r["Date"].strftime("%Y-%m-%d"), "opponent": opp_name, "opponent_key": opp_key,
                "home": is_home, "comp": r["comp"], "competition_type": r["competition_type"],
                "gf": _num(gf), "ga": _num(ga), "result": result,
                "shots_for": _num(sf), "shots_against": _num(sa),
                "poss_for": _num(pf), "xg_for": _num(xgf), "xg_against": _num(xga),
            })
    return recent


def build_standings(m: pd.DataFrame) -> list[dict]:
    lg = m[m["competition_type"] == "league"]
    tables = []
    for (season, comp), grp in lg.groupby(["season", "comp"]):
        stats: dict[str, dict] = {}
        for _, r in grp.iterrows():
            h, a, gh, ga_ = r["h"], r["a"], r["FTHG"], r["FTAG"]
            sh = stats.setdefault(h, {"name": r["HomeTeam"], "p": 0, "w": 0, "d": 0, "l": 0, "gf": 0.0, "ga": 0.0})
            sa = stats.setdefault(a, {"name": r["AwayTeam"], "p": 0, "w": 0, "d": 0, "l": 0, "gf": 0.0, "ga": 0.0})
            sh["p"] += 1; sa["p"] += 1
            sh["gf"] += gh; sh["ga"] += ga_
            sa["gf"] += ga_; sa["ga"] += gh
            if gh > ga_: sh["w"] += 1; sa["l"] += 1
            elif gh < ga_: sa["w"] += 1; sh["l"] += 1
            else: sh["d"] += 1; sa["d"] += 1
        rows = []
        for tk, s in stats.items():
            rows.append({
                "team_key": tk, "name": s["name"], "played": s["p"], "w": s["w"], "d": s["d"], "l": s["l"],
                "gf": _num(s["gf"]), "ga": _num(s["ga"]), "gd": _num(s["gf"] - s["ga"]),
                "pts": s["w"] * 3 + s["d"],
            })
        rows.sort(key=lambda x: (-x["pts"], -x["gd"], -x["gf"]))
        for i, row in enumerate(rows):
            row["rank"] = i + 1
        # for competition_type=='league' rows, `comp` is already the league display name
        tables.append({"season": season, "league": comp, "rows": rows})
    return tables


def build_cup_finals(m: pd.DataFrame) -> list[dict]:
    cups = m[m["competition_type"].isin(CUP_TYPES) & m["season"].isin(COMPLETE_SEASONS)]
    finals = []
    for (season, comp), grp in cups.groupby(["season", "comp"]):
        if len(grp) < 2:
            continue  # too little coverage to trust "last match" as the final
        last = grp.sort_values("Date").iloc[-1]
        gh, ga_ = last["FTHG"], last["FTAG"]
        decided_by_pens = gh == ga_
        winner_key = None if decided_by_pens else (last["h"] if gh > ga_ else last["a"])
        winner_name = None
        if winner_key == last["h"]:
            winner_name = last["HomeTeam"]
        elif winner_key == last["a"]:
            winner_name = last["AwayTeam"]
        finals.append({
            "season": season, "comp": comp, "competition_type": last["competition_type"],
            "date": last["Date"].strftime("%Y-%m-%d"),
            "home": last["HomeTeam"], "away": last["AwayTeam"],
            "home_key": last["h"], "away_key": last["a"],
            "score": f"{int(gh)}-{int(ga_)}",
            "decided_by_penalties": bool(decided_by_pens),
            "winner_key": winner_key, "winner_name": winner_name,
        })
    return finals


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


def build_full_schedule(asof: pd.Timestamp) -> dict[str, list[dict]]:
    """Every remaining 2026-27 league fixture per team (not just the next one) -
    fuel for the projected final table."""
    sched: dict[str, list[dict]] = {}
    for league_name in LEAGUES.values():
        live = load_upcoming_fixtures(asof, league_name)
        for _, r in live.iterrows():
            sched.setdefault(r["h"], []).append({"opp_key": r["a"], "home": True})
            sched.setdefault(r["a"], []).append({"opp_key": r["h"], "home": False})
    return sched


def _team_exp_points(model, tk: str, fx: dict) -> float:
    """Expected points from one remaining fixture, from `tk`'s perspective:
    3*P(win) + 1*P(draw), using the same Dixon-Coles model as the match odds."""
    if fx["home"]:
        pA, pD, pH = match_probs(model, tk, fx["opp_key"])
        return float(3 * pH + pD)
    pA, pD, pH = match_probs(model, fx["opp_key"], tk)
    return float(3 * pA + pD)


def build_projected_table(model, standings: list[dict], schedule: dict) -> list[dict]:
    """Current 2026-27 points + expected points (not simulated results) from
    each team's remaining fixtures, run through the match-odds model."""
    tables = []
    for t in standings:
        if t["season"] != "2026-27":
            continue
        rows = []
        for row in t["rows"]:
            tk = row["team_key"]
            remaining = schedule.get(tk, [])
            exp_pts = sum(_team_exp_points(model, tk, fx) for fx in remaining) if model is not None else 0.0
            rows.append({
                "team_key": tk, "name": row["name"], "current_pts": row["pts"],
                "current_rank": row["rank"], "played": row["played"],
                "games_remaining": len(remaining), "projected_pts": round(row["pts"] + exp_pts, 1),
            })
        rows.sort(key=lambda r: -r["projected_pts"])
        for i, r in enumerate(rows):
            r["projected_rank"] = i + 1
        tables.append({"league": t["league"], "rows": rows})
    return tables


MIN_LISTED_FOR_LEADERBOARD = 1_500_000  # floor so a nominal sub-1M TM listing can't produce a 10x+ "bargain"


def build_value_leaderboard(all_players: list[dict], n: int = 8) -> dict:
    """Biggest gaps between the model's predicted value and the listed value,
    both directions - a live demo of the value model, not just a single-player tile."""
    pool = [p for p in all_players if p.get("value") and (p["current"]["min"] or 0) >= MIN_MIN_PROJECT
            and (p["value"]["listed_value_eur"] or 0) >= MIN_LISTED_FOR_LEADERBOARD]

    def slim(p: dict) -> dict:
        return {
            "player": p["player"], "squad": p["squad"], "league": p["league"], "pos": p["pos"],
            "listed_value_eur": p["value"]["listed_value_eur"], "predicted_eur": p["value"]["predicted_eur"],
            "ratio": p["value"]["ratio"],
        }
    bargains = sorted(pool, key=lambda p: p["value"]["ratio"], reverse=True)[:n]
    overpriced = sorted(pool, key=lambda p: p["value"]["ratio"])[:n]
    return {"bargains": [slim(p) for p in bargains], "overpriced": [slim(p) for p in overpriced]}


def build_methodology_example(model, fixtures: list[dict], blended_df: pd.DataFrame,
                               all_players: list[dict]) -> dict | None:
    """A fully worked example of every number on the site, computed for one real
    upcoming fixture and one real player, so 'how was this made' has an actual
    answer instead of just a paragraph."""
    fx = None
    for c in sorted(fixtures, key=lambda f: f["date"]):
        if model is not None and c["home_key"] in model["idx"] and c["away_key"] in model["idx"]:
            fx = c
            break
    if fx is None:
        return None
    h, a = fx["home_key"], fx["away_key"]
    att_h, def_h = float(model["att"][model["idx"][h]]), float(model["dfn"][model["idx"][h]])
    att_a, def_a = float(model["att"][model["idx"][a]]), float(model["dfn"][model["idx"][a]])
    home_adv, rho = float(model["home"]), float(model["rho"])
    lh, la = expected_goals(model, h, a)
    pA, pD, pH = match_probs(model, h, a)

    squad = blended_df[blended_df["team_key"] == h].copy()
    med = squad["min_pct"].median()
    squad["minute_frac"] = (squad["min_pct"].fillna(med if pd.notna(med) else 50.0) / 100.0).clip(0.05, 1.0)
    squad["atk_weight"] = squad["npxg90"].fillna(0).clip(lower=0) * squad["minute_frac"]
    atk_total = float(squad["atk_weight"].sum())
    prop_example = None
    if len(squad) and atk_total > 0:
        top = squad.sort_values("atk_weight", ascending=False).iloc[0]
        share = float(top["atk_weight"] / atk_total)
        lam_player = lh * share
        prop_example = {
            "player": top["player"], "team": fx["home"],
            "npxg90_blended": _num(top["npxg90"]), "minute_frac": _num(top["minute_frac"]),
            "atk_weight": _num(top["atk_weight"]), "squad_atk_total": _num(atk_total),
            "share": _num(share), "team_lambda": _num(lh),
            "lambda_player": _num(lam_player), "p_anytime_goal": _num(1 - np.exp(-lam_player)),
        }

    withval = [p for p in all_players if p.get("value")]
    value_example = None
    if withval:
        vp = max(withval, key=lambda p: p["value"]["listed_value_eur"] or 0)
        value_example = {
            "player": vp["player"], "squad": vp["squad"], "league": vp["league"], "age": vp["age"],
            "current": vp["current"], "listed_value_eur": vp["value"]["listed_value_eur"],
            "predicted_eur": vp["value"]["predicted_eur"], "ratio": vp["value"]["ratio"],
            "season": vp["value"]["as_of_season"],
        }

    return {
        "fixture": {
            "home": fx["home"], "away": fx["away"], "league": fx["league"], "date": fx["date"],
            "att_home": _num(att_h), "def_home": _num(def_h), "att_away": _num(att_a), "def_away": _num(def_a),
            "home_adv": _num(home_adv), "rho": _num(rho),
            "lambda_home": _num(lh), "lambda_away": _num(la),
            "p_home": _num(pH), "p_draw": _num(pD), "p_away": _num(pA),
        },
        "prop_example": prop_example,
        "value_example": value_example,
    }


# ---------------------------------------------------------------------------
# Games (see site/template.html): a match simulator and a "predict the season"
# streak game, both fed entirely by data already computed above.
# ---------------------------------------------------------------------------

STREAK_SEASONS = ["2020-21", "2021-22", "2022-23", "2023-24", "2024-25",
                  "2025-26", "2026-27"]
SIM_SQUAD_N = 16   # players kept per team for the goalscorer draw


def build_games_data(model, blended_df: pd.DataFrame, all_matches: pd.DataFrame,
                     teams_list: list[dict], all_players: list[dict]) -> dict:
    key2league = {t["team_key"]: t["league"] for t in teams_list}
    key2name = {t["team_key"]: t["name"] for t in teams_list}
    # every club with a 2026-27 player in the site data - these are "current"
    current_keys = {p["team_key"] for p in all_players}
    _NAME_FIX = {"leipzig": "RB Leipzig", "cologne": "FC Koln", "hamburger": "Hamburger SV",
                 "arminia bielefeld": "Arminia", "greuther furth": "Greuther Furth"}

    def _name(k: str) -> str:
        nm = key2name.get(k)
        if nm and "�" not in nm:
            return nm
        return _NAME_FIX.get(k) or k.replace("_", " ").title()

    # --- Game 1: match simulator -------------------------------------------
    # ONLY teams with a current-season squad (a relegated club has a stale
    # rating but no 2026-27 players to score the goals). Each gets its
    # Dixon-Coles attack/defence rating and its goalscorer weights (the same
    # npxG90 x minutes-share recipe as the anytime-goal props).
    squads: dict[str, list[dict]] = {}
    for tk, sq in blended_df.groupby("team_key"):
        med = sq["min_pct"].median()
        mf = (sq["min_pct"].fillna(med if pd.notna(med) else 50.0) / 100.0).clip(0.05, 1.0)
        w = (sq["npxg90"].fillna(0).clip(lower=0).to_numpy() * mf.to_numpy())
        tot = float(w.sum())
        if tot <= 0:
            continue
        rows = sorted(
            ({"p": r["player"], "pos": r["pos"], "w": round(float(ww / tot), 4)}
             for (_, r), ww in zip(sq.iterrows(), w)),
            key=lambda x: -x["w"])[:SIM_SQUAD_N]
        squads[tk] = rows

    # a club just promoted this season may have no player over the 45-minute
    # cut yet - build its squad straight from the raw table (any minutes) so
    # it still shows up in the simulator.
    missing = current_keys - set(squads)
    if missing:
        raw = pd.read_csv(PROC / "fbref_player_season_stats.csv",
                          low_memory=False, encoding="latin-1")
        raw = _fix_names(raw, ["Player", "Squad"])
        raw = raw[raw["season"] == "2026-27"].copy()
        raw["tk"] = raw["Squad"].map(normalize_team)
        raw["mn"] = pd.to_numeric(raw["standard__Playing Time_Min"], errors="coerce")
        raw["nx"] = pd.to_numeric(raw.get("understat__np_xg"), errors="coerce").fillna(0)
        raw["gl"] = pd.to_numeric(raw.get("standard__Performance_Gls"), errors="coerce").fillna(0)
        for tk in missing:
            r = raw[(raw["tk"] == tk) & (raw["mn"] > 0)]
            if r.empty:
                continue
            wcol = (r["nx"] + 0.4 * r["gl"] + 0.02 * r["mn"] / 90.0)
            tot = float(wcol.sum())
            if tot <= 0:
                continue
            squads[tk] = sorted(
                ({"p": rr["Player"], "pos": str(rr["Pos"]).split(",")[0],
                  "w": round(float(ww / tot), 4)}
                 for (_, rr), ww in zip(r.iterrows(), wcol)),
                key=lambda x: -x["w"])[:SIM_SQUAD_N]

    sim_teams = []
    for tk in sorted(squads):
        if model is None or tk not in model["idx"] or tk not in current_keys:
            continue
        i = model["idx"][tk]
        sim_teams.append({
            "key": tk, "name": _name(tk), "league": key2league.get(tk, ""),
            "att": round(float(model["att"][i]), 4),
            "def": round(float(model["dfn"][i]), 4),
        })
    squads = {t["key"]: squads[t["key"]] for t in sim_teams}
    sim = {
        "home_adv": round(float(model["home"]), 4) if model else 0.25,
        "rho": round(float(model["rho"]), 4) if model else -0.05,
        "teams": sim_teams, "squads": squads,
    }

    # --- Game 2: "predict the season" streak game -------------------------
    # every league match, in date order, with the actual score and the
    # leak-free Dixon-Coles model's pre-match W/D/L probabilities.
    dc = pd.read_csv(PROC / "dixon_coles_probs.csv", encoding="utf-8")
    lg = all_matches[all_matches["competition_type"] == "league"].merge(
        dc[["season", "HomeTeam", "AwayTeam", "dc_pH", "dc_pD", "dc_pA"]],
        on=["season", "HomeTeam", "AwayTeam"], how="left")
    lg = lg[lg["season"].isin(STREAK_SEASONS)].sort_values("Date")

    tks = sorted(set(lg["h"]) | set(lg["a"]))
    tk_idx = {k: i for i, k in enumerate(tks)}
    streak_teams = [[k, _name(k), key2league.get(k, "")] for k in tks]
    fixtures = []
    for r in lg.itertuples(index=False):
        pH = r.dc_pH if pd.notna(r.dc_pH) else 0.45
        pD = r.dc_pD if pd.notna(r.dc_pD) else 0.26
        pA = r.dc_pA if pd.notna(r.dc_pA) else 0.29
        fixtures.append([
            STREAK_SEASONS.index(r.season), tk_idx[r.h], tk_idx[r.a],
            int(r.FTHG), int(r.FTAG),
            round(float(pH), 3), round(float(pD), 3), round(float(pA), 3),
        ])
    streak = {"seasons": STREAK_SEASONS, "teams": streak_teams, "fixtures": fixtures}

    return {
        "sim": sim,
        "streak": streak,
        "notes": {
            "sim": "Draws each team's goals from a Poisson on the Dixon-Coles "
                   "expected goals for the matchup, then hands each goal to a "
                   "player picked by his npxG-per-90 x minutes share. Random "
                   "single game - hit re-run, or simulate 1,000 for the spread.",
            "streak": "Predict a real team's whole season one match at a time. "
                      "The reveal shows the actual result and what the leak-free "
                      "Dixon-Coles model (fitted only on earlier matches) gave "
                      "pre-match. One wrong call ends the run.",
        },
    }


def main() -> None:
    # Dixon-Coles is fit once, combined across all competitions (league + cup +
    # European) - that's what lets Bundesliga/La Liga/Premier League teams share
    # a single attack/defence scale via their Champions/Europa League meetings.
    dc_matches = prep_dc_matches()
    asof = dc_matches["Date"].max() + pd.Timedelta(days=1)
    model = fit(dc_matches, asof)

    all_players: list[dict] = []
    all_blended = []
    for src_league, league_name in LEAGUES.items():
        players, _, blended_df = build_players_payload(src_league, league_name)
        all_players.extend(players)
        blended_df["league"] = league_name
        all_blended.append(blended_df)
    blended_df = pd.concat(all_blended, ignore_index=True)

    fixtures = []
    for src_league, league_name in LEAGUES.items():
        live = load_upcoming_fixtures(asof, league_name)
        nxt = next_fixture_per_team(live)
        for _, r in nxt.iterrows():
            lh, la = expected_goals(model, r["h"], r["a"])
            pA, pD, pH = match_probs(model, r["h"], r["a"])
            fx = {
                "date": r["Date"].strftime("%Y-%m-%d %H:%M"),
                "league": league_name,
                "home": r["HomeTeam"], "away": r["AwayTeam"],
                "home_key": r["h"], "away_key": r["a"],
                "p_home_win": _num(pH), "p_draw": _num(pD), "p_away_win": _num(pA),
                "exp_goals_home": _num(lh), "exp_goals_away": _num(la),
                "home_props": player_props_for_fixture(blended_df, r["h"], lh),
                "away_props": player_props_for_fixture(blended_df, r["a"], la),
            }
            fixtures.append(fx)

    all_matches = load_all_matches()
    teams = build_teams_list()
    recent_matches = build_recent_matches(all_matches)
    standings = build_standings(all_matches)
    cup_finals = build_cup_finals(all_matches)

    full_schedule = build_full_schedule(asof)
    projected_table = build_projected_table(model, standings, full_schedule)
    value_leaderboard = build_value_leaderboard(all_players)
    methodology_example = build_methodology_example(model, fixtures, blended_df, all_players)
    games = build_games_data(model, blended_df, all_matches, teams, all_players)

    payload = {
        "generated_asof": asof.strftime("%Y-%m-%d"),
        "leagues": list(LEAGUES.values()),
        "players": all_players,
        "fixtures": fixtures,
        "teams": teams,
        "recent_matches": recent_matches,
        "standings": standings,
        "cup_finals": cup_finals,
        "projected_table": projected_table,
        "value_leaderboard": value_leaderboard,
        "methodology_example": methodology_example,
        "games": games,
        "notes": {
            "current_stats": "2026-27 FBref season-to-date stats (min 45 minutes played).",
            "last_season": "2025-26 full-season stats for the same player, where available.",
            "projected_38": "Simple pace projection: current per-90 rate x projected minutes over a 38-game season. Not a trained model.",
            "value": "Predicted market value from the trained value-regression model (current season preferred, else last season; R2(log) 0.89, MAE EUR4.9M, within-2x 91%) vs listed market value.",
            "match_odds": "Win/draw/loss odds from a Dixon-Coles attack/defence model fit on all competitions through the date above.",
            "player_props": "Anytime goal/assist odds: team's Dixon-Coles expected goals split across the matchday squad by each player's (non-penalty xG90 or xA90, shrunk toward last season's rate early in the current season) x season minutes-share, then Poisson P(>=1).",
            "cup_finals": "Each competition's final is inferred as the last-dated match of that season/competition in the results data - not read from an official bracket. When it ended level (decided on penalties/extra time not recorded here), no winner is shown. The in-progress 2026-27 season is excluded.",
            "standings": "Full league tables computed directly from match results (3 pts/win). The 2026-27 table is the live in-progress standing.",
            "projected_table": "Current points + expected points (3xP(win)+P(draw) per game, not simulated results) from each team's remaining fixtures, using the same Dixon-Coles model as the match odds. A projection, not a guarantee - form, injuries and transfers between now and kickoff aren't in it.",
            "value_leaderboard": "The value model's biggest gaps between predicted and listed value, both directions, among players with at least 180 minutes this season.",
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    payload_json = json.dumps(payload, indent=None, separators=(",", ":"))
    OUT.write_text(payload_json, encoding="utf-8")
    print(f"wrote {OUT}  ({len(all_players)} players, {len(fixtures)} fixtures, {len(teams)} teams, "
          f"{len(standings)} standings tables, {len(cup_finals)} cup finals, "
          f"{len(projected_table)} projected tables, {len(games['sim']['teams'])} sim teams / "
          f"{len(games['streak']['fixtures'])} streak matches)  size={OUT.stat().st_size/1024:.0f} KB")

    splice_index_html(payload_json)


def splice_index_html(payload_json: str) -> None:
    """site/template.html + this run's data -> the committed, self-contained
    site/index.html. Escaping </script so a name/comp string can't break out of
    the embedded JSON <script> block (e.g. a club literally named "</script>")."""
    template = TEMPLATE.read_text(encoding="utf-8")
    if PLACEHOLDER not in template:
        raise RuntimeError(f"{TEMPLATE} is missing the {PLACEHOLDER} placeholder - can't splice data in")
    safe_json = payload_json.replace("</script", "<\\/script")
    INDEX.write_text(template.replace(PLACEHOLDER, safe_json), encoding="utf-8")
    print(f"wrote {INDEX}  size={INDEX.stat().st_size/1024:.0f} KB")


if __name__ == "__main__":
    main()
