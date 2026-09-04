"""Parse cached FBref player season-stats HTML into one wide table.

Browser-free: reads ``data/raw/fbref/player_stats/<COMP>_<season>_<category>.html``
and joins the ~11 stat categories into
``data/processed/fbref_player_season_stats.csv`` - one row per (player, squad,
season), with per-90 and totals for the value model.

A player who moved mid-season shows once per club stint (FBref splits them), plus
a league total row (Squad ends in ``2 Squads`` etc.) which we drop.

Run:
    py -3.11 src/parse_fbref_player_stats.py
"""

from __future__ import annotations

import re
import sys
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from live.schema import deaccent, normalize_team  # name folding + club resolver


def _norm_name(s) -> str:
    """Loose key for player/club names across sources: transliterated to bare
    ascii (o slash -> o, sharp s -> ss ... - see live.schema.deaccent), no
    punctuation, lowercase, collapsed spaces."""
    if not isinstance(s, str):
        return ""
    s = deaccent(s).lower().replace("'", "")  # O'Shea -> oshea, to match the FBref slug
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    return " ".join(s.split())

PROJECT_ROOT = Path(__file__).resolve().parent.parent
STATS_DIR = PROJECT_ROOT / "data" / "raw" / "fbref" / "player_stats"
OUT_PATH = PROJECT_ROOT / "data" / "processed" / "fbref_player_season_stats.csv"

FNAME_RE = re.compile(r"^(?P<comp>[A-Z]{3}\d)_(?P<season>\d{4}-\d{2})_(?P<cat>[a-z_]+)\.html$")

CATEGORY_TABLE = {
    "standard": "stats_standard", "shooting": "stats_shooting",
    "passing": "stats_passing", "passing_types": "stats_passing_types",
    "gca": "stats_gca", "defense": "stats_defense", "possession": "stats_possession",
    "playing_time": "stats_playing_time", "misc": "stats_misc",
    "keeper": "stats_keeper", "keeper_adv": "stats_keeper_adv",
}

# data-stat names FBref renders before the stat numbers load / that carry no signal
_SKELETON_STATS = {
    "ranker", "player", "nationality", "position", "team", "age", "birth_year",
    "minutes_90s", "games", "matches", "assists",
}

# identity columns present in every category table
ID_COLS = ["Player", "Nation", "Pos", "Squad", "Age", "Born"]
JOIN_KEYS = ["season", "src_league", "Player", "Squad", "Born"]

_SEASON_ORD = {f"{y}-{str(y + 1)[-2:]}": y for y in range(2013, 2028)}


def _pos1(p) -> str:
    return {"GK": "GK", "DF": "DF", "MF": "MF", "FW": "FW"}.get(
        str(p).split(",")[0], "MF")


def _flatten(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = [
            c[-1] if (not str(c[0]) or str(c[0]).startswith("Unnamed"))
            else f"{c[0]}_{c[-1]}".strip("_")
            for c in df.columns
        ]
    return df


def _player_ids(html_doc: str, table_id: str) -> dict[tuple[str, str], str]:
    """(Player, Squad) -> FBref player id, scraped from data-append-csv rows."""
    m = re.search(rf'id="{table_id}".*?</table>', html_doc, re.S)
    if not m:
        return {}
    out: dict[tuple[str, str], str] = {}
    for row in re.findall(r"<tr.*?</tr>", m.group(0), re.S):
        pid = re.search(r'data-append-csv="([^"]+)"', row)
        pname = re.search(r'data-append-csv="[^"]+">([^<]+)</a>', row)
        squad = re.search(r'/squads/[0-9a-f]{8}/[^"]*">([^<]+)</a>', row)
        if pid and pname:
            out[(pname.group(1), squad.group(1) if squad else "")] = pid.group(1)
    return out


def _page_has_stats(doc: str, table_id: str) -> bool:
    """FBref serves the advanced (Opta-derived) player tables as empty skeletons
    to scrapers - real numbers appear in >=6 distinct non-skeleton stat columns."""
    m = re.search(rf'id="{table_id}".*?</table>', doc, re.S)
    if not m:
        return False
    got = set(re.findall(r'data-stat="([a-z_0-9]+)"[^>]*>\s*[\d.]', m.group(0)))
    return len(got - _SKELETON_STATS) >= 6


def load_category(cat: str) -> pd.DataFrame:
    table_id = CATEGORY_TABLE[cat]
    frames, empty = [], 0
    for path in sorted(STATS_DIR.glob(f"*_{cat}.html")):
        m = FNAME_RE.match(path.name)
        if not m:
            continue
        doc = path.read_text(encoding="utf-8").replace("<!--", "").replace("-->", "")
        if not _page_has_stats(doc, table_id):
            empty += 1
            continue
        try:
            cands = pd.read_html(StringIO(doc), attrs={"id": table_id})
        except (ValueError, IndexError):
            continue
        # FBref ships some tables twice (visible + a copy for its export widget);
        # uncommenting exposes both under the same id. Keep the fuller one.
        df = max(cands, key=lambda t: t.notna().to_numpy().sum())
        df = _flatten(df)
        # FBref's served HTML mangles accented display names ("Mbapp�"); the
        # href slug is clean - use it as the join key.
        seg = re.search(rf'id="{table_id}".*?</table>', doc, re.S)
        slugs = re.findall(
            r'data-append-csv="[0-9a-f]+"[^>]*>\s*<a href="/en/players/[0-9a-f]+/([A-Za-z0-9-]+)"',
            seg.group(0) if seg else doc)
        df = df[df["Player"].notna() & (df["Player"] != "Player")]
        df = df[~df["Squad"].astype(str).str.contains("Squads", na=False)]  # drop league-total rows
        if len(slugs) == len(df):
            df.insert(0, "player_slug", [s.replace("-", " ") for s in slugs])
        else:
            df.insert(0, "player_slug", df["Player"].astype(str))
        df.insert(0, "season", m["season"])
        df.insert(1, "src_league", m["comp"])
        # non-identity stat columns get a category prefix to stay distinct
        ren = {c: f"{cat}__{c}" for c in df.columns
               if c not in (ID_COLS + ["season", "src_league", "player_slug", "Rk", "Matches"])}
        df = df.rename(columns=ren).drop(columns=["Rk", "Matches"], errors="ignore")
        frames.append(df)
    if empty:
        print(f"  ({cat}: {empty} pages had an empty/skeleton table - skipped)")
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def main() -> None:
    if not STATS_DIR.exists() or not any(STATS_DIR.glob("*.html")):
        raise SystemExit(f"no player-stats pages in {STATS_DIR} - run pull_fbref_player_stats.py")

    base = load_category("standard")
    if base.empty:
        raise SystemExit("no standard-category pages cached yet")

    combined = base
    for cat in CATEGORY_TABLE:
        if cat == "standard":
            continue
        df = load_category(cat)
        if df.empty:
            print(f"({cat}: no pages, skipping)")
            continue
        on = [k for k in JOIN_KEYS if k in df.columns and k in combined.columns]
        drop_dupe_ids = [c for c in ID_COLS + ["player_slug"]
                         if c not in on and c in df.columns]
        df = df.drop(columns=drop_dupe_ids).drop_duplicates(subset=on)
        combined = combined.merge(df, on=on, how="left")

    for col in ("Age", "Born"):
        combined[col] = pd.to_numeric(combined[col], errors="coerce")

    # advanced stats FBref gated from the scrape - big-5, 2020-21..2022-23 - come
    # from the worldfootballR mirror (src/pull_wfr_advanced.py). Left-join what's
    # there; rows/leagues/seasons the mirror doesn't cover keep NaN advanced cols.
    wfr = OUT_PATH.parent / "wfr_player_advanced.csv"
    if wfr.exists():
        adv = pd.read_csv(wfr)
        adv["Born"] = pd.to_numeric(adv["Born"], errors="coerce")
        on = ["season", "src_league", "Player", "Squad", "Born"]
        adv = adv.drop(columns=[c for c in ("Nation", "Pos", "Age", "fbref_player_id")
                                if c in adv.columns]).drop_duplicates(subset=on)
        new = [c for c in adv.columns if c not in combined.columns or c in on]
        combined = combined.merge(adv[new], on=on, how="left")
        matched = combined.filter(regex=r"^(passing|defense|possession|gca)__").notna().any(axis=1).sum()
        print(f"merged worldfootballR advanced stats ({matched} rows got xG/progressive/etc.)")
    else:
        print("(no wfr_player_advanced.csv - run pull_wfr_advanced.py for xG etc.)")

    # xG for every season/league from Understat (src/pull_understat.py). The
    # mirror above only covers 2020-22; Understat covers 2020-21..now for all
    # three leagues. Joined on season+league+normalised player+team.
    us = OUT_PATH.parent / "understat_player_season.csv"
    if us.exists():
        ux = pd.read_csv(us)
        keep = ["season", "src_league", "player", "team", "xg", "np_xg", "xa",
                "np_goals", "shots", "key_passes", "xg_chain", "xg_buildup"]
        ux = ux[[c for c in keep if c in ux.columns]].rename(
            columns={c: f"understat__{c}" for c in keep if c not in ("season", "src_league")})
        ux["_pk"] = ux["understat__player"].map(_norm_name)
        ux["_tk"] = ux["understat__team"].map(normalize_team)
        ux = ux.drop(columns=["understat__player", "understat__team"]).drop_duplicates(
            subset=["season", "src_league", "_pk", "_tk"])
        combined["_pk"] = combined["player_slug"].map(_norm_name)
        combined["_tk"] = combined["Squad"].map(normalize_team)
        combined = combined.merge(ux, on=["season", "src_league", "_pk", "_tk"], how="left")
        got = combined["understat__xg"].notna().sum()
        combined = combined.drop(columns=["_pk", "_tk"])
        print(f"merged Understat xG ({got}/{len(combined)} rows)")
    else:
        print("(no understat_player_season.csv - run pull_understat.py for xG)")

    # Transfermarkt market value - the value model's TARGET.
    #   tm_player_values.csv    worldfootballR mirror, 2020-21..2022-23
    #   tm_values_scraped.csv   nodriver scrape, 2023-24..2025-26
    tm_parts = []
    mv = OUT_PATH.parent / "tm_player_values.csv"
    if mv.exists():
        m = pd.read_csv(mv)
        m["contract_expiry"] = pd.to_datetime(m.get("contract_expiry"), errors="coerce")
        tm_parts.append(m[["season", "src_league", "player_name", "squad",
                           "tm_player_id", "player_dob", "player_foot",
                           "contract_expiry", "market_value_eur"]].rename(
            columns={"squad": "club", "player_dob": "tm_dob", "player_foot": "tm_foot"}))
    sc = OUT_PATH.parent / "tm_values_scraped.csv"
    if sc.exists():
        s = pd.read_csv(sc)
        s["club"] = s["squad_slug"].str.replace("-", " ")
        s["contract_expiry"] = pd.NaT
        tm_parts.append(s[["season", "src_league", "player_name", "club",
                           "tm_player_id", "contract_expiry", "market_value_eur"]])
    if tm_parts:
        tv = pd.concat(tm_parts, ignore_index=True)
        tv["_pk"] = tv["player_name"].map(_norm_name)
        tv["_tk"] = tv["club"].map(normalize_team)
        tv = tv.drop(columns=["player_name", "club"]).drop_duplicates(
            subset=["season", "src_league", "_pk", "_tk"])
        combined["_pk"] = combined["player_slug"].map(_norm_name)
        combined["_tk"] = combined["Squad"].map(normalize_team)
        combined = combined.merge(tv, on=["season", "src_league", "_pk", "_tk"], how="left")
        got = combined["market_value_eur"].notna().sum()
        combined = combined.drop(columns=["_pk", "_tk"])
        print(f"merged Transfermarkt value ({got}/{len(combined)} rows, 2020-26)")
    else:
        print("(no tm value files - run pull_transfermarkt*.py)")

    # Sofascore fills the value for the current season (Transfermarkt mirror stops
    # at 2022-23; Sofascore has no history so it's current-season only).
    sf = OUT_PATH.parent / "sofascore_values.csv"
    if sf.exists():
        sv = pd.read_csv(sf)[["season", "src_league", "player_name", "club",
                              "sofascore_id", "contract_until", "market_value_eur"]].rename(
            columns={"market_value_eur": "_sf_value"})
        sv["_pk"] = sv["player_name"].map(_norm_name)
        sv["_tk"] = sv["club"].map(normalize_team)
        sv = sv.drop(columns=["player_name", "club"]).drop_duplicates(
            subset=["season", "src_league", "_pk", "_tk"])
        combined["_pk"] = combined["player_slug"].map(_norm_name)
        combined["_tk"] = combined["Squad"].map(normalize_team)
        combined = combined.merge(sv, on=["season", "src_league", "_pk", "_tk"], how="left")
        if "market_value_eur" not in combined.columns:
            combined["market_value_eur"] = pd.NA
        combined["market_value_eur"] = combined["market_value_eur"].fillna(combined["_sf_value"])
        if "contract_expiry" in combined.columns:
            combined["contract_expiry"] = combined["contract_expiry"].fillna(
                pd.to_datetime(combined["contract_until"], errors="coerce"))
        got = combined["_sf_value"].notna().sum()
        combined = combined.drop(columns=["_pk", "_tk", "_sf_value"])
        print(f"merged Sofascore value ({got} rows, current season)")
    else:
        print("(no sofascore_values.csv - run pull_sofascore_values.py)")

    # Fallback for the still-unmatched: mostly (a) mid-season transfers / loans -
    # FBref splits the season row by club, the value feeds carry one club - and
    # (b) name-form drift: "Thiago" vs "Thiago Alcantara", "Son Heung-min" vs
    # "Heung-min Son", "Andy Robertson" vs "Andrew Robertson", "Max Kilman" vs
    # "Maximilian Kilman", "Illia Zabarnyi" vs "Ilya Zabarnyi". Fill from a
    # pooled feed, only where the value comes out unambiguous.
    if combined["market_value_eur"].isna().any():
        pool = []
        for path, club_col in [(mv, "squad"), (sc, "squad_slug"), (sf, "club")]:
            if not path.exists():
                continue
            p = pd.read_csv(path)
            if "market_value_eur" not in p.columns or club_col not in p.columns:
                continue
            p = p[p["market_value_eur"].notna()]
            pool.append(pd.DataFrame({
                "season": p["season"], "src_league": p["src_league"],
                "_pk": p["player_name"].map(_norm_name),
                "_tk": p[club_col].map(lambda c: normalize_team(str(c).replace("-", " "))),
                "v": p["market_value_eur"]}))
        vp = pd.concat(pool, ignore_index=True)
        vp = vp[vp["_pk"] != ""]
        vp["_toks"] = vp["_pk"].map(lambda s: s.split())
        vp["_sig"] = vp["_toks"].map(frozenset)
        vp_sl = {k: g for k, g in vp.groupby(["season", "src_league"])}
        vp_slt = {k: g for k, g in vp.groupby(["season", "src_league", "_tk"])}

        # too common to identify a player on their own (Jose Luis Garcia Vaya
        # must not collect German Garcia's value just because both end ...garcia)
        _COMMON = {
            "garcia", "rodriguez", "fernandez", "gonzalez", "lopez", "perez",
            "sanchez", "martinez", "gomez", "diaz", "martin", "jimenez", "ruiz",
            "hernandez", "moreno", "alvarez", "romero", "alonso", "torres",
            "navarro", "dominguez", "vazquez", "ramos", "serrano", "castro",
            "silva", "santos", "costa", "pereira", "ferreira", "oliveira",
        }

        def _one_value(g) -> "float | None":
            vals = g["v"].unique()
            return float(vals[0]) if len(vals) == 1 else None

        def _same_person(fb: list, fd: list) -> bool:
            """fb = FBref slug tokens, fd = feed name tokens - same player under
            first-name drift (Andy/Andrew, Max/Maximilian) or a mononym feed
            name (Thiago, Emerson). Requires the surnames to line up, so
            "James Bree" never grabs "James Ward-Prowse"."""
            if not fb or not fd:
                return False
            if frozenset(fb) == frozenset(fd):
                return True
            if len(fd) == 1:                       # feed is a mononym
                return fd[0] == fb[0] or fd[0] == fb[-1]
            if len(fb) == 1:
                return fb[0] == fd[0] or fb[0] == fd[-1]
            if fb[-1] == fd[-1]:                   # shared surname
                return True
            if fb[0] == fd[0] and (fb[-1] in fd or fd[-1] in fb):  # shared first + surname nested
                return True
            # a shared non-leading name of real length: "Isi Palazon" vs
            # "Isaac Palazon Camacho", "Gabri Veiga" vs "Gabriel Veiga". The
            # leading token is excluded so "James Bree" can't seize "James ...".
            if any(len(t) >= 5 and t not in _COMMON
                   for t in set(fb[1:]) & set(fd[1:])):
                return True
            return False

        need = combined["market_value_eur"].isna() & combined["player_slug"].notna()
        newv = {"xfer": 0, "surname": 0}
        fills: dict = {}
        for idx, slug, tk, s, lg in zip(combined.index[need],
                                        combined.loc[need, "player_slug"],
                                        combined.loc[need, "Squad"].map(normalize_team),
                                        combined.loc[need, "season"],
                                        combined.loc[need, "src_league"]):
            fb = _norm_name(slug).split()
            rt = frozenset(fb)
            if not fb:
                continue
            # (b) league-wide: same token set (reordered) or exactly one name apart
            g = vp_sl.get((s, lg))
            if g is not None:
                sigs = list(g["_sig"])

                def _near(ct: frozenset) -> bool:
                    if ct == rt:
                        return True
                    if not (ct <= rt or rt <= ct) or len(ct ^ rt) > 1:
                        return False
                    if ct < rt:  # feed carries a mononym - only if nothing extends it
                        return not any(ct < o for o in sigs)
                    return True
                v = _one_value(g[g["_sig"].map(_near)])
                if v is not None:
                    fills[idx] = v
                    newv["xfer"] += 1
                    continue
            # (a) team-scoped: same player by surname, unique value within that squad
            gt = vp_slt.get((s, lg, tk))
            if gt is not None:
                hit = gt[gt["_toks"].map(lambda fd: _same_person(fb, fd))]
                v = _one_value(hit)
                if v is not None:
                    fills[idx] = v
                    newv["surname"] += 1
        if fills:
            combined.loc[list(fills), "market_value_eur"] = pd.Series(fills)
        print(f"fuzzy-filled {newv['xfer']} (transfer / name-form) + "
              f"{newv['surname']} (same-club surname) more values")

    # ---------------------------------------------------------------------------
    # Guarantee a value for everyone who played. Two last-resort passes:
    #   3. carry a player's most recent known value forward - across a transfer
    #      or a season the feeds skipped (Adeyemi, Anthony Gordon, Rodri, keepers
    #      the name join missed).
    #   4. peer-median impute so no one with minutes is left blank - EXCEPT
    #      players on a club promoted into the league this season, who may
    #      genuinely have no market history.
    # ---------------------------------------------------------------------------
    combined["market_value_imputed"] = 0
    combined["_ord"] = combined["season"].map(_SEASON_ORD)
    combined["_pk"] = combined["player_slug"].map(_norm_name)
    combined["_tk"] = combined["Squad"].map(normalize_team)
    combined["_min"] = pd.to_numeric(
        combined.get("standard__Playing Time_Min"), errors="coerce")

    # value history: full big-5 mirror (2015-22, all leagues) + scrape + Sofascore
    # + every value we have so far - one number per (player, season).
    hist = []
    try:
        from build_value_history import _from_mirror
        mh = _from_mirror().rename(columns={"player_key": "_pk", "market_value_eur": "v"})
        mh["_pk"] = mh["_pk"].map(_norm_name)
        hist.append(mh[["_pk", "season", "v"]])
    except Exception as exc:  # noqa: BLE001 - mirror is optional
        print(f"(carry-forward: no full mirror - {exc})")
    for path, namecol in [(sc, "player_name"), (sf, "player_name")]:
        if path.exists():
            p = pd.read_csv(path)
            if "market_value_eur" in p.columns:
                p = p[p["market_value_eur"].notna()]
                hist.append(pd.DataFrame({"_pk": p[namecol].map(_norm_name),
                                          "season": p["season"],
                                          "v": p["market_value_eur"]}))
    have = combined.loc[combined["market_value_eur"].notna(), ["_pk", "season", "market_value_eur"]]
    hist.append(have.rename(columns={"market_value_eur": "v"}))
    H = pd.concat(hist, ignore_index=True)
    H = H[(H["_pk"] != "") & H["v"].notna()].copy()
    H["_ord"] = H["season"].map(_SEASON_ORD)
    H = H.dropna(subset=["_ord"]).groupby(["_pk", "_ord"], as_index=False)["v"].max()

    need = combined["market_value_eur"].isna() & combined["_pk"].ne("") & combined["_ord"].notna()
    carried = 0
    if need.any():
        by_pk = {k: g.sort_values("_ord") for k, g in H.groupby("_pk")}
        fills = {}
        for idx, pk, o in zip(combined.index[need], combined.loc[need, "_pk"],
                              combined.loc[need, "_ord"]):
            g = by_pk.get(pk)
            if g is None:
                continue
            prior = g[g["_ord"] < o]
            if len(prior):
                fills[idx] = float(prior.iloc[-1]["v"])
        if fills:
            combined.loc[list(fills), "market_value_eur"] = pd.Series(fills)
            carried = len(fills)
    print(f"carried a prior value forward for {carried} players (transfers / feed gaps)")

    # clubs new to their league this season - their lower-division arrivals are
    # the one group allowed to stay unvalued. (Not applied to the earliest
    # season, where every club looks new for lack of a prior year.)
    seen = set(zip(combined["_ord"], combined["_tk"]))
    ords = set(combined["_ord"].dropna())
    newcomer = {(o, tk) for o, tk in seen
                if (o - 1) in ords and (o - 1, tk) not in seen}

    lab = combined[combined["market_value_eur"].notna()].copy()
    lab["_pos"] = lab["Pos"].map(_pos1)
    lab["_ab"] = (pd.to_numeric(lab["Age"], errors="coerce") // 3).clip(6, 12)
    lab["_lv"] = np.log1p(lab["market_value_eur"])
    med3 = lab.groupby(["src_league", "_pos", "_ab"])["_lv"].median()
    med2 = lab.groupby(["src_league", "_pos"])["_lv"].median()
    med1 = lab.groupby("src_league")["_lv"].median()

    need = (combined["market_value_eur"].isna() & (combined["_min"] > 0)
            & combined["_pk"].ne("") & combined["_ord"].notna())
    imp = {}
    for idx, r in combined.loc[need].iterrows():
        if (r["_ord"], r["_tk"]) in newcomer:
            continue  # promoted-club newcomer - allowed to have no value
        pos, ab = _pos1(r["Pos"]), np.clip(pd.to_numeric(r["Age"], errors="coerce") // 3, 6, 12)
        lv = med3.get((r["src_league"], pos, ab))
        if pd.isna(lv):
            lv = med2.get((r["src_league"], pos), med1.get(r["src_league"]))
        if pd.notna(lv):
            imp[idx] = float(np.expm1(lv))
    if imp:
        combined.loc[list(imp), "market_value_eur"] = pd.Series(imp)
        combined.loc[list(imp), "market_value_imputed"] = 1
    print(f"peer-median imputed {len(imp)} more (non-promoted players with minutes)")

    still = combined["market_value_eur"].isna() & (combined["_min"] > 0)
    print(f"remaining unvalued with minutes: {still.sum()} "
          f"(promoted-club arrivals with no market history)")
    combined = combined.drop(columns=["_ord", "_pk", "_tk", "_min"])

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(OUT_PATH, index=False)
    print(f"wrote {OUT_PATH}  ({len(combined)} rows, {combined.shape[1]} cols)")
    print(f"players: {combined['Player'].nunique()}  "
          f"seasons: {sorted(combined['season'].dropna().unique())}")
    print(f"rows per season:\n{combined.groupby('season').size().to_string()}")


if __name__ == "__main__":
    main()
