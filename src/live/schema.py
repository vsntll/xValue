"""Shared contract for the current-season match sources.

Every source module exposes ``fetch(comp_codes, season) -> pd.DataFrame`` with the
columns in ``MATCH_COLS`` (missing stats = NaN). Downstream code only ever sees
this schema, so sources can be swapped / reordered without touching anything else.
"""

from __future__ import annotations

import re
import unicodedata

import pandas as pd

# aligned with data/processed/match_features.csv, plus source/id/possession/xG
MATCH_COLS = [
    "source", "season", "league", "tier", "competition_type", "comp_code",
    "Date", "Time", "HomeTeam", "AwayTeam", "status",
    "FTHG", "FTAG", "FTR", "HTHG", "HTAG", "HTR",
    "HS", "AS", "HST", "AST", "HC", "AC", "HF", "AF", "HY", "AY", "HR", "AR",
    "HPoss", "APoss", "HxG", "AxG", "match_id",
]

# our three leagues + the cups their clubs play in. Per-source competition codes;
# None = that source doesn't carry it.
LEAGUES: dict[str, dict] = {
    "ENG1": dict(name="Premier League", tier=1, competition_type="league",
                 espn="eng.1", fdorg="PL"),
    "GER1": dict(name="Bundesliga", tier=1, competition_type="league",
                 espn="ger.1", fdorg="BL1"),
    "ESP1": dict(name="La Liga", tier=1, competition_type="league",
                 espn="esp.1", fdorg="PD"),
}
CUPS: dict[str, dict] = {
    "UCL": dict(name="Champions League", tier=1, competition_type="european",
                espn="uefa.champions", fdorg="CL"),
    "UEL": dict(name="Europa League", tier=1, competition_type="european",
                espn="uefa.europa", fdorg=None),
    "UECL": dict(name="Europa Conference League", tier=1, competition_type="european",
                 espn="uefa.europa.conf", fdorg=None),
    "FA": dict(name="FA Cup", tier=1, competition_type="domestic_cup",
               espn="eng.fa", fdorg=None),
    "EFL": dict(name="EFL Cup", tier=1, competition_type="league_cup",
                espn="eng.league_cup", fdorg=None),
    "DFB": dict(name="DFB-Pokal", tier=1, competition_type="domestic_cup",
                espn="ger.dfb_pokal", fdorg=None),
    "CDR": dict(name="Copa del Rey", tier=1, competition_type="domestic_cup",
                espn="esp.copa_del_rey", fdorg=None),
}
ALL_COMPS = {**LEAGUES, **CUPS}
DEFAULT_COMPS = list(LEAGUES) + ["UCL", "FA", "EFL", "DFB", "CDR"]


def blank_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=MATCH_COLS)


def finalize(rows: list[dict], source: str) -> pd.DataFrame:
    """rows -> a MATCH_COLS frame: fill missing cols, derive FTR, order columns."""
    df = pd.DataFrame(rows)
    for c in MATCH_COLS:
        if c not in df.columns:
            df[c] = pd.NA
    df["source"] = source
    if len(df):
        h, a = pd.to_numeric(df["FTHG"], errors="coerce"), pd.to_numeric(df["FTAG"], errors="coerce")
        df["FTR"] = df["FTR"].where(
            df["FTR"].notna(),
            pd.Series(["H"] * len(df)).where(h > a, pd.Series(["A"] * len(df)).where(h < a, "D")),
        )
        df.loc[h.isna() | a.isna(), "FTR"] = pd.NA
    return df[MATCH_COLS]


# --- team-name normalisation ---------------------------------------------------

_DROP_TOKENS = {
    "fc", "cf", "afc", "sc", "ac", "cd", "cp", "ssc", "rc", "sd", "ca", "ud",
    "sv", "vfb", "vfl", "tsg", "fsv", "sg", "bsc", "kv", "rb",
    "club", "de", "futbol", "the", "calcio", "balompie", "cp",
    "1", "07", "05", "04", "09", "08", "06", "1899", "1846", "1904", "1900", "1846",
}
_ALIASES = {
    "man united": "manchester united", "man utd": "manchester united",
    "man city": "manchester city", "spurs": "tottenham hotspur", "tottenham": "tottenham hotspur",
    "wolves": "wolverhampton wanderers", "wolverhampton": "wolverhampton wanderers",
    "newcastle": "newcastle united", "west ham": "west ham united",
    "brighton": "brighton hove albion", "brighton and hove albion": "brighton hove albion",
    "nottm forest": "nottingham forest", "nottingham": "nottingham forest",
    "sheffield utd": "sheffield united", "leeds": "leeds united",
    "leicester": "leicester city", "norwich": "norwich city", "ipswich": "ipswich town",
    "luton": "luton town", "hull": "hull city", "stoke": "stoke city",
    "bayern": "bayern munich", "bayern munchen": "bayern munich",
    "dortmund": "borussia dortmund", "bvb": "borussia dortmund",
    "gladbach": "borussia monchengladbach", "monchengladbach": "borussia monchengladbach",
    "leverkusen": "bayer leverkusen", "hoffenheim": "tsg hoffenheim",
    "frankfurt": "eintracht frankfurt", "koln": "cologne", "fc koln": "cologne",
    "mainz": "mainz 05", "fsv mainz": "mainz 05", "mainz 05": "mainz 05",
    "rb leipzig": "leipzig", "rasenballsport leipzig": "leipzig",
    "borussia m gladbach": "borussia monchengladbach",
    "monchengladbach": "borussia monchengladbach", "gladbach": "borussia monchengladbach",
    "werder bremen": "werder bremen", "werder": "werder bremen",
    "elversberg": "elversberg", "paderborn": "paderborn", "hamburg": "hamburger",
    "hamburg sv": "hamburger", "hamburger sv": "hamburger",
    "st pauli": "st pauli", "heidenheim": "heidenheim",
    "atletico": "atletico madrid", "atletico de madrid": "atletico madrid",
    "athletic": "athletic club", "athletic bilbao": "athletic club",
    "real sociedad": "real sociedad", "betis": "real betis", "real betis balompie": "real betis",
    "villarreal": "villarreal", "sevilla": "sevilla", "celta": "celta vigo",
    "celta de vigo": "celta vigo", "deportivo alaves": "alaves", "cadiz": "cadiz",
    "espanyol": "espanyol", "espanol": "espanyol",
    "rayo": "rayo vallecano", "vallecano": "rayo vallecano", "girona": "girona",
    "las palmas": "las palmas", "leganes": "leganes", "valladolid": "real valladolid",
    "almeria": "almeria", "getafe": "getafe", "osasuna": "osasuna", "mallorca": "mallorca",
    # football-data.co.uk short names -> the canonical (Understat-ish) key
    "ath bilbao": "athletic club", "ath madrid": "atletico madrid",
    "sociedad": "real sociedad", "celta": "celta vigo", "alaves": "alaves",
    "man city": "manchester city", "man united": "manchester united",
    "nottm forest": "nottingham forest", "sheffield united": "sheffield united",
    "tottenham": "tottenham hotspur", "west ham": "west ham united",
    "wolves": "wolverhampton wanderers", "leicester": "leicester city",
    "leeds": "leeds united", "newcastle": "newcastle united",
    "dortmund": "borussia dortmund", "ein frankfurt": "eintracht frankfurt",
    "gladbach": "borussia monchengladbach", "leverkusen": "bayer leverkusen",
    "stuttgart": "stuttgart", "hoffenheim": "tsg hoffenheim", "koln": "cologne",
    "fc koln": "cologne", "hertha": "hertha berlin", "st pauli": "st pauli",
    "bielefeld": "arminia bielefeld", "greuther furth": "greuther furth",
}


def normalize_team(name: str) -> str:
    """A loose canonical key for cross-source joins - lowercase, no diacritics,
    no corporate tokens. Not a display name."""
    if not name:
        return ""
    s = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode()
    s = s.lower().replace("&", " and ").replace("-", " ").replace(".", " ")
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    toks = [t for t in s.split() if t and t not in _DROP_TOKENS]
    s = " ".join(toks).strip()
    s = _ALIASES.get(s, s)
    return _ALIASES.get(s, s)  # two hops: alias may map onto another alias key


def team_tokens(name: str) -> frozenset:
    return frozenset(normalize_team(name).split())


def teams_match(a: str, b: str) -> bool:
    """True if two team names plausibly refer to the same club - exact
    normalized, or a decisive token overlap (handles residual spelling drift)."""
    na, nb = normalize_team(a), normalize_team(b)
    if na and na == nb:
        return True
    ta, tb = set(na.split()), set(nb.split())
    if not ta or not tb:
        return False
    inter = ta & tb
    return bool(inter) and len(inter) >= min(len(ta), len(tb))
