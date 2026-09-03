"""FotMob unofficial JSON API - stub.

FotMob's ToS now restricts programmatic use; kept as a low-volume fallback only.
Endpoints (subject to change, may need an ``x-mas`` signed header):
    https://www.fotmob.com/api/matches?date=YYYYMMDD
    https://www.fotmob.com/api/matchDetails?matchId=<id>   (shots, xG, possession)

Implement ``fetch(comp_codes, season, with_stats)`` to return a schema.MATCH_COLS
frame when a source above it goes dark.
"""

from __future__ import annotations

from .schema import blank_frame


def fetch(comp_codes: list[str], season: str, with_stats: bool = True) -> "object":
    raise NotImplementedError(
        "fotmob source not implemented - enable only as a fallback. See module docstring."
    )
    return blank_frame()  # noqa: unreachable - keeps the interface obvious
