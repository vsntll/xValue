"""Sofascore unofficial JSON API - stub, third fallback.

Endpoints (subject to change; often need a browser-like User-Agent):
    https://api.sofascore.com/api/v1/sport/football/scheduled-events/YYYY-MM-DD
    https://api.sofascore.com/api/v1/event/<id>/statistics   (shots, possession, xG)

Implement ``fetch(comp_codes, season, with_stats)`` to return a schema.MATCH_COLS
frame if both ESPN and FotMob are unavailable.
"""

from __future__ import annotations

from .schema import blank_frame


def fetch(comp_codes: list[str], season: str, with_stats: bool = True) -> "object":
    raise NotImplementedError(
        "sofascore source not implemented - enable only as a fallback. See module docstring."
    )
    return blank_frame()  # noqa: unreachable
