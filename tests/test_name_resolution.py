"""live/schema.py name resolution - the cross-source team-name join every
pipeline step leans on (football-data, FBref, ESPN, FotMob, Understat spell
clubs differently)."""

from __future__ import annotations

import pytest

from live.schema import deaccent, normalize_team, teams_match


@pytest.mark.parametrize("names,key", [
    (["Man Utd", "Man United", "Manchester United", "Manchester United FC"], "manchester united"),
    (["Wolves", "Wolverhampton", "Wolverhampton Wanderers"], "wolverhampton wanderers"),
    (["Nott'm Forest", "Nottingham Forest"], "nottingham forest"),
    (["Brighton", "Brighton & Hove Albion", "Brighton and Hove Albion"], "brighton hove albion"),
    (["Bayern München", "FC Bayern München", "Bayern Munich"], "bayern munich"),
    (["Borussia M.Gladbach", "Borussia Mönchengladbach", "Gladbach"], "borussia monchengladbach"),
    (["RB Leipzig", "RasenBallsport Leipzig"], "leipzig"),
    (["Sport-Club Freiburg", "SC Freiburg"], "freiburg"),
    (["1. FC Köln", "FC Köln"], "cologne"),
    (["Paris SG", "Paris Saint-Germain", "Paris Saint Germain"], "paris saint germain"),
    (["Inter", "Internazionale"], "inter"),
    (["Atlético Madrid", "Atletico Madrid"], "atletico madrid"),
])
def test_spellings_resolve_to_one_key(names, key):
    assert {normalize_team(n) for n in names} == {key}


def test_paris_fc_is_not_psg():
    # "Paris FC" must not collapse onto PSG once corporate tokens are dropped
    assert normalize_team("Paris FC") != normalize_team("Paris SG")
    assert not teams_match("Paris FC", "Paris Saint-Germain")


def test_same_city_clubs_stay_apart():
    assert not teams_match("Manchester City", "Manchester United")


def test_teams_match_handles_spelling_drift():
    assert teams_match("Wolves", "Wolverhampton Wanderers")
    assert teams_match("Tottenham", "Tottenham Hotspur")


def test_empty_and_accents():
    assert normalize_team("") == ""
    assert normalize_team(None) == ""
    assert deaccent("Mönchengladbach") == "Monchengladbach"
