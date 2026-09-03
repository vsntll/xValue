"""Current-season match ingestion - pluggable unofficial sources behind one schema.

See src/pull_live.py for the CLI. Each source module (espn, football_data_org,
fotmob, sofascore) exposes ``fetch(comp_codes, season, with_stats) -> DataFrame``
with columns from ``schema.MATCH_COLS``.
"""
