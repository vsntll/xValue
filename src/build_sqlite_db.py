"""Sync data/processed/*.csv into a SQLite database for ad-hoc SQL queries.

Read-only mirror: every processed CSV becomes a same-named table, fully
replaced each run. The CSVs stay the pipeline's source of truth - nothing
else reads from or writes to this database.
"""
import sqlite3
from pathlib import Path

import pandas as pd

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
DB_PATH = PROCESSED_DIR / "xvalue.db"


def main() -> None:
    csv_paths = sorted(PROCESSED_DIR.glob("*.csv"))
    conn = sqlite3.connect(DB_PATH)
    try:
        for csv_path in csv_paths:
            table = csv_path.stem
            df = pd.read_csv(csv_path, low_memory=False)
            df.to_sql(table, conn, if_exists="replace", index=False)
            print(f"{table}: {len(df):,} rows")
    finally:
        conn.close()
    print(f"\nWrote {len(csv_paths)} tables to {DB_PATH}")


if __name__ == "__main__":
    main()
