"""Is the client's database connected, and does it have what the voice bot needs? (sourav, 2026-10-24)

Why: the client pastes their URL into clinic-api/client_database.env and wants to know, in plain words, whether the bot will now use THEIR data, and what is still missing.
Read-only: it only runs SELECT 1 and reads table / column names. It never creates, alters or deletes anything.

    python tools/check_client_db.py                 # uses clinic-api/client_database.env (or CLIENT_DATABASE_URL)
    python tools/check_client_db.py --url postgresql+psycopg2://user:pw@host/db
"""

from __future__ import annotations

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CLINIC = os.path.join(os.path.dirname(HERE), "clinic-api")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--url", help="check this URL instead of the one in client_database.env")
    a = ap.parse_args(argv)
    sys.path.insert(0, CLINIC)
    os.environ.setdefault("CLINIC_DB_PATH", os.path.join(CLINIC, "_check_unused.db"))
    import db as dbmod

    url = a.url or dbmod._client_database_url()
    if not url:
        print("No client database URL set (clinic-api/client_database.env is empty).")
        print("-> The voice bot is using OUR OWN data (local SQLite). Nothing to check.")
        return 0
    from sqlalchemy import inspect, text

    try:
        eng = dbmod._make_engine(url)
        with eng.connect() as conn:
            conn.execute(text("SELECT 1"))
    except Exception as e:  # noqa: BLE001
        print(f"NOT CONNECTED: {type(e).__name__}: {str(e)[:200]}")
        print("-> The voice bot keeps using OUR OWN data until this connects.")
        return 1
    print("CONNECTED. The voice bot will use the CLIENT'S data.")
    from models import Base

    insp = inspect(eng)
    have = set(insp.get_table_names())
    missing_tables, missing_cols = [], []
    for t in Base.metadata.sorted_tables:
        if t.name not in have:
            missing_tables.append(t.name)
            continue
        cols = {c["name"] for c in insp.get_columns(t.name)}
        missing_cols += [f"{t.name}.{c.name}" for c in t.columns if c.name not in cols]
    if not missing_tables and not missing_cols:
        print("Schema: every table and column the bot needs is present.")
        return 0
    if missing_tables:
        print(f"Missing tables ({len(missing_tables)}): {', '.join(missing_tables)}")
    if missing_cols:
        print(f"Missing columns ({len(missing_cols)}): {', '.join(missing_cols[:30])}{' ...' if len(missing_cols) > 30 else ''}")
    print("-> Nothing was changed. To let the API create what is missing, start it ONCE with CLIENT_DB_INIT=1 (it also seeds our sample catalogue if the database is empty).")
    return 2


if __name__ == "__main__":
    sys.exit(main())
