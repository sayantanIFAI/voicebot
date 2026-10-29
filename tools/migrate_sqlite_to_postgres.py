"""Copy the clinic database from SQLite to PostgreSQL, table by table, and check the counts.

    python tools/migrate_sqlite_to_postgres.py --sqlite /workspace/clinic.db --target postgresql+psycopg2://clinic:PW@127.0.0.1:5432/clinic --replace

* The target is created from the same models the API uses (nothing is seeded: the ids and links of your real data are kept exactly as they are in SQLite).
* `--replace` drops and recreates the target's tables first; without it the target must have no rows.
* The SQLite file is only READ. Take a snapshot first when the API is running (deploy/gateway/backup_db.sh) and point --sqlite at the snapshot.
* Integer id sequences are moved past the highest copied id so new rows do not collide.
* Prints one line per table (SQLite rows / PostgreSQL rows) and exits 1 if any differ.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "clinic-api"))

CHUNK = 1000


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sqlite", required=True)
    ap.add_argument("--target", required=True, help="a SQLAlchemy URL, postgresql+psycopg2://user:password@host:5432/database")
    ap.add_argument("--replace", action="store_true")
    args = ap.parse_args(argv)

    os.environ["DATABASE_URL"] = "sqlite:///" + args.sqlite  # models are imported with the SQLite engine; the target engine is built here
    from models import Base
    from sqlalchemy import create_engine, func, select, text

    src = create_engine("sqlite:///" + args.sqlite)
    dst = create_engine(args.target)
    if not args.target.startswith("postgresql"):
        print("the target is not PostgreSQL")
        return 2
    if args.replace:
        Base.metadata.drop_all(dst)
    Base.metadata.create_all(dst)

    known = {t.name for t in Base.metadata.sorted_tables}
    with src.connect() as c:
        extra = [r[0] for r in c.execute(text("select name from sqlite_master where type='table' and name not like 'sqlite_%'")) if r[0] not in known]
    if extra:
        print("NOT copied (not in the models):", ", ".join(extra))

    bad = 0
    for table in Base.metadata.sorted_tables:
        with src.connect() as s, dst.begin() as d:
            n_src = s.execute(select(func.count()).select_from(table)).scalar_one()
            if not args.replace and d.execute(select(func.count()).select_from(table)).scalar_one():
                print(f"{table.name}: the target already has rows (use --replace)")
                return 2
            result = s.execution_options(stream_results=True).execute(select(table))
            while True:
                rows = result.fetchmany(CHUNK)
                if not rows:
                    break
                d.execute(table.insert(), [dict(r._mapping) for r in rows])
            pk = [c for c in table.primary_key.columns if c.type.python_type is int and c.autoincrement is not False]
            if len(pk) == 1:
                col = pk[0].name
                d.execute(text(f'SELECT setval(pg_get_serial_sequence(\'"{table.name}"\', \'{col}\'), COALESCE((SELECT MAX("{col}") FROM "{table.name}"), 1), (SELECT MAX("{col}") IS NOT NULL FROM "{table.name}"))'))
        with dst.connect() as d:
            n_dst = d.execute(select(func.count()).select_from(table)).scalar_one()
        flag = "" if n_src == n_dst else "   <-- DIFFERENT"
        bad += n_src != n_dst
        print(f"{table.name:32s} sqlite {n_src:7d}   postgres {n_dst:7d}{flag}")
    print("RESULT:", "every table matches" if not bad else f"{bad} table(s) differ")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
