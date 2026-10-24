"""Show what the voice agent has recorded: every booking with patient, age, address / police station, doctor, day, time and confirmation number. (sourav, 2026-10-24)

Why: after a test call on Colab / RunPod you want proof that the booking was really logged. This reads the `booking_log` view (clinic-api/booking_log.sql) straight from the
clinic database -- the system of record, not a copy.

    python tools/booking_log.py                       # the latest 20 bookings, as a table
    python tools/booking_log.py --watch               # refresh every 3 seconds while you make a test call
    python tools/booking_log.py --csv out.csv         # export everything
    python tools/booking_log.py --db /workspace/clinic.db --limit 5

The view is created if missing, so this works on a database the API has not started on yet (as long as the tables exist).
"""

from __future__ import annotations

import argparse
import csv
import os
import sqlite3
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
SQL_PATH = os.path.join(os.path.dirname(HERE), "clinic-api", "booking_log.sql")
SHOW = ("confirmation_id", "booked_at", "patient_name", "age", "address", "doctor", "tests", "visit_date", "visit_day", "time_slot", "status", "paid", "serial_number")


def connect(path: str) -> sqlite3.Connection:
    """sourav: open the database read-write only to (re)create the view; every query after is a SELECT."""
    con = sqlite3.connect(path)
    con.row_factory = sqlite3.Row
    with open(SQL_PATH, encoding="utf-8") as f:
        con.executescript(f.read())
    return con


def fetch(con: sqlite3.Connection, limit: int | None) -> list[sqlite3.Row]:
    sql = "SELECT * FROM booking_log ORDER BY booked_at DESC"
    return con.execute(sql + (f" LIMIT {int(limit)}" if limit else "")).fetchall()


def table(rows: list[sqlite3.Row]) -> str:
    if not rows:
        return "(no bookings yet)"
    data = [[("" if r[c] is None else str(r[c])) for c in SHOW] for r in rows]
    widths = [max(len(c), *(len(row[i]) for row in data)) for i, c in enumerate(SHOW)]
    line = lambda cells: "  ".join(c.ljust(w) for c, w in zip(cells, widths))  # noqa: E731
    return "\n".join([line(SHOW), line(["-" * w for w in widths]), *(line(r) for r in data)])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--db", default=os.environ.get("CLINIC_DB_PATH", "/workspace/clinic.db"))
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--watch", action="store_true")
    ap.add_argument("--csv", metavar="FILE")
    a = ap.parse_args(argv)
    if not os.path.exists(a.db):
        print(f"database not found: {a.db}", file=sys.stderr)
        return 1
    con = connect(a.db)
    if a.csv:
        rows = fetch(con, None)
        with open(a.csv, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            if rows:
                w.writerow(rows[0].keys())
            w.writerows([tuple(r) for r in rows])
        print(f"{len(rows)} bookings written to {a.csv}")
        return 0
    while True:
        print(table(fetch(con, a.limit)))
        if not a.watch:
            return 0
        print(f"\n[{time.strftime('%H:%M:%S')}] watching -- Ctrl+C to stop\n")
        time.sleep(3)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        pass
