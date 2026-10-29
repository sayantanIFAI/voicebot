"""The booking log: see, in one place, that a call really recorded the booking. (sourav, 2026-10-24)

Why: the owner runs the voice bot on Colab / RunPod, makes a booking, and wants to SEE that the patient name, age, police station / address, doctor, day, time and the
automatically generated confirmation number were actually recorded. Two things make that visible:
  1. the SQL view `booking_log` (booking_log.sql) -- one readable row per booking, joined from the real tables (never a copy, so it cannot disagree with them);
  2. an append-only CSV file, one line written the moment a booking is confirmed, openable in Excel / Colab (`BOOKING_LOG_PATH`, default `booking_log.csv` next to the database;
     set BOOKING_LOG_PATH=off to switch the file off).
The command-line viewer is tools/booking_log.py.

Safety: logging is a side effect and must NEVER break a booking. Every public function here swallows its own errors (they are logged as warnings), and nothing in the booking
path reads what these write. The police station is whatever the caller said as part of the patient's address -- the patient table has no separate police-station column.
"""

from __future__ import annotations

import contextlib
import csv
import logging
import os
import re
import threading

from sqlalchemy import text

logger = logging.getLogger("clinic-api.booking_log")

SQL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "booking_log.sql")
COLUMNS = (
    "kind", "confirmation_id", "booked_at", "patient_name", "age", "sex", "address", "patient_phone", "caller_phone", "doctor", "department", "tests",
    "visit_date", "visit_day", "time_slot", "status", "paid", "serial_number",
)  # fmt: skip
_lock = threading.Lock()


def for_dialect(sql: str, dialect: str) -> str:
    """The view is written once, in SQLite's words (booking_log.sql). For PostgreSQL (2026-10-29) the two SQLite-only functions are spelled the PostgreSQL way: strftime('%w', d) is the day of
    the week 0-6 from Sunday, GROUP_CONCAT(x, ', ') is STRING_AGG(x, ', ')."""
    if dialect != "postgresql":
        return sql
    sql = re.sub(r"strftime\(\s*'%+w'\s*,\s*([A-Za-z_.]+)\s*\)", r"EXTRACT(DOW FROM \1::date)::int::text", sql)
    return sql.replace("GROUP_CONCAT(", "STRING_AGG(")


def _statements(dialect: str = "sqlite") -> list[str]:
    """sourav: booking_log.sql split into statements (comment lines dropped). Flow: the file is the single definition of the view; this only feeds it to the database."""
    with open(SQL_PATH, encoding="utf-8") as f:
        body = "\n".join(line for line in f.read().splitlines() if not line.strip().startswith("--"))
    return [for_dialect(s.strip(), dialect) for s in body.split(";") if s.strip()]


def ensure_view(engine) -> bool:
    """sourav, 2026-10-24: (re)create the `booking_log` view. Called at API start. Why: DROP + CREATE keeps the view in step with this file after an update; the view holds no data, so
    there is nothing to lose. Returns False (and logs) if the database is not SQLite or anything fails -- the API starts regardless."""
    try:
        with engine.begin() as conn:
            for stmt in _statements(engine.dialect.name):
                conn.execute(text(stmt))
        return True
    except Exception as e:  # noqa: BLE001 - never stop the API for a convenience view
        logger.warning("booking_log view not created: %s", e)
        return False


def _ensure_view_in(db) -> None:
    """sourav, 2026-10-24: create the view through the SESSION that is booking (a second connection would be locked out by SQLite while this session holds a read).
    Why: the sample-patient seed books during start-up, before ensure_view has run; the first log line then creates the view itself instead of being lost."""
    for stmt in _statements(db.get_bind().dialect.name):
        db.execute(text(stmt))
    db.commit()


def log_path() -> str | None:
    """sourav: where the CSV goes, or None when switched off. Default: next to the SQLite file (CLINIC_DB_PATH), else the working directory."""
    configured = os.environ.get("BOOKING_LOG_PATH")
    if configured and configured.lower() in ("off", "none", "0", "false"):
        return None
    if configured:
        return configured
    db_path = os.environ.get("CLINIC_DB_PATH", "/workspace/clinic.db")
    return os.path.join(os.path.dirname(os.path.abspath(db_path)) or ".", "booking_log.csv")


def rows_for(db, confirmation_id: str) -> list[dict]:
    """sourav: the view's row(s) for one confirmation number."""
    result = db.execute(text("SELECT * FROM booking_log WHERE confirmation_id = :c"), {"c": confirmation_id})
    return [dict(r._mapping) for r in result]


def append(db, confirmation_id: str) -> None:
    """sourav, 2026-10-24: write the line for a booking that was just confirmed. Flow: read the row back from the view (so the file shows what the DATABASE holds, not what the caller
    asked for), append it to the CSV (header on a new file), under a lock. Any error is logged and dropped: the booking has already succeeded."""
    path = log_path()
    if path is None:
        return
    try:
        try:
            rows = rows_for(db, confirmation_id)
        except Exception:  # noqa: BLE001 - the view does not exist yet (first booking after a fresh start): make it, then read again
            db.rollback()
            import db as _db_module

            if _db_module.USING_CLIENT_DB and not _db_module.CLIENT_DB_INIT:
                return  # sourav: never create objects in the client's database on our own
            _ensure_view_in(db)
            rows = rows_for(db, confirmation_id)
        if not rows:
            return
        with _lock:
            new = not os.path.exists(path) or os.path.getsize(path) == 0
            with open(path, "a", newline="", encoding="utf-8-sig") as f:  # utf-8-sig: Excel shows Bengali names correctly
                w = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
                if new:
                    w.writeheader()
                for r in rows:
                    w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in COLUMNS})
    except Exception as e:  # noqa: BLE001 - logging must never break a booking
        logger.warning("booking log not written for %s: %s", confirmation_id, e)
        with contextlib.suppress(Exception):
            db.rollback()  # PostgreSQL leaves the session's transaction unusable after an error; the booking itself is already committed
