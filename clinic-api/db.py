"""DB session setup.

Defaults to SQLite ON THE NETWORK VOLUME, not Postgres. That is a
deliberate reversal, and the reasoning is worth keeping:

Postgres cannot live on /workspace. It is a MooseFS mount that reports
every file as root:root no matter what it is chowned to, and Postgres
refuses to start unless its data directory is owned by the postgres user
-- a check with no override. So Postgres could only ever run on the pod's
LOCAL overlay filesystem, which RunPod wipes on every restart. The result
was absurd: the clinic database, the one component whose entire job is to
remember things, was the only component in the stack that could not
survive a reboot. It was lost to three separate restarts, each time
needing a reinstall and a reseed.

SQLite has no ownership model to violate -- it is one file. It sits on
/workspace and persists. For this workload the things Postgres is better
at simply do not apply: a 74-row read-mostly catalogue, one writer
process, no concurrent writers, no replication.

Journal mode is left at the default (DELETE) rather than WAL on purpose.
WAL needs a shared-memory index file alongside the database, and that is
exactly the primitive a network filesystem is least reliable at. DELETE
journalling costs write throughput this workload does not need.

DATABASE_URL still overrides everything, so pointing this back at a real
Postgres for production is a one-line environment change.
"""

from __future__ import annotations

import os

from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker

import logging

DEFAULT_SQLITE_PATH = os.environ.get("CLINIC_DB_PATH", "/workspace/clinic.db")
LOCAL_DATABASE_URL = f"sqlite:///{DEFAULT_SQLITE_PATH}"

# ---- sourav, 2026-10-24: THE CLIENT'S OWN DATABASE, by one line in one file ------------------------------------------------------------------------------------------
# Why: the client should only have to paste their database URL into `client_database.env` (next to this file) -- or set CLIENT_DATABASE_URL -- and the clinic API reads and
# writes THEIR data from then on. Until a URL is there AND the database answers, nothing changes: the API keeps working on our own data (the local SQLite file above),
# exactly as before. Flow: DATABASE_URL (the existing override) wins; else the client URL if it connects (SELECT 1, 5 s); else the local SQLite file. A client URL that does
# not connect is logged and ignored -- the API never fails to start because of it. USING_CLIENT_DB tells main.py to NOT seed our sample catalogue or alter their tables
# unless CLIENT_DB_INIT=1 is set once on purpose (tools/check_client_db.py explains what is missing).
CLIENT_DB_FILE = os.environ.get("CLIENT_DB_FILE", os.path.join(os.path.dirname(os.path.abspath(__file__)), "client_database.env"))
_log = logging.getLogger("clinic-api.db")


def _client_url_from_file(path: str) -> str:
    """sourav: the value of CLIENT_DATABASE_URL in the small key=value file (comment lines start with #); empty when the file or the key is missing."""
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, _, value = line.partition("=")
                    if key.strip() == "CLIENT_DATABASE_URL":
                        return value.strip().strip("\"'")
    except OSError:
        pass
    return ""


def _client_database_url() -> str:
    return os.environ.get("CLIENT_DATABASE_URL", "").strip() or _client_url_from_file(CLIENT_DB_FILE)


def _make_engine(url: str):
    sqlite = url.startswith("sqlite")
    return create_engine(
        url,
        pool_pre_ping=True,
        # FastAPI serves requests from a thread pool, so the connection that
        # opens a session is not always the one that closes it.
        connect_args={"check_same_thread": False, "timeout": 30} if sqlite else {"connect_timeout": 5},
    )


def _connects(url: str) -> bool:
    from sqlalchemy import text

    try:
        eng = _make_engine(url)
        with eng.connect() as conn:
            conn.execute(text("SELECT 1"))
        eng.dispose()
        return True
    except Exception as e:  # noqa: BLE001 - any driver / network / credential problem means "not connected"
        _log.warning("client database not reachable, using the local data instead: %s", type(e).__name__)
        return False


USING_CLIENT_DB = False
if os.environ.get("DATABASE_URL"):
    DATABASE_URL = os.environ["DATABASE_URL"]
else:
    _client_url = _client_database_url()
    if _client_url and _connects(_client_url):
        DATABASE_URL, USING_CLIENT_DB = _client_url, True
        _log.info("using the CLIENT database")
    else:
        DATABASE_URL = LOCAL_DATABASE_URL
CLIENT_DB_INIT = os.environ.get("CLIENT_DB_INIT", "") == "1"  # one deliberate run that may create missing tables / columns and seed an EMPTY client database

_is_sqlite = DATABASE_URL.startswith("sqlite")

engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    # FastAPI serves requests from a thread pool, so the connection that
    # opens a session is not always the one that closes it.
    connect_args={"check_same_thread": False, "timeout": 30} if _is_sqlite else {},
)

if _is_sqlite:

    @event.listens_for(engine, "connect")
    def _sqlite_pragmas(dbapi_conn, _record):
        cur = dbapi_conn.cursor()
        # Wait rather than raising "database is locked" the instant another
        # connection holds it -- on a network filesystem, lock handoff is
        # measured in milliseconds, not microseconds.
        cur.execute("PRAGMA busy_timeout=30000")
        cur.execute("PRAGMA foreign_keys=ON")
        cur.close()


SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
