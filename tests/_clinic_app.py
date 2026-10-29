"""A real clinic API (its own SQLite file, seeded with the sample patients) for tests that need one.

    with clinic_app() as (app, client):       # client is a starlette TestClient; app is the ASGI app

The clinic-api modules (main, models, registry, ...) share names with nothing else, except `main`, which
is also the orchestrator: this loads clinic-api's copy, keeps the app object, and puts sys.path and
sys.modules back afterwards, so it can be used next to tests/_pod_stubs.py.
"""

import contextlib
import os
import sys
import tempfile

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CLINIC_MODULES = (
    "main",
    "db",
    "models",
    "seed",
    "booking_service",
    "lay_departments",
    "booking_migrate",
    "enquiry_service",
    "enquiry_migrate",
    "i18n_content",
    "phonetic_match",
    "patient_context",
    "disclosure",
    "registry",
    "agent_messages",
    "patient_seed",
    "gazetteer",
    "idempotency",
    "lab_days",  # sourav, 2026-10-24: new clinic-api modules, dropped like the rest so each app starts clean
    "booking_log",
)


def _pg_url(name: str) -> str:
    base = os.environ["PG_TEST_ADMIN_URL"]  # e.g. postgresql+psycopg2://clinic:PASSWORD@127.0.0.1:5432/postgres (the role needs CREATEDB)
    return base.rsplit("/", 1)[0] + "/" + name


def _pg_admin():
    import psycopg2

    url = os.environ["PG_TEST_ADMIN_URL"].replace("postgresql+psycopg2://", "postgresql://")
    conn = psycopg2.connect(url)
    conn.autocommit = True
    return conn


def _make_pg_database() -> str | None:
    if not os.environ.get("PG_TEST_ADMIN_URL"):
        return None
    import uuid

    name = "t_" + uuid.uuid4().hex[:12]
    conn = _pg_admin()
    try:
        conn.cursor().execute(f'CREATE DATABASE "{name}"')
    finally:
        conn.close()
    return name


def _drop_pg_database(name: str) -> None:
    try:
        import db as _db  # the app's engine must let go of its connections first

        _db.engine.dispose()
    except Exception:  # noqa: BLE001
        pass
    conn = _pg_admin()
    try:
        cur = conn.cursor()
        cur.execute("SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname = %s", (name,))
        cur.execute(f'DROP DATABASE IF EXISTS "{name}"')
    finally:
        conn.close()


@contextlib.contextmanager
def clinic_app(sample_patients: bool = True):
    fd, db_path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    saved_env = {k: os.environ.get(k) for k in ("CLINIC_DB_PATH", "DATABASE_URL", "CLINIC_SEED_SAMPLE_PATIENTS")}
    os.environ["CLINIC_DB_PATH"] = db_path
    os.environ.pop("DATABASE_URL", None)
    pg_name = _make_pg_database()  # PG_TEST_ADMIN_URL set (2026-10-29): the same tests run against PostgreSQL, each in a database of its own
    if pg_name:
        os.environ["DATABASE_URL"] = _pg_url(pg_name)
    os.environ["CLINIC_SEED_SAMPLE_PATIENTS"] = "1" if sample_patients else "0"
    saved_path = list(sys.path)
    saved_mods = {k: sys.modules.pop(k) for k in _CLINIC_MODULES if k in sys.modules}
    sys.path.insert(0, os.path.join(REPO_ROOT, "clinic-api"))
    try:
        from fastapi.testclient import TestClient

        import main as clinic_main

        with TestClient(clinic_main.app) as client:
            yield clinic_main.app, client
    finally:
        for k in _CLINIC_MODULES:
            sys.modules.pop(k, None)
        sys.modules.update(saved_mods)
        sys.path[:] = saved_path
        for k, v in saved_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        if pg_name:
            _drop_pg_database(pg_name)
        try:
            os.remove(db_path)
        except OSError:
            pass
