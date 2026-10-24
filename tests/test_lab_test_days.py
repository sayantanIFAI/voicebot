"""The weekdays a lab test is done on (models.LabTest.available_days), sourav 2026-10-24.

"CT scan of the brain is done every day except Thursday" is stored as "0,1,2,4,5,6" and answered from the table. These tests cover the pure day rules, the migration of an
existing database, the seed, and the HTTP endpoint the voice agent calls.

    python -m pytest tests/test_lab_test_days.py -v
"""

import datetime
import os
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (os.path.join(REPO_ROOT, "tests"),):  # sourav: clinic-api is NOT put on sys.path permanently (other files would then import the wrong `main`); lab_days is loaded by file path below
    if p not in sys.path:
        sys.path.insert(0, p)

import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location("lab_days", os.path.join(REPO_ROOT, "clinic-api", "lab_days.py"))
lab_days = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lab_days)
from _clinic_app import clinic_app  # noqa: E402

THURSDAY = datetime.date(2026, 10, 29)  # a Thursday
FRIDAY = datetime.date(2026, 10, 30)


def test_the_example_every_day_except_thursday():
    days = lab_days.parse_days("0,1,2,4,5,6")
    assert lab_days.closed_days(days) == [3]
    assert not lab_days.is_open_on(days, THURSDAY) and lab_days.is_open_on(days, FRIDAY)
    assert lab_days.next_open_date(days, THURSDAY, include_from=False) == FRIDAY
    assert lab_days.next_open_date(days, FRIDAY) == FRIDAY


@pytest.mark.parametrize("raw", [None, "", "   ", "x,y", "7,8,-1", ",,,"])
def test_an_empty_or_unreadable_column_means_every_day_so_no_test_goes_missing_by_a_typo(raw):
    assert lab_days.parse_days(raw) == [0, 1, 2, 3, 4, 5, 6]


def test_parsing_is_tolerant_sorted_and_without_repeats():
    assert lab_days.parse_days("4, 2,2 ;0") == [0, 2, 4]
    assert lab_days.format_days([4, 0, 2, 2]) == "0,2,4"
    assert lab_days.format_days("6,5") == "5,6"


def test_the_default_is_every_day():
    assert lab_days.DEFAULT_DAYS == "0,1,2,3,4,5,6" and lab_days.parse_days(lab_days.DEFAULT_DAYS) == list(lab_days.ALL_DAYS)


def test_next_open_date_walks_forward_to_the_next_listed_weekday():
    only_monday = [0]
    assert lab_days.next_open_date(only_monday, THURSDAY, include_from=False) == datetime.date(2026, 11, 2)


@pytest.fixture(scope="module")
def client():
    with clinic_app() as (_app, c):
        yield c


def test_every_seeded_test_has_the_column_and_an_unrestricted_test_is_done_every_day(client):
    r = client.get("/api/v1/tests/availability", params={"name": "Lipid Profile"}).json()
    assert r["found"] and r["every_day"] and r["available_days"] == [0, 1, 2, 3, 4, 5, 6] and r["closed_days"] == []


def test_a_restricted_test_lists_its_days(client):
    r = client.get("/api/v1/tests/availability", params={"name": "TMT"}).json()
    assert r["found"] and r["available_days"] == [0, 2, 4] and r["closed_days"] == [1, 3, 5, 6] and not r["every_day"]


def test_a_date_is_answered_yes_or_no_with_the_next_day_it_is_done(client):
    monday, tuesday = "2026-10-26", "2026-10-27"
    ok = client.get("/api/v1/tests/availability", params={"name": "TMT", "date": monday}).json()
    assert ok["available_on_date"] is True and ok["next_available_date"] == monday
    no = client.get("/api/v1/tests/availability", params={"name": "TMT", "date": tuesday}).json()
    assert no["available_on_date"] is False and no["next_available_date"] == "2026-10-28"  # Wednesday


def test_a_bad_date_is_rejected_not_guessed(client):
    assert client.get("/api/v1/tests/availability", params={"name": "TMT", "date": "tomorrow"}).status_code == 422


def test_an_unknown_test_is_not_found_with_the_usual_suggestions(client):
    r = client.get("/api/v1/tests/availability", params={"name": "zzzzqqqq"}).json()
    assert r["found"] is False


def test_the_existing_search_answer_is_unchanged(client):
    r = client.get("/api/v1/tests/search", params={"name": "TMT"}).json()
    assert r["found"] and "available_days" not in r  # nothing that already read this answer sees a new field


def test_an_owner_edit_of_the_column_is_served_and_is_not_overwritten_by_the_backfill(client):
    import db as db_mod
    import seed
    from models import LabTest

    db = db_mod.SessionLocal()
    try:
        t = db.query(LabTest).filter(LabTest.name.like("ECG%")).first()
        t.available_days = lab_days.format_days([0, 1, 2, 4, 5, 6])  # every day except Thursday
        db.commit()
    finally:
        db.close()
    r = client.get("/api/v1/tests/availability", params={"name": "ECG", "date": THURSDAY.isoformat()}).json()
    assert r["closed_days"] == [3] and r["available_on_date"] is False and r["next_available_date"] == FRIDAY.isoformat()
    seed.backfill_test_days()  # runs on every boot: must leave the edit (and TMT, already restricted) alone
    r = client.get("/api/v1/tests/availability", params={"name": "ECG"}).json()
    assert r["closed_days"] == [3]


def test_an_existing_database_gets_the_column_and_the_sample_days():
    """A database seeded before this column existed: the migration adds it (every day), the backfill then gives the sample days to the tests that have them."""
    with clinic_app() as (_app, c):
        import db as db_mod  # imported inside: clinic_app() loads a fresh copy of the clinic modules pointing at its own temporary database
        import enquiry_migrate
        import seed
        from sqlalchemy import text

        with db_mod.engine.begin() as conn:
            conn.execute(text("ALTER TABLE lab_tests DROP COLUMN available_days"))
        with db_mod.engine.connect() as conn:  # a plain PRAGMA, not an Inspector: an Inspector caches what it saw and would hide the column from the migration's own look
            assert "available_days" not in {row[1] for row in conn.execute(text("PRAGMA table_info(lab_tests)"))}
        assert "lab_tests.available_days" in enquiry_migrate.add_enquiry_columns()
        assert seed.backfill_test_days() == 2
        assert c.get("/api/v1/tests/availability", params={"name": "TMT"}).json()["available_days"] == [0, 2, 4]
        assert c.get("/api/v1/tests/availability", params={"name": "Lipid Profile"}).json()["every_day"] is True
