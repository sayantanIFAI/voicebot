"""The booking log (sourav, 2026-10-24): a confirmed booking shows up with name, age, address / police station, doctor, day, time and its automatic confirmation number,
both in the `booking_log` view and in the CSV file; and logging can never break a booking.

    python -m pytest tests/test_booking_log.py -v
"""

import csv
import os
import sys
import tempfile

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# sourav: clinic-api is deliberately NOT added to sys.path (clinic_app() does it per test); a permanent entry makes other test files import the wrong `main`
for p in (os.path.join(REPO_ROOT, "tests"),):
    if p not in sys.path:
        sys.path.insert(0, p)

from _clinic_app import clinic_app  # noqa: E402




@pytest.fixture
def log_file(monkeypatch):
    path = os.path.join(tempfile.mkdtemp(), "booking_log.csv")
    monkeypatch.setenv("BOOKING_LOG_PATH", path)
    return path


def _make_booking(name, age, address):
    import booking_service as bs
    from db import SessionLocal
    from models import Doctor

    db = SessionLocal()
    try:
        doctor = db.query(Doctor).first()
        import datetime

        for off in range(1, 21):
            d = (bs._now().date() + datetime.timedelta(days=off)).isoformat()
            slots = bs.available_slots(db, doctor.id, d)
            if slots:
                held = bs.hold_slot(db, doctor.id, d, slots[0])
                done = bs.confirm_booking(db, held["hold_token"], doctor.id, d, slots[0], name, "9811111111", "9811111111", age, patient_address=address)
                return done, d, slots[0]
        raise AssertionError("no free slot found")
    finally:
        db.close()


def test_a_booking_is_logged_with_everything_the_agent_captured(log_file):
    with clinic_app(sample_patients=False):
        done, date, slot = _make_booking("Rina Das", 34, "Behala, Behala Police Station")
        assert done["success"]
        import booking_log
        from db import SessionLocal

        db = SessionLocal()
        try:
            row = booking_log.rows_for(db, done["confirmation_id"])[0]
        finally:
            db.close()
        assert (row["patient_name"], row["age"], row["address"]) == ("Rina Das", 34, "Behala, Behala Police Station")
        assert row["visit_date"] == date and row["time_slot"] == slot and row["doctor"] and row["visit_day"]
        assert row["confirmation_id"] == done["confirmation_id"] and row["paid"] == "no"
        with open(log_file, encoding="utf-8-sig") as f:
            lines = list(csv.DictReader(f))
        assert lines[-1]["confirmation_id"] == done["confirmation_id"] and lines[-1]["patient_name"] == "Rina Das"


def test_the_file_log_can_be_switched_off_and_a_bad_path_never_breaks_a_booking(monkeypatch):
    with clinic_app(sample_patients=False):
        monkeypatch.setenv("BOOKING_LOG_PATH", "off")
        assert _make_booking("A One", 30, "x")[0]["success"]
        monkeypatch.setenv("BOOKING_LOG_PATH", "/nonexistent-dir/deeper/log.csv")
        assert _make_booking("B Two", 31, "y")[0]["success"]


def test_the_command_line_viewer_reads_the_view(log_file):
    with clinic_app(sample_patients=False):
        done, _d, _s = _make_booking("Rina Das", 34, "Behala")
        import booking_log as cli  # noqa: F401
        import importlib.util

        spec = importlib.util.spec_from_file_location("booking_log_cli", os.path.join(REPO_ROOT, "tools", "booking_log.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        con = mod.connect(os.environ["CLINIC_DB_PATH"])
        text = mod.table(mod.fetch(con, 5))
        assert done["confirmation_id"] in text and "Rina Das" in text
