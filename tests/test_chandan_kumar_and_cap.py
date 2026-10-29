"""Owner, 2026-10-29: five patients a day for every doctor; Dr. Chandan Kumar (gastroenterologist, every weekday) is full today and tomorrow, so the first free day is later."""

import datetime
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (REPO_ROOT, os.path.join(REPO_ROOT, "tests")):
    if p not in sys.path:
        sys.path.insert(0, p)

from _clinic_app import clinic_app  # noqa: E402


def test_five_a_day_and_chandan_is_full_today_and_tomorrow():
    with clinic_app(sample_patients=False) as (_app, client):
        import booking_service as bs
        import clinic_clock
        import clinic_extras
        from db import SessionLocal
        from models import Doctor, DoctorSchedule

        db = SessionLocal()
        try:
            clinic_extras.apply_extras(db)
            clinic_extras.apply_extras(db)  # idempotent: the second run adds nothing
            caps = {r.max_bookings_per_day for r in db.query(DoctorSchedule).all()}
            assert caps == {5}
            d = db.query(Doctor).filter_by(name="Dr. C. Kumar").first()
            assert d is not None and d.department.name == "Gastroenterology"
            assert {s.weekday for s in db.query(DoctorSchedule).filter_by(doctor_id=d.id).all()} == {0, 1, 2, 3, 4}
            today = clinic_clock.today()
            assert bs.available_slots(db, d.id, today.isoformat()) == []
            assert bs.available_slots(db, d.id, (today + datetime.timedelta(days=1)).isoformat()) == []
            first = bs.earliest_available(db, d.id, today)
            assert first is not None
            fd = datetime.date.fromisoformat(first["date"])
            assert fd >= today + datetime.timedelta(days=2) and fd.weekday() < 5  # a weekday, after tomorrow
            # another doctor still has today's/next free slots capped at five
            other = db.query(Doctor).filter_by(name="Dr. S. Mukherjee").first()
            for day in range(0, 8):
                dd = (today + datetime.timedelta(days=day)).isoformat()
                assert len(bs.available_slots(db, other.id, dd)) <= 5
        finally:
            db.close()
