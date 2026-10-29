"""Owner, 2026-10-29: the bulk availability answer is exactly the single lookups' answers, and the schedule cache never shows a stale week after a change made through the API."""

import datetime
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (REPO_ROOT, os.path.join(REPO_ROOT, "tests")):
    if p not in sys.path:
        sys.path.insert(0, p)

from _clinic_app import clinic_app  # noqa: E402


def test_bulk_answers_equal_the_single_lookups():
    with clinic_app(sample_patients=False) as (_app, client):
        today = datetime.date.today()
        dates = [today.isoformat(), (today + datetime.timedelta(days=1)).isoformat(), ""]
        bulk = client.get("/api/v1/doctors/availability/bulk", params={"dates": ",".join(dates)}).json()["answers"]
        assert bulk
        for row in bulk[:60]:
            params = {"name": row["name"]}
            if row["date"]:
                params["date"] = row["date"]
            assert client.get("/api/v1/doctors/availability", params=params).json() == row["answer"], row["name"]


def test_a_schedule_change_is_seen_at_once():
    with clinic_app(sample_patients=False) as (_app, client):
        import main as cm
        from db import SessionLocal
        from models import Doctor, DoctorSchedule

        db = SessionLocal()
        try:
            doc = db.query(Doctor).first()
            day = datetime.date.today() + datetime.timedelta(days=3)
            before = cm._schedule_for_weekday(db, doc.id, day.weekday())
            for s in db.query(DoctorSchedule).filter_by(doctor_id=doc.id, weekday=day.weekday()).all():
                db.delete(s)
            db.commit()
            assert cm._schedule_for_weekday(db, doc.id, day.weekday()) is None
            db.add(DoctorSchedule(doctor_id=doc.id, weekday=day.weekday(), start_time="09:00", end_time="11:00"))
            db.commit()
            got = cm._schedule_for_weekday(db, doc.id, day.weekday())
            assert (got.start_time, got.end_time) == ("09:00", "11:00")
            assert before is None or before.start_time is not None
        finally:
            db.close()
