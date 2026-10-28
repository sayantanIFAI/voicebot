"""Owner, 2026-10-28: Dr. Subrata Banerjee sits every day; a date next month is answered with the first day next month he sits, and only that."""

import datetime
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (REPO_ROOT, os.path.join(REPO_ROOT, "tests")):
    if p not in sys.path:
        sys.path.insert(0, p)

from _clinic_app import clinic_app  # noqa: E402

from agent.reply_templates import too_far_ahead_reply  # noqa: E402


def test_the_doctor_sits_on_all_seven_days():
    with clinic_app(sample_patients=False) as (_app, client):
        import clinic_extras
        from db import SessionLocal
        from models import Doctor, DoctorSchedule

        db = SessionLocal()
        try:
            clinic_extras.apply_extras(db)
            d = db.query(Doctor).filter_by(name="Dr. S. Banerjee").first()
            days = {s.weekday for s in db.query(DoctorSchedule).filter_by(doctor_id=d.id).all()}
            assert days == set(range(7))
            again = clinic_extras.apply_extras(db)
            assert again["weekdays_added"] == 0  # idempotent
        finally:
            db.close()


def test_a_date_next_month_gets_the_first_sitting_of_next_month():
    with clinic_app(sample_patients=False) as (_app, client):
        import clinic_extras
        import main as cm
        from db import SessionLocal
        from models import Doctor

        db = SessionLocal()
        try:
            clinic_extras.apply_extras(db)
            doc = db.query(Doctor).filter_by(name="Dr. S. Banerjee").first()
            today = datetime.date.today()
            first = (today.replace(day=1) + datetime.timedelta(days=32)).replace(day=1)
            asked = first + datetime.timedelta(days=9)
            r = cm._validate_not_too_far_ahead(asked.isoformat(), db, doc)
            assert r["reason"] == "too_far_ahead"
            assert r["next_month_first"]["date"] == first.isoformat()  # sits every day: the 1st
            # two months away, or this month beyond the window: the old answer, no next-month line
            far = first + datetime.timedelta(days=45)
            assert "next_month_first" not in cm._validate_not_too_far_ahead(far.isoformat(), db, doc)
        finally:
            db.close()


def test_the_reply_names_the_first_date_in_three_languages():
    r = {"max_advance_days": 3, "next_month_first": {"date": "2026-11-01", "start": "18:00", "end": "20:00"}}
    assert "2026-11-01" in too_far_ahead_reply(r, "en") and "first availability" in too_far_ahead_reply(r, "en")
    assert "2026-11-01" in too_far_ahead_reply(r, "bn") and "পরের মাসের" in too_far_ahead_reply(r, "bn")
    assert "2026-11-01" in too_far_ahead_reply(r, "hi") and "अगले महीने" in too_far_ahead_reply(r, "hi")
