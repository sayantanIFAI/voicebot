"""Owner, 2026-10-29: a doctor booked only ON the day he sits ("appointment on the day" = true) is not booked for a later date; the caller is told when he sits and to phone at 7 am."""

import datetime
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (REPO_ROOT, os.path.join(REPO_ROOT, "tests")):
    if p not in sys.path:
        sys.path.insert(0, p)

from _clinic_app import clinic_app  # noqa: E402

from agent.reply_templates import book_on_the_day_reply  # noqa: E402


def _next_weekday(start: datetime.date, weekday_set: set[int]) -> datetime.date:
    d = start + datetime.timedelta(days=1)
    while d.weekday() not in weekday_set:
        d += datetime.timedelta(days=1)
    return d


def test_a_future_date_for_an_on_the_day_doctor_is_refused_with_his_next_sitting():
    with clinic_app(sample_patients=False) as (_app, client):
        import clinic_extras
        from db import SessionLocal
        from models import Doctor

        db = SessionLocal()
        try:
            clinic_extras.apply_extras(db)
            haq = db.query(Doctor).filter_by(name="Dr. Z. Haq").first()
            assert haq.appointment_on_the_day is True
            other = db.query(Doctor).filter_by(name="Dr. S. Mukherjee").first()
            assert other.appointment_on_the_day is False
        finally:
            db.close()
        target = _next_weekday(datetime.date.today(), {0, 1, 2, 3, 4})
        r = client.post("/api/v1/bookings/hold", json={"doctor_name": "Dr. Ziaul Haq", "date": target.isoformat(), "time_slot": "10:00"}, headers={"Idempotency-Key": "otd1"}).json()
        assert r["success"] is False and r["reason"] == "book_on_the_day"
        assert r["sitting_date"] == target.isoformat() and r["opens_at"] == "07:00" and r["is_today"] is False
        # a doctor not marked is not refused for THIS reason
        r2 = client.post("/api/v1/bookings/hold", json={"doctor_name": "Dr. Sourav Mukherjee", "date": target.isoformat(), "time_slot": "10:00"}, headers={"Idempotency-Key": "otd2"}).json()
        assert r2.get("reason") != "book_on_the_day"


def test_the_sentence_is_said_in_three_languages():
    res = {"doctor_name": "Dr. Ziaul Haq", "doctor_name_bn": "জিয়াউল হক", "doctor_name_hi": "ज़ियाउल हक़", "sitting_date": "2026-11-02", "is_today": False}
    assert "সকাল সাতটায় ফোন" in book_on_the_day_reply(res, "bn") and "জিয়াউল হক" in book_on_the_day_reply(res, "bn")
    assert "सुबह सात बजे" in book_on_the_day_reply(res, "hi")
    assert "7 AM" in book_on_the_day_reply(res, "en")
    assert "আগের মতোই" in book_on_the_day_reply(res, "bn", rescheduling=True)
