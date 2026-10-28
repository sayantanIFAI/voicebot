"""Doctor appointments can be taken at most 3 days ahead (booking AND rescheduling); a later date is refused and the caller is asked to call again nearer the day (sourav, 2026-10-24).
Also: the client-database file keeps the API on our own data until a URL is set AND reachable.

    python -m pytest tests/test_booking_window.py -q
"""

import datetime
import os
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (REPO_ROOT, os.path.join(REPO_ROOT, "tests")):
    if p not in sys.path:
        sys.path.insert(0, p)

from _clinic_app import clinic_app  # noqa: E402
from test_orchestrator_booking_flow import FakeTools, env, m  # noqa: E402,F401

from agent import visit_cues as v  # noqa: E402
from agent.reply_templates import booking_reply, reschedule_reply, too_far_ahead_reply  # noqa: E402

FAR = {"success": False, "reason": "too_far_ahead", "max_advance_days": 3, "last_bookable_date": "2026-10-27"}


@pytest.mark.parametrize("lang,must", [("en", "up to three days ahead"), ("bn", "তিন দিন আগে পর্যন্ত"), ("hi", "तीन दिन पहले तक")])
def test_the_refusal_is_worded_in_all_three_languages_for_booking_and_rescheduling(lang, must):
    assert must in booking_reply({}, FAR, lang)
    resched = reschedule_reply(FAR, lang)
    assert must in resched
    assert any(w in resched for w in ("unchanged", "আগের মতোই", "वैसी ही"))  # the old appointment stays
    assert "call" in too_far_ahead_reply(FAR, "en") and "ফোন" in too_far_ahead_reply(FAR, "bn") and "फ़ोन" in too_far_ahead_reply(FAR, "hi")


def test_the_number_of_days_comes_from_the_clinic_answer_not_from_the_wording():
    assert "five days" in too_far_ahead_reply({"max_advance_days": 5}, "en")


# ----------------------------------------------------------------------------------------------------------------------------------------------- backend
def _sitting_days(db, doctor_id):
    from models import DoctorSchedule

    return {s.weekday for s in db.query(DoctorSchedule).filter_by(doctor_id=doctor_id).all()}


def test_the_window_is_today_plus_three_days_for_hold_and_for_reschedule():
    with clinic_app(sample_patients=False) as (_app, client):
        import booking_service as bs
        from db import SessionLocal
        from models import Doctor, DoctorSchedule

        db = SessionLocal()
        try:
            sched = db.query(DoctorSchedule).first()
            doctor = db.get(Doctor, sched.doctor_id)
            sitting = _sitting_days(db, doctor.id)
            today = datetime.date.today()
            inside = [today + datetime.timedelta(days=i) for i in range(0, 4) if (today + datetime.timedelta(days=i)).weekday() in sitting]
            outside = [today + datetime.timedelta(days=i) for i in range(4, 15) if (today + datetime.timedelta(days=i)).weekday() in sitting]
            slot = bs._generate_slots(sched.start_time, sched.end_time)[-1]
        finally:
            db.close()
        # beyond the window: refused, nothing held
        far = outside[0]
        r = client.post("/api/v1/bookings/hold", json={"doctor_name": doctor.name, "date": far.isoformat(), "time_slot": slot}).json()
        assert r["success"] is False and r["reason"] == "too_far_ahead" and r["max_advance_days"] == 3
        # the last allowed day (today + 3) is accepted when the doctor sits (otherwise it is the usual not-sitting answer, never too_far_ahead)
        edge = today + datetime.timedelta(days=3)
        r = client.post("/api/v1/bookings/hold", json={"doctor_name": doctor.name, "date": edge.isoformat(), "time_slot": slot}).json()
        assert r.get("reason") != "too_far_ahead"
        if edge.weekday() in sitting:
            assert r["success"] is True
        # day four is refused even though everything else is fine
        day4 = today + datetime.timedelta(days=4)
        r = client.post("/api/v1/bookings/hold", json={"doctor_name": doctor.name, "date": day4.isoformat(), "time_slot": slot}).json()
        assert r["reason"] == "too_far_ahead"
        # rescheduling: an existing booking cannot be moved beyond the window; it stays where it was
        db = SessionLocal()
        try:
            day = inside[0]
            slots = bs.available_slots(db, doctor.id, day.isoformat())
            held = bs.hold_slot(db, doctor.id, day.isoformat(), slots[0])
            done = bs.confirm_booking(db, held["hold_token"], doctor.id, day.isoformat(), slots[0], "Pt One", "9811111111", "9811111111", 30)
            assert done["success"]
        finally:
            db.close()
        r = client.post("/api/v1/bookings/reschedule", json={"confirmation_id": done["confirmation_id"], "new_date": far.isoformat(), "new_time_slot": slot, "caller_phone": "9811111111"}, headers={"Idempotency-Key": "w1"}).json()
        assert r["success"] is False and r["reason"] == "too_far_ahead"
        db = SessionLocal()
        try:
            from models import Appointment

            assert db.query(Appointment).filter_by(confirmation_id=done["confirmation_id"]).one().date == day.isoformat()
        finally:
            db.close()


def test_lab_test_booking_is_not_limited():
    with clinic_app(sample_patients=False) as (_app, client):
        far = (datetime.date.today() + datetime.timedelta(days=20)).isoformat()
        r = client.post("/api/v1/bookings/tests", json={"test_names": ["CBC"], "date": far, "patient_name": "Pt", "phone": "9811111111", "caller_phone": "9811111111"}, headers={"Idempotency-Key": "w2"}).json()
        assert r.get("reason") != "too_far_ahead"


# ----------------------------------------------------------------------------------------------------------------------------------- the real turn handler
class _Tools(FakeTools):
    async def hold_slot(self, doctor, date, slot):
        return dict(FAR)

    async def reschedule_appointment(self, confirmation_id, new_date, new_time_slot):
        return dict(FAR)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "lang,must,opening,date_said",
    [("en", "three days ahead", "I want to book", "twentieth of December at ten"), ("bn", "তিন দিন আগে পর্যন্ত", "আমি বুক করতে চাই", "বিশে ডিসেম্বর সকাল দশটায়"), ("hi", "तीन दिन पहले तक", "मुझे बुक करना है", "बीस दिसंबर सुबह दस बजे")],
)
async def test_a_far_date_in_a_booking_is_refused_in_the_callers_language_and_the_date_is_asked_again(env, m, lang, must, opening, date_said):
    env.state["lang"] = lang
    env.session.lang = lang
    env.session.disclosed_langs.add(lang)
    m._tools = _Tools()
    await env.say(opening, "book_appointment", {"doctor_name": "Sen"})
    out = " ".join(await env.say(date_said, "book_appointment", {"date": "2026-12-20", "time_slot": "10:00"}))
    assert must in out, out
    assert "date" not in env.session.booking.slots  # cleared, so a nearer day can be given


@pytest.mark.parametrize("t", ["koto din age booking hoy", "kitne din pehle appointment mil jata hai", "how many days in advance can I book", "next month er booking hobe", "কত দিন আগে থেকে বুক করা যায়", "अगले महीने की बुकिंग हो जाएगी"])
def test_how_far_ahead_questions_are_recognised(t):
    assert v.asks_booking_window(t)


@pytest.mark.parametrize("t", ["I want to book an appointment", "doctor sen kobe bose", "booking korbo aj"])
def test_an_ordinary_booking_sentence_is_not_a_window_question(t):
    assert not v.asks_booking_window(t)


@pytest.mark.asyncio
@pytest.mark.parametrize("lang,text,must", [("en", "how many days in advance can I book", "three days ahead"), ("bn", "কত দিন আগে থেকে বুক করা যায়", "তিন দিন আগে পর্যন্ত"), ("hi", "कितने दिन पहले बुक कर सकते हैं", "तीन दिन पहले तक")])
async def test_the_window_question_gets_the_rule(env, m, lang, text, must):
    env.state["lang"] = lang
    env.session.lang = lang
    env.session.disclosed_langs.add(lang)
    assert must in " ".join(await env.say(text))


# ------------------------------------------------------------------------------------------------------------------- the client database file
def _fresh_db_module(monkeypatch, tmp_path, url_line, env_url=None):
    import importlib.util

    f = tmp_path / "client_database.env"
    f.write_text(f"# comment\nCLIENT_DATABASE_URL={url_line}\n")
    monkeypatch.setenv("CLIENT_DB_FILE", str(f))
    monkeypatch.setenv("CLINIC_DB_PATH", str(tmp_path / "ours.db"))
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("CLIENT_DATABASE_URL", raising=False)
    if env_url:
        monkeypatch.setenv("CLIENT_DATABASE_URL", env_url)
    spec = importlib.util.spec_from_file_location("db_probe", os.path.join(REPO_ROOT, "clinic-api", "db.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_an_empty_client_file_keeps_our_own_data(monkeypatch, tmp_path):
    mod = _fresh_db_module(monkeypatch, tmp_path, "")
    assert not mod.USING_CLIENT_DB and mod.DATABASE_URL.endswith("ours.db")


def test_a_reachable_client_url_is_used(monkeypatch, tmp_path):
    mod = _fresh_db_module(monkeypatch, tmp_path, f"sqlite:///{tmp_path / 'client.db'}")
    assert mod.USING_CLIENT_DB and mod.DATABASE_URL.endswith("client.db")


def test_an_unreachable_client_url_falls_back_to_our_data_and_never_stops_the_api(monkeypatch, tmp_path):
    mod = _fresh_db_module(monkeypatch, tmp_path, "postgresql+psycopg2://nobody:x@127.0.0.1:1/none")
    assert not mod.USING_CLIENT_DB and mod.DATABASE_URL.endswith("ours.db")


def test_the_environment_variable_works_too_and_an_explicit_database_url_still_wins(monkeypatch, tmp_path):
    mod = _fresh_db_module(monkeypatch, tmp_path, "", env_url=f"sqlite:///{tmp_path / 'e.db'}")
    assert mod.USING_CLIENT_DB
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'forced.db'}")
    import importlib.util

    spec = importlib.util.spec_from_file_location("db_probe2", os.path.join(REPO_ROOT, "clinic-api", "db.py"))
    mod2 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod2)
    assert mod2.DATABASE_URL.endswith("forced.db") and not mod2.USING_CLIENT_DB
