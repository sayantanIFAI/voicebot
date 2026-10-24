"""Afternoon visit, "I want to show my report" (7-day rule) and "what will my serial be if I book today" (sourav, 2026-10-24).

English, Bengali and Hindi -- script, romanized and mixed, plain and lame sentences -- first as pure cues, then through the REAL turn handler, then the backend number.

    python -m pytest tests/test_visit_cues.py -q
"""

import os
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (REPO_ROOT, os.path.join(REPO_ROOT, "tests")):
    if p not in sys.path:
        sys.path.insert(0, p)

from test_orchestrator_booking_flow import FakeTools, env, m  # noqa: E402,F401

from agent import visit_cues as v  # noqa: E402
from agent.booking_flow import new_state  # noqa: E402

AFTERNOON = [
    "ami bikaler dike jabo", "ami bikel e asbo", "bikele jaabo", "dupure jabo", "sondhye jabo", "ami sondhyar dike jacchi", "I will come in the evening", "I'll go in the afternoon",
    "can I come in the afternoon", "main shaam ko aaunga", "main dopahar mein aaungi", "hum shaam ko aayenge", "mai sham ko ayunga", "आज शाम को आऊंगा", "दोपहर में जाऊंगा",
    "আমি বিকেলে যাব", "বিকালের দিকে আসব", "আমি দুপুরে আসছি", "সন্ধ্যায় আসব", "aaj sham ko jaunga doctor ke paas",
]  # fmt: skip
NOT_AFTERNOON = ["doctor sen bikel e koyta theke bose?", "bikel e doctor ache?", "shaam ko doctor kab baithte hain?", "bikel ta kemon", "what is the price of CBC", "ami jabo", "I want to book for the evening"]  # fmt: skip

REPORT = [
    "ami report dekhabo", "report dekhate chai", "doctor ke report dekhate hobe tini bolechilen", "amar report ta doctor ke dekhabo", "riport dekhabo", "reports dekhate asbo",
    "mujhe doctor ka report dikhana hai unhone bulaya tha", "main report dikhane aana hai", "report dikhana hai", "I want to show my report to the doctor", "I have to show the report",
    "doctor told me to come with the report to show", "রিপোর্ট দেখাতে আসব", "আমি রিপোর্ট দেখাব", "ডাক্তার রিপোর্ট নিয়ে আসতে বলেছিলেন", "रिपोर्ट दिखाने आना है", "मुझे रिपोर्ट दिखानी है",
    "डॉक्टर ने रिपोर्ट के साथ बुलाया था",
]  # fmt: skip
NOT_REPORT = ["is my report ready", "report collect korbo", "report nite asbo", "amar report pabo kobe", "रिपोर्ट तैयार है क्या", "রিপোর্ট রেডি", "send my report on whatsapp", "price of report", "doctor dekhabo"]  # fmt: skip

SERIAL_Q = [
    "aj book korle serial koto hobe", "ami ekhon book korle kon number pabo", "kal book korle serial koto", "today booking korle ki serial pabo", "jodi ekhon book kori tahole serial number koto",
    "if I book today what will my serial number be", "what will be my serial if I book now", "agar aaj book karun to serial number kya milega", "aaj booking karu to serial kitna hoga",
    "আজ বুক করলে সিরিয়াল কত হবে", "আজকে নিলে সিরিয়াল কোনটা পাব", "आज बुक करूं तो सीरियल कितना होगा",
]  # fmt: skip
NOT_SERIAL_Q = ["what is the serial number", "serial number ki", "serial number kaise milega booking ke baad", "I booked but did not get the serial", "booking korechi serial paini", "serial 5 hole kokhon asbo", "सीरियल नंबर क्या होता है", "সিরিয়াল নম্বর কি"]  # fmt: skip


@pytest.mark.parametrize("t", AFTERNOON)
def test_afternoon_plan_is_recognised(t):
    assert v.says_will_come_in_the_afternoon(t)


@pytest.mark.parametrize("t", NOT_AFTERNOON)
def test_afternoon_words_alone_or_a_question_are_not_the_plan(t):
    assert not v.says_will_come_in_the_afternoon(t)


@pytest.mark.parametrize("t", REPORT)
def test_showing_a_report_is_recognised(t):
    assert v.wants_to_show_a_report(t)


@pytest.mark.parametrize("t", NOT_REPORT)
def test_collecting_a_report_or_no_report_is_not_showing_one(t):
    assert not v.wants_to_show_a_report(t)


@pytest.mark.parametrize("t", SERIAL_Q)
def test_the_next_serial_question_is_recognised(t):
    assert v.asks_next_serial(t)


@pytest.mark.parametrize("t", NOT_SERIAL_Q)
def test_other_serial_sentences_keep_their_own_answers(t):
    assert not v.asks_next_serial(t)


@pytest.mark.parametrize(
    "t,yn,expected",
    [("ha", "yes", "old"), ("haan", "yes", "old"), ("na", "no", "recent"), ("nahi", "no", "recent"), ("3 din age", None, "recent"), ("পাঁচ দিন আগে", None, "recent"), ("छह दिन पहले", None, "recent"),
     ("10 days ago", None, "old"), ("সাত দিন আগে", None, "old"), ("ek mash age", None, "old"), ("दो हफ्ते पहले", None, "old"), ("2 weeks", None, "old"), ("kal", None, "recent"), ("পরশু", None, "recent"),
     ("कल ही", None, "recent"), ("onek din age", None, "old"), ("bahut pehle", None, "old"), ("hmm", None, None)],
)  # fmt: skip
def test_how_long_ago_is_read_from_a_yes_no_or_the_time_itself(t, yn, expected):
    assert v.visit_age_answer(t, yn) == expected


def test_every_reply_exists_in_all_three_languages():
    for lang in ("bn", "hi", "en"):
        for s in (v.afternoon_reply(lang), v.ask_visit_reply(lang), v.repeat_visit_reply(lang), v.old_visit_reply(lang), v.recent_visit_reply(lang), v.next_serial_reply("Sen", 4, "today", lang), v.next_serial_not_sitting_reply("Sen", "today", None, lang)):
            assert s and "{" not in s
    assert "keeping some time in hand" in v.afternoon_reply("en") and "serial" in v.afternoon_reply("en")
    assert "4" in v.next_serial_reply("Sen", 4, "today", "bn")


# ------------------------------------------------------------------------------------------------------------------------ the real turn handler
class _Tools(FakeTools):
    def __init__(self, base):
        self.__dict__.update(base.__dict__)
        self.available = True

    async def next_serial(self, name, date):
        return {"found": True, "doctor_name": "Dr. A. Sen", "doctor_name_bn": "ডাঃ এ. সেন", "doctor_name_hi": "डॉ. ए. सेन", "date": date, "available": self.available,
                "chamber_hours": "10:00-12:00" if self.available else None, "next_serial": 6 if self.available else None, "next_available_date": None if self.available else "2026-11-05"}  # fmt: skip


def _setup(env, m, lang):
    env.state["lang"] = lang
    env.session.lang = lang
    env.session.disclosed_langs.add(lang)
    env.session.last_doctor_enquiry = {"doctor_name": "Dr. A. Sen", "date": None, "turn": 0}
    t = _Tools(env.tools)
    m._tools = t
    return t


AFTERNOON_TURNS = [("en", "ami bikaler dike jabo", "keeping some time in hand"), ("en", "main shaam ko aaunga", "check your serial number"), ("bn", "আমি বিকেলে যাব", "হাতে সময় নিয়ে"), ("hi", "आज शाम को आऊंगा", "हाथ में समय लेकर")]


@pytest.mark.asyncio
@pytest.mark.parametrize("lang,text,must", AFTERNOON_TURNS)
async def test_afternoon_plan_gets_keep_time_in_hand_and_check_serial(env, m, lang, text, must):
    _setup(env, m, lang)
    out = " ".join(await env.say(text))
    assert must in out, out
    assert not env.session.ws.closed


@pytest.mark.asyncio
async def test_the_afternoon_answer_never_interrupts_a_booking(env, m):
    _setup(env, m, "en")
    st = new_state("book_appointment")
    st.stage = "collecting"
    env.session.booking = st
    out = " ".join(await env.say("ami bikel e jabo"))
    assert "keeping some time in hand" not in out


REPORT_TURNS = [("en", "ami report dekhabo", "yes", "book an appointment and pay"), ("en", "mujhe doctor ka report dikhana hai unhone bulaya tha", "3 din age", "show the report. Please check your serial"),
                ("bn", "আমি রিপোর্ট দেখাব", "হ্যাঁ", "পেমেন্ট করে আসতে হবে"), ("bn", "ডাক্তার রিপোর্ট নিয়ে আসতে বলেছিলেন", "দুই দিন আগে", "রিপোর্ট দেখাতে আসতে পারেন"),
                ("hi", "मुझे रिपोर्ट दिखानी है", "हाँ", "पेमेंट करके आना होगा"), ("hi", "डॉक्टर ने रिपोर्ट के साथ बुलाया था", "दो दिन पहले", "रिपोर्ट दिखाने आ सकते हैं")]  # fmt: skip


@pytest.mark.asyncio
@pytest.mark.parametrize("lang,ask,answer,must", REPORT_TURNS)
async def test_showing_a_report_asks_the_visit_age_then_applies_the_seven_day_rule(env, m, lang, ask, answer, must):
    _setup(env, m, lang)
    first = " ".join(await env.say(ask))
    assert ("seven days" in first) or ("সাত দিন" in first) or ("सात दिन" in first), first
    assert env.session.report_visit_asked is not None
    second = " ".join(await env.say(answer))
    assert must in second, second
    assert env.session.report_visit_asked is None and not env.session.ws.closed


@pytest.mark.asyncio
async def test_an_unclear_visit_answer_is_asked_once_more_then_let_go(env, m):
    _setup(env, m, "en")
    await env.say("I want to show my report")
    again = " ".join(await env.say("I do not remember exactly"))
    assert "yes or no" in again and env.session.report_visit_asked is not None


@pytest.mark.asyncio
async def test_a_yes_to_the_visit_question_does_not_end_the_call(env, m):
    _setup(env, m, "en")
    await env.say("report dekhabo")
    await env.say("ok")  # a bare 'ok' with the question open is an answer attempt, not a goodbye
    assert not env.session.ws.closed


SERIAL_TURNS = [("en", "if I book today what will my serial number be", "serial number will be 6"), ("en", "aj book korle serial koto hobe", "serial number will be 6"),
                ("bn", "আজ বুক করলে সিরিয়াল কত হবে", "সিরিয়াল নম্বর হবে 6"), ("hi", "आज बुक करूं तो सीरियल कितना होगा", "सीरियल नंबर 6 होगा")]  # fmt: skip


@pytest.mark.asyncio
@pytest.mark.parametrize("lang,text,must", SERIAL_TURNS)
async def test_the_next_serial_comes_from_the_table(env, m, lang, text, must):
    _setup(env, m, lang)
    out = " ".join(await env.say(text))
    assert must in out, out


@pytest.mark.asyncio
async def test_no_serial_is_promised_for_a_day_the_doctor_does_not_sit(env, m):
    t = _setup(env, m, "en")
    t.available = False
    out = " ".join(await env.say("aj book korle serial koto hobe"))
    assert "not sitting" in out and "serial number will be" not in out


@pytest.mark.asyncio
async def test_without_a_doctor_the_doctor_is_asked_for(env, m):
    _setup(env, m, "en")
    env.session.last_doctor_enquiry = None
    out = " ".join(await env.say("if I book today what will my serial be"))
    assert "doctor" in out.lower() and env.session.pending_serial is not None


@pytest.mark.asyncio
async def test_the_plain_serial_question_still_gets_the_old_whatsapp_answer(env, m):
    _setup(env, m, "en")
    out = " ".join(await env.say("what is the serial number"))
    assert "WhatsApp" in out


# --------------------------------------------------------------------------------------------------------------------------------- the backend
def test_every_booking_gets_the_next_serial_and_the_endpoint_tells_it_before_booking():
    import datetime

    from _clinic_app import clinic_app

    with clinic_app(sample_patients=False) as (_app, client):
        import booking_service as bs
        from db import SessionLocal
        from models import Doctor, DoctorSchedule

        db = SessionLocal()
        try:
            sched = db.query(DoctorSchedule).first()
            doctor = db.get(Doctor, sched.doctor_id)
            day = next(d for d in (bs._now().date() + datetime.timedelta(days=i) for i in range(1, 15)) if d.weekday() == sched.weekday)
            r = client.get("/api/v1/bookings/next-serial", params={"name": doctor.name, "date": day.isoformat()}).json()
            assert r["found"] and r["available"] and r["next_serial"] == 1
            slots = bs.available_slots(db, doctor.id, day.isoformat())
            serials = []
            for i in range(2):
                held = bs.hold_slot(db, doctor.id, day.isoformat(), slots[i])
                done = bs.confirm_booking(db, held["hold_token"], doctor.id, day.isoformat(), slots[i], f"Pt {i}", f"98000000{i}0", f"98000000{i}0", 30 + i)
                assert done["success"]
                from models import Appointment

                serials.append(db.query(Appointment).filter_by(confirmation_id=done["confirmation_id"]).one().token_number)
            assert serials == [1, 2]
            r = client.get("/api/v1/bookings/next-serial", params={"name": doctor.name, "date": day.isoformat()}).json()
            assert r["next_serial"] == 3
            off = next(d for d in (day + datetime.timedelta(days=i) for i in range(1, 8)) if not db.query(DoctorSchedule).filter_by(doctor_id=doctor.id, weekday=d.weekday()).first())
            r = client.get("/api/v1/bookings/next-serial", params={"name": doctor.name, "date": off.isoformat()}).json()
            assert r["available"] is False and r["next_serial"] is None and r["next_available_date"]
        finally:
            db.close()


def test_an_unknown_doctor_has_no_serial():
    from _clinic_app import clinic_app

    with clinic_app(sample_patients=False) as (_app, client):
        assert client.get("/api/v1/bookings/next-serial", params={"name": "zzzzqqqq"}).json()["found"] is False
