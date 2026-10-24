"""sourav, 2026-10-24 -- the owner's six items, English / Bengali / Hindi, script + romanised + mixed + "lame" wording.

  1. "kobe kobe bose?" -> the doctor's WEEKDAYS (som budh sukro ...)
  2. "naam ta booking kora jabe?" -> "Yes, which doctor?" (the date questions follow after the doctor is named)
  3. lay phrases -> department (haarer doctor = orthopaedics, mathar doctor = neurology ...), all three languages
  4. a patient of 13 or more is not sent to the paediatrician
  5. dates are SAID by nearness: 18th -> "আঠারো তারিখ"; next month -> "সামনের মাসের দুই তারিখ"
  6. "do we have to pay now?" -> not paying now is fine, book later at the clinic, no guarantee of slots

    python -m pytest tests/test_lay_items.py -q
"""

import datetime
import importlib.util
import os
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (REPO_ROOT, os.path.join(REPO_ROOT, "tests")):
    if p not in sys.path:
        sys.path.insert(0, p)

from test_orchestrator_booking_flow import FakeTools, env, m  # noqa: E402,F401

from agent import age_cues, date_phrase, doctor_enquiry_cues  # noqa: E402
from agent import early_routing as E  # noqa: E402
from agent.booking_flow import new_state  # noqa: E402


def _lay():
    spec = importlib.util.spec_from_file_location("lay_departments_t", os.path.join(REPO_ROOT, "clinic-api", "lay_departments.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


LAY = _lay()

# ------------------------------------------------------------------------------------------------------------- items 1, 2, 6: routing
ROUTING = [
    ("kobe kobe bose doctor sen", "en", E.DOCTOR_WEEKDAYS), ("doctor sen kobe kobe boshen", "en", E.DOCTOR_WEEKDAYS), ("ডাক্তার সেন কবে কবে বসেন", "bn", E.DOCTOR_WEEKDAYS),
    ("doctor sen kab kab baithte hain", "en", E.DOCTOR_WEEKDAYS), ("डॉक्टर सेन कब कब बैठते हैं", "hi", E.DOCTOR_WEEKDAYS), ("kaun kaun se din aate hain", "en", E.DOCTOR_WEEKDAYS),
    ("naam ta booking kora jabe", "en", E.WRITE_NAME), ("naam ta booking kora jabe te", "en", E.WRITE_NAME), ("amar naam ta book hobe", "en", E.WRITE_NAME),
    ("can a name be booked", "en", E.WRITE_NAME), ("নামটা বুকিং করা যাবে", "bn", E.WRITE_NAME), ("নাম বুক করা যাবে কি", "bn", E.WRITE_NAME),
    ("naam book ho jayega", "en", E.WRITE_NAME), ("नाम बुक हो जाएगा क्या", "hi", E.WRITE_NAME), ("नाम बुकिंग करनी है", "hi", E.WRITE_NAME),
    ("do we have to pay now", "en", E.PAY_NOW), ("do I need to pay now", "en", E.PAY_NOW), ("is payment required now", "en", E.PAY_NOW),
    ("ekhon payment korte hobe ki", "en", E.PAY_NOW), ("ekhoni taka dite hobe", "en", E.PAY_NOW), ("payment ekhon i korte hobe naki", "en", E.PAY_NOW),
    ("এখনই পেমেন্ট করতে হবে", "bn", E.PAY_NOW), ("এখন টাকা দিতে হবে কি", "bn", E.PAY_NOW),
    ("abhi payment karna padega kya", "en", E.PAY_NOW), ("abhi pay karna hoga", "en", E.PAY_NOW), ("अभी पेमेंट करना पड़ेगा", "hi", E.PAY_NOW), ("अभी पैसे देने होंगे क्या", "hi", E.PAY_NOW),
    ("pore pay korle hobe na", "en", E.PAY_NOW),
]  # fmt: skip


@pytest.mark.parametrize("text,lang,expected", ROUTING, ids=[f"{l}:{t[:30]}" for t, l, _ in ROUTING])
def test_routing(text, lang, expected):
    assert E.classify(text, lang) == expected


@pytest.mark.parametrize(
    "text",
    ["which day will he sit", "I will pay now", "I am paying now", "payment link paini", "PhonePe hobena", "the fee depends on age", "what is my serial number", "dues baki ache ki"],
)
def test_nothing_else_is_swallowed(text):
    assert E.classify(text, "en") not in (E.PAY_NOW, E.DOCTOR_WEEKDAYS, E.WRITE_NAME) or text == "what is my serial number" and E.classify(text, "en") not in (E.PAY_NOW,)


def test_weekday_answer_is_the_days_not_dates():
    assert "সোম বুধ শুক্র" in doctor_enquiry_cues.weekdays_reply("সেন", [0, 2, 4], None, "bn")
    assert "রবি মঙ্গল" in doctor_enquiry_cues.weekdays_reply("সেন", [6, 1], None, "bn") or "মঙ্গল রবি" in doctor_enquiry_cues.weekdays_reply("সেন", [6, 1], None, "bn")


# ------------------------------------------------------------------------------------------------------------- items 2, 6: the replies
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "lang,text,must",
    [
        ("en", "do we have to pay now", "no guarantee of slots"),
        ("en", "ekhon payment korte hobe ki", "no guarantee of slots"),
        ("bn", "এখনই পেমেন্ট করতে হবে", "স্লট পাওয়ার কোনো নিশ্চয়তা নেই"),
        ("hi", "अभी पेमेंट करना पड़ेगा", "स्लॉट मिलने की कोई गारंटी नहीं"),
        ("en", "naam ta booking kora jabe te", "Yes,"),
        ("en", "can a name be booked", "Which doctor"),
        ("bn", "নামটা বুকিং করা যাবে", "হ্যাঁ,"),
        ("hi", "नाम बुक हो जाएगा क्या", "जी हाँ,"),
    ],
    ids=lambda v: str(v)[:26],
)
async def test_the_reply(env, m, lang, text, must):
    env.state["lang"] = lang
    env.session.lang = lang
    env.session.disclosed_langs.add(lang)
    out = " ".join(await env.say(text))
    assert must in out, out


@pytest.mark.asyncio
async def test_pay_now_in_the_middle_of_a_booking_keeps_the_booking(env, m):
    env.state["lang"] = "en"
    st = new_state("book_appointment")
    st.stage = "collecting"
    st.slots.update(doctor_name="Sen", date="2026-10-30", time_slot="10:00", patient_name="Rahul Dey", phone="9876543210")
    env.session.booking = st
    out = " ".join(await env.say("do I have to pay now"))
    assert "no guarantee of slots" in out and env.session.booking is st and not env.session.ws.closed


# ------------------------------------------------------------------------------------------------------------- item 3: lay phrase -> department
OWNER_MAP = [
    ("bachhader haarer doctor", "Paediatrics"), ("haarer doctor", "Orthopaedics"), ("booker doctor", "Cardiology"), ("heart er doctor", "Cardiology"),
    ("peter doctor", "Diabetology & Endocrinology"), ("pain er doctor", "General Medicine"), ("plain doctor", "General Medicine"), ("emni physian", "General Medicine"),
    ("mathar doctor", "Neurology"), ("pecchaper doctor", "Urology"), ("operation er doctor", "General Surgery"), ("kata chherar doctor", "General Surgery"),
    ("meyeder doctor", "Gynaecology & Obstetrics"), ("meyeder problem er doctor", "Gynaecology & Obstetrics"), ("nerve er doctor", "Neurology"),
    # Bengali script
    ("বাচ্চাদের হাড়ের ডাক্তার", "Paediatrics"), ("হাড়ের ডাক্তার", "Orthopaedics"), ("বুকের ডাক্তার", "Cardiology"), ("হার্টের ডাক্তার", "Cardiology"),
    ("পেটের ডাক্তার", "Diabetology & Endocrinology"), ("ব্যথার ডাক্তার", "General Medicine"), ("এমনি ফিজিশিয়ান", "General Medicine"), ("মাথার ডাক্তার", "Neurology"),
    ("প্রস্রাবের ডাক্তার", "Urology"), ("অপারেশনের ডাক্তার", "General Surgery"), ("কাটাছেঁড়ার ডাক্তার", "General Surgery"), ("মেয়েদের ডাক্তার", "Gynaecology & Obstetrics"),
    ("মেয়েদের সমস্যার ডাক্তার", "Gynaecology & Obstetrics"), ("নার্ভের ডাক্তার", "Neurology"),
    # Hindi script
    ("बच्चों की हड्डी का डॉक्टर", "Paediatrics"), ("हड्डी का डॉक्टर", "Orthopaedics"), ("दिल का डॉक्टर", "Cardiology"), ("पेट का डॉक्टर", "Diabetology & Endocrinology"),
    ("दर्द का डॉक्टर", "General Medicine"), ("सिर का डॉक्टर", "Neurology"), ("पेशाब का डॉक्टर", "Urology"), ("ऑपरेशन का डॉक्टर", "General Surgery"),
    ("औरतों का डॉक्टर", "Gynaecology & Obstetrics"), ("नस का डॉक्टर", "Neurology"), ("बच्चों का डॉक्टर", "Paediatrics"),
    # romanised Hindi / mixed / lame
    ("haddi ka doctor", "Orthopaedics"), ("dil ka doctor", "Cardiology"), ("pet ka doctor", "Diabetology & Endocrinology"), ("sir ka doctor", "Neurology"),
    ("peshab ka doctor", "Urology"), ("operation wala doctor", "General Surgery"), ("aurton ka doctor", "Gynaecology & Obstetrics"), ("bachchon ka doctor", "Paediatrics"),
    ("amar ekta mathar daktar lagbe", "Neurology"), ("haarer ekta doctor dekhabo", "Orthopaedics"), ("ami meyeder daktar dekhate chai", "Gynaecology & Obstetrics"),
]  # fmt: skip


@pytest.mark.parametrize("text,dept", OWNER_MAP, ids=[t[:30] for t, _ in OWNER_MAP])
def test_lay_phrase_names_its_department(text, dept):
    assert LAY.resolve(text) == [dept]


@pytest.mark.parametrize("text", ["I want a doctor", "book a doctor", "sir I need an appointment", "Dr Sen", "amar doctor lagbe", "hater byatha", "ok thik ache"])
def test_an_ordinary_sentence_names_no_lay_department(text):
    assert LAY.resolve(text) == []


def test_a_specific_phrase_beats_the_ordinary_physician_one():
    assert LAY.resolve("peter pain er doctor") == ["Diabetology & Endocrinology"]


# ------------------------------------------------------------------------------------------------------------- item 4: 13 and above
@pytest.mark.parametrize("text", ["bachhader doctor", "বাচ্চাদের ডাক্তার", "बच्चों का डॉक्टर", "bachchon ka doctor", "chhotoder doctor"])
@pytest.mark.parametrize("age,dept", [(5, "Paediatrics"), (12, "Paediatrics"), (13, "General Medicine"), (15, "General Medicine"), (17, "General Medicine")])
def test_thirteen_and_above_go_to_the_normal_doctor(text, age, dept):
    assert LAY.resolve(text, age) == [dept]


def test_a_teen_with_a_named_specialty_keeps_that_specialty():
    assert LAY.resolve("bachhader haarer doctor", 15) == ["Orthopaedics"] and LAY.resolve("bachhader haarer doctor", 6) == ["Paediatrics"]


@pytest.mark.parametrize(
    "text,age",
    [("amar 15 bochorer chheler jor", 15), ("15 saal ka beta", 15), ("বয়স ১৪", 14), ("ছেলের বয়স পনেরো", 15), ("age is 16", 16), ("tero bochor", 13), ("my son is thirteen years old", 13),
     ("उम्र 14 है", 14), ("umar 14 hai", 14), ("sholo bochor", 16), ("10 years old girl", 10), ("bachhader doctor", None), ("amar 2 ta chele", None)],
)
def test_age_is_read_from_the_sentence(text, age):
    assert age_cues.age_in(text) == age


def test_the_clinic_api_applies_the_age_rule():
    from _clinic_app import clinic_app

    with clinic_app() as (app, client):
        def dept(q, **kw):
            r = client.get("/api/v1/departments/route", params={"query": q, "lang": "en", **kw}).json()
            return r.get("department_name")

        assert dept("bachhader doctor") == "Paediatrics"
        assert dept("bachhader doctor", age=12) == "Paediatrics"
        assert dept("bachhader doctor", age=13) == "General Medicine"
        assert dept("বাচ্চাদের ডাক্তার", age=16) == "General Medicine"
        assert dept("child fever", age=14) in ("General Medicine", None)
        for text, want in [("mathar doctor", "Neurology"), ("pecchaper doctor", "Urology"), ("kata chherar doctor", "General Surgery"), ("haarer doctor", "Orthopaedics"),
                           ("peter doctor", "Diabetology & Endocrinology"), ("booker doctor", "Cardiology"), ("meyeder doctor", "Gynaecology & Obstetrics"), ("plain doctor", "General Medicine")]:
            assert dept(text) == want, text
        # a department named by a lay phrase also lists its real doctors
        r = client.get("/api/v1/departments/doctors", params={"name": "mathar doctor", "lang": "en"}).json()
        assert r["matched"] and r["department_name"] == "Neurology"
        assert client.get("/api/v1/doctors/department", params={"name": "Dr. N. Biswas"}).json()["department_name"] == "Paediatrics"
        assert client.get("/api/v1/doctors/department", params={"name": "Dr. A. Sen"}).json()["department_name"] == "General Medicine"


class _Tools:
    def __init__(self, dept):
        self.dept = dept

    async def doctor_department(self, name):
        return {"found": True, "department_name": self.dept}

    async def department_doctors(self, name, lang="bn", **kw):
        return {"matched": True, "department_name": name, "doctors": ["Dr. P. Ghosh"], "sitting": [{"name": "Dr. P. Ghosh"}], "doctors_today": ["Dr. P. Ghosh"]}


@pytest.mark.asyncio
@pytest.mark.parametrize("age,moved", [(15, True), (13, True), (9, False)])
async def test_a_teen_booked_with_the_paediatrician_moves_to_the_normal_doctor(env, m, monkeypatch, age, moved):
    monkeypatch.setattr(m, "_tools", _Tools("Paediatrics"))

    async def no_proposal(*a, **k):
        return False

    monkeypatch.setattr(m, "_propose_slot", no_proposal)
    env.state["lang"] = "en"
    st = new_state("book_appointment")
    st.slots.update(doctor_name="Biswas", date="2026-10-30", time_slot="10:00", patient_age=age)
    st.hold_token = "tok"
    env.session.booking = st
    await m._redirect_teen_from_paediatrics(env.session, st, "en")
    assert (st.slots["doctor_name"] == "P. Ghosh") is moved
    assert (st.hold_token is None) is moved


@pytest.mark.asyncio
async def test_a_teen_with_a_non_paediatric_doctor_is_left_alone(env, m, monkeypatch):
    monkeypatch.setattr(m, "_tools", _Tools("General Medicine"))
    st = new_state("book_appointment")
    st.slots.update(doctor_name="Sen", date="2026-10-30", patient_age=15)
    assert await m._redirect_teen_from_paediatrics(env.session, st, "en") is False and st.slots["doctor_name"] == "Sen"


# ------------------------------------------------------------------------------------------------------------- item 5: dates
TODAY = datetime.date(2026, 10, 24)


@pytest.mark.parametrize(
    "iso,bn,hi,en",
    [
        ("2026-10-28", "আটাশ তারিখ", "अट्ठाईस तारीख", "the 28th"),
        ("2026-10-18", "আঠারো তারিখ", "अठारह तारीख", "the 18th"),
        ("2026-11-02", "সামনের মাসের দুই তারিখ", "अगले महीने की दो तारीख", "the 2nd of next month"),
        ("2026-09-25", "গত মাসের পঁচিশ তারিখ", "पिछले महीने की पच्चीस तारीख", "the 25th of last month"),
        ("2027-01-03", "জানুয়ারির তিন তারিখ", "जनवरी की तीन तारीख", "the 3rd of January"),
        ("2026-10-31", "একত্রিশ তারিখ", "इकतीस तारीख", "the 31st"),
        ("2026-12-01", "ডিসেম্বরের এক তারিখ", "दिसंबर की एक तारीख", "the 1st of December"),
    ],
)
def test_the_date_is_said_by_nearness(iso, bn, hi, en):
    assert (date_phrase.say(iso, "bn", TODAY), date_phrase.say(iso, "hi", TODAY), date_phrase.say(iso, "en", TODAY)) == (bn, hi, en)


def test_december_to_january_is_next_month():
    assert date_phrase.say("2027-01-02", "bn", datetime.date(2026, 12, 20)) == "সামনের মাসের দুই তারিখ"


@pytest.mark.parametrize(
    "text,lang,want",
    [
        ("ডাক্তার সেন আগামীকাল, 2026-10-25 তারিখে বসবেন।", "bn", "ডাক্তার সেন আগামীকাল, পঁচিশ তারিখে বসবেন।"),
        ("ডাক্তার সেন 2026-11-02 তারিখে বসবেন", "bn", "ডাক্তার সেন সামনের মাসের দুই তারিখে বসবেন"),
        ("डॉक्टर सेन 2026-11-02 को बैठेंगे।", "hi", "डॉक्टर सेन अगले महीने की दो तारीख को बैठेंगे।"),
        ("कल, 2026-10-25 तारीख को बैठेंगे", "hi", "कल, पच्चीस तारीख को बैठेंगे"),
        ("Doctor Sen sits on 2026-11-02 at 10:00.", "en", "Doctor Sen sits on the 2nd of next month at 10:00."),
        ("Doctor Sen sits tomorrow, 2026-10-25.", "en", "Doctor Sen sits tomorrow, the 25th."),
        ("nothing to change here", "en", "nothing to change here"),
        ("KCD-2026-10-25-0001", "en", "KCD-2026-10-25-0001"),
    ],
)
def test_a_reply_with_an_iso_date_is_reworded(text, lang, want):
    assert date_phrase.speak_dates(text, lang, TODAY) == want


@pytest.mark.asyncio
async def test_the_spoken_reply_carries_no_iso_date(env, m):
    """End to end: whatever the reply, the caller never hears a YYYY-MM-DD string."""
    env.state["lang"] = "en"
    sent = []
    orig = env.session.send_json

    async def spy(kind, text, *a, **k):
        sent.append(text)
        return await orig(kind, text, *a, **k)

    env.session.send_json = spy
    await m._speak(env.session, "Doctor Sen sits on 2026-10-30 at 10:00.", "en")
    assert sent and "2026-10-30" not in sent[0] and "30th" in sent[0] or "of next month" in sent[0] or "of last month" in sent[0]
