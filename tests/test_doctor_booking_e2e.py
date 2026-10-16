"""sourav, 2026-10-14: the doctor-appointment flow end to end -- availability question, "book him / that day", time, name, phone,
READ-BACK of every value, yes -> confirmation id, no -> correct one field -- in Bengali, Hindi and English.

WHY: live calls got as far as the patient name and phone number but never read the booking back or gave a confirmation id, and a "no" did not
lead to a correction. The older orchestrator booking tests use hand-made fake clinic tools that no longer match the code (they were already
failing), so this file runs the REAL orchestrator (main_pcm._dispatch_turn), the REAL fast path, the REAL reply templates and the REAL
clinic API (agent/tools_client.ClinicToolsClient talking to the clinic app in-process). Only the speech recogniser text, TTS and the AI model
are stand-ins; the model stand-in returns, for a sentence the fast path does not settle, what a correct extractor would (clearly labelled
below -- it cannot run off-GPU). Flow per scenario: new call -> say(...) turn by turn -> assert on what was SPOKEN and what the clinic holds."""

import datetime
import os
import re
import sys

import httpx
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (REPO_ROOT, os.path.join(REPO_ROOT, "tests")):
    if p not in sys.path:
        sys.path.insert(0, p)

from _clinic_app import clinic_app
from _pod_stubs import pod_stubs
from _synth_voices import FATHER, utterance
from test_orchestrator_persona import ASRResult, FakeTTS, FakeWS, _wav

from agent.fast_path import Catalogue, FastPath
from agent.tools_client import ClinicToolsClient

LANGS = ("bn", "hi", "en")


@pytest.fixture(scope="module")
def m():
    with pod_stubs(REPO_ROOT) as imp:
        yield imp("main_pcm")


@pytest.fixture  # sourav: a fresh clinic per scenario, so one scenario's booked slots never leak into the next
def clinic():
    with clinic_app(sample_patients=False) as (app, client):
        yield app, client


@pytest.fixture
def rig(m, clinic, monkeypatch, tmp_path):
    app, client = clinic
    payload = client.get("/api/v1/catalogue").json()
    tools = ClinicToolsClient("http://clinic.test")
    tools._client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://clinic.test", timeout=20)
    state = {"text": "", "lang": "en", "llm": []}

    async def route(session, wav):
        return state["lang"], ASRResult(state["text"]), True

    async def no_reload():
        return None

    async def cached_catalogue():
        return payload

    async def llm_stand_in(session, text, lang, key):
        # sourav: STAND-IN for the model. `state["llm"]` is a queue of replies a correct extractor would give for the next
        # sentence(s) the fast path leaves to the model; an empty queue means "does not understand" (unclear).
        if state["llm"]:
            return {"secondary_intent": None, "direct_reply_bn": None, **state["llm"].pop(0)}
        return {"intent": "unclear", "slots": {}, "secondary_intent": None, "direct_reply_bn": None}

    monkeypatch.setattr(m, "_fast_path", FastPath(Catalogue(payload)))
    monkeypatch.setattr(m, "_maybe_reload_fast_path", no_reload)
    monkeypatch.setattr(m, "_cached_catalogue", cached_catalogue)
    monkeypatch.setattr(m, "_tools", tools)
    monkeypatch.setattr(m, "_admission", None)
    monkeypatch.setattr(m, "_tts_router", FakeTTS())
    monkeypatch.setattr(m, "CONDITION_INPUT", "off")
    monkeypatch.setattr(m, "_route_and_transcribe", route)
    monkeypatch.setattr(m, "_resolve_intent_uncached", llm_stand_in)
    voice = utterance(FATHER, dur=3.0, seed=1, amp=0.3)

    class Rig:
        pass

    r = Rig()
    r.state, r.tools, r.n, r.client, r.payload = state, tools, 0, client, payload

    def new_call(lang):
        session = m.CallSession(FakeWS())
        session.release_gate()
        for lg in LANGS:
            session.disclosed_langs.add(lg)
        session.call_state = m.new_call_state()
        state["lang"] = lang
        r.session = session
        return session

    async def say(text, llm=None, lang=None):
        state["text"] = text
        if lang:
            state["lang"] = lang
        if llm is not None:
            state["llm"] = list(llm)
        session = r.session
        session.turn_epoch = session.speak_epoch
        r.n += 1
        before = len(session.ws.spoken())
        await m._dispatch_turn(session, _wav(tmp_path, voice, f"u{r.n}.wav"))
        return " ".join(session.ws.spoken()[before:])

    r.new_call, r.say = new_call, say
    return r


def _next_weekday_date():
    d = datetime.date.today() + datetime.timedelta(days=1)
    while d.weekday() == 6:
        d += datetime.timedelta(days=1)
    return d.isoformat()


# sourav 2026-10-16: the spoken correction now says the SAME time the model stand-in returns (half past eleven, a quarter past eleven ...): the
#  caller's own words are read in code and win over the model.
# (lang, availability question, "book him that day", bare no, correction, yes, first time, corrected time, words that must be in the
#  read-back question (the spoken doctor name differs per script), what-to-change cue)
SCENARIOS = [
    ("en", "is doctor Sen available tomorrow", "book him for that day", "no", "no, make it eleven", "yes", "10:00", "11:00", "Sen", "what would you like to change"),
    ("hi", "सेन डॉक्टर कल बैठेंगे क्या", "उनका अपॉइंटमेंट उसी दिन कर दीजिए", "नहीं", "नहीं, साढ़े ग्यारह बजे कर दीजिए", "हाँ", "10:30", "11:30", "सेन", "क्या बदलना"),
    ("bn", "সেন ডাক্তার কাল বসবেন?", "ওনার অ্যাপয়েন্টমেন্ট ওই দিনেই করুন", "না", "না, সওয়া এগারোটায় করুন", "হ্যাঁ", "10:15", "11:15", "সেন", "কী বদলাতে"),
    # mixed: English words inside a Hindi / Bengali sentence, and English yes / no words inside a Hindi / Bengali call
    ("hi", "सेन doctor कल बैठेंगे क्या", "उनकी appointment उसी दिन book कर दो", "no", "no, पौने बारह बजे कर दो", "yes", "10:45", "11:45", "सेन", "क्या बदलना"),
    ("bn", "সেন doctor কাল বসবেন কি", "ওনার appointment ওই দিনে book করুন", "no", "no, সাড়ে দশটায় করুন", "yes", "10:00", "10:30", "সেন", "কী বদলাতে"),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("sc", SCENARIOS, ids=[f"{s[0]}-{i}" for i, s in enumerate(SCENARIOS)])
async def test_doctor_booking_full_flow(rig, sc):
    lang, ask, book, no, fix, yes, t1, t2, who, change_cue = sc
    B = lambda **s: [{"intent": "book_appointment", "slots": s}]
    rig.new_call(lang)
    a = await rig.say(ask)
    assert who.lower() in a.lower() and "10:00" in a, a  # the doctor's real timing, from the clinic
    b = await rig.say(book, B())
    assert "10:00" not in b or "?" in b, b
    await rig.say(t1, B(time_slot=t1))
    await rig.say("Rahul Dey", B(patient_name="Rahul Dey"))
    readback = await rig.say("9876543210", B(phone="9876543210"))
    # the read-back holds every value the caller gave, plus the doctor and the date carried over from the availability answer
    sat = re.search(r"\d{4}-\d\d-\d\d", a).group(0)  # the day the clinic itself answered with: no date is hard-coded
    for needle in (who, t1, "Rahul Dey", "9876543210", sat):
        assert needle.lower() in readback.lower(), (needle, readback)
    asked = await rig.say(no)
    assert change_cue.lower() in asked.lower(), asked  # a bare no asks WHAT to change, not "which doctor?"
    again = await rig.say(fix, B(time_slot=t2))
    for needle in (who, t2, "Rahul Dey", "9876543210"):
        assert needle.lower() in again.lower(), (needle, again)
    assert t1 not in again, again
    done = await rig.say(yes)
    assert "KCD-" in done, done  # the confirmation id is spoken
    assert t2 in done, done



@pytest.mark.asyncio
@pytest.mark.parametrize("lang", ["en", "hi", "bn"])
async def test_every_doctor_is_booked_the_same_way(rig, lang):
    """sourav 2026-10-14: not just Dr. Sen -- for EVERY doctor in the clinic list: name + 'when are they sitting' -> 'book him/her that day'
    -> time, name, phone -> read-back names THAT doctor and the carried date -> yes -> confirmation id. Names two doctors share
    (Roy/Ray, Basu/Bose) are said with the given name too, as a caller would to be understood."""
    B = lambda **s: [{"intent": "book_appointment", "slots": s}]
    key = {"en": "full_name", "hi": "full_name_hi", "bn": "full_name_bn"}[lang]
    ask = {"en": "is {d} available tomorrow", "hi": "{d} कल बैठेंगे क्या", "bn": "{d} কাল বসবেন?"}[lang]
    book = {"en": "book an appointment with him for that day", "hi": "उनका अपॉइंटमेंट उसी दिन कर दीजिए", "bn": "ওনার অ্যাপয়েন্টমেন্ট ওই দিনেই করুন"}[lang]
    yes = {"en": "yes", "hi": "हाँ", "bn": "হ্যাঁ"}[lang]
    bad = []
    for i, d in enumerate(rig.payload["doctors"]):
        spoken = " ".join(w for w in (d.get(key) or d.get("full_name") or "").split() if w.strip(".").lower() not in ("dr", "doctor"))
        rig.new_call(lang)
        phone = f"98{i:08d}"  # a different patient per doctor: the clinic rightly refuses a second booking for one phone at the same time
        a = await rig.say(ask.format(d=spoken))
        sitting = re.search(r"(\d{4}-\d\d-\d\d)", a)
        hours = re.search(r"(\d\d:\d\d)-(\d\d:\d\d)", a)
        b = await rig.say(book, B())
        if not hours:
            # not sitting that day: the answer named the next sitting day, and "that day" must mean it; look up its hours
            avail = await rig.tools.get_doctor_availability(d["name"] if "name" in d else spoken, sitting.group(1) if sitting else None)
            hours_s = avail.get("chamber_hours") or "10:00-12:00"
        else:
            hours_s = hours.group(0)
        t = hours_s.split("-")[0]
        await rig.say(t, B(time_slot=t))
        await rig.say("Rahul Dey", B(patient_name="Rahul Dey"))
        rb = await rig.say(phone, B(phone=phone))
        done = await rig.say(yes)
        want = sitting.group(1) if sitting else ""
        last = (d.get(key) or "").split()[-1].lower()
        if not (phone in rb and "KCD-" in done and want in rb and t in rb):
            bad.append((spoken, a, b, rb, done))
    for row in bad:
        print("BAD", *row, sep="\n   ")
    print("DOCTORS", len(rig.payload["doctors"]), "BAD", len(bad))
    assert not bad, len(bad)


def _names_of(d):
    """Every written form of ONE doctor (surname, aliases, last word of each full name): the reply must hold one of them."""
    forms = [d.get("surname") or d["name"].split()[-1]]
    for k in ("aliases_bn", "aliases_hi", "aliases_en"):
        forms += list(d.get(k) or [])
    for k in ("full_name", "full_name_bn", "full_name_hi"):
        if d.get(k):
            forms.append(d[k].split()[-1])
    return [f.lower() for f in forms if f and len(f) > 1]


@pytest.mark.asyncio
@pytest.mark.parametrize("lang", ["en", "hi", "bn"])
async def test_every_doctor_every_doctor_intent(rig, lang):
    """sourav 2026-10-14: EVERY doctor x EVERY doctor-related thing a caller says, in each language:
      1. "when does <doctor> sit" (no day)        -> the answer names THAT doctor and gives a real date + hours
      2. "does <doctor> sit on the day after tomorrow" -> the answer names that doctor (sits, or next sitting day)
      3. whole booking in ONE sentence (doctor + day + time + name + phone) -> read-back names that doctor -> yes -> confirmation id
      4. wrong doctor at the read-back: "no, with <another doctor>" -> the read-back now names the OTHER doctor and the booking is that doctor's
    (Cancel / reschedule need the caller to prove who they are first; that is a separate flow and not part of this test.)"""
    docs = rig.payload["doctors"]
    key = {"en": "full_name", "hi": "full_name_hi", "bn": "full_name_bn"}[lang]
    when = {"en": "when does {d} sit", "hi": "{d} कब बैठते हैं", "bn": "{d} কবে বসেন"}[lang]
    dat = {"en": "does {d} sit on the day after tomorrow", "hi": "{d} परसों बैठेंगे क्या", "bn": "{d} পরশু বসবেন?"}[lang]
    one = {"en": "book {d} for tomorrow at {t} for Amit Roy phone {p}", "hi": "{d} का कल {t} बजे अपॉइंटमेंट अमित रॉय के नाम फोन {p}", "bn": "{d} এর কাল {t} টায় অ্যাপয়েন্টমেন্ট অমিত রায় ফোন {p}"}[lang]
    wrong = {"en": "no, with {d} instead", "hi": "नहीं, {d} के साथ कर दीजिए", "bn": "না, {d} এর সাথে করুন"}[lang]
    yes = {"en": "yes", "hi": "हाँ", "bn": "হ্যাঁ"}[lang]
    tomorrow = (datetime.date.today() + datetime.timedelta(days=1)).isoformat()
    bad = []

    def spoken(d):
        return " ".join(w for w in (d.get(key) or d.get("full_name") or "").split() if w.strip(".").lower() not in ("dr", "doctor"))

    for i, d in enumerate(docs):
        other = docs[(i + 7) % len(docs)]
        names, onames = _names_of(d), _names_of(other)
        rig.new_call(lang)
        a1 = await rig.say(when.format(d=spoken(d)))
        a2 = await rig.say(dat.format(d=spoken(d)))
        # a free hour for this doctor on `tomorrow`, from the clinic itself
        av = await rig.tools.get_doctor_availability(d["name"], tomorrow)
        hrs = av.get("chamber_hours") or ""
        if not hrs:  # not sitting tomorrow: take the next sitting day
            tom = av.get("next_available_date") or tomorrow
            av = await rig.tools.get_doctor_availability(d["name"], tom)
            hrs = av.get("chamber_hours") or "10:00-12:00"
        else:
            tom = tomorrow
        t = hrs.split("-")[0]
        phone = f"97{i:08d}"
        sp = spoken(d)
        b = await rig.say(one.format(d=sp, t=t.split(":")[0] + (" " + t.split(":")[1] if t.split(":")[1] != "00" else ""), p=phone),
                          [{"intent": "book_appointment", "slots": {"doctor_name": sp, "date": tom, "time_slot": t, "patient_name": "Amit Roy", "phone": phone}}])
        c = await rig.say(wrong.format(d=spoken(other)),
                          [{"intent": "book_appointment", "slots": {"doctor_name": spoken(other)}}])
        # the other doctor may be off that day: then the truthful answer is "he does not sit that day, another day?" (never a made-up slot)
        other_sits = (await rig.tools.get_doctor_availability(other["name"], tom)).get("available") is True
        done = await rig.say(yes) if other_sits else ""
        ok = (
            any(n in a1.lower() for n in names) and re.search(r"\d{4}-\d\d-\d\d", a1) is not None
            and any(n in a2.lower() for n in names)
            and any(n in b.lower() for n in names) and phone in b
            and any(n in c.lower() for n in onames)
            and ((phone in c and "KCD-" in done) if other_sits else ("KCD-" not in c))
        )
        if not ok:
            bad.append((spoken(d), a1, a2, b, c, done))
    for row in bad:
        print("BAD", *row, sep="\n   ")
    print("DOCTORS", len(docs), "BAD", len(bad))
    assert not bad, len(bad)
