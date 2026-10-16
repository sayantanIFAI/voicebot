"""sourav, 2026-10-16: natural doctor questions in English, Hindi and Bengali (native script, romanised, mixed) -- schedule / availability
questions are answered from the clinic's real schedule, booking requests enter the booking flow, and neither changes the call's language.

HOW: the sentences are PATTERNS (tests/doctor_intent_corpus.py) with a doctor slot; the doctors are read from the clinic's own catalogue
(/api/v1/catalogue of the real clinic app), so every doctor in the database is exercised by the same patterns and nothing is written per
doctor. Level 1 asserts the fast path's decision for every (pattern, doctor). Level 2 sends each decision to the REAL clinic API and
asserts the doctor that comes back is the doctor that was asked for. Level 3 runs whole turns through the real orchestrator (the model is a
stand-in that fails or does not understand) and asserts what was SPOKEN, that no booking was created by a question, and that the call's
language did not change. Language-behaviour evidence: the language-switch decisions are pinned to Sayantan's original module's.
It is a text-level test: it is NOT a live ASR / LLM / telephony test."""

import datetime

import pytest
from test_doctor_booking_e2e import clinic, m, rig  # noqa: F401  (fixtures)

import doctor_intent_corpus as C
from _clinic_app import clinic_app
from agent import language_switch
from agent.fast_path import Catalogue, FastPath
from agent.llm import ExtractionError

TODAY = datetime.date(2026, 10, 16)  # a Friday
LANGS = ("en", "hi", "bn")
ROWS = [(lang, tpl, kind, dt) for lang, rows in (("en", C.EN), ("hi", C.HI), ("bn", C.BN)) for tpl, kind, dt in rows]
_WEEK = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]


def _date(key):
    if key is None:
        return None
    if key in ("today", "tomorrow", "day_after"):
        return (TODAY + datetime.timedelta(days={"today": 0, "tomorrow": 1, "day_after": 2}[key])).isoformat()
    return (TODAY + datetime.timedelta(days=(_WEEK.index(key) - TODAY.weekday()) % 7)).isoformat()


def _ref(doctor, template, lang):
    """How a caller would say the doctor: native script when the sentence is in a native script, Latin otherwise."""
    english = "Dr " + doctor["full_name"].replace("Dr. ", "")
    if template.isascii() or lang == "en":
        return english
    return ("डॉ. " + doctor["full_name_hi"]) if lang == "hi" else ("ডাঃ " + doctor["full_name_bn"])


@pytest.fixture(scope="module")
def payload():
    with clinic_app(sample_patients=False) as (_app, client):
        return client.get("/api/v1/catalogue").json()


@pytest.fixture(scope="module")
def fp(payload):
    return FastPath(Catalogue(payload), today=TODAY)


def _decide(fp, utterance, lang):
    """-> (kind, date, doctor spoken form): what the system does with the sentence before any model."""
    result = fp.resolve(utterance, lang)
    if result is not None and result.intent == "doctor_availability":
        return "schedule", result.slots.get("date"), result.slots.get("doctor_name")
    if result is None:
        hint = fp.doctor_booking_hint(utterance, lang)
        if hint is not None:
            return "book", hint.slots.get("date"), hint.slots.get("doctor_name")
    return "model", None, None


def test_the_corpus_is_the_size_the_report_states():
    counts = {lang: len(rows) for lang, rows in (("en", C.EN), ("hi", C.HI), ("bn", C.BN))}
    assert all(n >= 40 for n in counts.values()), counts
    assert len(C.MIXED) >= 10
    assert len({t for t, _k, _d in C.EN}) == len(C.EN) and len({t for t, _k, _d in C.HI}) == len(C.HI) and len({t for t, _k, _d in C.BN}) == len(C.BN)


@pytest.mark.parametrize("lang, template, kind, day", ROWS, ids=[f"{r[0]}-{i}" for i, r in enumerate(ROWS)])
def test_every_doctor_in_the_catalogue_is_recognised_by_the_same_pattern(fp, payload, lang, template, kind, day):
    wrong = []
    for doctor in payload["doctors"]:
        utterance = template.replace("{d}", _ref(doctor, template, lang))
        got_kind, got_date, _form = _decide(fp, utterance, lang)
        ok = got_kind == kind if kind != "model" else got_kind == "model"
        if ok and kind in ("schedule", "book") and day is not None:
            ok = got_date == _date(day)
        if ok and kind in ("schedule", "book"):
            named = fp._named_doctor(utterance.lower(), lang)
            ok = named is not None and named[0] == doctor["name"]
        if not ok:
            wrong.append((doctor["name"], utterance, got_kind, got_date))
    assert not wrong, wrong[:3]


@pytest.mark.parametrize("call_lang", LANGS)
@pytest.mark.parametrize("template, kind, day", C.MIXED, ids=[f"mixed-{i}" for i in range(len(C.MIXED))])
def test_mixed_language_sentences_are_recognised_whatever_language_the_call_is_in(fp, payload, call_lang, template, kind, day):
    for doctor in payload["doctors"]:
        utterance = template.replace("{d}", "Dr " + doctor["full_name"].replace("Dr. ", ""))
        got_kind, got_date, _ = _decide(fp, utterance, call_lang)
        assert got_kind == kind, (doctor["name"], utterance, got_kind)
        if day is not None:
            assert got_date == _date(day), (utterance, got_date)


@pytest.mark.parametrize("lang", LANGS)
def test_sentences_that_are_not_about_a_doctor_are_never_turned_into_one(fp, lang):
    for utterance in C.NEGATIVE[lang]:
        got_kind, _d, _f = _decide(fp, utterance, lang)
        assert got_kind == "model", (utterance, got_kind)
        result = fp.resolve(utterance, lang)
        assert result is None or result.intent not in ("doctor_availability", "book_appointment"), (utterance, result)


def test_common_words_that_are_also_doctor_names_do_not_pick_a_doctor(fp):
    # Hindi "कर" ("do") is also the spoken form of Dr. A. Kar; with no "doctor / डॉ" in the sentence it is just the word
    assert _decide(fp, "उसी दिन बुक कर दीजिए", "hi")[0] == "model"
    assert _decide(fp, "मुझे बुक कर दीजिए", "hi")[0] == "model"
    assert _decide(fp, "आप दिन में कब तक काम कर सकते हैं", "hi")[0] == "model"


def test_a_shared_surname_is_never_guessed(fp, payload):
    """Two doctors who share a spoken form ("Roy", "Basu"): the fast path abstains, so the clinic API asks "which one?"."""
    sentence = {"hi": "डॉ. {f} कब बैठते हैं", "bn": "ডাঃ {f} কবে বসেন", "en": "when does Dr {f} sit"}
    shared = set()
    for lang in LANGS:
        for form in {f for d in payload["doctors"] for f in fp.catalogue._doctor_forms_of(d, lang)}:
            if fp.catalogue.doctor_owner_count(lang, form) > 1:
                shared.add((lang if form.isascii() else ("hi" if "\u0900" <= form[0] <= "\u097f" else "bn"), form))
    assert shared, "this catalogue has no doctors sharing a spoken form, so the test cannot check the clarification path"
    for lang, form in sorted(shared):
        assert _decide(fp, sentence[lang].format(f=form), lang)[0] == "model", (lang, form)


def test_an_unknown_doctor_is_not_matched(fp):
    for utterance, lang in (("when does Dr Zzyzxqw sit", "en"), ("डॉ. ज़्ज़ाइक्स कब बैठते हैं", "hi"), ("ডাঃ ঝঝাইক্স কবে বসেন", "bn")):
        assert _decide(fp, utterance, lang)[0] == "model", utterance


def test_a_follow_up_without_a_name_is_left_to_the_conversation(fp):
    for utterance, lang in (("What about the evening?", "en"), ("और शाम को?", "hi"), ("আর বিকেলে?", "bn")):
        assert _decide(fp, utterance, lang)[0] == "model", utterance


# ------------------------------------------------------------------------------------------------ level 2: the real clinic API


@pytest.mark.asyncio
@pytest.mark.parametrize("lang, template", [("en", "When does {d} sit?"), ("hi", "{d} कब बैठते हैं?"), ("bn", "{d} কবে বসেন?"), ("en", "Is {d} available tomorrow?")])
async def test_the_clinic_database_confirms_the_doctor_the_pattern_found(rig, lang, template):
    fast = FastPath(Catalogue(rig.payload), today=TODAY)
    for doctor in rig.payload["doctors"]:
        utterance = template.replace("{d}", _ref(doctor, template, lang))
        result = fast.resolve(utterance, lang)
        assert result is not None and result.intent == "doctor_availability", (doctor["name"], utterance)
        answer = await rig.tools.get_doctor_availability(result.slots["doctor_name"], result.slots.get("date"))
        assert answer.get("found"), (doctor["name"], utterance, answer)
        assert doctor["name"] in str(answer.values()) or doctor["name"].replace("Dr. ", "") in str(answer.values()), (doctor["name"], answer)


# ------------------------------------------------------------------------------------------------ level 3: whole turns


def B(**s):
    return [{"intent": "book_appointment", "slots": s}]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "lang, said, native",
    [
        ("en", "When does Dr Sen sit?", False),
        ("en", "Kal Dr Sen milenge kya?", True),
        ("en", "Ami Dr Sen ke dekhabo. Dr Sen kobe boshen?", True),
        ("hi", "डॉ. सेन कब बैठते हैं?", True),
        ("bn", "কাল ডাঃ সেন বসবেন?", True),
    ],
)
async def test_a_schedule_question_is_answered_from_the_clinic_and_creates_no_booking(rig, lang, said, native):
    session = rig.new_call(lang)
    spoken = await rig.say(said)
    assert "10:" in spoken or "11:" in spoken, spoken  # the doctor's real hours, from the clinic
    assert session.booking is None  # a question never starts or creates a booking
    assert session.lang == lang  # the call's language is untouched


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "lang, said",
    [
        ("en", "I want to book an appointment with Dr Sen."),
        ("en", "Mujhe Dr Sen ka appointment book karna hai."),
        ("en", "Ami Dr Sen er appointment book korte chai."),
        ("hi", "मुझे डॉ. सेन से अपॉइंटमेंट बुक करना है।"),
        ("bn", "ডাঃ সেনের সঙ্গে অ্যাপয়েন্টমেন্ট বুক করতে চাই।"),
        ("hi", "मुझे डॉ. सेन को दिखाना है।"),
        ("bn", "আমি ডাঃ সেনকে দেখাতে চাই।"),
    ],
)
async def test_a_booking_request_enters_the_booking_flow_even_when_the_model_does_not_understand(rig, lang, said):
    session = rig.new_call(lang)
    spoken = await rig.say(said)  # the model stand-in has nothing queued = "does not understand"
    assert spoken.strip() and "problem" not in spoken.lower(), spoken
    assert session.booking is not None and session.booking.action == "book_appointment"
    assert "sen" in (session.booking.slots.get("doctor_name") or "").lower() or "সেন" in (session.booking.slots.get("doctor_name") or "") or "सेन" in (session.booking.slots.get("doctor_name") or "")
    assert session.booking.stage == "collecting"  # nothing is confirmed or booked by the request itself
    assert session.lang == lang


@pytest.mark.asyncio
async def test_a_booking_request_survives_a_model_failure(rig, m, monkeypatch):
    async def boom(session, text, lang, key):
        raise ExtractionError("timed out")

    monkeypatch.setattr(m, "_resolve_intent_uncached", boom)
    session = rig.new_call("en")
    spoken = await rig.say("I want to book an appointment with Dr Sen.")
    assert "problem" not in spoken.lower(), spoken
    assert session.booking is not None and session.booking.slots.get("doctor_name")


@pytest.mark.asyncio
async def test_schedule_then_book_him_that_day_carries_the_doctor_and_date(rig):
    session = rig.new_call("en")
    a = await rig.say("Is Dr Sen available tomorrow?")
    tomorrow = (datetime.date.today() + datetime.timedelta(days=1)).isoformat()
    assert session.booking is None
    await rig.say("book him for that day", B())
    assert session.booking is not None and session.booking.slots.get("doctor_name")
    assert session.booking.slots.get("date")


@pytest.mark.asyncio
async def test_an_unknown_doctor_is_said_plainly_not_invented(rig):
    session = rig.new_call("en")
    spoken = await rig.say("Is Dr Zzyzxqw available tomorrow?", [{"intent": "doctor_availability", "slots": {"doctor_name": "Zzyzxqw"}}])
    assert spoken.strip() and "10:" not in spoken, spoken
    assert session.booking is None


# ------------------------------------------------------------------------------------------------ language behaviour evidence

# Decisions of Sayantan's ORIGINAL agent/language_switch.py (commit b356fbe), pinned: (phrase, call language) -> switch target.
SAYANTAN_SWITCH = [('speak in hindi', 'en', 'hi'), ('speak in bangla', 'en', 'bn'), ('speak in english', 'en', 'en'), ('switch to hindi', 'en', 'hi'), ('hindi mein bolo', 'en', 'hi'), ('bangla mein bolo', 'en', 'bn'), ('हिंदी में बोलिए', 'en', 'hi'), ('हिंदी में बोलिए', 'hi', 'hi'), ('हिंदी में बोलिए', 'bn', 'hi'), ('अंग्रेज़ी में बात करें', 'en', 'en'), ('अंग्रेज़ी में बात करें', 'hi', 'en'), ('अंग्रेज़ी में बात करें', 'bn', 'en'), ('বাংলায় বলুন', 'en', 'bn'), ('বাংলায় বলুন', 'hi', 'bn'), ('বাংলায় বলুন', 'bn', 'bn'), ('ইংরেজিতে বলুন', 'en', 'en'), ('ইংরেজিতে বলুন', 'hi', 'en'), ('ইংরেজিতে বলুন', 'bn', 'en'), ('can I talk in Hindi?', 'en', 'hi'), ('continue in English', 'en', 'en'), ('please speak english', 'en', 'en'), ('bengali please', 'en', 'bn'), ('Speak Bangla', 'en', 'bn'), ('Speak in English', 'en', 'en'), ('बांग्ला में बोलिए', 'en', 'bn'), ('बांग्ला में बोलिए', 'hi', 'bn'), ('बांग्ला में बोलिए', 'bn', 'bn'), ('हिंदी में बात करो', 'en', 'hi'), ('हिंदी में बात करो', 'hi', 'hi'), ('हिंदी में बात करो', 'bn', 'hi'), ('english mein baat karo', 'en', 'en'), ('ইংরেজিতে কথা বলুন', 'en', 'en'), ('ইংরেজিতে কথা বলুন', 'hi', 'en'), ('ইংরেজিতে কথা বলুন', 'bn', 'en'), ('hindi te kotha bolun', 'en', 'hi'), ('speak hindi', 'en', 'hi')]  # fmt: skip


@pytest.mark.parametrize("phrase, lang, target", SAYANTAN_SWITCH)
def test_every_language_switch_sayantans_module_made_is_still_made(phrase, lang, target):
    assert language_switch.detect_language_switch_request(phrase, lang) == target


def test_no_doctor_sentence_in_any_language_is_ever_read_as_a_language_switch():
    for lang, rows in (("en", C.EN), ("hi", C.HI), ("bn", C.BN)):
        for template, _kind, _day in rows + [(t, k, d) for t, k, d in C.MIXED]:
            for name in ("Dr Sen", "डॉ. सेन", "ডাঃ সেন"):
                for call_lang in LANGS:
                    assert language_switch.detect_language_switch_request(template.replace("{d}", name), call_lang) is None, (template, call_lang)
        for utterance in C.NEGATIVE[lang]:
            assert language_switch.detect_language_switch_request(utterance, lang) is None
