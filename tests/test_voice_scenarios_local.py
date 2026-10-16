"""sourav, 2026-10-16 (final verification): the doctor-booking call scenarios A-H, driven turn by turn through the REAL orchestrator
(main_pcm._dispatch_turn), the REAL fast path, the REAL reply templates, the REAL agent/tools_client.ClinicToolsClient (httpx) and the REAL
clinic API with a temporary database.

WHAT IS AND IS NOT REAL (stated plainly, because this is not a live call):
  real      : orchestrator, language/session handling after ASR, fast path + doctor cues, booking state machine, templates, the HTTP client,
              the clinic API and its database (a temporary SQLite file per scenario, test-only data, never a real patient or number).
  stand-in  : the speech recogniser (the caller's words are given as text), TTS (a fake that records what would be spoken) and the intent MODEL
              (a queue of what a correct extractor returns; empty = "does not understand"). No ASR / LLM / TTS / telephony / websocket here.
The HTTP faults in scenario G are injected by a transport that wraps the real clinic app, so the real client code maps them to errors.
Flow per scenario: new call -> say(...) turn by turn -> assert on what was SPOKEN and on what the clinic database really holds."""

import re

import httpx
import pytest
from test_booking_capture_e2e import _ASK, _BOOK, _FIRST_TIME, B, _bookings  # noqa: F401
from test_doctor_booking_e2e import clinic, m, rig  # noqa: F401  (fixtures)

PHONE = "9000012345"  # test-only number
PHONE_WORDS = {
    "en": "nine zero zero zero zero one two three four five",
    "hi": "नौ शून्य शून्य शून्य शून्य एक दो तीन चार पाँच",
    "bn": "নয় শূন্য শূন্য শূন্য শূন্য এক দুই তিন চার পাঁচ",
}
YES = {"en": "yes", "hi": "हाँ", "bn": "হ্যাঁ"}
NO = {"en": "no", "hi": "नहीं", "bn": "না"}
WHO = {"en": "Sen", "hi": "सेन", "bn": "সেন"}


async def _reach_read_back(rig, lang):
    rig.new_call(lang)
    a = await rig.say(_ASK[lang])
    await rig.say(_BOOK[lang], B())
    await rig.say(_FIRST_TIME[lang], B(time_slot=_FIRST_TIME[lang]))
    await rig.say("Rahul Dey", B(patient_name="Rahul Dey"))
    readback = await rig.say(PHONE_WORDS[lang])  # model queue empty: deterministic capture only
    return a, readback


# ---------------------------------------------------------------------------------------------- scenarios A, B, C


@pytest.mark.asyncio
@pytest.mark.parametrize("lang", ["en", "hi", "bn"])
async def test_scenario_abc_schedule_then_booking_ends_in_a_real_clinic_booking(rig, lang):
    a, readback = await _reach_read_back(rig, lang)
    day = re.search(r"\d{4}-\d\d-\d\d", a).group(0)
    truth = await rig.tools.get_doctor_availability("Sen", day)
    assert truth.get("chamber_hours") and truth["chamber_hours"] in a, (truth, a)  # the spoken schedule IS the clinic's answer
    for needle in (WHO[lang], day, _FIRST_TIME[lang], "Rahul Dey", PHONE):
        assert needle.lower() in readback.lower(), (needle, readback)
    assert not await _bookings(rig, PHONE)  # nothing is booked before the explicit yes
    done = await rig.say(YES[lang])
    cid = re.search(r"KCD-[A-Za-z0-9-]+", done)
    assert cid, done
    rows = await _bookings(rig, PHONE)
    assert len(rows) == 1 and rows[0]["confirmation_id"] == cid.group(0), (rows, done)  # the spoken id is the stored id
    row = rows[0]
    assert row["date"] == day and str(row["time_slot"]).startswith(_FIRST_TIME[lang]) and "Rahul Dey" in str(row.get("patient_name", "Rahul Dey"))
    assert rig.session.booking is None or rig.session.booking.stage == "done"


# ------------------------------------------------------------------------------------------------------ scenario D

ENGLISH_CALL_PHRASES = [  # as typed by the task, romanised, in an English-tagged call
    "When does Dr Sen sit?",
    "Mujhe Dr Sen ko dikhana hai.",
    "Ami Dr Sen ke dekhate chai.",
    "Dr Sen kal available hain kya?",
    "Dr Sen er next slot kobe?",
]
NATIVE_PHRASES = {
    "hi": ["सेन डॉक्टर कब बैठते हैं", "मुझे डॉक्टर सेन को दिखाना है", "डॉक्टर सेन कल उपलब्ध हैं क्या", "डॉक्टर सेन का अगला स्लॉट कब है"],
    "bn": ["সেন ডাক্তার কবে বসেন", "আমি ডাক্তার সেনকে দেখাতে চাই", "ডাক্তার সেন কাল আছেন কি", "ডাক্তার সেনের পরের স্লট কবে"],
}


@pytest.mark.asyncio
@pytest.mark.parametrize("phrase", ENGLISH_CALL_PHRASES)
async def test_scenario_d_natural_romanised_phrases_reach_dr_sen_without_the_model(rig, phrase):
    rig.new_call("en")
    said = await rig.say(phrase)  # model queue EMPTY: only the deterministic path can answer
    assert "sen" in said.lower() and not said.lower().startswith("sorry"), said
    if rig.session.booking is None:  # a schedule question: the clinic's own date and hours
        day = re.search(r"\d{4}-\d\d-\d\d", said).group(0)
        assert (await rig.tools.get_doctor_availability("Sen", day))["chamber_hours"] in said
    else:  # "I want to see Dr Sen": a booking with that doctor is opened and the day is asked
        assert rig.session.booking.slots.get("doctor_name") and "day" in said.lower()
    assert rig.session.lang == "en"


@pytest.mark.asyncio
@pytest.mark.parametrize("lang", ["hi", "bn"])
async def test_scenario_d_natural_native_script_phrases_reach_dr_sen_without_the_model(rig, lang):
    for phrase in NATIVE_PHRASES[lang]:
        rig.new_call(lang)
        said = await rig.say(phrase)
        assert WHO[lang] in said, (phrase, said)
        assert re.search(r"\d{4}-\d\d-\d\d", said) or rig.session.booking is not None, (phrase, said)
        assert rig.session.lang == lang


# ------------------------------------------------------------------------------------------------------ scenario E


@pytest.mark.asyncio
@pytest.mark.parametrize("lang", ["en", "hi", "bn"])
async def test_scenario_e_an_incomplete_number_is_never_booked_and_the_rest_is_asked_for(rig, lang):
    rig.new_call(lang)
    await rig.say(_ASK[lang])
    await rig.say(_BOOK[lang], B())
    await rig.say(_FIRST_TIME[lang], B(time_slot=_FIRST_TIME[lang]))
    await rig.say("Rahul Dey", B(patient_name="Rahul Dey"))
    ask = await rig.say(PHONE_WORDS[lang].rsplit(" ", 4)[0])  # six of the ten digits
    assert "6" in ask and rig.session.booking.stage == "collecting", ask
    assert not await _bookings(rig, PHONE) and "KCD-" not in ask
    rest = await rig.say(" ".join(PHONE_WORDS[lang].split(" ")[-4:]))
    assert PHONE in rest and rig.session.booking.stage == "confirming"  # only now the read-back
    assert not await _bookings(rig, PHONE)


# ------------------------------------------------------------------------------------------------------ scenario F


@pytest.mark.asyncio
@pytest.mark.parametrize("lang, fix, newtime", [("en", "no, make it eleven", "11:00"), ("hi", "नहीं, साढ़े ग्यारह बजे कर दीजिए", "11:30"), ("bn", "না, সওয়া এগারোটায় করুন", "11:15")])
async def test_scenario_f_a_correction_at_the_read_back_changes_only_that_field(rig, lang, fix, newtime):
    # sourav 2026-10-16: the spoken words now say the SAME time the model stand-in returns (half past eleven / a quarter past eleven): the caller's
    # own words are read in code and win over the model, so words and stand-in must agree.
    _, readback = await _reach_read_back(rig, lang)
    asked = await rig.say(NO[lang])
    assert "KCD-" not in asked and not await _bookings(rig, PHONE)
    again = await rig.say(fix, B(time_slot=newtime))
    for needle in (WHO[lang], newtime, "Rahul Dey", PHONE):
        assert needle.lower() in again.lower(), (needle, again)
    assert _FIRST_TIME[lang] not in again
    done = await rig.say(YES[lang])
    assert "KCD-" in done
    rows = await _bookings(rig, PHONE)
    assert len(rows) == 1 and str(rows[0]["time_slot"]).startswith(newtime), rows


# ------------------------------------------------------------------------------------------------------ scenario G


class FaultTransport(httpx.AsyncBaseTransport):
    """Wraps the real clinic app. `rule(request)` returns None (pass through), "500", "503", "timeout" (never reaches the clinic), "connect",
    or "timeout_after_commit" (the clinic DOES process the request, then the answer is lost)."""

    def __init__(self, inner, rule):
        self.inner, self.rule, self.hits = inner, rule, []

    async def handle_async_request(self, request):
        mode = self.rule(request)
        if mode:
            self.hits.append((request.method, request.url.path, mode))
        if mode in ("500", "503"):
            return httpx.Response(int(mode), json={"detail": "injected test fault"}, request=request)
        if mode == "timeout":
            raise httpx.ReadTimeout("injected timeout", request=request)
        if mode == "connect":
            raise httpx.ConnectError("injected connection failure", request=request)
        response = await self.inner.handle_async_request(request)
        if mode == "timeout_after_commit":
            await response.aread()
            raise httpx.ReadTimeout("injected: answer lost after the clinic committed", request=request)
        return response


def _fault_on_confirm(rig, mode_holder):
    inner = rig.tools._client._transport
    transport = FaultTransport(inner, lambda req: mode_holder["mode"] if req.url.path == "/api/v1/bookings/confirm" else None)
    rig.tools._client = httpx.AsyncClient(transport=transport, base_url="http://clinic.test", timeout=20)
    return transport


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["500", "503", "timeout", "connect"])
@pytest.mark.parametrize("lang", ["en", "hi", "bn"])
async def test_scenario_g_a_backend_failure_never_claims_a_booking_or_invents_an_id(rig, lang, mode):
    holder = {"mode": None}
    await _reach_read_back(rig, lang)
    transport = _fault_on_confirm(rig, holder)
    holder["mode"] = mode
    said = await rig.say(YES[lang])
    assert transport.hits, "the fault was never exercised"
    assert "KCD-" not in said, said  # no confirmation id is spoken or invented
    holder["mode"] = None
    assert not await _bookings(rig, PHONE)  # and the clinic really holds no booking
    assert rig.session.booking is not None and rig.session.booking.stage == "confirming"  # the details are kept for a retry
    done = await rig.say(YES[lang])  # backend healthy again: the retry books exactly once
    assert "KCD-" in done and len(await _bookings(rig, PHONE)) == 1


@pytest.mark.asyncio
async def test_scenario_g_an_answer_lost_after_the_clinic_committed_is_found_not_rebooked(rig):
    holder = {"mode": None}
    await _reach_read_back(rig, "en")
    _fault_on_confirm(rig, holder)
    holder["mode"] = "timeout_after_commit"
    said = await rig.say("yes")
    holder["mode"] = None
    rows = await _bookings(rig, PHONE)
    assert len(rows) == 1 and rows[0]["confirmation_id"] in said, (rows, said)  # the REAL id, read back from the clinic


# ------------------------------------------------------------------------------------------------------ scenario H


@pytest.mark.asyncio
@pytest.mark.parametrize("lang", ["en", "hi", "bn"])
async def test_scenario_h_doctor_intent_turns_do_not_change_language_greet_or_disclose(rig, lang):
    rig.new_call(lang)
    before = (rig.session.lang, rig.session.lang_router.previous_language, set(rig.session.disclosed_langs), len(rig.session.lang_history))
    spoken = []
    for text in ([ENGLISH_CALL_PHRASES[0]] if lang == "en" else NATIVE_PHRASES[lang][:2]) + [_ASK[lang], _BOOK[lang]]:
        spoken.append(await rig.say(text, B() if text == _BOOK[lang] else None))
        assert rig.session.lang == lang
    assert rig.session.lang_router.previous_language in (before[1], lang)
    assert rig.session.disclosed_langs == before[2]  # nothing was newly "disclosed" (no greeting replay)
    assert not any("Sonoscan Vaani" in s or "সোনোস্ক্যান" in s or "सोनोस्कैन" in s for s in spoken)  # no greeting / disclosure spoken
    assert all("language" not in s.lower() or lang != "en" for s in spoken)  # no language-switch offer
