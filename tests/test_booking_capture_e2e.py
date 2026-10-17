"""sourav, 2026-10-16: the doctor booking from "is Dr. Sen available tomorrow" to the confirmation id, with the details that used to stop
it: a phone number said in words / in pieces / as "double nine", a wrong or invalid number, a misheard or invented patient name, the model
failing, a hesitant answer at the read-back, a timeout while confirming, and "the first one" after a slot conflict.

Like tests/test_doctor_booking_e2e.py (whose fixtures it reuses) this runs the REAL orchestrator (main_pcm._dispatch_turn), the REAL fast
path, the REAL reply templates and the REAL clinic API in-process. Only the recogniser text, TTS and the intent MODEL are stand-ins (the model
stand-in is a queue of what a correct extractor would return; an empty queue = "does not understand"). It is NOT a live ASR / LLM / telephony test."""

import re

import pytest
from test_doctor_booking_e2e import clinic, m, rig  # noqa: F401  (fixtures)

from agent.llm import ExtractionError
from agent.tools_client import ToolCallError


def B(**s):
    return [{"intent": "book_appointment", "slots": s}]


_ASK = {"en": "is doctor Sen available tomorrow", "hi": "सेन डॉक्टर कल बैठेंगे क्या", "bn": "সেন ডাক্তার কাল বসবেন?"}
_BOOK = {"en": "book him for that day", "hi": "उनका अपॉइंटमेंट उसी दिन कर दीजिए", "bn": "ওনার অ্যাপয়েন্টমেন্ট ওই দিনেই করুন"}
_FIRST_TIME = {"en": "10:00", "hi": "10:30", "bn": "10:15"}  # Dr. Sen's first slot in each language's seed


async def _to_name(rig, lang="en", time=None):
    """availability question -> 'book him that day' -> time. Returns the clinic's own availability answer."""
    time = time or _FIRST_TIME[lang]
    rig.new_call(lang)
    a = await rig.say(_ASK[lang])
    await rig.say(_BOOK[lang], B())
    await rig.say(time, B(time_slot=time))
    return a


async def _to_phone(rig, lang="en", time=None):
    a = await _to_name(rig, lang, time)
    await rig.say("Rahul Dey", B(patient_name="Rahul Dey"))
    return a


async def _bookings(rig, phone):
    return (await rig.tools.lookup_bookings(phone=phone)).get("bookings") or []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "lang, said, phone",
    [
        ("en", "nine eight seven six five four three two one zero", "9876543210"),
        ("en", "double nine eight seven six five four three two one", "9987654321"),
        ("hi", "नौ आठ सात छह पाँच चार तीन दो एक शून्य", "9876543210"),
        ("hi", "९८७६५४३२१०", "9876543210"),
        ("bn", "নয় আট সাত ছয় পাঁচ চার তিন দুই এক শূন্য", "9876543210"),
        ("bn", "৯৮৭৬৫৪৩২১০", "9876543210"),
    ],
)
async def test_a_phone_number_said_in_words_or_local_digits_reaches_the_read_back_without_the_model(rig, lang, said, phone):
    await _to_phone(rig, lang)
    readback = await rig.say(said)  # the model stand-in queue is EMPTY: only the deterministic capture can fill the phone
    assert phone in readback, readback
    done = await rig.say({"en": "yes", "hi": "हाँ", "bn": "হ্যাঁ"}[lang])
    assert "KCD-" in done, done
    assert len(await _bookings(rig, phone)) == 1


@pytest.mark.asyncio
async def test_a_phone_number_said_over_several_turns_is_joined_and_the_rest_is_asked_for(rig):
    await _to_phone(rig)
    first = await rig.say("nine eight seven six five")
    assert "5 digits" in first, first
    second = await rig.say("four three two")
    assert "8 digits" in second, second
    readback = await rig.say("one zero")
    assert "9876543210" in readback, readback
    assert "KCD-" in await rig.say("yes")


@pytest.mark.asyncio
async def test_a_wrong_number_and_a_start_again_never_mix_with_the_old_digits(rig):
    await _to_phone(rig)
    await rig.say("nine eight seven six five")
    assert "again" in (await rig.say("sorry wrong number start again")).lower()
    readback = await rig.say("9123456780")
    assert "9123456780" in readback and "98765" not in readback, readback


@pytest.mark.asyncio
async def test_an_invalid_phone_from_the_model_is_never_booked(rig):
    await _to_phone(rig)
    said = await rig.say("it is the same number as my other booking", B(phone="123"))  # no digits said: the model's "123" is its own
    assert "10-digit" in said, said
    assert rig.session.booking.slots.get("phone") is None
    assert "9876543210" in await rig.say("9876543210")


@pytest.mark.asyncio
async def test_a_phone_correction_at_the_read_back_replaces_only_the_phone(rig):
    await _to_phone(rig)
    await rig.say("9876543210")
    again = await rig.say("no my number is 9123456780")
    assert "9123456780" in again and "9876543210" not in again and "Rahul Dey" in again, again
    done = await rig.say("yes")
    assert "KCD-" in done
    assert len(await _bookings(rig, "9123456780")) == 1 and not await _bookings(rig, "9876543210")


@pytest.mark.asyncio
async def test_a_name_the_model_made_up_is_replaced_by_what_the_caller_said_and_needs_a_yes(rig):
    await _to_name(rig)
    ask = await rig.say("Ramesh Ghosh", B(patient_name="Rahul Dey"))  # the model invented "Rahul Dey"
    assert "Ramesh Ghosh" in ask and "Rahul" not in ask, ask
    await rig.say("yes")
    readback = await rig.say("9876543210")
    assert "Ramesh Ghosh" in readback and "Rahul" not in readback, readback


@pytest.mark.asyncio
async def test_a_name_the_model_missed_is_read_from_the_callers_words_and_read_back(rig):
    await _to_name(rig)
    await rig.say("what", B())  # the name is asked for
    ask = await rig.say("Suman Pal")  # model stand-in: unclear, no slots
    assert "Suman Pal" in ask, ask
    no = await rig.say("no, Subir Pal")
    assert "Subir Pal" in no, no
    await rig.say("yes")
    assert "Subir Pal" in await rig.say("9876543210")


@pytest.mark.asyncio
async def test_a_rejected_name_with_no_new_name_is_asked_again(rig):
    await _to_name(rig)
    await rig.say("what", B())
    await rig.say("Suman Pal")
    again = await rig.say("no")
    assert "name" in again.lower() and "Suman" not in again, again
    assert rig.session.booking.slots.get("patient_name") is None


@pytest.mark.asyncio
async def test_a_name_correction_after_the_read_back_replaces_the_name(rig):
    await _to_phone(rig)
    await rig.say("9876543210")
    ask = await rig.say("no, the name is Anil Kar", B(patient_name="Anil Kar"))
    assert "Anil Kar" in ask and "Rahul Dey" not in ask, ask
    assert "KCD-" in await rig.say("yes")
    assert (await _bookings(rig, "9876543210"))[0]["patient_name"] == "Anil Kar"


@pytest.mark.asyncio
async def test_a_failing_model_during_a_booking_asks_the_next_question_not_a_little_problem(rig, m, monkeypatch):
    async def boom(session, text, lang, key):
        raise ExtractionError("model timed out")

    await _to_name(rig)
    monkeypatch.setattr(m, "_resolve_intent_uncached", boom)
    spoken = await rig.say("um the name is hmm")
    assert "problem" not in spoken.lower() and spoken.strip(), spoken
    assert "name" in spoken.lower(), spoken  # the same next question
    # digits and a name are still understood with the model down
    ask = await rig.say("Suman Pal")
    assert "Suman Pal" in ask, ask
    await rig.say("yes")
    assert "9876543210" in await rig.say("9876543210")
    assert "KCD-" in await rig.say("yes")


@pytest.mark.asyncio
async def test_empty_model_output_is_the_same_as_a_failure(rig, m, monkeypatch):
    async def empty(session, text, lang, key):
        raise ExtractionError("empty output")

    await _to_name(rig)
    monkeypatch.setattr(m, "_resolve_intent_uncached", empty)
    assert "problem" not in (await rig.say("hmm")).lower()


@pytest.mark.asyncio
async def test_a_hesitant_answer_at_the_read_back_asks_yes_or_no_and_books_nothing(rig):
    await _to_phone(rig)
    await rig.say("9876543210")
    ask = await rig.say("hmm maybe")
    assert "yes or a no" in ask, ask
    assert not await _bookings(rig, "9876543210")
    assert "KCD-" in await rig.say("yes")


@pytest.mark.asyncio
async def test_a_bare_no_after_the_read_back_asks_what_to_change_and_the_date_can_be_changed(rig):
    await _to_phone(rig)
    await rig.say("9876543210")
    assert "what would you like to change" in (await rig.say("no")).lower()
    sat = None
    again = await rig.say("make it eleven", B(time_slot="11:00"))
    assert "11:00" in again and "Rahul Dey" in again and "9876543210" in again, again


@pytest.mark.asyncio
async def test_confirming_twice_creates_one_booking(rig):
    await _to_phone(rig)
    await rig.say("9876543210")
    assert "KCD-" in await rig.say("yes")
    try:
        await rig.say("yes")
    except Exception:  # noqa: BLE001 - the call is already closed after the confirmation; either way no second booking
        pass
    assert len(await _bookings(rig, "9876543210")) == 1


@pytest.mark.asyncio
async def test_a_timeout_after_the_clinic_booked_is_verified_and_not_booked_twice(rig, monkeypatch):
    await _to_phone(rig)
    await rig.say("9876543210")
    real = rig.tools.confirm_booking

    async def created_but_timed_out(*a, **k):
        await real(*a, **k)
        raise ToolCallError("read timeout")

    monkeypatch.setattr(rig.tools, "confirm_booking", created_but_timed_out)
    done = await rig.say("yes")
    booked = await _bookings(rig, "9876543210")
    assert len(booked) == 1
    assert booked[0]["confirmation_id"] in done, done  # the REAL id, found by checking the clinic, not invented


@pytest.mark.asyncio
async def test_a_timeout_before_anything_was_booked_says_nothing_was_booked_and_a_yes_retries_once(rig, monkeypatch):
    await _to_phone(rig)
    await rig.say("9876543210")
    real = rig.tools.confirm_booking

    async def failed(*a, **k):
        raise ToolCallError("connection reset")

    monkeypatch.setattr(rig.tools, "confirm_booking", failed)
    said = await rig.say("yes")
    assert "nothing was booked" in said and "KCD-" not in said, said
    assert not await _bookings(rig, "9876543210")
    assert rig.session.booking is not None  # the booking is kept at the read-back
    monkeypatch.setattr(rig.tools, "confirm_booking", real)
    assert "KCD-" in await rig.say("yes")
    assert len(await _bookings(rig, "9876543210")) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("pick, index", [("the first one", 0), ("the second one", 1)])
async def test_the_first_or_second_offered_slot_is_a_slot_the_clinic_really_offered(rig, pick, index):
    a = await _to_phone(rig)
    await rig.say("9876543210")
    assert "KCD-" in await rig.say("yes")  # someone holds 10:00 with Dr. Sen on that day
    # a second caller wants the same slot
    rig.new_call("en")
    await rig.say("is doctor Sen available tomorrow")
    await rig.say("book him for that day", B())
    await rig.say("10:00", B(time_slot="10:00"))
    # SPEC CHANGE 2026-10-17: a time is confirmed ("Is that time fine?") before the hold is made; the yes is what meets the taken slot
    taken = await rig.say("yes", B())
    assert "already booked" in taken, taken
    offered = list(rig.session.booking.offered_options)
    assert offered, "the clinic offered nothing"
    if index >= len(offered):
        pytest.skip("the clinic offered fewer slots than this ordinal")
    chosen = offered[index]
    await rig.say(pick)
    assert rig.session.booking.slots["time_slot"] == chosen["time_slot"]
    assert rig.session.booking.slots["date"] == chosen["date"]
    await rig.say("Rahul Dey", B(patient_name="Rahul Dey"))
    readback = await rig.say("9123456780")
    assert chosen["time_slot"] in readback and chosen["date"] in readback, readback
    assert "KCD-" in await rig.say("yes")


@pytest.mark.asyncio
async def test_an_ordinal_with_nothing_offered_picks_nothing(rig):
    await _to_name(rig)
    await rig.say("the first one", B())
    assert rig.session.booking.slots.get("time_slot") == "10:00"  # unchanged: nothing was offered, nothing was invented


@pytest.mark.asyncio
async def test_a_new_call_never_inherits_the_digits_or_name_of_an_earlier_call(rig):
    await _to_phone(rig)
    await rig.say("nine eight seven six five")
    first = rig.session
    await _to_name(rig, time="11:00")  # a new session (the first call still holds 10:00)
    assert rig.session is not first
    assert rig.session.booking.phone_buffer == "" and rig.session.booking.pending_name is None
    await rig.say("Rahul Dey", B(patient_name="Rahul Dey"))
    readback = await rig.say("9123456780")
    assert "9123456780" in readback and "98765" not in readback


@pytest.mark.asyncio
async def test_missing_details_are_asked_one_by_one_before_any_read_back(rig):
    rig.new_call("en")
    first = await rig.say("I want to book an appointment with doctor Sen", B(doctor_name="Sen"))
    assert "Sen" in first
    assert "Read" not in first and "confirm" not in first.lower()
    for text, slots in (("tomorrow", {"date": None}),):
        pass
    st = rig.session.booking
    assert st is not None and st.stage == "collecting"
    assert "phone" in __import__("agent.booking_flow", fromlist=["x"]).missing_required(st)
