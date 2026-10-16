"""sourav, 2026-10-16: regression tests for the three real calls (one Hindi, two Bengali) where a booking fell apart:
  * Bengali / Hindi spoken times ("দশটার সময়", "10 ta", "दस बजे", a garbled "Doster") were not understood, and a bare "ten" became 22:00 although
    the clinic sits in the daytime -- then "that time is not right" was said with no hint of the hours;
  * a bare "कल" (tomorrow) after the doctor's sitting time lost the booking ("which doctor?");
  * a ten-digit number was refused as "not 10 digits", digit words written in Bengali / Devanagari letters ("ডাবল সিক্স") were not read, and a
    sentence with the name AND the number was hijacked by a booking lookup ("no upcoming appointment found").
Unit tests on the pure functions, then the same cases through the REAL orchestrator / fast path / templates / clinic API (stand-ins: only the
recogniser text, TTS and the intent model -- NOT a live call). Phone numbers here are synthetic test values."""

import datetime

import pytest
from test_booking_capture_e2e import _ASK, _BOOK, _bookings, B  # noqa: F401
from test_doctor_booking_e2e import clinic, m, rig  # noqa: F401  (fixtures)

from agent import booking_capture as bc

TODAY = datetime.date(2026, 10, 16)
CHAMBER = [(10 * 60, 12 * 60)]  # a doctor who sits 10:00-12:00


# ------------------------------------------------------------------------------------------------------------ spoken times


@pytest.mark.parametrize(
    "text, want",
    [
        ("দশটার সময়", "10:00"),  # Bengali, the word for time is সময়
        ("দশটা", "10:00"),
        ("১০ টা", "10:00"),  # Bengali numerals + ta
        ("10 ta", "10:00"),  # Latin letters
        ("ten ta", "10:00"),
        ("এগারোটায়", "11:00"),
        ("সাড়ে দশটা", "10:30"),
        ("दस बजे", "10:00"),
        ("साढ़े दस बजे", "10:30"),
        ("ग्यारह बजे", "11:00"),
        ("ten o'clock", "10:00"),
        ("eleven o'clock", "11:00"),
        ("ten thirty", "10:30"),
        ("at 11", "11:00"),
        ("Doster", "10:00"),  # the recogniser's spelling of দশটার
        ("Agar uta", "11:00"),  # এগারোটা -- the sound also fits চার (4) but 4 is not inside the doctor's hours
        ("Tiger Uta", "10:00"),
    ],
)
def test_a_spoken_time_is_read_as_the_daytime_hour_inside_the_chamber(text, want):
    got = bc.read_when(text, True, window=CHAMBER, today=TODAY)
    assert got.time == want and got.pure, got


def test_an_explicit_evening_word_is_respected_and_a_bare_hour_is_never_night():
    assert bc.read_when("রাত দশটা", True, today=TODAY).time == "22:00"
    assert bc.read_when("5 pm", True, today=TODAY).time == "17:00"
    assert bc.read_when("सुबह दस बजे", True, today=TODAY).time == "10:00"
    assert bc.read_when("ten", True, today=TODAY).time == "10:00"  # no window known: the clinic's own 8-20 day
    assert bc.read_when("2 ta", True, today=TODAY).time == "14:00"  # 2 can only be the afternoon in an 8-20 day


@pytest.mark.parametrize("text", ["I have 3 days of fever", "what does the CBC cost", "book kar do", "Ravi Das", "is the doctor in"])
def test_ordinary_sentences_are_never_taken_for_a_time(text):
    assert bc.read_when(text, True, window=CHAMBER, today=TODAY).time is None


def test_the_model_reading_ten_as_22_is_moved_back_to_the_daytime_only_when_no_evening_was_said():
    assert bc.fix_model_time("22:00", "ten", CHAMBER) == "10:00"
    assert bc.fix_model_time("22:00", "দশটার সময়", None) == "10:00"
    assert bc.fix_model_time("22:00", "ten pm", CHAMBER) == "22:00"  # the caller said pm: not our call
    assert bc.fix_model_time("11:00", "eleven", CHAMBER) == "11:00"


def test_a_bare_tomorrow_in_each_language_is_tomorrow():
    tomorrow = (TODAY + datetime.timedelta(days=1)).isoformat()
    for text in ("कल", "কাল", "tomorrow", "কাল সকাল দশটায়"):
        assert bc.read_when(text, False, window=CHAMBER, today=TODAY).date == tomorrow, text
    assert bc.read_when("day after tomorrow", False, today=TODAY).date == (TODAY + datetime.timedelta(days=2)).isoformat()


def test_the_chamber_hours_are_spoken_from_the_clinics_own_slot_list():
    assert bc.hours_span(["10:00", "10:15", "11:45"]) == ("10:00 AM", "12:00 PM")
    assert bc.hours_span(None) is None


# --------------------------------------------------------------------------------------------------------------------- phone


@pytest.mark.parametrize(
    "said",
    [
        "Seven Eight Zero Eight Seven Nine Seven Nine Double Six",
        "সেভেন এইট জিরো এইট সেভেন নাইন সেভেন নাইন ডাবল সিক্স",
        "सेवन एट जीरो एट सेवन नाइन सेवन नाइन डबल सिक्स",
        "সাত আট শূন্য আট সাত নয় সাত নয় ডাবল ছয়",
        "7808797966",
    ],
)
def test_a_ten_digit_number_is_read_in_every_spelling_the_recogniser_uses(said):
    step = bc.absorb("", said, True)
    assert step.status == "complete" and step.number == "7808797966", step


def test_ten_digits_that_start_with_zero_are_refused_for_the_right_reason():
    step = bc.absorb("", "zero one two three four five six seven eight nine", True)
    assert step.status == "invalid" and step.reason == "start" and step.n == 0
    assert "starts with 6, 7, 8 or 9" in bc.phone_invalid_prompt(step, "en")
    assert "10-digit" not in bc.phone_invalid_prompt(step, "en")  # the old, untrue "not a 10-digit number"
    step = bc.absorb("", "9 8 7 6 5 4 3 2 1 0 1 2", True)
    assert step.status == "invalid" and step.reason == "length" and step.n == 12
    assert "12 digits" in bc.phone_invalid_prompt(step, "en")


def test_a_number_said_as_a_number_is_asked_again_digit_by_digit_never_read_back_wrong():
    step = bc.absorb("", "the number is four five hundred seven eighty-nine", True)
    assert step.status == "compound"
    assert bc.absorb("", "I have fever since twenty days", False).status == "none"
    assert bc.absorb("", "eleven", True).status == "none"  # an hour, not a number


# ------------------------------------------------------------------------- the same cases through the real orchestrator


_WANT = {
    "en": "I want to book an appointment with Dr. Sen",
    "hi": "डॉक्टर सेन के साथ अपॉइंटमेंट बुक करना है",
    "bn": "ডাক্তার সেনের অ্যাপয়েন্টমেন্ট বুক করতে চাই",
}


async def _to_time_question(rig, lang):
    rig.new_call(lang)
    await rig.say(_ASK[lang])
    return await rig.say(_BOOK[lang], B())


@pytest.mark.asyncio
@pytest.mark.parametrize("lang, day_word", [("hi", "कल"), ("bn", "কাল"), ("en", "tomorrow")])
async def test_a_bare_day_after_the_doctors_sitting_time_stays_in_the_booking(rig, lang, day_word):
    rig.new_call(lang)
    await rig.say(_WANT[lang], B(doctor_name="Sen"))  # the doctor's sitting time is spoken, the day is asked
    # the model stand-in labels the bare day the way the live model did: a doctor-availability question with NO doctor
    reply = await rig.say(day_word, [{"intent": "doctor_availability", "slots": {}}])
    st = rig.session.booking
    assert st is not None and st.action == "book_appointment", reply
    assert not any(w in reply.lower() for w in ("which doctor", "कौन से डॉक्टर", "কোন ডাক্তার")), reply


@pytest.mark.asyncio
async def test_ten_o_clock_is_booked_at_ten_even_if_the_model_says_22(rig):
    await _to_time_question(rig, "en")
    reply = await rig.say("ten o'clock", B(time_slot="22:00"))  # the live model's reading of a bare "ten"
    st = rig.session.booking
    assert st.slots.get("time_slot") == "10:00", (st.slots, reply)


@pytest.mark.asyncio
async def test_a_time_outside_the_chamber_is_answered_with_the_real_hours(rig):
    await _to_time_question(rig, "en")
    reply = await rig.say("eight pm", B(time_slot="20:00"))
    assert "sits from" in reply and "AM" in reply and "PM" in reply, reply


@pytest.mark.asyncio
async def test_the_ten_digit_number_with_double_six_reaches_the_read_back(rig):
    rig.new_call("en")
    await rig.say(_ASK["en"])
    await rig.say(_BOOK["en"], B())
    await rig.say("ten o'clock", B(time_slot="10:00"))
    await rig.say("Rahul Dey", B(patient_name="Rahul Dey"))
    reply = await rig.say("Seven Eight Zero Eight Seven Nine Seven Nine Double Six")
    assert "7808797966" in reply, reply


@pytest.mark.asyncio
async def test_a_name_and_a_number_said_as_a_number_is_not_hijacked_by_a_lookup(rig):
    rig.new_call("en")
    await rig.say(_ASK["en"])
    await rig.say(_BOOK["en"], B())
    await rig.say("ten o'clock", B(time_slot="10:00"))
    reply = await rig.say(
        "The patient's name is Saurabh and the number is four five hundred seven eighty-nine",
        [{"intent": "lookup_booking", "slots": {"patient_name": "Saurabh"}}],
    )
    low = reply.lower()
    assert "no upcoming appointment" not in low and "did you say the number" not in low, reply
    assert "digit" in low, reply  # asked for the number digit by digit
    st = rig.session.booking
    assert st.action == "book_appointment" and st.slots.get("patient_name") == "Saurabh"
