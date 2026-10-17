"""sourav, 2026-10-16: unit tests for agent/booking_capture.py -- phone digits (words, scripts, "double nine", pieces), name reading,
ordinal slot pick, hedges. Pure functions: no clinic, no model."""

import pytest

from agent import booking_capture as bc
from agent.booking_flow import BookingState


@pytest.mark.parametrize(
    "said, number",
    [
        ("nine eight seven six five four three two one zero", "9876543210"),
        ("9876543210", "9876543210"),
        ("my number is 98765 43210", "9876543210"),
        ("plus 91 98765 43210", "9876543210"),
        ("double nine eight seven six five four three two one", "9987654321"),
        ("nine nine eight triple seven six five four", None),  # 9+9+8+777+6+5+4 = 9 digits: not a number
        ("triple eight seven six five four three two one", "8887654321"),
        ("नौ आठ सात छह पाँच चार तीन दो एक शून्य", "9876543210"),
        ("९८७६५४३२१०", "9876543210"),
        ("নয় আট সাত ছয় পাঁচ চার তিন দুই এক শূন্য", "9876543210"),
        ("৯৮৭৬৫৪৩২১০", "9876543210"),
        ("nau aath saat chhe paanch char teen do ek shunya", "9876543210"),
        ("ডাবল নয় আট সাত ছয় পাঁচ চার তিন দুই এক", "9987654321"),
        ("डबल नौ आठ सात छह पाँच चार तीन दो एक", "9987654321"),
    ],
)
def test_spoken_phone_numbers_are_read_digit_by_digit(said, number):
    assert bc.normalise_phone(said) == number


@pytest.mark.parametrize("bad", ["1234567", "5876543210", "98765432101", "", None, "twenty five", "call me tomorrow"])
def test_a_value_that_is_not_a_ten_digit_mobile_number_is_rejected(bad):
    assert bc.normalise_phone(bad) is None


def test_a_number_said_over_several_turns_is_joined():
    buf = ""
    for part, status in (("nine eight seven six five", "partial"), ("four three two", "partial"), ("one zero", "complete")):
        step = bc.absorb(buf, part, asked=True)
        assert step.status == status
        buf = step.buffer
    assert step.number == "9876543210"


def test_a_repeated_full_number_replaces_it_and_is_never_appended():
    first = bc.absorb("", "9876543210", asked=True)
    again = bc.absorb("", "9876543210", asked=True)
    assert first.number == again.number == "9876543210"
    # a restated half is not added twice
    assert bc.absorb("98765", "98765", asked=True).buffer == "98765"
    assert bc.absorb("98765", "98765 43", asked=True).buffer == "9876543"


def test_a_corrected_number_replaces_the_old_one():
    assert bc.absorb("", "no it is 9123456780", asked=False).number == "9123456780"


def test_digits_of_an_unrelated_sentence_are_not_taken_unless_the_phone_was_asked():
    assert bc.absorb("", "I want 5 pm on the 3rd", asked=False).status == "none"
    assert bc.absorb("", "at five", asked=False).status == "none"


def test_ten_digits_that_are_not_a_mobile_number_ask_again_and_keep_nothing():
    step = bc.absorb("", "1 2 3 4 5 6 7 8 9 0", asked=True)
    assert step.status == "invalid" and step.buffer == ""


def test_start_again_clears_the_kept_digits():
    assert bc.absorb("98765", "sorry wrong number", asked=True).status == "cleared"
    assert bc.absorb("98765", "ভুল হয়েছে", asked=True).status == "cleared"


def test_the_phone_is_masked_in_logs():
    assert bc.mask("9876543210") == "********10"


@pytest.mark.parametrize(
    "said, name",
    [
        ("my name is rahul das", "Rahul Das"),
        ("Rahul Das", "Rahul Das"),
        ("rahul das hai", "Rahul Das"),
        ("mera naam Priya Sen hai", "Priya Sen"),
        ("amar naam suman pal", "Suman Pal"),
        ("আমার নাম রাহুল দাস", "রাহুল দাস"),
        ("मेरा नाम प्रिया सेन है", "प्रिया सेन"),
        ("Anirban", "Anirban"),
    ],
)
def test_a_name_is_read_from_the_callers_own_words(said, name):
    assert bc.candidate_name(said) == name


@pytest.mark.parametrize("said", ["yes", "no", "tomorrow", "book appointment", "what is your name?", "9876543210", "my number is 98765", "", "hmm", "হ্যাঁ", "नहीं", "one two three four five"])
def test_things_that_are_not_names_are_never_taken_as_one(said):
    assert bc.candidate_name(said) is None


def test_the_models_name_must_be_in_the_callers_words():
    assert bc.name_is_grounded("Rahul Dey", "Rahul Dey")
    assert bc.name_is_grounded("Rahul Dey", "my name is Rahul Dei")
    assert not bc.name_is_grounded("Rahul Dey", "ok thanks")
    assert not bc.name_is_grounded("Rahul Dey", "রমেশ ঘোষ")


@pytest.mark.parametrize(
    "said, index",
    [("the first one", 0), ("second one please", 1), ("the last one", 2), ("পহেলা", 0), ("पहला वाला", 0), ("दूसरा", 1), ("শেষেরটা", 2), ("pehla", 0), ("doosra", 1)],
)
def test_ordinal_picks_one_of_the_slots_actually_offered(said, index):
    offered = ["10:30", "11:00", "11:30"]
    assert bc.pick_offered_slot(said, offered) == offered[index]


def test_an_ordinal_past_the_end_or_none_picks_nothing():
    assert bc.pick_offered_slot("the third one", ["10:30", "11:00"]) is None
    assert bc.pick_offered_slot("ok", ["10:30"]) is None
    assert bc.pick_offered_slot("the first one", []) is None


def test_options_from_a_refused_hold_come_from_the_clinic_only():
    slots = {"date": "2026-10-20"}
    assert bc.options_from_hold({"alternative_slots": ["10:30", "11:00"]}, slots) == [
        {"date": "2026-10-20", "time_slot": "10:30"},
        {"date": "2026-10-20", "time_slot": "11:00"},
    ]
    assert bc.options_from_hold({"alternative_options": [{"date": "2026-10-21", "time_slot": "10:00"}]}, slots) == [{"date": "2026-10-21", "time_slot": "10:00"}]
    assert bc.options_from_hold({}, slots) == []


@pytest.mark.parametrize("said", ["hmm", "maybe", "I am not sure", "शायद", "হুম"])
def test_a_hedge_is_neither_yes_nor_no(said):
    assert bc.is_hedge(said)


@pytest.mark.parametrize("said", ["make it eleven", "no", "yes", "change the date to Monday please"])
def test_a_correction_or_a_plain_answer_is_not_a_hedge(said):
    assert not bc.is_hedge(said)


def _booking(**slots):
    st = BookingState(action="book_appointment")
    st.slots.update(slots)
    return st


def test_a_pure_digit_sentence_fills_the_phone_without_the_model():
    st = _booking(doctor_name="Sen", date="2026-10-20", time_slot="10:00", patient_name="Rahul Dey")
    st.asked_fields.add("phone")
    plan = bc.before_model(st, "98765 43210", "en")
    assert plan.skip_model and plan.slots == {"phone": "9876543210"}


def test_partial_digits_are_kept_on_this_booking_only_and_the_rest_is_asked_for():
    st = _booking(doctor_name="Sen", date="2026-10-20", time_slot="10:00", patient_name="Rahul Dey")
    st.asked_fields.add("phone")
    plan = bc.before_model(st, "98765", "en")
    assert st.phone_buffer == "98765" and plan.speak == "Okay, go on."  # SPEC CHANGE 2026-10-18: an "ok" in the gap, not a count
    other = _booking(doctor_name="Rao")
    assert other.phone_buffer == ""  # a second caller's booking starts empty


def test_the_models_bad_phone_never_reaches_the_booking():
    st = _booking(doctor_name="Sen")
    slots, ask = bc.after_model(st, "my number is one two three", "en", {"phone": "123"}, bc.TurnPlan())
    assert "phone" not in slots and "10-digit" in ask


def test_a_name_the_caller_never_said_is_read_back_before_it_counts():
    # sourav, 2026-10-16 (final verification): the read-back happens when the NAME is the question being asked (doctor, day and time are
    # already known, so patient_name is the next missing field). Same assertions as before; only the booking state now says so.
    st = _booking(doctor_name="Sen", date="2026-10-20", time_slot="10:00")
    # the model says "Rahul Dey" but the caller said "Ramesh Ghosh": the caller's words win, and are read back
    slots, ask = bc.after_model(st, "Ramesh Ghosh", "en", {"patient_name": "Rahul Dey"}, bc.TurnPlan())
    assert "patient_name" not in slots and "Ramesh Ghosh" in ask and "Rahul" not in ask and st.pending_name == "Ramesh Ghosh"
    yes = bc.before_model(st, "yes", "en")
    assert yes.skip_model and yes.slots == {"patient_name": "Ramesh Ghosh"} and st.name_confirmed == "Ramesh Ghosh"
    # no name in the caller's words at all: the model's guess is dropped and the name is asked again
    st2 = _booking(doctor_name="Sen", date="2026-10-20", time_slot="10:00")
    slots, ask = bc.after_model(st2, "ok thanks", "en", {"patient_name": "Rahul Dey"}, bc.TurnPlan())
    assert "patient_name" not in slots and st2.pending_name is None and "name again" in ask


def test_an_invented_name_is_dropped_silently_when_the_name_is_not_what_is_being_asked():
    """sourav, 2026-10-16 (final verification): "all of it" / "that is all" said while the booking still needs the day must never become
    "I heard the name All Of It. Is that correct?". The model's invented name is dropped, nothing is read back, and the flow carries on."""
    st = _booking(doctor_name="Sen")  # the day and time are still missing: the name is not due
    slots, ask = bc.after_model(st, "all of it", "en", {"patient_name": "Ravi Das", "date": "2026-10-20"}, bc.TurnPlan())
    assert "patient_name" not in slots and slots["date"] == "2026-10-20" and ask is None and st.pending_name is None


def test_a_stale_or_finished_booking_is_left_alone():
    st = _booking(doctor_name="Sen")
    st.stage = "done"
    assert bc.before_model(st, "98765 43210", "en") == bc.TurnPlan()
    assert bc.before_model(None, "98765 43210", "en") == bc.TurnPlan()


def test_name_in_another_script_is_grounded_by_sound_but_an_invented_name_is_not():
    """sourav 2026-10-16: the model writes the caller's Devanagari/Bengali name in Latin letters -> same name, accept; a name never said -> not."""
    from agent import booking_capture as bc

    assert bc.name_is_grounded("Amit Roy", "कल दस बजे अपॉइंटमेंट अमित रॉय के नाम")
    assert bc.name_is_grounded("Amit Roy", "কাল দশটায় অ্যাপয়েন্টমেন্ট অমিত রায় ফোন")
    assert not bc.name_is_grounded("Rahul Sharma", "कल दस बजे अपॉइंटमेंट अमित रॉय के नाम")
