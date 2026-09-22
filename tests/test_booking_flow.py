"""agent/booking_flow.py: the cross-turn slot-filling state machine that
KCD-357, 358, 361 (relative-date reuse), 364, 366, 367, 368 all depend on.
Pure and offline -- no network, no database, no GPU.

    python -m pytest tests/test_booking_flow.py -v
"""
import time

from agent.booking_flow import (
    classify_yes_no,
    effective_phone,
    is_ready_to_confirm,
    mark_confirming,
    merge_slots,
    merge_spelling,
    missing_required,
    new_state,
    reopen_for_correction,
    try_assemble_spelling,
)


def test_one_sentence_booking_fills_every_slot_at_once():
    # KCD-357: caller gives doctor, date, time, name, phone all in one turn.
    state = new_state("book_appointment")
    changed = merge_slots(state, {
        "doctor_name": "Sen", "date": "2026-10-01", "time_slot": "18:15",
        "patient_name": "Ravi", "phone": "9800000001",
    })
    assert set(changed) == {"doctor_name", "date", "time_slot", "patient_name", "phone"}
    assert missing_required(state) == []
    assert is_ready_to_confirm(state)


def test_guided_flow_fills_one_slot_per_turn_without_forgetting_earlier_ones():
    # KCD-358: doctor only, then the rest across later turns.
    state = new_state("book_appointment")
    merge_slots(state, {"doctor_name": "Sen"})
    assert missing_required(state) == ["date", "time_slot", "patient_name", "phone"]

    merge_slots(state, {"date": "2026-10-01", "time_slot": "18:15"})
    assert state.slots["doctor_name"] == "Sen"          # not forgotten
    assert missing_required(state) == ["patient_name", "phone"]

    merge_slots(state, {"patient_name": "Ravi", "phone": "9800000001"})
    assert is_ready_to_confirm(state)


def test_changing_mind_mid_booking_updates_only_that_slot():
    # KCD-366: caller picks a different doctor halfway through.
    state = new_state("book_appointment")
    merge_slots(state, {"doctor_name": "Sen", "date": "2026-10-01", "time_slot": "18:15"})
    merge_slots(state, {"patient_name": "Ravi", "phone": "9800000001"})
    assert is_ready_to_confirm(state)

    mark_confirming(state)
    changed = merge_slots(state, {"doctor_name": "Ghosh"})
    assert changed == ["doctor_name"]
    assert state.slots["date"] == "2026-10-01"           # everything else kept
    assert state.slots["patient_name"] == "Ravi"
    assert state.stage == "collecting"                    # re-opened, not committed blind


def test_correction_after_confirmation_reopens_collection_and_keeps_the_rest():
    # KCD-367: caller spots a mistake between "shall I confirm" and commit.
    state = new_state("book_appointment")
    merge_slots(state, {"doctor_name": "Sen", "date": "2026-10-01", "time_slot": "18:15",
                         "patient_name": "Ravi", "phone": "9800000001"})
    mark_confirming(state)
    assert state.stage == "confirming"

    reopen_for_correction(state, "phone", "9800000099")
    assert state.stage == "collecting"
    assert state.slots["phone"] == "9800000099"
    assert state.slots["doctor_name"] == "Sen"            # nothing else lost
    assert is_ready_to_confirm(state)                      # ready again immediately


def test_multi_test_booking_accumulates_across_turns_without_duplicates():
    # KCD-365: several tests named across turns.
    state = new_state("book_test")
    merge_slots(state, {"test_names": ["Uric Acid", "Lipid Profile"]})
    merge_slots(state, {"test_names": ["Lipid Profile", "CBC"]})   # a repeat + a new one
    assert state.test_names == ["Uric Acid", "Lipid Profile", "CBC"]


def test_reschedule_and_cancel_and_add_test_have_their_own_required_sets():
    r = new_state("reschedule_appointment")
    merge_slots(r, {"confirmation_id": "KCD-1"})
    assert missing_required(r) == ["new_date", "new_time_slot"]

    c = new_state("cancel_appointment")
    assert missing_required(c) == ["confirmation_id"]

    a = new_state("add_test_booking")
    merge_slots(a, {"confirmation_id": "KCD-1"})
    assert missing_required(a) == ["test_name"]


def test_lookup_accepts_either_phone_or_confirmation_id():
    l1 = new_state("lookup_booking")
    assert missing_required(l1) == ["phone"]
    merge_slots(l1, {"phone": "9800000001"})
    assert missing_required(l1) == []

    l2 = new_state("lookup_booking")
    merge_slots(l2, {"confirmation_id": "KCD-1"})
    assert missing_required(l2) == []


def test_state_becomes_stale_after_the_idle_timeout(monkeypatch):
    state = new_state("book_appointment")
    assert not state.is_stale()
    state.last_updated = time.monotonic() - 999
    assert state.is_stale()


def test_classify_yes_no_across_languages():
    assert classify_yes_no("হ্যাঁ ঠিক আছে", "bn") == "yes"
    assert classify_yes_no("না না", "bn") == "no"
    assert classify_yes_no("हाँ बुक कर दीजिए", "hi") == "yes"
    assert classify_yes_no("नहीं रद्द करो", "hi") == "no"
    assert classify_yes_no("yes that's correct", "en") == "yes"
    assert classify_yes_no("no that's wrong", "en") == "no"
    assert classify_yes_no("what did you say", "en") is None


def test_classify_yes_no_checks_no_before_yes_for_negated_correct():
    # "that's not correct" contains "correct" (a yes-word) but must read as no.
    assert classify_yes_no("that's not correct", "en") == "no"
    assert classify_yes_no("ঠিক না", "bn") == "no"


def test_spelling_assembly_ignores_non_letters():
    assert try_assemble_spelling(["r", "a", "3", "v", "", "i"]) == "ravi"
    assert try_assemble_spelling(["", "  "]) is None


def test_spelling_merges_across_turns():
    state = new_state("book_appointment")
    merge_spelling(state, ["r", "a"])
    combined = merge_spelling(state, ["v", "i"])
    assert combined == "ravi"


def test_a_declined_phone_number_stops_blocking_the_booking():
    # KCD-370: refusal still completes the booking, never blocks forever.
    state = new_state("book_appointment")
    merge_slots(state, {"doctor_name": "Sen", "date": "2026-10-01", "time_slot": "18:15",
                         "patient_name": "Ravi"})
    assert missing_required(state) == ["phone"]
    state.phone_declined = True
    assert missing_required(state) == []
    assert is_ready_to_confirm(state)


def test_effective_phone_prefers_contact_phone_then_falls_back_to_sentinel():
    state = new_state("book_appointment")
    assert effective_phone(state) == "not_provided"
    merge_slots(state, {"phone": "9800000001"})
    assert effective_phone(state) == "9800000001"
    merge_slots(state, {"contact_phone": "9800000099"})
    assert effective_phone(state) == "9800000099"
