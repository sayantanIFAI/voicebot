"""agent/llm.py's schema extension for Epic E26's new intents. Only
exercises the offline, deterministic parts (VALID_INTENTS, _validate) --
no Ollama call, so this runs anywhere.

    python -m pytest tests/test_llm_booking_intents.py -v
"""
from agent.llm import VALID_INTENTS, _validate


def test_new_booking_intents_are_recognised():
    for intent in ("book_test", "reschedule_appointment", "cancel_appointment",
                    "lookup_booking", "add_test_booking", "resend_confirmation",
                    "department_query"):
        assert intent in VALID_INTENTS


def _full_slots(**overrides) -> dict:
    slots = {k: None for k in (
        "test_name", "test_names", "doctor_name", "date", "time_slot", "new_date",
        "new_time_slot", "confirmation_id", "patient_name", "patient_age", "phone",
        "contact_phone", "relationship", "spelled_letters", "symptom_description", "faq_topic",
    )}
    slots.update(overrides)
    return slots


def test_reschedule_intent_validates():
    data = {"intent": "reschedule_appointment",
            "slots": _full_slots(confirmation_id="KCD-1", new_date="2026-10-05", new_time_slot="18:15"),
            "direct_reply_bn": None}
    ok, errors = _validate(data)
    assert ok, errors


def test_department_query_validates_with_symptom_description():
    data = {"intent": "department_query",
            "slots": _full_slots(symptom_description="chest pain since morning"),
            "direct_reply_bn": None}
    ok, errors = _validate(data)
    assert ok, errors


def test_invalid_intent_is_rejected():
    data = {"intent": "delete_all_bookings", "slots": _full_slots(), "direct_reply_bn": None}
    ok, errors = _validate(data)
    assert not ok
    assert any("invalid intent" in e for e in errors)


def test_model_overstepping_direct_reply_on_a_non_smalltalk_intent_is_stripped():
    data = {"intent": "book_test", "slots": _full_slots(test_names=["Uric Acid"]),
            "direct_reply_bn": "your test costs 250 taka"}
    ok, _ = _validate(data)
    assert ok
    assert data["direct_reply_bn"] is None    # the model never gets to state a fact
