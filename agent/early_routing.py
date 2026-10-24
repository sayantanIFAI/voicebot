"""The ORDER in which a caller's turn is checked for the requests that must never reach the intent model: one place, used by BOTH
orchestrator entry points (main.py and main_pcm.py), so the two cannot drift apart again.

Why this exists: the cue functions (asks_for_a_person, wants_doctor_directly, wants_to_complain, asks_about_dues, report_pickup,
referral_cues, owner_answers.asks_about_health_package) were first wired into main.py only. main_pcm.py -- the entry point the Twilio phone
bridge runs -- never called them, so over a real phone call a complaint, a "money left to pay?" question or a health-package question went
to the intent model (a health-package question was answered with the insurance text). A cue
function that nothing calls fixes nothing; a single `classify()` that both entry points call, with its order tested, does.

This module only DECIDES which request it is. What is said or done about it (a handoff, a fixed reply, a queued callback) stays in the
orchestrator, because that needs the live session. Pure functions, no I/O, no model: unit-testable without the voice stack.

Order (first match wins), strongest reason first:
  1. PERSON          -- "let me talk to someone" (KCD-353): honoured immediately.
  2. DOCTOR_DIRECT   -- a talking verb joined to "the doctor" (KCD-lead-10): a human arranges it.
  3. COMPLAINT       -- (KCD-lead-12): a human takes it.
  3b. LINK_MISSING   -- "I did not get the payment link" (owner, 2026-10-23): network issue, wait 15 minutes, call back.
  4. DUES            -- "do I owe anything / is a payment pending" (KCD-lead-9): proof on WhatsApp + callback, never a balance.
  5. HEALTH_PACKAGE  -- Sonoscan offers none (KCD-lead-13): said BEFORE the FAQ matcher or the intent model can send it to the insurance answer.
  6. REPORT_PICKUP   -- "is my report ready" (KCD-lead-8).
  7. REFERRAL        -- another center passing a number (KCD-lead-1).
  8. GOOD_DOCTOR / DOCTOR_WEEKDAYS / NEXT_IF_ABSENT -- three doctor questions answered by rule (agent/doctor_enquiry_cues.py, owner 2026-10-23).
  9. WHEN_TO_COME / SERIAL_NUMBER -- "when should I come" (sitting start time + one to two hours in hand) and "what is my serial number" (it comes on WhatsApp
     after payment); skipped when the sentence carries a token number, which owner_answers answers more exactly.
  10. CANNOT_PAY_ONLINE (item 17, checked right after LINK_MISSING), SITTING_HOURS (item 16, "kota theke kota bose"), COME_TO_CLINIC (item 17, book at the
     clinic) and WRITE_NAME (item 22, "naam lekhanor chhilo": asks which doctor) -- owner, 2026-10-23.
  11. RARE_TEST (item 7, after HEALTH_PACKAGE): a scan / test of a body part the lab table has none for, or a TB test -- a person takes it (agent/rare_test_cues.py).
      DAY_AVAILABILITY (item 25, before DOCTOR_WEEKDAYS): "samner robibar doctor x ke pabo" -- the doctor on that coming weekday (agent/weekday_cues.py).
Genuine insurance / cashless questions match none of these and keep using the existing FAQ flow.
"""

from __future__ import annotations

from agent import doctor_enquiry_cues, owner_answers, rare_test_cues, referral_cues, report_pickup, weekday_cues
from agent.billing_cues import asks_about_dues, payment_link_missing
from agent.human_request import asks_for_a_person, wants_doctor_directly, wants_to_complain

PERSON = "person"
DOCTOR_DIRECT = "doctor_direct"
COMPLAINT = "complaint"
DUES = "dues"
HEALTH_PACKAGE = "health_package"
REPORT_PICKUP = "report_pickup"
REFERRAL = "referral"
LINK_MISSING = "link_missing"
GOOD_DOCTOR = "good_doctor"
DOCTOR_WEEKDAYS = "doctor_weekdays"
NEXT_IF_ABSENT = "next_if_absent"
WHEN_TO_COME = "when_to_come"
SERIAL_NUMBER = "serial_number"
CANNOT_PAY_ONLINE = "cannot_pay_online"
SITTING_HOURS = "sitting_hours"
COME_TO_CLINIC = "come_to_clinic"
WRITE_NAME = "write_name"
PAY_NOW = "pay_now"  # sourav, 2026-10-24: "do we have to pay now?" -> not paying now is fine, book later at the clinic, no slot guarantee
RARE_TEST = "rare_test"
DAY_AVAILABILITY = "day_availability"

ORDER = (
    PERSON,
    DOCTOR_DIRECT,
    COMPLAINT,
    LINK_MISSING,
    CANNOT_PAY_ONLINE,
    PAY_NOW,
    DUES,
    HEALTH_PACKAGE,
    RARE_TEST,
    REPORT_PICKUP,
    REFERRAL,
    GOOD_DOCTOR,
    DAY_AVAILABILITY,
    DOCTOR_WEEKDAYS,
    NEXT_IF_ABSENT,
    SITTING_HOURS,
    WHEN_TO_COME,
    SERIAL_NUMBER,
    COME_TO_CLINIC,
    WRITE_NAME,
)

# The reason code handed to the human-handoff system for the kinds that hand off (existing codes kept: "caller_requested" is KCD-353's).
HANDOFF_REASON = {
    PERSON: "caller_requested",
    DOCTOR_DIRECT: "wants_doctor_personally",
    COMPLAINT: "complaint",
    RARE_TEST: "unlisted_test",
}


def classify(text: str, lang: str = "en") -> str | None:
    """The first early request this sentence is, or None (the turn then goes on to the normal path, unchanged)."""
    if not (text or "").strip():
        return None
    if asks_for_a_person(text, lang):
        return PERSON
    if wants_doctor_directly(text, lang):
        return DOCTOR_DIRECT
    if wants_to_complain(text, lang):
        return COMPLAINT
    if payment_link_missing(text, lang):
        return LINK_MISSING
    if doctor_enquiry_cues.cannot_pay_online(text, lang):
        return CANNOT_PAY_ONLINE
    if doctor_enquiry_cues.asks_pay_now(text, lang):
        return PAY_NOW
    if asks_about_dues(text, lang):
        return DUES
    if owner_answers.asks_about_health_package(text):
        return HEALTH_PACKAGE
    if rare_test_cues.asks_for_unlisted_test(text, lang):
        return RARE_TEST
    if report_pickup.asks_about_report_pickup(text, lang):
        return REPORT_PICKUP
    if referral_cues.asks_to_pass_a_number(text, lang):
        return REFERRAL
    if doctor_enquiry_cues.asks_good_doctor(text, lang):
        return GOOD_DOCTOR
    if weekday_cues.asks_day_availability(text, lang):
        return DAY_AVAILABILITY
    if doctor_enquiry_cues.asks_doctor_weekdays(text, lang):
        return DOCTOR_WEEKDAYS
    if doctor_enquiry_cues.asks_next_if_absent(text, lang):
        return NEXT_IF_ABSENT
    if doctor_enquiry_cues.asks_sitting_hours(text, lang):
        return SITTING_HOURS
    # A sentence that carries a token / serial NUMBER, or asks to move it up, keeps the exact answers in owner_answers (token_arrival_reply, token_push_reply).
    if not (owner_answers.asks_token_arrival(text) or owner_answers.asks_token_push(text)):
        if doctor_enquiry_cues.asks_when_to_come(text, lang):
            return WHEN_TO_COME
        if doctor_enquiry_cues.asks_serial_number(text, lang):
            return SERIAL_NUMBER
    if doctor_enquiry_cues.asks_come_to_clinic(text, lang):
        return COME_TO_CLINIC
    if doctor_enquiry_cues.asks_to_write_name(text, lang):
        return WRITE_NAME
    return None
