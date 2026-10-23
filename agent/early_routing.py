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
  4. DUES            -- "do I owe anything / is a payment pending" (KCD-lead-9): proof on WhatsApp + callback, never a balance.
  5. HEALTH_PACKAGE  -- Sonoscan offers none (KCD-lead-13): said BEFORE the FAQ matcher or the intent model can send it to the insurance answer.
  6. REPORT_PICKUP   -- "is my report ready" (KCD-lead-8).
  7. REFERRAL        -- another center passing a number (KCD-lead-1).
Genuine insurance / cashless questions match none of these and keep using the existing FAQ flow.
"""

from __future__ import annotations

from agent import owner_answers, referral_cues, report_pickup
from agent.billing_cues import asks_about_dues
from agent.human_request import asks_for_a_person, wants_doctor_directly, wants_to_complain

PERSON = "person"
DOCTOR_DIRECT = "doctor_direct"
COMPLAINT = "complaint"
DUES = "dues"
HEALTH_PACKAGE = "health_package"
REPORT_PICKUP = "report_pickup"
REFERRAL = "referral"

ORDER = (PERSON, DOCTOR_DIRECT, COMPLAINT, DUES, HEALTH_PACKAGE, REPORT_PICKUP, REFERRAL)

# The reason code handed to the human-handoff system for the kinds that hand off (existing codes kept: "caller_requested" is KCD-353's).
HANDOFF_REASON = {
    PERSON: "caller_requested",
    DOCTOR_DIRECT: "wants_doctor_personally",
    COMPLAINT: "complaint",
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
    if asks_about_dues(text, lang):
        return DUES
    if owner_answers.asks_about_health_package(text):
        return HEALTH_PACKAGE
    if report_pickup.asks_about_report_pickup(text, lang):
        return REPORT_PICKUP
    if referral_cues.asks_to_pass_a_number(text, lang):
        return REFERRAL
    return None
