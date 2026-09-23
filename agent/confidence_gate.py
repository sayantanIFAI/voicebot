"""KCD-442/KCD-447: what a low-ASR-confidence turn is allowed to do.

agent/asr.py already computes `ASRResult.decoder_agreement` (CTC vs RNNT
agreement, 0.0-1.0) but nothing downstream ever reads it once language
routing is done -- exactly the gap both stories point at (see their Repo
Evidence). This module is the one place that decides what "low
confidence" means and what it implies, so main.py's dispatch consults a
single flag instead of re-deriving the threshold at each call site.

Two distinct consequences, matching the two stories:
  - KCD-442: a low-confidence turn asking a FACTUAL question (test_rate,
    doctor_availability, test_prep, clinic_faq) does not run the lookup
    at all -- a misheard entity name silently returning a confident,
    wrong answer for a DIFFERENT test/doctor is exactly the kind of
    quietly-wrong fact CLAUDE.md's truth boundary exists to prevent.
    agent/reply_templates.py's insufficient_information_reply() is
    spoken instead, distinct from "not found" (KCD-443, already Done)
    and from an infrastructure apology.
  - KCD-447: a low-confidence turn is never enough on its own to commit a
    WRITE (a booking hold/confirm/cancel/reschedule). Booking actions in
    this codebase already never write without passing through
    booking_flow.mark_confirming + an explicit classify_yes_no "yes" on
    a LATER turn (see tests/test_confidence_gate.py's
    test_booking_never_writes_without_a_separate_confirm_turn for the
    proof this already holds structurally); this module's role for
    writes is only to make that guarantee explicit and testable, not to
    add a second gate on top of it.

THRESHOLD is REASONED, not measured -- there is no real-call corpus
locally to calibrate the boundary between "ASR was genuinely unsure" and
"ASR was fine", the same "measured vs reasoned" caveat fast_path.py's
FAQ_COMMIT_FLOOR already carries. Recalibrate against real call audio
before trusting this past the pilot.
"""
from __future__ import annotations

# Below this, the two decoders disagreed on more than half the turn's
# words. agent/asr.py's dual-decoder agreement check already treats 0.0
# (CTC-fallback-only, RNNT unavailable) as "trust it anyway" for routing
# purposes -- this threshold is deliberately ABOVE that special case, so
# a genuine low-agreement turn (both decoders ran, and disagreed a lot)
# is caught, not the "only one decoder was even available" case.
LOW_CONFIDENCE_FLOOR = 0.5

# Read-only factual lookups a low-confidence turn must not silently act
# on -- acting on a misheard test/doctor name would produce a confident,
# wrong answer about something else entirely. Booking actions are
# deliberately excluded: they already require a separate confirm turn
# regardless of confidence (see module docstring).
GATED_FACTUAL_INTENTS = frozenset({"test_rate", "doctor_availability", "test_prep", "clinic_faq"})


def is_low_confidence(decoder_agreement: float) -> bool:
    """decoder_agreement == 0.0 without a genuine disagreement measurement
    (agent/asr.py's CTC-fallback path) is NOT treated as low confidence
    here -- that path already carries its own explicit "no RNNT decoder
    to compare against" meaning, distinct from "compared and disagreed"."""
    return 0.0 < decoder_agreement < LOW_CONFIDENCE_FLOOR


def should_withhold_factual_answer(intent: str, decoder_agreement: float) -> bool:
    return intent in GATED_FACTUAL_INTENTS and is_low_confidence(decoder_agreement)
