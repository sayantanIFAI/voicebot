"""KCD-353: the patient-facing disclosure that the agent is automated.

A patient has the right to know they are speaking to an automated system, so that
they can decide how to use it. The disclosure is:

  * SPOKEN at the start of every call -- inside the greeting, in the language the
    call opens in (Bengali), and AGAIN, once, in the caller's own language the
    first time they answer in a different one (Hindi or English);
  * STATED on every text channel: every outbound message the clinic-api queues
    carries the notice in clinic-api/disclosure.py (kept in step with this file's
    version by tests/test_disclosure_and_human_request.py);
  * FOLLOWED by an honoured request for a person: agent/human_request.py catches
    "let me talk to someone" before any model is consulted and hands the call over.

The wording is VERSIONED and PENDING REVIEW. The story requires the clinical and
legal leads to review it; that is not something this codebase can do, and the
words below were drafted by the engineering side. REVIEW_STATUS says so and the
release gate reads it. Changing a word means bumping DISCLOSURE_VERSION, so an
audit can say which wording a given call heard (the version is logged per call).
"""
from __future__ import annotations

DISCLOSURE_VERSION = "1.0-draft"
REVIEW_STATUS = "pending_clinical_and_legal_review"

# What is spoken. One sentence to say it is automated, one to say a person is
# available -- so the caller knows both what they are talking to and that they
# are not stuck with it.
DISCLOSURE = {
    "bn": "আমি কলকাতা কেয়ারের স্বয়ংক্রিয় সহকারী, মানুষ নই। চাইলে যেকোনো সময় স্টাফের সাথে কথা বলতে পারবেন।",
    "hi": "मैं कोलकाता केयर की स्वचालित सहायक हूँ, कोई इंसान नहीं। आप चाहें तो कभी भी स्टाफ़ से बात कर सकते हैं।",
    "en": "I am the automated assistant of Kolkata Care, not a person. You can ask for our staff at any time.",
}


def disclosure_for(lang: str) -> str:
    return DISCLOSURE.get(lang) or DISCLOSURE["bn"]


def insert_into_greeting(greeting: str, lang: str) -> str:
    """The greeting is "welcome. question?". The disclosure goes between them, so
    the call still ends its opening on the question that hands the caller the floor."""
    import re
    parts = [p for p in re.split(r"(?<=[।.!?])\s+", greeting.strip()) if p]
    if len(parts) < 2:
        return f"{greeting.strip()} {disclosure_for(lang)}"
    return " ".join([parts[0], disclosure_for(lang)] + parts[1:])
