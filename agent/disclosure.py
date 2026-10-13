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

DISCLOSURE_VERSION = "2.1-draft"  # 2.1: the spoken greeting now carries only the identity sentence
REVIEW_STATUS = "pending_clinical_and_legal_review"

# What is spoken, and the built-in default: the operator can change it from the database without a
# deploy (agent/messages.py, key "disclosure"), and the call record stores which wording was heard.
# Who the caller is speaking with, that it is automated, and that a person is available -- one short
# sentence each, so the caller knows what they are talking to and that they are not stuck with it.
# Native script for the brand name in Bengali and Hindi: a Latin word inside Bengali text is dropped
# by the Bengali voice (agent/bn_normalize.py).
# OWNER'S INSTRUCTION, 2026-10-13: all three languages say ONLY the identity sentence and the help offer --
# nothing about being automated, not a person, or staff being reachable at any time. Kept verbatim-identical
# per language with agent/phrases.GREETING_IDENTITY so the opening greeting (Bengali) and this once-only
# language-switch restatement (Hindi/English) never say two different things about what the agent is.
# NOTE: this is the same wording the "automated, not human" notice existed to give the caller (REVIEW_STATUS
# below is "pending_clinical_and_legal_review" -- that review has not happened for this wording either, same
# as it had not for the text it replaces).
DISCLOSURE = {
    "bn": "আমি মনোরমা বাণী। বলুন, কীভাবে সাহায্য করতে পারি?",
    "hi": "मैं मनोरमा वाणी हूँ। बताइए, मैं आपकी क्या मदद कर सकती हूँ?",
    "en": "I am Manorama Vaani. How can I help you?",
}


def disclosure_for(lang: str) -> str:
    from agent import messages

    return messages.text("disclosure", lang if lang in DISCLOSURE else "bn", DISCLOSURE.get(lang) or DISCLOSURE["bn"])


def version_label() -> str:
    """The wording version to record for a call: the database's when it supplied the text."""
    from agent import messages

    return f"{DISCLOSURE_VERSION}+{messages.label()}"


def insert_into_greeting(greeting: str, lang: str) -> str:
    """The greeting is "welcome. question?". The disclosure goes between them, so
    the call still ends its opening on the question that hands the caller the floor."""
    import re

    parts = [p for p in re.split(r"(?<=[।.!?])\s+", greeting.strip()) if p]
    if len(parts) < 2:
        return f"{greeting.strip()} {disclosure_for(lang)}"
    return " ".join([parts[0], disclosure_for(lang)] + parts[1:])
