""""What is the CBC rate, do I need to fast?" is TWO questions, and both must be answered.

Real call 2026-10-14: "সিবিসি টেস্টের দাম কত ফাস্টিং করতে হবে" came back from the intent model as a plain price question
(`test_rate`, no `secondary_intent`), so the caller heard the price and the fasting half of the same sentence was dropped.
The caller had to ask again.

The repo already answers a second question when the model flags one (main.py `_finish_enquiry_turn`, KCD-395). The model
does not flag it reliably, and "price AND preparation" is by far the commonest pair on a lab line. This module is the
deterministic backstop: when the primary question is the price of a test and the SAME sentence plainly also asks about
preparation (or the reverse), the missing second question is added for the test already named. The model still decides
nothing about facts: both answers are the usual lookups from the clinic's table.

It adds only when the model left the second question empty, never replaces one it set, and needs a test named in the
primary slots, so a sentence about something else is untouched.
"""

from __future__ import annotations

import re
import unicodedata

_PUNCT = re.compile(r"[.,;:!?\"'()\[\]{}।॥\-–—_/]+")

_PREP = (
    # Bengali
    "ফাস্টিং", "ফাস্ট", "খালি পেটে", "খালি পেট", "না খেয়ে", "উপবাস", "প্রস্তুতি", "কিছু খেতে", "খাওয়া যাবে", "জল খেতে",
    # Hindi
    "फास्टिंग", "फास्ट", "खाली पेट", "भूखे पेट", "बिना खाए", "उपवास", "तैयारी", "कुछ खा", "खाना खा",
    # English
    "fasting", "fast ", "empty stomach", "preparation", "prepare", "eat before", "without eating",
)
_PRICE = (
    "দাম", "রেট", "খরচ", "কত টাকা", "দর", "ফি ", "চার্জ", "price", "rate", "cost", "how much", "fee", "charge",
    "दाम", "रेट", "कीमत", "कितना", "कितने", "खर्च", "फीस", "चार्ज", "भाव", "पैसे",
)


def _norm(text: str) -> str:
    s = unicodedata.normalize("NFC", text or "").lower().replace("़", "").replace("়", "")
    return " ".join(_PUNCT.sub(" ", s).split()) + " "


def _has(norm_text: str, cues: tuple[str, ...]) -> bool:
    padded = " " + norm_text
    for cue in cues:
        c = unicodedata.normalize("NFC", cue).lower()
        if c.isascii():
            if (" " + c.strip() + " ") in padded or (len(c.strip()) > 4 and c.strip() in norm_text):
                return True
        elif c in norm_text:
            return True
    return False


def asks_about_preparation(text: str) -> bool:
    return _has(_norm(text), _PREP)


def asks_about_price(text: str) -> bool:
    return _has(_norm(text), _PRICE)


def add_implied_second_question(text: str, data: dict) -> dict:
    """`data` is the intent object ({"intent", "slots", "secondary_intent", "secondary_slots", ...}). Returns it unchanged,
    or a copy with the missing second question added. Never edits the dict it is given."""
    if not isinstance(data, dict) or data.get("secondary_intent"):
        return data
    intent = data.get("intent")
    slots = data.get("slots") or {}
    names = [n for n in (slots.get("test_names") or []) if n]
    test = slots.get("test_name") or (names[0] if len(names) == 1 else None)
    if not test:
        return data
    if intent == "test_rate" and asks_about_preparation(text):
        second = "test_prep"
    elif intent == "test_prep" and asks_about_price(text):
        second = "test_rate"
    else:
        return data
    return {**data, "secondary_intent": second, "secondary_slots": {"test_name": test}}
