"""KCD-lead-9: "do I owe you anything?" is never answered with a figure read over the phone -- a caller asking
about dues is told to send payment proof on WhatsApp and that a human will call back (agent/owner_answers.py's
billing_dues_reply). So the detection here only needs to recognise the QUESTION, never look up a balance; it runs
deterministically, the same discipline as agent/human_request.py's "a request for a person is honoured
immediately" -- a billing question is never routed through a model that could misread it as, say, a price enquiry.

A payment-due question has two parts that are only meaningful together: a MONEY word (money, payment, bill, balance,
টাকা, पैसे, taka ...) and a DUE word (owe, left, pending, due, outstanding, বাকি, बाकी, baki ...). Either alone is not
enough ("what is the payment method" / "I will pay tomorrow" must not fire). A few phrases are complete on their own
("how much do I owe", "outstanding", "বকেয়া", "बकाया").

Phrase lists are provisional and were written by a non-native speaker (agent/persona.py REVIEW_STATUS): expect a
native reviewer to add more, same caveat agent/human_request.py already carries.
"""

from __future__ import annotations

import re
import unicodedata

_I = re.I

# ---- English -------------------------------------------------------------
_EN = [
    re.compile(
        r"\b(do|did|will|would) i (still |really |actually )?(owe|have to pay|need to pay|have any (money|payment|balance|dues?))\b",
        _I,
    ),
    re.compile(
        r"\bhow much (do|did|will|would)? ?i (still |yet )?(owe|need to pay|have to pay|have left to pay|have to pay)\b",
        _I,
    ),
    re.compile(
        r"\b(money|amount|balance|payment|payments|dues?|bill|bills|fees?|charges?)\b.{0,25}\b(left|remaining|pending|due|outstanding|unpaid|overdue|to pay|to be paid)\b",
        _I,
    ),
    re.compile(
        r"\b(left|remaining|pending|outstanding|unpaid|overdue|due)\b.{0,20}\b(money|amount|balance|payment|payments|dues?|bill|bills|fees?|charges?)\b",
        _I,
    ),
    re.compile(r"\b(shows?|showing|showed)\b.{0,12}\b(pending|due|unpaid|outstanding)\b", _I),
    re.compile(r"\boutstanding (amount|balance|payment|bill|dues?)\b", _I),
    re.compile(r"\b(any|some) (money|dues?|balance) (left|remaining|pending)\b", _I),
]
# ---- Bengali -------------------------------------------------------------
_MONEY_BN = r"(টাকা|টাকার|পেমেন্ট|পেমেন্টের|বিল|বিলের|খরচ|ফি|পাওনা)"
_DUE_BN = r"(বাকি|বাকী|বকেয়া|পেন্ডিং|পেন্ডিং|দিতে হবে|দেওয়ার আছে|পাওনা)"
_BN = [
    re.compile(_MONEY_BN + r".{0,30}" + _DUE_BN),
    re.compile(_DUE_BN + r".{0,20}" + _MONEY_BN),
    re.compile(r"বকেয়া"),
    re.compile(r"(আর|আরও|আরো|আরেক)\s*টাকা\s*(দিতে|লাগবে)"),
]
# ---- Hindi -----------------------------------------------------------------
_MONEY_HI = r"(पैसा|पैसे|रुपये|रूपये|रकम|पेमेंट|पेमेन्ट|भुगतान|बिल|फीस|शुल्क)"
_DUE_HI = r"(बाकी|बकाया|पेंडिंग|देने हैं|देना है|देने है|चुकाना|बचा)"
_HI = [
    re.compile(_MONEY_HI + r".{0,30}" + _DUE_HI),
    re.compile(_DUE_HI + r".{0,20}" + _MONEY_HI),
    re.compile(r"बकाया"),
]
# ---- romanized Bengali / Hindi ----------------------------------------------
_MONEY_RM = r"(taka|tk|payment|bill|bills|poisa|paisa|paise|fee|fees|rupaye|rupees?)"
_DUE_RM = r"(baki|bakee|baaki|bokeya|bokeyaa|bakaya|pending|dite hobe|dena hai|dene hain|due)"
_RM = [
    re.compile(rf"\b{_MONEY_RM}\b.{{0,30}}\b{_DUE_RM}\b", _I),
    re.compile(rf"\b{_DUE_RM}\b.{{0,20}}\b{_MONEY_RM}\b", _I),
    re.compile(r"\b(bokeya|bakaya)\b", _I),
]

_PATTERNS = {"en": _EN + _RM, "bn": _BN + _RM, "hi": _HI + _RM}

# "no, nothing is due" / "কোনো বকেয়া নেই" must not be read as the question itself.
_NEGATED_EN = re.compile(
    r"\b(no|not|nothing|none|without|never)\b.{0,15}\b(owe|dues?|balance|pending|outstanding|left to pay)\b|\bnothing (is |left )?(due|pending|to pay)\b",
    _I,
)
_NEGATED_BN = re.compile(r"(বকেয়া নেই|কিছু বাকি নেই|কোনো টাকা বাকি নেই|বাকি নেই|বাকি নাই|দিতে হবে না|দিতে হবেনা)")
_NEGATED_HI = re.compile(r"(बकाया नहीं|कुछ बाकी नहीं|कोई पैसा बाकी नहीं|बाकी नहीं|कुछ नहीं देना|देना नहीं)")
_NEGATED_RM = re.compile(r"\b(baki nei|baki nai|baki nei|bokeya nei|baki nahi|nothing due)\b", _I)
_NEGATED = {"en": _NEGATED_EN, "bn": _NEGATED_BN, "hi": _NEGATED_HI}


def _nfc(t: str) -> str:
    return unicodedata.normalize("NFC", t or "")


def asks_about_dues(text: str, lang: str = "en") -> bool:
    """True if the caller is asking whether they owe money, or reporting that a payment still shows as due -- never
    merely whether they already PAID, nor that nothing is due. All three languages are always checked (a caller may
    switch mid-call); `lang` only orders them."""
    t = _nfc(text).strip()
    if not t:
        return False
    if _NEGATED_RM.search(t):
        return False
    for lg in dict.fromkeys((lang, "en", "bn", "hi")):
        if any(p.search(t) for p in _PATTERNS.get(lg, [])):
            if _NEGATED.get(lg) and _NEGATED[lg].search(t):
                continue
            return True
    return False
