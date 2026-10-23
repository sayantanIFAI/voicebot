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


# ---- "I did not get the payment link" (owner, 2026-10-23) -----------------------------------------------------------------------------------------------
# The link is sent to WhatsApp and SMS right after a doctor booking. A caller who says it has not come is told there is a network issue, to wait fifteen
# minutes, and to call back if it still has not come (owner_answers.payment_link_missing_reply). Nothing is looked up, nothing is promised about the link.
_LINK_MISSING = [
    re.compile(r"\blinks?\b.{0,25}\b(not|n'?t|never|yet)\b.{0,15}\b(receive|received|get|got|come|came|coming|arrive|arrived|see|seen|working|opening)", _I),
    re.compile(r"\b(didn'?t|did not|haven'?t|have not|not|never|can'?t|cannot)\b.{0,15}\b(get|got|receive|received|find|see)\b.{0,15}\b(the |a |any |my )?(payment )?links?\b", _I),
    re.compile(r"\b(no|where is (the|my)|where'?s (the|my))\s+(payment\s+)?links?\b", _I),
    re.compile(r"\blinks?\s+(is\s+)?(missing|not here|not there)\b", _I),
    re.compile(r"\blink\b.{0,15}\b(pai\s*ni|painy|painai|ashe\s*ni|asheni|ase\s*ni|aschhe\s*na|asche\s*na|aschena|pacchi\s*na|pachhi\s*na|asenai)\b", _I),
    re.compile(r"\blink\b.{0,15}\b(nahi|nhi|nahin)\s*(aaya|aya|aayi|mila|mili|aa\s*raha|aa\s*rahi)\b", _I),
    re.compile(r"লিংক.{0,18}(পাইনি|পাই\s*নি|আসেনি|আসে\s*নি|আসছে\s*না|আসেনা|পাচ্ছি\s*না|যায়নি|পাঠায়নি)"),
    re.compile(r"(লিংক|লিঙ্ক).{0,12}(কই|কোথায়)|(হোয়াটসঅ্যাপে|এসএমএস|মেসেজ).{0,20}(কিছু|লিংক).{0,12}(আসেনি|আসে\s*নি|আসছে\s*না)"),
    re.compile(r"लिंक.{0,18}(नहीं\s*आया|नही\s*आया|नहीं\s*मिला|नही\s*मिला|नहीं\s*आ\s*रहा|नहीं\s*आई|नहीं\s*भेजा|नहीं\s*दिख)"),
    re.compile(r"(व्हाट्सऐप|व्हाट्सएप|एसएमएस|मैसेज).{0,20}(कुछ|लिंक).{0,12}(नहीं\s*आया|नही\s*आया|नहीं\s*आ\s*रहा)"),
]


# The word "link" is written many ways by a Bengali / Hindi recogniser: লিংক / লিঙ্ক / লিন্ক, लिंक / लिन्क. Every spelling is read as the one the patterns above use.
_LINK_SPELLINGS = (
    (re.compile(r"লি(?:ঙ্ক|ন্ক|ংক্|নক|ঙ্গ)"), "লিংক"),
    (re.compile(r"लि(?:न्क|ंक्|नक|ङ्क)"), "लिंक"),
)


def payment_link_missing(text: str, lang: str = "en") -> bool:
    """True when the caller says the payment link has not arrived. All three languages are always checked (a caller may switch mid-call)."""
    t = _nfc(text).strip()
    for spelling, canonical in _LINK_SPELLINGS:
        t = spelling.sub(canonical, t)
    return bool(t) and any(p.search(t) for p in _LINK_MISSING)


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

# Round 6 (owner, 2026-10-23): romanized Bengali / Hindi wordings of "the link has not come" that testing found unrecognised ("link pai nai", "ashe nai", "pelam na").
_LINK_MISSING.extend(
    [
        re.compile(
            r"\blinks?\b.{0,20}\b(?:pai\s*nai|pai\s*na|painai|ashe\s*nai|ase\s*nai|ashenai|ashena|asena|aseni|pelam\s*na|pelam\s*nai|paini|pacchi\s*nai|pachhi\s*nai|aschena|aschhena|asche\s*nai|aschhe\s*nai)\b",
            _I,
        ),
        re.compile(r"\blink\b.{0,15}\b(?:nai|na)\s*(?:aaya|aya|aayi|mila|mili|aa\s*raha|aa\s*rahi)\b", _I),
    ]
)
