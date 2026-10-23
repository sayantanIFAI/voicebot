"""KCD-lead-1: a caller who is STAFF at another Sonoscan center, passing along a patient's number so an outbound
call can be made later to arrange a doctor appointment -- a different caller from an ordinary patient, and the
AI's job here is only to capture enough for a human to work from (never to book anything itself): the patient's
name, phone number, and which doctor/department, logged for a callback. Deterministic, pre-model, same discipline
as agent/human_request.py -- distinguishing "another center is calling" from an ordinary patient call matters more
than catching every possible phrasing, so both signals (another branch/center, AND passing a number/referral) are
required together, except for the word "referral" itself, unambiguous on its own in this domain.

Phrase lists are provisional and were written by a non-native speaker (agent/persona.py REVIEW_STATUS): expect a
native reviewer to add more.
"""

from __future__ import annotations

import re
import unicodedata

# ---- English -------------------------------------------------------------
_EN_OTHER_CENTER = re.compile(r"\b(another|other|different)\s+(branch|center|centre|sonoscan)\b", re.I)
_EN_PASS_NUMBER = re.compile(r"\b(pass|give|share)\b.{0,15}\b(a|the|her|his|their)\s*number\b", re.I)
_EN_REFERRAL = re.compile(r"\breferral\b", re.I)
# ---- Bengali -------------------------------------------------------------
_BN_OTHER_CENTER = re.compile(r"(অন্য|আরেক)\s*(সেন্টার|ব্রাঞ্চ|শাখা|সোনোস্ক্যান)")
_BN_PASS_NUMBER = re.compile(r"নম্বর(টা)?\s*(দিতে চাই|দিচ্ছি|পাঠাচ্ছি|দেব)")
_BN_REFERRAL = re.compile(r"রেফারেল")
# ---- Hindi -----------------------------------------------------------------
_HI_OTHER_CENTER = re.compile(r"(दूसरी|दूसरे|और)\s*(ब्रांच|सेंटर|शाखा|सोनोस्कैन)")
_HI_PASS_NUMBER = re.compile(r"नंबर\s*(देना चाहता|दे रहा|भेज रहा|दूंगा)")
_HI_REFERRAL = re.compile(r"रेफरल")

_PATTERNS = {
    "en": (_EN_OTHER_CENTER, _EN_PASS_NUMBER, _EN_REFERRAL),
    "bn": (_BN_OTHER_CENTER, _BN_PASS_NUMBER, _BN_REFERRAL),
    "hi": (_HI_OTHER_CENTER, _HI_PASS_NUMBER, _HI_REFERRAL),
}

_ASK = {
    "bn": "ঠিক আছে, রোগীর নাম, ফোন নম্বর, এবং কোন ডাক্তার বা বিভাগে অ্যাপয়েন্টমেন্ট চান, সেটা বলবেন?",
    "hi": "ठीक है, मरीज़ का नाम, फ़ोन नंबर, और किस डॉक्टर या विभाग में अपॉइंटमेंट चाहिए, यह बता दीजिए?",
    "en": "Sure, could you tell me the patient's name, phone number, and which doctor or department they'd like an appointment with?",
}
_LOGGED = {
    "bn": "ঠিক আছে, নোট করে নিলাম। আমাদের কেউ পরে ফোন করে অ্যাপয়েন্টমেন্টের ব্যবস্থা করে দেবে।",
    "hi": "ठीक है, नोट कर लिया। हमारी तरफ़ से कोई बाद में फ़ोन करके अपॉइंटमेंट की व्यवस्था कर देगा।",
    "en": "All right, I've noted that down. Someone from our side will call back later to arrange the appointment.",
}


def _nfc(t: str) -> str:
    return unicodedata.normalize("NFC", t or "")


def asks_to_pass_a_number(text: str, lang: str = "en") -> bool:
    """True if the caller is staff at another center passing along a patient's number for a later outbound call.
    All three languages are always checked (a caller may switch mid-call); `lang` only orders them."""
    t = _nfc(text).strip()
    if not t:
        return False
    for lg in dict.fromkeys((lang, "en", "bn", "hi")):
        other_center, pass_number, referral = _PATTERNS.get(lg, (None, None, None))
        if referral and referral.search(t):
            return True
        if other_center and pass_number and other_center.search(t) and pass_number.search(t):
            return True
    return False


def referral_ask_reply(lang: str) -> str:
    return _ASK.get(lang, _ASK["en"])


def referral_logged_reply(lang: str) -> str:
    return _LOGGED.get(lang, _LOGGED["en"])
