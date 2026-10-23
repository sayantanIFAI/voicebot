"""KCD-lead-8: a caller asking whether their report is ready is never told a guessed ready-status -- whether the
reporting doctor has actually come in that day is something only a human on site can confirm. So the caller is
told to call back on the day of their test/doctor's visit to check; and when THIS call is already happening on
that day (the caller says so), it is routed to a human immediately instead of a deferral reply that would make no
sense for a call already placed on the right day.

Deliberately conservative: this only recognises an EXPLICIT "today" in the caller's own words (mentions_today) --
it does not look up or guess which day a booking's visit date is. A caller who asks about a report without saying
when their test was always gets the deferral reply, never a guess. Deterministic, pre-model, same discipline as
agent/human_request.py and agent/billing_cues.py. Phrase lists are provisional and were written by a non-native
speaker (agent/persona.py REVIEW_STATUS): expect a native reviewer to add more.
"""

from __future__ import annotations

import re
import unicodedata

# ---- English -------------------------------------------------------------
_EN = [
    re.compile(r"\b(is|when will|has) my report\b.{0,20}\b(ready|available|done|come)\b", re.I),
    re.compile(r"\b(can i|could i|want to) (get|collect|have|pick up) my report\b", re.I),
    re.compile(r"\breport\b.{0,10}\b(ready|collection|pick ?up)\b", re.I),
]
# ---- Bengali -------------------------------------------------------------
_BN = [
    re.compile(r"রিপোর্ট\s*(টা)?\s*(রেডি|তৈরি|পাব|পাওয়া যাবে|নিতে পারব|কালেক্ট|এসেছে)"),
]
# ---- Hindi -----------------------------------------------------------------
_HI = [
    re.compile(r"रिपोर्ट\s*(तैयार|रेडी|मिल जाएगी|मिलेगी|कलेक्ट|आ गई)"),
]
_PATTERNS = {"en": _EN, "bn": _BN, "hi": _HI}

_TODAY_EN = re.compile(r"\btoday\b", re.I)
_TODAY_BN = re.compile(r"আজ(ই)?")
_TODAY_HI = re.compile(r"आज")
_TODAY = {"en": _TODAY_EN, "bn": _TODAY_BN, "hi": _TODAY_HI}

_DEFER = {
    "bn": "রিপোর্টটা রেডি কিনা নিশ্চিত হতে, আপনার টেস্ট বা ডাক্তার দেখানোর দিনে আমাদের আবার ফোন করে জিজ্ঞাসা করে নেবেন।",
    "hi": "रिपोर्ट तैयार है या नहीं, यह जानने के लिए अपने टेस्ट या डॉक्टर दिखाने वाले दिन हमें दोबारा फ़ोन करके पूछ लीजिए।",
    "en": "To check whether your report is ready, please call us back on the day of your test or doctor's visit and ask then.",
}


def _nfc(t: str) -> str:
    return unicodedata.normalize("NFC", t or "")


def asks_about_report_pickup(text: str, lang: str = "en") -> bool:
    """True if the caller is asking about report readiness/collection. All three languages are always checked
    (a caller may switch mid-call); `lang` only orders them."""
    t = _nfc(text).strip()
    if not t:
        return False
    for lg in dict.fromkeys((lang, "en", "bn", "hi")):
        if any(p.search(t) for p in _PATTERNS.get(lg, [])):
            return True
    return False


def mentions_today(text: str, lang: str = "en") -> bool:
    """True only if the caller's OWN words say "today" -- never inferred from a lookup. See module docstring."""
    t = _nfc(text).strip()
    if not t:
        return False
    for lg in dict.fromkeys((lang, "en", "bn", "hi")):
        pat = _TODAY.get(lg)
        if pat and pat.search(t):
            return True
    return False


def report_pickup_defer_reply(lang: str) -> str:
    return _DEFER.get(lang, _DEFER["en"])
