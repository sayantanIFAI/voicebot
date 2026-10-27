"""A caller asking what THEIR test result was (owner, 2026-10-27).

Real calls: "আমার কত সুগার এসছিল টেস্ট এর রিপোর্ট টা এখানে বলুন আমাকে ফোনে", "আমার সুগার টেস্ট করেছিলাম ওখানে আগের মাসে ওটা কি রিসুল্ট এসছে বলেন না". The agent cannot read out a personal lab
result over the phone, and it must say so plainly in the caller's language and send them to the counter -- not ask which test they want the preparation for, or for the number they booked from.

(What a result MEANS -- "is it dangerous?" -- is a different question, answered by agent/clinical_safety.py with an offer of a clinician. This module is only "tell me my result".)

Rule-based, with three families that must all hold: a RESULT word (or "how much did it come to"), no PRICE word (a price question is not this), and no booking / future wording that makes it a
different request ("book a sugar test"). REASONED, NOT MEASURED; a native reader should extend the lists, and tests/test_result_request.py is the place to record the phrasings.
"""

from __future__ import annotations

import re
import unicodedata

_RESULT = (
    # Bengali: the recogniser writes "result" many ways
    "রিসুল্ট", "রেজাল্ট", "রিজাল্ট", "রেসাল্ট", "রিজল্ট", "রেজাল্টটা", "ফলাফল", "রিপোর্টে কী", "রিপোর্টে কি", "রিপোর্টে কত", "রিপোর্ট কী এসেছ", "রিপোর্ট কি এসেছ", "রিপোর্টটা বলুন", "রিপোর্ট বলুন",
    "রিপোর্টটা বলে", "রিপোর্ট বলে দিন", "রিপোর্টটা পড়ে",
    # Hindi
    "रिजल्ट", "रिज़ल्ट", "रिजल्ट", "नतीजा", "परिणाम", "रिपोर्ट में क्या", "रिपोर्ट में कितना", "रिपोर्ट बता", "रिपोर्ट पढ़कर", "रिपोर्ट पढ़ के",
)
_RESULT_LATIN = re.compile(
    r"\b(?:result|results|my report says|what did my report|read (?:out )?my report|tell me my report|report result|reading of)\b",
    re.I,
)
# "কত সুগার এসেছিল" / "कितना शुगर आया था" / "how much did my sugar come": the number that came out
_CAME_OUT = re.compile(r"(?:কত|কতটা)\s+[^\s]+\s*(?:এসেছিল|এসছিল|এসেছে|এসছে|এল|এলো|ছিল|আছে|দেখাচ্ছে)|(?:कितना|कितनी|कितने)\s+[^\s]+\s*(?:आया|आई|आये|निकला|था|थी|आता)|how (?:much|high|low) (?:did|was)", re.I)
_PRICE = (
    "দাম", "রেট", "কত টাকা", "টাকা", "খরচ", "ফি", "কত লাগ", "price", "rate", "cost", "fee", "charge", "how much does", "how much is", "कीमत", "रेट", "दाम", "कितने रुपये", "कितना खर्च", "कितना लगेगा",
)
_BOOK = ("বুক", "করাতে চাই", "করাতে হবে", "করাতে", "book", "appointment", "অ্যাপয়েন্টমেন্ট", "बुक", "अपॉइंटमेंट", "करवाना", "कराना")

REPLY = {
    "bn": "ফোনে টেস্টের রেজাল্ট বলা যাবে না। দয়া করে কাউন্টার থেকে রিপোর্ট সংগ্রহ করে নিন।",
    "hi": "फ़ोन पर टेस्ट का रिज़ल्ट नहीं बताया जा सकता। कृपया रिपोर्ट काउंटर से ले लीजिए।",
    "en": "Test results cannot be given over the phone. Please collect your report from the counter.",
}


def _norm(text: str) -> str:
    return " ".join(unicodedata.normalize("NFC", text or "").lower().split())


def asks_for_own_result(text: str) -> bool:
    """True when the caller asks to be TOLD their own test result (and is not asking a price or booking a test)."""
    t = _norm(text)
    if not t:
        return False
    said_result = any(w in t for w in _RESULT) or _RESULT_LATIN.search(t) is not None or _CAME_OUT.search(t) is not None
    if not said_result:
        return False
    if any(w in t for w in _PRICE):
        return False
    # a booking word only blocks it when no result word was said in a clear form: "রেজাল্ট" beside "করাতে" is still a result question ("করেছিলাম" is past, not "করাতে")
    if any(w in t for w in _BOOK) and not any(w in t for w in ("রিসুল্ট", "রেজাল্ট", "রিজাল্ট", "ফলাফল", "result", "रिजल्ट", "रिज़ल्ट", "नतीजा", "परिणाम")):
        return False
    return True
