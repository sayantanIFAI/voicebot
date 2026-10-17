"""'What tests do you do?' -- answered from the clinic's own catalogue, grouped by kind, with no model call.

Real call 2026-10-17: the caller asked, five times and in different words, what kinds of tests the clinic does ("আপনাদের ওখানে কি কি টেস্ট হয়",
"এইসব ধরনের টেস্ট হয়"). There was no answer path for a question that names no test, so the AI kept asking "which test do you mean?".
The answer is built ONLY from the catalogue (names, kinds): nothing is invented, and a test the clinic stopped doing disappears from it.
It lists the kinds the clinic offers with a few examples of each and invites the caller to name one; the next name is then priced as usual.
"""

from __future__ import annotations

import re
import unicodedata

_TEST_WORD = re.compile(r"টেস্ট|পরীক্ষা|টেস্টের|टेस्ट|जांच|जाँच|\btests?\b|\bscans?\b|স্ক্যান", re.IGNORECASE)

# a marker that the question is about the WHOLE range of tests, not one
_GENERAL = (
    # Bengali
    "কি কি", "কী কী", "কোন কোন", "কত রকম", "কত ধরনের", "সব ধরনের", "এইসব ধরনের", "এই সব ধরনের", "সব রকম", "কি ধরনের", "কী ধরনের",
    "কি রকম", "কী রকম", "টেস্টের লিস্ট", "টেস্টের তালিকা", "পরীক্ষার তালিকা", "সব টেস্ট", "সবরকম", "সব ধরণের", "এইসব ধরণের",
    # Hindi
    "क्या क्या", "कौन कौन", "कौन-कौन", "कौन से", "किस किस", "कितने तरह", "कितने प्रकार", "सब तरह के", "हर तरह के", "टेस्ट की लिस्ट", "टेस्ट की सूची",
    "सभी टेस्ट", "सारे टेस्ट", "किस तरह के", "कैसे कैसे",
    # English
    "what kind of test", "what type of test", "what kinds of test", "what sort of test", "what tests", "which tests", "what kind of tests", "what kinds of tests", "what type of tests", "what types of tests", "types of tests",
    "kinds of tests", "list of tests", "all the tests", "all tests", "what all tests", "what sort of tests", "tests do you", "tests are available",
    "tests you do", "tests you have", "tests available",
)
# a question that also wants a price, a booking or a preparation is a different question
_OTHER_QUESTION = (
    "দাম", "রেট", "কত টাকা", "খরচ", "বুক", "অ্যাপয়েন্টমেন্ট", "করাতে চাই", "প্রস্তুতি", "ফাস্টিং", "रेट", "दाम", "कीमत", "कितना", "बुक", "अपॉइंटमेंट", "तैयारी",
    "price", "rate", "cost", "how much", "book", "appointment", "prepare", "preparation", "fasting", "where is", "address",
)


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFC", s or "").lower()
    return " ".join(re.sub(r"[.,;:!?\"'()।॥]+", " ", s).split())


def asks_which_tests(text: str) -> bool:
    t = _norm(text)
    if not t or len(t.split()) > 25 or not _TEST_WORD.search(t):
        return False
    if any(c in t for c in _OTHER_QUESTION):
        return False
    return any(m in t for m in _GENERAL)


_ORDER = ("Blood", "Urine", "Cardiac", "Imaging")
_LIMIT = {"Blood": 6, "Urine": 1, "Cardiac": 3, "Imaging": 3}


def _spoken(test: dict, lang: str) -> str | None:
    """The name to SAY: the first catalogue alias written in the call's script; for English the test's own name without its bracket."""
    if lang == "bn":
        for a in test.get("aliases_bn") or []:
            if any("ঀ" <= c <= "৿" for c in a):
                return a
        return None
    if lang == "hi":
        for a in test.get("aliases_hi") or []:
            if any("ऀ" <= c <= "ॿ" for c in a):
                return a
        return None
    name = str(test.get("name") or "")
    head = name.split("(")[0].strip()
    return head or name or None


def _join(items: list[str], lang: str) -> str:
    if len(items) <= 1:
        return "".join(items)
    word = {"bn": "আর", "hi": "और", "en": "and"}.get(lang, "and")
    return ", ".join(items[:-1]) + f" {word} " + items[-1]


_TEMPLATES = {
    "bn": {
        "Blood": "রক্তের টেস্ট হয়, যেমন {x}",
        "Urine": "প্রস্রাবের টেস্ট হয়",
        "Cardiac": "হার্টের টেস্ট হয়, যেমন {x}",
        "Imaging": "ছবি তোলার টেস্ট হয়, যেমন {x}",
        "head": "আমাদের এখানে ",
        "tail": "। কোন টেস্টের কথা জানতে চান?",
        "sep": "। ",
    },
    "hi": {
        "Blood": "खून की जांच होती है, जैसे {x}",
        "Urine": "पेशाब की जांच होती है",
        "Cardiac": "दिल की जांच होती है, जैसे {x}",
        "Imaging": "इमेजिंग होती है, जैसे {x}",
        "head": "हमारे यहाँ ",
        "tail": "। आप किस टेस्ट के बारे में जानना चाहेंगे?",
        "sep": "। ",
    },
    "en": {
        "Blood": "blood tests, such as {x}",
        "Urine": "urine tests",
        "Cardiac": "heart tests, such as {x}",
        "Imaging": "imaging, such as {x}",
        "head": "We do ",
        "tail": ". Which test would you like to know about?",
        "sep": "; ",
    },
}


def reply(tests: list[dict], lang: str) -> str | None:
    """The spoken overview, or None if the catalogue gives nothing to say (the turn then goes on as before)."""
    lang = lang if lang in _TEMPLATES else "en"
    t = _TEMPLATES[lang]
    parts: list[str] = []
    for kind in _ORDER:
        names = [n for n in (_spoken(x, lang) for x in tests if x.get("sample_type") == kind) if n]
        names = list(dict.fromkeys(names))[: _LIMIT[kind]]
        if names:
            parts.append(t[kind].format(x=_join(names, lang)))
    if not parts:
        return None
    return t["head"] + t["sep"].join(parts) + t["tail"]
