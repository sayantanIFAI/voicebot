"""Lost and found: something left behind, or someone to be found (owner, 2026-10-27).

Real call: "আগের দিন ওখানে ডাক্তার দেখাতে কি আমি একটা বাগ ফেলে এসছি বাগ টা কোথ থেকে পাবো" ("I left a bag there the other day, where can I get it?") was answered "sorry, I did not understand" and then
"which doctor do you mean?". The agent knows nothing about belongings or whereabouts and must never guess: such a call goes straight to a person (main.py `_handoff_to_human`, reason
`lost_and_found`, which is the human-routing placeholder).

Two shapes, both need a clear sign: a LOST VERB ("left behind", "lost", "forgot") beside a thing or a "where can I get it" question; or a PERSON (father, mother, the patient ...) that cannot be found.
Rule-based and deliberately narrow; a native reader should extend the lists, and tests/test_lost_found.py is the place to record phrasings.
"""

from __future__ import annotations

import re
import unicodedata

_LOST_VERB = (
    # Bengali (the recogniser writes "ফেলে এসেছি" many ways)
    "ফেলে এস", "ফেলে গেছি", "ফেলে গিয়ে", "ফেলে রেখে", "ফেলে আস", "হারিয়ে ফেল", "হারিয়ে গেছে", "হারিয়ে গিয়েছে", "ভুলে ফেলে", "ভুলে রেখে", "রেখে এস", "রেখে গেছি", "রয়ে গেছে", "পড়ে আছে",
    # Hindi
    "भूल आया", "भूल आई", "भूल गया", "भूल गई", "छोड़ आया", "छोड़ आई", "छूट गया", "छूट गई", "खो गया", "खो गई", "गुम हो", "रह गया", "रह गई",
    # English
    "left my", "left a", "left behind", "lost my", "lost a", "forgot my", "forgotten my", "misplaced",
)
_THING = (
    "বক", "বাগ", "ব্যাগ", "ব্যাগটা", "ফোন", "মোবাইল", "চশমা", "পার্স", "ছাতা", "চাবি", "ওয়ালেট", "জিনিস", "ঘড়ি", "কাগজ", "ফাইল", "প্রেসক্রিপশন", "রিপোর্ট", "কার্ড", "টাকা", "ছড়ি", "লাঠি", "টিফিন",
    "बैग", "फोन", "मोबाइल", "चश्मा", "पर्स", "छाता", "चाबी", "वॉलेट", "सामान", "घड़ी", "कागज", "फाइल", "पर्ची", "रिपोर्ट", "कार्ड", "पैसे",
    "bag", "phone", "mobile", "glasses", "spectacles", "purse", "umbrella", "key", "keys", "wallet", "watch", "file", "documents", "stick", "belongings", "things",
)
_RETRIEVE = (
    "কোথা থেকে পাব", "কোথায় পাব", "কোথ থেকে পাব", "কোথ থেকে পাবো", "পাওয়া যাবে", "পাওয়া গেছে", "কেউ পেয়েছে", "পেয়েছেন কি", "ফেরত", "জমা আছে", "लौटा", "कहाँ मिलेगा", "मिल सकता", "मिला क्या", "मिल गया",
    "where can i get", "where do i get", "did anyone find", "has anyone found", "lost and found", "can i get it back",
)
# a person who cannot be found: "my father is missing / I cannot find him"
_PERSON = ("বাবা", "মা", "মাকে", "মায়ের", "ছেলে", "মেয়ে", "স্বামী", "স্ত্রী", "বউ", "দাদু", "ঠাকুমা", "দিদা", "রোগী", "আত্মীয়", "पिता", "पापा", "माँ", "मां", "बेटा", "बेटी", "पति", "पत्नी", "मरीज", "my father", "my mother", "my son", "my daughter", "my husband", "my wife", "the patient")
_PERSON_LOST = ("খুঁজে পাচ্ছি না", "খুজে পাচ্ছি না", "হারিয়ে গেছে", "নিখোঁজ", "কোথায় গেল", "কোথায় চলে গেল", "কোথায় গেছে", "ढूंढ नहीं पा", "नहीं मिल रहे", "गुम हो गए", "कहाँ चले गए", "can't find", "cannot find", "is missing", "went missing")

_PUNCT = re.compile(r"[?!.,;:()\"“”‘’|/\\।…]+")


def _norm(text: str) -> str:
    t = unicodedata.normalize("NFC", text or "").lower()
    return " ".join(_PUNCT.sub(" ", t).split())


def asks_lost_and_found(text: str) -> bool:
    """True when the caller says they left / lost something (or asks where to get it back), or cannot find a person."""
    t = _norm(text)
    if not t:
        return False
    lost = any(v in t for v in _LOST_VERB)
    if lost and (any(w in t for w in _THING) or any(r in t for r in _RETRIEVE)):
        return True
    if any(r in t for r in _RETRIEVE) and any(w in t for w in _THING) and ("ফেলে" in t or "হারি" in t or "left" in t or "lost" in t or "भूल" in t or "खो" in t):
        return True
    return _names_a_person(t) and any(l in t for l in _PERSON_LOST)


def _names_a_person(t: str) -> bool:
    """A family member / the patient named as a WORD (not a piece of a word: "মা" is inside "আমার")."""
    tokens = t.split()
    for p in _PERSON:
        if " " in p:
            if p in t:
                return True
        elif any(tok == p or (len(p) >= 3 and tok.startswith(p)) for tok in tokens):
            return True
    return False
