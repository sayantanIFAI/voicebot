"""Questions about THE doctor being talked about: which department / speciality, the fee, and the menu of what can be asked (owner, 2026-10-29).

"রাজশ্রী দত্ত কিসের ডাক্তার" with Dr. Rajarshi Dutta already the subject is "what is HE a doctor of?" -- answered with his department, never with "there is no doctor of that name". The same for the fee when a
recogniser writes it in English letters in a Bengali sentence ("rather see the ten fees for"). Only words are read here; the department and the fee come from the clinic's own tables (main.py).
"""

from __future__ import annotations

import re

_SPECIALITY = re.compile(
    r"কিসের\s*ডাক্তার|কীসের\s*ডাক্তার|কি\s*ডাক্তার|কী\s*ডাক্তার|কিসের\s*(?:ডা|ডক্টর)|কোন\s*(?:বিভাগ|ডিপার্টমেন্ট)|কি\s*বিভাগ|কী\s*বিভাগ|বিভাগের|ডিপার্টমেন্ট|স্পেশালিস্ট|স্পেশালিটি|"
    r"किस\s*(?:चीज़|चीज|बीमारी)\s*के\s*डॉक्टर|किसके\s*डॉक्टर|कौन\s*से\s*विभाग|किस\s*विभाग|विभाग|स्पेशलिस्ट|स्पेशलिटी|"
    r"\b(?:what\s+(?:kind|sort|type)\s+of\s+doctor|which\s+department|what\s+department|specialist|speciality|specialty|department)\b",
    re.I,
)
_FEE = re.compile(
    r"ফি(?!\S)|ফিস|ফীস|ফিজ|ভিজিট|"
    r"फीस|फ़ीस|विज़िट|विजिट|"
    r"\b(?:fee|fees|feez|visit)\b",
    re.I,
)
_TEST_WORD = re.compile(r"টেস্ট|পরীক্ষা|রিপোর্ট|এক্স\s*রে|स्कैन|टेस्ट|जांच|\b(?:test|tests|report|scan|x-?ray|mri|ct)\b", re.I)

MENU = {
    "bn": "{name} সম্পর্কে কী জানতে চান? কোন বিভাগের ডাক্তার, কবে বসেন, ভিজিট কত, কলকাতার কি না, কত ভিড়, নাকি সিরিয়াল নম্বর?",
    "hi": "{name} के बारे में क्या जानना है? कौन से विभाग के हैं, कब बैठते हैं, विज़िट फ़ीस, कोलकाता के हैं या नहीं, कितनी भीड़ है, या सीरियल नंबर?",
    "en": "What would you like to know about {name}? Their department, when they sit, the visit fee, whether they are from Kolkata, how crowded it is, or the serial number?",
}


# Owner, 2026-10-30 ("কোন ডাক্তার এটা" right after "উনি সামনের মাসের দুই তারিখে বসবেন" was answered "কোন ডাক্তারের কথা জিজ্ঞেস করছেন?"): "which doctor is this / who is he / what is his name" about the doctor JUST
# spoken of is answered with that doctor's name, never asked back.
_WHICH_DOCTOR = re.compile(
    r"কোন\s*ডাক্তার\s*(?:এটা|ওটা|এটি|এই|ওই|উনি|ওনি|এনি|এনার|ওনার|সেটা|সে)(?![ঀ-৿])|(?:এটা|ওটা|উনি|ওনি|এনি|সে)\s*কোন\s*ডাক্তার|(?:উনি|ওনি|এনি|ইনি)\s*কে(?![ঀ-৿])|(?:ডাক্তারের|ওনার|এনার|ওনারা)\s*নাম|কার\s*কথা\s*বলছ|"
    r"कौन\s*(?:सा\s*)?डॉक्टर\s*(?:है|हैं|ये|यह|वो)|इनका\s*नाम|उनका\s*नाम|ये\s*कौन|वो\s*कौन|"
    r"\bwhich\s+doctor\s+(?:is\s+)?(?:this|that|he|she|it)\b|\bwho\s+(?:is|are)\s+(?:he|she|this|that)\b|\bdoctor'?s\s+name\b|\bwhat'?s\s+(?:his|her)\s+name\b",
    re.I,
)


_QUALITY_WORDS = frozenset("ভালো ভাল ভালই ভাল্লো good best अच्छा अच्छे अच्छी बढ़िया".split())
_QUALITY_FILLER = frozenset(
    "তো কি কী উনি ওনি ইনি ওনারা এনারা ওরা ও এই ওই এ সে ডাক্তার ডাক্তারবাবু সব সবাই হ্যাঁ আচ্ছা আর তাহলে ক্যামন কেমন "
    "is he she it they are the doctor doctors so then ok okay really "
    "तो क्या ये वो वह डॉक्टर सब है हैं और फिर".split()
)


def asks_quality_bare(text: str) -> bool:
    """"তো কি ভালো", "ডাক্তার কি ভালো", "is he good" -- a short question about how good the doctor being spoken of is, made only of a quality word and little words (owner, 2026-10-31)."""
    toks = [w.strip(".,?!।").lower() for w in (text or "").split() if w.strip(".,?!।")]
    return 1 <= len(toks) <= 4 and any(t in _QUALITY_WORDS for t in toks) and all(t in _QUALITY_WORDS or t in _QUALITY_FILLER for t in toks)


def asks_which_doctor(text: str) -> bool:
    t = text or ""
    return len(t.split()) <= 8 and _WHICH_DOCTOR.search(t) is not None


def asks_speciality(text: str) -> bool:
    return bool(_SPECIALITY.search(text or ""))


def mentions_fee(text: str) -> bool:
    """A fee word (ফি / fees / ভিজিট / फीस) and no lab-test word: with a doctor being talked about, that is the doctor's visit fee."""
    t = text or ""
    return bool(_FEE.search(t)) and not _TEST_WORD.search(t)


def speciality_reply(name: str, department: str, lang: str) -> str:
    """"ডাঃ রাজর্ষি দত্ত জেনারেল মেডিসিন বিভাগের ডাক্তার।" -- the department IS the speciality."""
    return {
        "bn": f"{name} {department} বিভাগের ডাক্তার।",
        "hi": f"{name} {department} विभाग के डॉक्टर हैं।",
        "en": f"{name} is a doctor in the {department} department.",
    }.get(lang, f"{name} - {department}")


def menu(name: str, lang: str) -> str:
    return MENU.get(lang, MENU["en"]).format(name=name)
