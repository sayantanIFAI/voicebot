"""A patient's age said in the caller's own words, and the paediatric age limit (owner, 2026-10-24; credit: sourav).

Why: "bachhader doctor" is the paediatrician -- but the owner's rule is that a child of 13 YEARS OR MORE goes to the normal doctor (General Medicine), not the paediatrician.
The age comes from the sentence itself ("amar 15 bochorer chheler jor") or from the booking's own age answer. The routing decision is made by the clinic API
(clinic-api/lay_departments.py, `age` parameter); this module only READS the number and phrases the explanation. Digits in Latin / Bengali / Devanagari, the common
age words in the three languages, and the teen numbers 13-20 written as words (romanised Bengali / Hindi / English) -- a caller says "tero bochor" as often as "13".
"""

from __future__ import annotations

import re

PAEDIATRIC_MAX_AGE = 12  # 13 and above: the normal doctor (same constant as clinic-api/lay_departments.py)

_DIGITS = str.maketrans("০১২৩৪৫৬৭৮৯०१२३४५६७८९", "01234567890123456789")
_YEAR_WORDS = r"(?:years?|yrs?|yr|y/o|yo|year[\s-]*old|bochor\w*|bochhor\w*|bosor\w*|bochar|boyos\w*|boyes\w*|saal\w*|sal|baras|barsh|varsh|বছর\w*|বয়স\w*|বয়েস|সাল|साल|वर्ष|बरस|उम्र|उमर|आयु)"
_AGE_WORDS = r"(?:age|aged|boyos\w*|boyes\w*|umar|umr|umra|বয়স\w*|বয়েস|उम्र|उमर|आयु)"

# teen numbers written as words (13-20) in English, romanised Bengali, romanised Hindi
_WORD_NUMBERS = {
    13: ("thirteen", "tero", "terah", "tera", "তেরো", "তের", "तेरह"),
    14: ("fourteen", "choddo", "chowdo", "chaudah", "chaudha", "চোদ্দ", "চৌদ্দ", "चौदह"),
    15: ("fifteen", "ponero", "pondro", "pandrah", "পনেরো", "পনের", "पंद्रह", "पन्द्रह"),
    16: ("sixteen", "sholo", "solah", "soleh", "ষোল", "ষোলো", "सोलह"),
    17: ("seventeen", "satero", "shotero", "satrah", "সতেরো", "সতের", "सत्रह"),
    18: ("eighteen", "athero", "atharah", "আঠারো", "আঠার", "अठारह"),
    19: ("nineteen", "unish", "unnis", "ekunish", "উনিশ", "उन्नीस"),
    20: ("twenty", "kuri", "bish", "bees", "বিশ", "কুড়ি", "बीस"),
}

_NUM_BEFORE = re.compile(rf"(?<!\d)(\d{{1,3}})\s*[-]?\s*{_YEAR_WORDS}(?![a-z])", re.I)
_NUM_AFTER = re.compile(rf"{_AGE_WORDS}\s*(?:holo|hoyeche|hoyechhe|hocche|hai|hain|is|of|:|=|-)?\s*(\d{{1,3}})(?!\d)", re.I)
_BARE_AGED = re.compile(r"(?<!\d)(\d{1,3})\s*(?:yo|y\.o\.|yrs?)\b", re.I)


def age_in(text: str) -> int | None:
    """The patient's age in years if the sentence says one, else None. Never a guess: a number with no age word beside it is not an age."""
    t = (text or "").translate(_DIGITS).lower()
    if not t.strip():
        return None
    for pat in (_NUM_BEFORE, _NUM_AFTER, _BARE_AGED):
        m = pat.search(t)
        if m and 0 < int(m.group(1)) <= 120:
            return int(m.group(1))
    for n, words in _WORD_NUMBERS.items():
        for w in words:
            if re.search(rf"(?<![a-zঀ-৿ऀ-ॿ]){re.escape(w)}\s*{_YEAR_WORDS}(?![a-z])", t) or re.search(
                rf"{_AGE_WORDS}\s*(?:holo|hoyeche|hai|is|of)?\s*{re.escape(w)}(?![a-z])", t
            ):
                return n
    return None


def is_teen_or_older(age: int | None) -> bool:
    return age is not None and age > PAEDIATRIC_MAX_AGE


_REDIRECT = {
    "bn": "বাচ্চাদের ডাক্তার ১২ বছর পর্যন্ত দেখেন। ১৩ বছর বা তার বেশি বয়স হলে সাধারণ ডাক্তার দেখান, তাই আমি ডাক্তার {doctor}-এর সঙ্গে করছি।",
    "hi": "बच्चों के डॉक्टर 12 साल तक के बच्चों को देखते हैं। 13 साल या उससे ज़्यादा उम्र हो तो सामान्य डॉक्टर दिखाते हैं, इसलिए मैं डॉक्टर {doctor} के साथ कर रही हूँ।",
    "en": "The children's doctor sees patients up to 12 years. From 13 years the normal doctor is seen, so I am booking with Doctor {doctor}.",
}


def redirect_reply(doctor: str, lang: str) -> str:
    """Said when a patient of 13 or more was about to be booked with the paediatrician and the booking moves to a General Medicine doctor."""
    return _REDIRECT.get(lang, _REDIRECT["en"]).format(doctor=doctor)
