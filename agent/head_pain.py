"""A headache is asked about once before a department is named (owner, 2026-10-29: "mathar betha doctor").

"মাথায় ব্যথা" can be the ordinary doctor's (a knock, a cut, a fall) or the Neurology doctor's (a plain headache). The agent does not guess: it asks "only a headache, or did you get hurt on the head?",
and the answer decides -- hurt -> General Medicine, a plain headache -> Neurology. The clinic's own routing, the doctors and the sittings are used from there on (no new facts here); this module only
reads the words. All three languages, in their own scripts or in Latin letters.
"""

from __future__ import annotations

import re

_HEAD = re.compile(
    r"মাথ|মাতা|মাফা|माथ|सिर|\b(?:mathar?|mathay|mathai|mathate|matha|head|headache)\b",
    re.I,
)
_PAIN = re.compile(
    r"ব্যথা|ব্যাথা|বেথা|বেতা|যন্ত্রণা|ধরেছে|ধরে আছে|দপদপ|টনটন|दर्द|दरद|दुख|\b(?:pain|ache|aches|headache|byath\w*|beth\w*|dard|jontrona)\b",
    re.I,
)
_HURT = re.compile(
    r"লেগেছে|লেগে|লাগছে|লাগল|চোট|আঘাত|পড়ে|পরে গিয়ে|ফেটে|কেটে|রক্ত|ধাক্কা|ঠুকে|ঠোকা|অ্যাক্সিডেন্ট|এক্সিডেন্ট|মেরেছে|মারছে|"
    r"चोट|लगी|लगा|गिर|खून|फट|कट|टक्कर|एक्सीडेंट|एक्सिडेंट|"
    r"\b(?:hit|hurt|fell|fall|fallen|injur\w*|bleed\w*|blood|accident|bump\w*|knock\w*|cut|wound\w*|lege|legeche|legechhe|chot|choto?t)\b",
    re.I,
)


def asks_about_head_pain(text: str) -> bool:
    """A sentence about a head with pain in it and no sign of an injury: the one that is asked about."""
    t = text or ""
    return bool(_HEAD.search(t) and _PAIN.search(t) and not _HURT.search(t))


def hurt(text: str) -> bool:
    return bool(_HURT.search(text or ""))


def a_pain_answer(text: str) -> bool:
    """The answer "only a headache": it speaks of pain, or says "only / just / very much" about it. (Hurt is checked first by the caller.)"""
    t = text or ""
    return bool(_PAIN.search(t) or re.search(r"শুধু|শুধুই|কেবল|অনেক|খুব|ভীষণ|সারাদিন|सिर्फ|केवल|बहुत|ज्यादा|ज़्यादा|\b(?:only|just|very|much|lot)\b", t, re.I))


QUESTION = {
    "bn": "আপনার কি শুধু মাথায় ব্যথা আছে, নাকি মাথায় কোথাও লেগেছে?",
    "hi": "क्या आपको सिर्फ़ सिर में दर्द है, या सिर में कहीं चोट लगी है?",
    "en": "Do you only have a headache, or did you hurt your head?",
}


def question(lang: str) -> str:
    return QUESTION.get(lang, QUESTION["en"])
