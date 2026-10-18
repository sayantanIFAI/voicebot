""""Do you do the lipid profile test?" is a question about the clinic, not a request to book.

Real call 2026-10-14: "লিপিড প্রোফাইল টেস্ট হয়" and "लिपिड प्रोफाइल टेस्ट होता है" (is the lipid profile test done here?)
were both answered "bookings and payment happen at the counter, not on the phone". Run against the pod's own intent
model, those sentences -- and even the English "do you do the lipid profile test" -- come back as `book_test`: the model
has no intent for "is it available" and picks the nearest one, which is the booking intent, whose reply is the counter
message. The caller asked a yes/no question about the catalogue and got a policy statement about payment.

What the caller should hear: if the test is in the clinic's table, YES, and an invitation to ask more ("what would you
like to know about it?"). Only a caller who actually asks to BOOK it hears the counter message -- unchanged.

This module is the deterministic half: it decides whether a sentence is an availability question and words the answer.
The truth boundary is untouched: "yes" is said only after the clinic API found the test (main.py looks it up first), and
the reply states nothing but that the test exists -- no price, no preparation, no time.
"""

from __future__ import annotations

import re
import unicodedata

from agent import translit_frames

_PUNCT = re.compile(r"[.,;:!?\"'()\[\]{}।॥\-–—_/]+")


def _norm(text: str) -> str:
    s = unicodedata.normalize("NFC", text or "").lower().replace("़", "").replace("়", "")
    return " ".join(_PUNCT.sub(" ", s).split())


# The sentence ENDS in a word that asks "is it done / is it there / can it be had". A test name comes before it.
# Written nukta-free (hoi == হয with or without the dot under the ya).
_AVAILABLE_TAILS: dict[str, tuple[str, ...]] = {
    "bn": (
        "হয", "হয কি", "কি হয", "হয না", "হবে কি", "হবে", "আছে", "আছে কি", "আছে না", "কি আছে", "করেন", "করেন কি",
        "করেন না", "করা হয", "করা হয কি", "করা যায", "করা যায কি", "করানো যায", "করা যাবে", "পাওয়া যায",
        "পাওয়া যাবে", "পাওয়া যায কি", "হয তো", "হয তো কি",
    ),
    "hi": (
        "होता है", "होता है क्या", "क्या होता है", "होती है", "होती है क्या", "हो जाता है", "हो जाती है", "होगा",
        "होगा क्या", "है क्या", "क्या है", "मिलता है", "मिलता है क्या", "मिलती है", "करते हैं", "करते हैं क्या",
        "करते है", "क्या करते हैं", "किया जाता है", "कर सकते हैं", "हो सकता है", "हो सकती है", "उपलब्ध है",
        "उपलब्ध है क्या", "होता है कि नहीं", "होता है या नहीं",
    ),
    "en": (
        "available", "available here", "done here", "possible", "possible here", "do you do it", "do you do this",
        "do you do that", "do you have it", "do you offer it", "do you perform it", "do you conduct it",
    ),
}
# English also asks it at the START: "do you do the lipid profile test", "is the CBC test available".
_EN_STARTS = re.compile(r"^(?:please\s+)?(?:do|does)\s+(?:you|u)\s+(?:do|have|offer|perform|conduct|provide|run|take)\b")
_EN_IS_AVAILABLE = re.compile(r"\b(?:is|are)\b.*\b(?:available|done|possible|offered)\b")

# Anything that makes it NOT a plain availability question: a booking verb, a price, a preparation, a report time.
_NOT_AVAILABILITY = (
    # booking
    "book", "booking", "appointment", "schedule", "slot", "want to", "need to", "would like to", "i want", "i need",
    "বুক", "বুকিং", "করাতে", "করাব", "করাবো", "করতে চাই", "নিতে চাই", "চাই", "দরকার", "অ্যাপয়েন্টমেন্ট", "অ্যাপয়েন্টমেন্ট",
    "बुक", "बुकिंग", "करवान", "करान", "चाहिए", "चाहता", "चाहती", "अपॉइंटमेंट", "करना है", "करनी है", "करवाना",
    # price
    "rate", "price", "cost", "fee", "charge", "how much", "রেট", "দাম", "দর", "কত", "টাকা", "খরচ", "রেট",
    "रेट", "दाम", "कीमत", "कितना", "कितने", "पैसे", "खर्च", "फीस", "भाव",
    # preparation
    "prepare", "preparation", "prep", "fasting", "empty stomach", "প্রস্তুতি", "ফাস্টিং", "খালি পেটে", "तैयारी",
    "फास्टिंग", "खाली पेट",
    # report time
    "report", "রিপোর্ট কবে", "রিপোর্ট কখন", "रिपोर्ट कब", "when will",
)


def _contains_cue(norm_text: str, cues: tuple[str, ...]) -> bool:
    padded = f" {norm_text} "
    for cue in cues:
        c = _norm(cue)
        if c.isascii():
            if f" {c} " in padded:
                return True
        elif c in norm_text:
            return True
    return False


# "<what> is done / available at your place": the structural frame of a "do you have X?" question, whatever X is.
_HAVE_FRAME = re.compile(
    r"(?:আপনাদের|আপনার|ওদের|তোমাদের)?\s*(?:ওখানে|এখানে|কাছে)\s+(?P<x>.+?)\s+(?:হয|করেন|পাওয়া যায|আছে)$"
    r"|(?:आपके|आप लोगों के|आपका)\s*(?:यहाँ|यहां|वहाँ|वहां)\s+(?P<y>.+?)\s+(?:होता है|होते हैं|होती है|होता|हो जाता है|करते हैं)$"
)  # the Bengali verbs are written without the nukta: _norm() drops it ("হয়" -> "হয")


def have_frame_name(text: str) -> str | None:
    """The words in "<your place> X <is done>" ("আপনাদের ওখানে চিস্টিক্স রে হয়") -- X, with a trailing word for "test" removed -- or None."""
    m = _HAVE_FRAME.search(_norm(text))
    if not m:
        return None
    x = (m.group("x") or m.group("y") or "").split()
    while x and x[-1] in _TEST_WORD_FORMS:
        x.pop()
    return " ".join(x) if x and sum(len(w) for w in x) >= 3 else None


_DONE_HERE = (
    "হয় কি", "করা হয়", "করা যায়", "করবো হয়", "করানো যায়", "করেন কি", "পাওয়া যায়", "আছে কি", "হয় তো", "হয়?", "होता है क्या", "करते हैं क्या", "हो जाता है क्या", "मिलता है क्या",
    "available", "do you do", "do you have", "is it done", "can i get",
)


def mentions_done_here(text: str) -> bool:
    """Loosely: does the caller ask whether a test is DONE here ("করবো হয় কি", "होता है क्या", "do you do ...") -- anywhere in the sentence, however long. Not a router (that is
    asks_if_test_is_available, narrow on purpose): only decides to open the rate answer with "Yes, we do these tests"."""
    t = _norm(text)
    return bool(t) and _contains_cue(t, _DONE_HERE)


def asks_if_test_is_available(text: str, lang: str) -> bool:
    """True for "is the X test done here?" in any of the three languages. Deliberately narrow: a sentence that also
    asks a price, a preparation, a report time, or asks to book is NOT an availability question -- those have their
    own answers, and a sentence like "CBC hoi ar dam koto" must still be priced."""
    t = _norm(text)
    if not t or _contains_cue(t, _NOT_AVAILABILITY):
        return False
    if translit_frames.match(text) is not None:  # "do you do chest X-ray" written in Bengali/Devanagari letters
        return True
    if have_frame_name(text) is not None and len(t.split()) <= 9:  # "<your place> X <is done>": the intent layer decides whether X is a test
        return True
    words = t.split()
    if len(words) > 14:
        return False
    # Real call 2026-10-17: "ক্গ টেস্ট হয় ওখানে" / "कि टेस्ट हो अपना ..." -- the test NAME was lost by the recogniser but the question is plain: "is a ... test done
    # there?". The word "test" followed within three words by "done/happens" (হয়, होता, हो ...) is that question, whatever the name.
    if len(words) <= 9:
        for i, w in enumerate(words):
            if w in _TEST_WORD_FORMS and any(v in _DONE_VERBS for v in words[i + 1: i + 4]):
                return True
    # Every language's tails are tried: the recogniser's language label for one clip is not always right.
    for tails in _AVAILABLE_TAILS.values():
        for tail in tails:
            tail_n = _norm(tail)
            if t == tail_n:
                continue  # a bare "হয" / "available" with no test named is an answer, not a question
            if t.endswith(" " + tail_n):
                return True
    if lang == "en" or t.isascii():
        if _EN_STARTS.search(t) or _EN_IS_AVAILABLE.search(t):
            return True
    return False


_TEST_WORD_FORMS = frozenset(("টেস্ট", "টেস্টটা", "টেস্টগুলো", "टेस्ट", "test", "tests"))
_DONE_VERBS = frozenset(("হয়", "হয", "হোয়", "করেন", "করে", "होता", "होते", "होती", "हो", "करते", "करती", "do", "does"))

_TEST_WORDS = ("test", "টেস্ট", "টেস্টের", "টেস্টটা", "टेस्ट", "পরীক্ষা", "परीक्षा", "जांच", "scan", "স্ক্যান", "स्कैन")


def _has_test_word(name: str) -> bool:
    """"Lipid Profile test" / "লিপিড প্রোফাইল টেস্ট" already say it; "Lipid Profile" / "সিবিসি" do not."""
    low = _norm(name)
    return any(w in low.split() or low.endswith(w) for w in _TEST_WORDS)


def _join(names: list[str], lang: str) -> str:
    if len(names) == 1:
        return names[0]
    word = {"bn": "এবং", "hi": "और", "en": "and"}.get(lang, "and")
    return ", ".join(names[:-1]) + f" {word} " + names[-1]


def availability_reply(names: list[str], lang: str) -> str:
    """The answer, from template. `names` are the tests the clinic's table CONFIRMED, as the caller said them."""
    many = len(names) > 1
    word = {"bn": "টেস্ট", "hi": "टेस्ट", "en": "test"}
    spoken = _join([n if _has_test_word(n) else f"{n} {word.get(lang, 'test')}" for n in names], lang)
    if lang == "hi":
        head = f"जी हाँ, होता है। हमारे यहाँ {spoken} होते हैं।" if many else f"जी हाँ, होता है। हमारे यहाँ {spoken} होता है।"
        return f"{head} इसके बारे में और क्या जानना चाहेंगे? रेट, तैयारी, या रिपोर्ट कब मिलेगी?"
    if lang == "en":
        return (
            f"Yes, we do. We do the {spoken}. What would you like to know about {'them' if many else 'it'}? "
            "The rate, the preparation, or when the report is ready?"
        )
    return f"হ্যাঁ, হয়। আমাদের এখানে {spoken} হয়। এ সম্পর্কে আর কী জানতে চান? রেট, প্রস্তুতি, নাকি রিপোর্ট কবে পাবেন?"
