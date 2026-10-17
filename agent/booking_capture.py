"""Deterministic capture of the three booking details a language model must never be the only judge of.

sourav, 2026-10-16 -- WHY THIS MODULE EXISTS. A live call (sayantan, 2026-10-14) never reached the "shall I confirm?" read-back:
  * the phone number was taken ONLY from what the intent model returned. A number said in pieces, in Hindi/Bengali digit
    words, as "double nine", or garbled by the recogniser came back empty (so the slot never filled) or as a wrong string (and
    merge_slots() stored it unchecked, so a 7-digit "number" reached the read-back and the booking call);
  * a patient name the model invented was stored as the caller's identity;
  * when the model failed, the caller heard the generic "little problem" instead of the next booking question.
agent/spoken_codes.parse_phone existed but nothing called it. This module is the small, pure, tested layer that main_pcm.py /
main.py call BEFORE (digits) and AFTER (checks) the model, so the booking state machine in agent/booking_flow.py keeps its
exact shape (collecting -> confirming -> awaiting_charge_confirm -> done) and only its inputs become reliable.

FLOW (per turn, only while a doctor booking is `collecting`):
  phone : absorb(buffer, text, asked) -> complete (10-digit mobile, slot filled) | partial (keep digits, ask for the rest) |
          invalid (clear, ask again digit by digit) | cleared (caller said "wrong / start again") | none (not about the phone).
  name  : the model's name is kept only if the caller's words support it; otherwise (or when the model gave none) a name read
          from the caller's own words is read BACK and needs a yes before it becomes the patient's name.
  slot  : "the first one / second one" picks from the slots the clinic itself offered after a conflict -- never from anywhere else.
Nothing here decides a fact: no doctor, date, price or availability is produced. A digit string is only ever the caller's digits.
A full phone number is never logged (mask()).
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from agent import nato_letters, phonetic_match, spoken_codes, slot_grounding

# ----------------------------------------------------------------------------------------------------------------- digits

# spoken_codes._DIGIT_WORD already holds the English / Bengali / Devanagari digit words (one digit each). Romanised Hindi and
# Bengali digit words are added here: ASR for a Hindi/Bengali call often writes them in Latin letters ("nau paanch teen").
_ROMAN_DIGIT_WORD = {
    "ek": "1", "dui": "2", "do": "2", "teen": "3", "tin": "3", "char": "4", "chaar": "4", "paanch": "5", "panch": "5",
    "pach": "5", "chhe": "6", "chhoy": "6", "che": "6", "chhah": "6", "saat": "7", "sat": "7", "aath": "8", "aat": "8",
    "nau": "9", "noy": "9", "shunya": "0", "shunyo": "0", "sunno": "0", "jiro": "0", "zero": "0",
}  # fmt: skip
_MULTIPLIER = {
    "double": 2, "ডবল": 2, "dabal": 2, "dabl": 2, "ডাবল": 2, "डबल": 2, "দুবার": 2, "triple": 3, "tripal": 3, "ট্রিপল": 3, "ट्रिपल": 3,
}  # fmt: skip
# sourav, 2026-10-16 (real Bengali call): "Seven Eight ... Double Six" was refused as "not a 10-digit number" because the recogniser wrote
# English digit words in Bengali / Devanagari letters ("ডাবল সিক্স", "सेवन एट") and the table only knew the native words.
_EXTRA_DIGIT_WORD = {
    "জিরো": "0", "জিরও": "0", "ওয়ান": "1", "টু": "2", "থ্রি": "3", "ফোর": "4", "ফাইভ": "5", "সিক্স": "6", "সেভেন": "7", "এইট": "8",
    "নাইট": "8", "সেবেন": "7", "সেবন": "7", "সেভন": "7", "সিক্ষ": "6", "ফাইব": "5", "থ্রী": "3", "ওয়ন": "1", "ওন": "1", "নাইন": "9", "वन": "1", "टू": "2", "थ्री": "3", "फोर": "4", "फाइव": "5", "सिक्स": "6", "सेवन": "7", "एट": "8", "नाइन": "9",
    "ছ": "6", "শুন্য": "0", "शुन्य": "0", "नो": "9", "छे": "6", "छै": "6", "paach": "5", "saath": "7", "nine": "9", "aat": "8",
}  # fmt: skip
_DIGIT_WORDS = {**spoken_codes._DIGIT_WORD, **_ROMAN_DIGIT_WORD, **_EXTRA_DIGIT_WORD}
# Numbers said as numbers ("four hundred", "eighty nine", "twenty") are NOT digits: the caller is told to say the number digit by digit.
_COMPOUND_STRONG = frozenset({
    "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety", "hundred", "thousand", "sau", "hazar", "hajar", "bis",
    "tis", "chalis", "pachas", "sattar", "assi", "nabbe", "একশো", "শো", "হাজার", "কুড়ি", "বিশ", "ত্রিশ", "চল্লিশ", "পঞ্চাশ", "ষাট", "সত্তর",
    "আশি", "নব্বই", "सौ", "सौं", "हज़ार", "हजार", "बीस", "तीस", "चालीस", "पचास", "साठ", "सत्तर", "अस्सी", "नब्बे",
})  # fmt: skip
# "ten".."nineteen" are also the HOUR in a time answer ("eleven"), so they only count as 'a number said as a number' beside digits.
_COMPOUND_TEENS = frozenset({
    "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen", "দশ", "এগারো", "বারো",
    "दस", "ग्यारह", "बारह", "dosh", "das",
})  # fmt: skip
_COMPOUND_WORDS = _COMPOUND_STRONG | _COMPOUND_TEENS

# Said by a caller who wants the number taken again from the start.
_RESET_WORDS = (
    "wrong number", "start again", "start over", "from the beginning", "galat", "ग़लत", "गलत", "फिर से शुरू", "शुरू से",
    "ভুল", "আবার প্রথম থেকে", "নতুন করে",
)  # fmt: skip


# "five nines" / "পাঁচটা নাইন" (five of the digit nine): a COUNT followed by a digit. In Bengali the count carries the classifier "-টা/-টি/-টে" ("পাঁচটা"), which a
# plain digit does not ("পাঁচ"); in English it is the plural digit ("nines", "eights"). Real call 2026-10-17: "পাঁচটা নাইন পাঁচটা নাইট" (99999 88888) was read as 2 digits.
_CLASSIFIED_COUNT = {
    "দুটো": 2, "দুটি": 2, "দুইটা": 2, "তিনটে": 3, "তিনটা": 3, "তিনটি": 3, "চারটে": 4, "চারটা": 4, "চারটি": 4, "পাঁচটা": 5, "পাঁচটি": 5, "পাঁচটে": 5,
    "ছটা": 6, "ছয়টা": 6, "ছয়টি": 6, "সাতটা": 7, "সাতটি": 7, "আটটা": 8, "আটটি": 8, "নয়টা": 9, "নয়টি": 9,
}  # fmt: skip
_PLURAL_DIGIT = {"zeros": "0", "ones": "1", "twos": "2", "threes": "3", "fours": "4", "fives": "5", "sixes": "6", "sevens": "7", "eights": "8", "nines": "9"}
_COUNT_WORD = {"two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9}


def _digit_of(tok: str) -> str | None:
    tok = tok.lstrip("+")
    if tok.isdigit() and len(tok) == 1:
        return tok
    if tok in _DIGIT_WORDS:
        return _DIGIT_WORDS[tok]
    if tok in _PLURAL_DIGIT:
        return _PLURAL_DIGIT[tok]
    return None


def read_digits(text: str) -> tuple[str, int, int]:
    """(digits said, number of tokens that were digits, number of tokens). "double nine" -> "99"; "triple five" -> "555"; "five nines" -> "99999";
    "পাঁচটা নাইন" -> "99999"; a numeral token ("98765") is kept whole; any other word is skipped, never merged into the digits around it."""
    tokens = spoken_codes._tokens(text)
    out: list[str] = []
    hits = 0
    times = 1
    i = 0
    while i < len(tokens):
        tok = tokens[i].lstrip("+")
        following = _digit_of(tokens[i + 1]) if i + 1 < len(tokens) else None
        if following is not None and tok in _CLASSIFIED_COUNT:  # "পাঁচটা নাইন"
            out.append(following * _CLASSIFIED_COUNT[tok])
            hits += 2
            times = 1
            i += 2
            continue
        if following is not None and tok in _COUNT_WORD and tokens[i + 1] in _PLURAL_DIGIT:  # "five nines"
            out.append(following * _COUNT_WORD[tok])
            hits += 2
            times = 1
            i += 2
            continue
        i += 1
        if tok in _MULTIPLIER:
            times = _MULTIPLIER[tok]
            hits += 1
            continue
        if tok.isdigit():
            digit = tok
        elif tok in _DIGIT_WORDS:
            digit = _DIGIT_WORDS[tok]
        elif tok in _PLURAL_DIGIT:
            digit = _PLURAL_DIGIT[tok]
        else:
            times = 1
            continue
        out.append(digit * times if len(digit) == 1 else digit)
        times = 1
        hits += 1
    return "".join(out), hits, len(tokens)


def phone_from_digits(digits: str) -> str | None:
    """The 10-digit Indian mobile number in `digits`, or None -- the SAME rule as spoken_codes.parse_phone (a leading 0091 / 091 /
    91 / 0 dropped when exactly ten digits remain; must start 6-9), applied to digits already read."""
    for prefix in ("0091", "091", "91", "0"):
        if digits.startswith(prefix) and len(digits) - len(prefix) == 10:
            digits = digits[len(prefix) :]
            break
    return digits if len(digits) == 10 and digits[0] in "6789" else None


def normalise_phone(raw: object) -> str | None:
    """A phone value from ANY source (the model's slot, a caller's sentence) -> the valid 10-digit number or None."""
    if not isinstance(raw, str) or not raw.strip():
        return None
    digits, _, _ = read_digits(raw)
    return phone_from_digits(digits)


def mask(number: str | None) -> str:
    """For logs: never the whole number."""
    return "" if not number else ("*" * max(0, len(number) - 2)) + number[-2:]


@dataclass(frozen=True)
class PhoneStep:
    status: str  # complete | partial | invalid | compound | cleared | none
    number: str | None = None  # set when complete
    buffer: str = ""  # the digits kept for the next turn (partial) -- "" otherwise
    pure: bool = False  # the sentence was (almost) only digits: the model has nothing else to add
    reason: str = ""  # invalid only: "start" (10 digits, not starting 6-9) | "length" (not 10 digits)
    n: int = 0  # invalid: the number of digits heard ("length") ; "start": the first digit heard as an int


def absorb(buffer: str, text: str, asked: bool) -> PhoneStep:
    """One turn of phone capture. `buffer` = digits kept from earlier turns of THIS booking; `asked` = the phone is the question
    the caller is answering (so a sentence of a few digits is certainly the number). When it is not asked, only a complete
    valid number counts -- digits in an unrelated sentence ("at 5 pm", "3 days") are never appended to the buffer."""
    low = unicodedata.normalize("NFC", text or "").lower()
    digits, hits, total = read_digits(text)
    # sourav, 2026-10-16: "four five hundred seven eighty-nine" -- a number said as a number. Its digits cannot be trusted ("458789"
    # was read back to a real caller), so the caller is asked for the number one digit at a time instead of a wrong read-back.
    said_tokens = spoken_codes._tokens(text)
    said_as_number = any(t in (_COMPOUND_WORDS if digits else _COMPOUND_STRONG) for t in said_tokens)
    if not digits:
        if buffer and any(w in low for w in _RESET_WORDS):
            return PhoneStep("cleared")
        if said_as_number and asked and total <= 6:
            return PhoneStep("compound", buffer=buffer, pure=True)
        return PhoneStep("none", buffer=buffer)
    pure = total > 0 and hits / total >= 0.6
    whole = phone_from_digits(digits)
    if said_as_number and not whole and (asked or buffer or len(digits) >= 3):
        return PhoneStep("compound", buffer=buffer, pure=pure)
    if whole:  # a full number in one go -- also a REPLACEMENT or a repeat of the whole thing: never appended to old digits
        return PhoneStep("complete", number=whole, pure=pure)
    if not (asked or buffer):
        return PhoneStep("none")  # not the phone question, and not a full number: leave the sentence to the rest of the pipeline
    if any(w in low for w in _RESET_WORDS):
        buffer = ""
    if buffer and digits.startswith(buffer):
        candidate = digits  # the caller repeated what they already said, then went on
    elif buffer and buffer.endswith(digits):
        candidate = buffer  # the same tail repeated: do not add it twice
    else:
        candidate = buffer + digits
    done = phone_from_digits(candidate)
    if done:
        return PhoneStep("complete", number=done, pure=pure)
    if len(candidate) < 10:
        return PhoneStep("partial", buffer=candidate, pure=pure)
    # 10+ digits that are not a mobile number: ask again, keep nothing -- and say WHICH rule failed (a real caller who gave ten digits
    # was only told "not a 10-digit number", which was simply untrue).
    body = candidate
    for prefix in ("0091", "091", "91", "0"):
        if candidate.startswith(prefix) and len(candidate) - len(prefix) == 10:
            body = candidate[len(prefix) :]
            break
    if len(body) == 10:
        return PhoneStep("invalid", pure=pure, reason="start", n=int(body[0]))
    return PhoneStep("invalid", pure=pure, reason="length", n=len(candidate))


# ----------------------------------------------------------------------------------------------------------------- name

_NAME_LEADS = (
    "my name is", "my name's", "the name is", "name is", "this is", "it is", "it's", "i am", "i'm", "call me", "patient name is",
    "patient is", "its", "mera naam", "mera nam", "meri naam", "naam hai", "naam", "mera", "main", "mai", "amar naam", "amar nam",
    "amar", "ami", "ঠিক আছে", "হ্যাঁ", "হ্যা", "হাঁ", "আচ্ছা", "yes", "yeah", "haan", "han", "okay", "ok", "हाँ", "हां", "अच्छा", "আমার নাম", "আমার", "আমি", "নাম", "मेरा नाम", "मेरा", "मैं", "मै", "नाम", "पेशेंट का नाम", "मरीज़ का नाम",
)  # fmt: skip
_NAME_TAILS = (
    "hai", "hain", "है", "हैं", "হয়", "হলো", "হল", "please", "pls", "ji", "जी", "জি", "সাহেব", "thanks", "thank you",
)  # fmt: skip
_NOT_A_NAME = frozenset(
    {
        "yes", "no", "ok", "okay", "haan", "han", "nahi", "nahin", "na", "ji", "doctor", "dr", "appointment", "book", "booking",
        "today", "tomorrow", "kal", "aaj", "parso", "hello", "hi", "sorry", "what", "which", "when", "why", "how", "again",
        "repeat", "wait", "thanks", "thank", "morning", "evening", "afternoon", "clinic", "time", "date", "number", "phone",
        "হ্যাঁ", "না", "নয়", "ডাক্তার", "কাল", "আজ", "হ্যালো", "আবার", "কী", "কি", "কখন", "কেন",
        "हाँ", "हां", "नहीं", "नही", "डॉक्टर", "कल", "आज", "परसों", "हैलो", "क्या", "कब", "कैसे", "फिर", "दोबारा",
    }
)  # fmt: skip


def _strip_leads_and_tails(words: list[str]) -> list[str]:
    changed = True
    while changed and words:
        changed = False
        joined = " ".join(words).lower()
        for lead in sorted(_NAME_LEADS, key=len, reverse=True):
            if joined == lead or joined.startswith(lead + " "):
                words = words[len(lead.split()) :]
                changed = True
                break
        if not words:
            break
        joined = " ".join(words).lower()
        for tail in sorted(_NAME_TAILS, key=len, reverse=True):
            if joined == tail or joined.endswith(" " + tail):
                words = words[: len(words) - len(tail.split())]
                changed = True
                break
    return words


def _letters_only(word: str) -> bool:
    return bool(word) and all(unicodedata.category(c)[0] in ("L", "M") or c in "'.-" for c in word)


def candidate_name(text: str) -> str | None:
    """A patient name read from the caller's OWN words, or None. It accepts one to four words that are all letters (any script),
    after dropping "my name is / mera naam / amar naam / आमি ..." and a trailing "hai / please". A yes/no, a question, a sentence
    that holds digits or booking words is never a name. Latin names are Title-cased; other scripts are left exactly as said."""
    raw = (text or "").strip()
    if not raw or "?" in raw or "؟" in raw:
        return None
    words = [w for w in re.split(r"[\s,।|!]+", raw) if w]
    # Real call 2026-10-17: "হ্যাঁ সায়ন দাস নাইন সিক্স থ্রি ডবল ফাইভ ..." -- the name and the phone number in ONE breath. The number words are the phone's,
    # not the name's: with three or more of them in the sentence they are set aside and what is left is read as the name.
    phone_words = [w for w in words if w.lower() in _DIGIT_WORDS or w.lower() in _MULTIPLIER or any(ch.isdigit() for ch in w)]
    if len(phone_words) >= 3:
        words = [w for w in words if w not in phone_words]
    words = _strip_leads_and_tails([w.strip(".") for w in words if w.strip(".")])
    if not 1 <= len(words) <= 4:
        return None
    if any(not _letters_only(w) for w in words):
        return None
    if any(w.lower() in _NOT_A_NAME or w.lower() in _HEDGES for w in words):
        return None
    name = " ".join(words)
    return name.title() if name.isascii() else name


def name_is_grounded(name: str, text: str) -> bool:
    """Does the caller's sentence support the name the model gave? (the same literal / one-or-two-letters-off test the slot-grounding
    module uses for tests and doctors). False means: ask the caller to confirm it -- it is never silently stored as the patient."""
    if slot_grounding._literally_in(name, slot_grounding._norm(text)):
        return True
    # sourav, 2026-10-16: the caller said "अमित रॉय" / "অমিত রায়" and the model wrote it "Amit Roy" -- the SAME name in another script
    # (ordinary for a model that romanises). Literal matching alone called that "invented" and made the caller repeat their name.
    # Accept it only when some run of the caller's own words, as long as the name, SOUNDS the same (the consonant-skeleton check the
    # doctor matcher uses). A name the caller never said has no such run, so it is still not stored silently.
    said = [w for w in re.split(r"[\s,।.!?]+", unicodedata.normalize("NFC", text or "")) if w]
    n = len(name.split())
    if not n or n > len(said):
        return False
    return any(phonetic_match.phonetic_match(name, " ".join(said[i : i + n])) for i in range(len(said) - n + 1))


# ----------------------------------------------------------------------------------------------------- offered-slot pick

_ORDINALS = (
    (0, ("first", "1st", "pehla", "pehle", "pahla", "pahle", "pratham", "prothom", "প্রথম", "পহেলা", "पहला", "पहले", "पहली", "एक नंबर")),
    (1, ("second", "2nd", "doosra", "dusra", "dusri", "dosra", "dwitiyo", "ditiyo", "দ্বিতীয়", "দ্বিতীয়টা", "দুই নম্বর", "दूसरा", "दूसरे", "दूसरी")),
    (2, ("third", "3rd", "teesra", "tisra", "tritiyo", "তৃতীয়", "তৃতীয়টা", "तीसरा", "तीसरे", "तीसरी")),
    (-1, ("last", "akhri", "aakhri", "shesh", "sesh", "শেষ", "শেষেরটা", "आखिरी", "आख़िरी", "अंतिम")),
)  # fmt: skip


def pick_offered_slot(text: str, offered: list) -> object | None:
    """"The first one" / "second one" / "last one" (and the Hindi and Bengali words) -> that entry of `offered`, the list the
    clinic returned after a slot conflict. None when no ordinal was said, or it points past the end of what was offered."""
    if not offered:
        return None
    tokens = [t.lower() for t in re.split(r"[\s,।.!?]+", unicodedata.normalize("NFC", text or "")) if t]
    for index, words in _ORDINALS:
        if any(w in tokens for w in words if " " not in w) or any(w in (text or "").lower() for w in words if " " in w):
            try:
                return offered[index]
            except IndexError:
                return None
    return None


# ------------------------------------------------------------------------------------------------------- time and day
# sourav, 2026-10-16 -- WHY THIS SECTION EXISTS. Real calls (Hindi + two Bengali, 2026-10-14/15):
#   * "দশটার সময়" / "10 ta" reached the model as garbled text ("Doster", "ten ten") and, when it did parse, a bare "ten" came back as
#     22:00 -- the clinic sits in the daytime, so "ten o'clock" can only mean 10 in the morning. The caller was then told "that time is
#     not right" with no hint of the hours;
#   * a bare "कल" ("tomorrow") after the doctor's sitting time was read as a new doctor-availability question and the booking lost
#     its place.
# FLOW: read_when() reads the hour / minute / day from the caller's OWN words in Bengali, Hindi, English (script or Latin letters).
# A bare hour is resolved against the doctor's chamber window when we have it (else the clinic's 8-20 day): 10 -> 10:00 never 22:00.
# An hour garbled by the recogniser is matched by sound ONLY against the hour words and ONLY if exactly one hour fits the window.
# Nothing here decides availability -- the clinic API still validates every slot; this only turns words into "HH:MM".

_HOUR_WORDS: dict[str, int] = {}
for _h, _words in {
    1: "one ek ekta এক একটা ওয়ান एक वन",
    2: "two dui do duto দুই দু দুটো টু दो टू",
    3: "three tin teen তিন থ্রি तीन थ्री",
    4: "four char chaar চার ফোর चार फोर",
    5: "five panch paanch পাঁচ পাচ ফাইভ पाँच पांच फाइव",
    6: "six chhoy choy chhe che chhah ছয় ছ সিক্স छह छः छे सिक्स",
    7: "seven saat sat shat সাত সেভেন सात सेवन",
    8: "eight aat aath আট এইট आठ एट",
    9: "nine noy nau নয় ন নাইন नौ नाइन",
    10: "ten dosh dos das দশ টেন दस टेन",
    11: "eleven egaro gyarah gyaarah এগারো এগার ইলেভেন এলেভেন ग्यारह इलेवन",
    12: "twelve baro barah bara বারো বার টুয়েলভ बारह ट्वेल्व",
}.items():
    for _w in _words.split():
        _HOUR_WORDS.setdefault(_w, _h)

_WEAK_HOUR = frozenset({"do", "ek", "sat", "bara", "che", "one", "two", "das", "dos"})  # "Das" is also a very common surname  # also ordinary words: need a marker (বাজে/baje/o'clock)
_GLUED = ("টার", "টায়", "টাই", "টো", "টে", "টা", "tar", "tay", "ta", "te")  # "দশটার" = দশ + টার, "ten ta" glued
_MARKERS = frozenset(
    {"টা", "টার", "টায়", "টে", "টো", "ta", "tar", "tay", "baje", "bajey", "বাজে", "बजे", "oclock", "o'clock", "o’clock", "বাজতে", "bajte"}
)
_HALF = frozenset({"সাড়ে", "সাড়ে", "sade", "sare", "साढ़े", "साढे", "साड़े", "half"})
_QUARTER_PAST = frozenset({"সওয়া", "সোয়া", "sawa", "सवा"})
_QUARTER_TO = frozenset({"পৌনে", "poune", "pauna", "पौने"})
_MINUTE_WORD = {"thirty": 30, "fifteen": 15, "forty-five": 45, "fortyfive": 45}
_AM = frozenset({"am", "a.m", "morning", "সকাল", "সকালে", "sokal", "shokal", "सुबह", "subah"})
_PM = frozenset({"pm", "p.m", "afternoon", "evening", "night", "বিকেল", "বিকাল", "বিকেলে", "সন্ধ্যা", "সন্ধ্যে", "সন্ধে", "রাত", "রাতে", "bikel", "sondhe", "raat", "rat", "शाम", "रात", "sham"})  # fmt: skip
_NOONISH = frozenset({"দুপুর", "দুপুরে", "dupur", "दोपहर", "dopahar", "noon"})  # "দুপুর ১১টা" is late morning; "দুপুর ২টা" is 2 pm
_TOMORROW = frozenset({"tomorrow", "kal", "kaal", "কাল", "কালকে", "আগামীকাল", "আগামিকাল", "agamikal", "कल", "कलके"})
_DAY_AFTER = frozenset({"parso", "parson", "পরশু", "পরশুদিন", "porshu", "परसों", "परसो"})
_TODAY = frozenset({"today", "aaj", "aj", "আজ", "আজকে", "आज"})
_FILLER = frozenset(
    {"at", "for", "around", "about", "by", "the", "time", "slot", "please", "okay", "ok", "yes", "haan", "han", "ji", "accha", "achha",
     "theek", "thik", "to", "me", "mein", "में", "ko", "को", "ke", "ka", "की", "के", "का", "liye", "लिए", "jonno", "জন্য", "সময়", "সময়ে",
     "সময়ের", "somoy", "shomoy", "samay", "समय", "আমি", "আমার", "চাই", "chai", "चाहिए", "chahiye", "চাইছি", "দিন", "দয়া", "করে", "জি",
     "জী", "जी", "हाँ", "হ্যাঁ", "হাঁ", "আচ্ছা", "ঠিক", "तो", "তো", "এ", "তে", "e", "i", "want", "need", "would", "like", "can", "you",
     "book", "appointment", "बुक", "করুন", "কর", "কইরা", "kar", "do", "dijiye", "দিন", "দিয়ে", "দেবেন", "देना", "दीजिए", "uh", "um", "hmm",
     "and", "আর", "और", "aur", "also", "sorry", "wait"}
)  # fmt: skip


def _words(text: str) -> list[str]:
    """Lower-cased tokens; Bengali/Devanagari numerals already ASCII (spoken_codes._norm); punctuation and the danda dropped."""
    norm = spoken_codes._norm(text).replace("।", " ").replace("॥", " ")
    out = []
    for tok in re.findall(r"[^\s,;!?]+", norm):
        tok = tok.strip(".'\"()")
        if tok:
            out.append(tok)
    return out


def _hour_of(tok: str) -> tuple[int, int | None, bool] | None:
    """(hour, minute or None, glued marker) if `tok` is an hour: "10", "10:30", "10টা", "দশটার", "ten"."""
    m = re.fullmatch(r"(\d{1,2})(?:[:.](\d{2}))?(am|pm|a\.m|p\.m|টা|টার|টায়|টে|টো|ta|tar|tay|baje|বাজে|बजे)?", tok)
    if m:
        hour = int(m.group(1))
        if 0 <= hour <= 24 and (m.group(2) is None or int(m.group(2)) < 60):
            return hour, (int(m.group(2)) if m.group(2) else None), bool(m.group(2) or m.group(3))
        return None
    if tok in _HOUR_WORDS:
        return _HOUR_WORDS[tok], None, False
    for suffix in _GLUED:
        if tok.endswith(suffix) and tok[: -len(suffix)] in _HOUR_WORDS:
            return _HOUR_WORDS[tok[: -len(suffix)]], None, True
    return None


def parse_window(chamber_hours: object) -> list[tuple[int, int]] | None:
    """"10:00-12:00" (or "10:00-12:00, 16:00-18:00") -> [(600, 720), ...] minutes; None if there is nothing usable."""
    if not isinstance(chamber_hours, str):
        return None
    spans = []
    for a, b, c, d in re.findall(r"(\d{1,2}):(\d{2})\s*[-–to]+\s*(\d{1,2}):(\d{2})", chamber_hours):
        lo, hi = int(a) * 60 + int(b), int(c) * 60 + int(d)
        if lo < hi:
            spans.append((lo, hi))
    return spans or None


def window_from_slots(valid_slots: object, step: int = 15) -> list[tuple[int, int]] | None:
    """The same window from the clinic's own `valid_slots` list (["10:00", "10:15", ...])."""
    try:
        mins = sorted(int(s[:2]) * 60 + int(s[3:5]) for s in valid_slots)  # type: ignore[union-attr]
    except (TypeError, ValueError):
        return None
    if not mins:
        return None
    if len(mins) > 1:
        step = min((b - a for a, b in zip(mins, mins[1:]) if b > a), default=step)
    return [(mins[0], mins[-1] + step)]


_CLINIC_DAY = [(8 * 60, 20 * 60)]  # the clinic's own opening hours -- only used when the doctor's chamber is not known yet


def _inside(minute_of_day: int, window: list[tuple[int, int]] | None) -> bool:
    return any(lo <= minute_of_day < hi for lo, hi in (window or _CLINIC_DAY))


def _clock(hour: int, minute: int, period: str | None, window: list[tuple[int, int]] | None) -> str | None:
    """12-hour (or 24-hour) reading -> "HH:MM". `period`: "am" | "pm" | "noonish" | None. With no period said, the reading that falls
    inside the chamber window (else the clinic's day) wins: "ten" -> 10:00, never 22:00. Nothing fits -> the daytime reading, which the
    clinic then rejects with the real hours (never silently moved to another hour)."""
    if hour > 24 or hour < 0 or minute > 59:
        return None
    if hour >= 13 or hour == 0:
        return f"{hour % 24:02d}:{minute:02d}"
    if period == "am":
        pick = hour % 12
    elif period == "pm":
        pick = hour % 12 + 12
    elif period == "noonish":
        pick = hour if hour in (10, 11, 12) else hour % 12 + 12
    else:
        fits = [h for h in sorted({hour % 12, hour % 12 + 12}) if _inside(h * 60 + minute, window)]
        if len(fits) == 1:
            pick = fits[0]
        elif len(fits) > 1:  # both readings fit a wide window: the daytime one
            pick = hour if 8 <= hour <= 12 else hour % 12 + 12
        else:  # neither fits: keep the daytime reading; the clinic answers with its real hours
            pick = hour if hour >= 8 else hour + 12
    return f"{pick:02d}:{minute:02d}"


@dataclass
class When:
    time: str | None = None
    date: str | None = None
    pure: bool = False  # every word was part of the time / day / harmless filler: the model has nothing to add
    fuzzy: bool = False


def _period_of(words: list[str]) -> tuple[str | None, set[int]]:
    period, used = None, set()
    for i, w in enumerate(words):
        if w in _AM:
            period = "am"
        elif w in _PM:
            period = "pm"
        elif w in _NOONISH:
            period = period or "noonish"
        else:
            continue
        used.add(i)
    return period, used


def day_from_words(words: list[str], today) -> tuple[str | None, set[int]]:
    """"tomorrow" / "কাল" / "कल" (also "day after tomorrow", "today") -> ISO date relative to `today`. In a booking, a bare
    কাল / कल always means tomorrow (the words also mean 'yesterday', but nobody books yesterday)."""
    import datetime

    for i, w in enumerate(words):
        if w in _DAY_AFTER:
            return (today + datetime.timedelta(days=2)).isoformat(), {i}
        if w in _TOMORROW:
            if i >= 2 and words[i - 2 : i] == ["day", "after"]:
                return (today + datetime.timedelta(days=2)).isoformat(), {i, i - 1, i - 2}
            return (today + datetime.timedelta(days=1)).isoformat(), {i}
        if w in _TODAY:
            return today.isoformat(), {i}
    return None, set()


def _fuzzy_hours(words: list[str], used: set[int]) -> tuple[set[int], set[int]]:
    """Hours whose sound matches an unrecognised word ("Doster" ~ দশটার = ten). Sound classes only (phonetic_match), the suffix টা/টার
    allowed after the hour word. Several hours can share a sound (চার and এগারো), so the caller keeps it only if one hour fits."""
    found: set[int] = set()
    matched: set[int] = set()
    for i, w in enumerate(words):
        if i in used or len(w) < 2 or w.isdigit() or w in _FILLER or w in _HOUR_WORDS:
            continue
        key = phonetic_match.phonetic_key(w)
        if not key:
            continue
        for word, hour in _HOUR_WORDS.items():
            base = phonetic_match.phonetic_key(word)
            if base and len(base) >= 2 and key.startswith(base) and key[len(base) :] in ("", "3", "6", "36", "33", "63"):
                found.add(hour)
                matched.add(i)
    return found, matched


def read_when(text: str, asked_time: bool, window=None, today=None, want_day: bool = True) -> When:
    """The time (and day) in the caller's words. A time is taken when it is MARKED (বাজে/টা/baje/o'clock/am/pm/a period word/a minute) or
    when the time is exactly what was just asked and there is a single candidate -- so '3 days' or a phone digit is never a time."""
    import datetime

    words = _words(text)
    out = When()
    if not words:
        return out
    used: set[int] = set()
    if want_day:
        out.date, day_used = day_from_words(words, today or datetime.date.today())
        used |= day_used
    period, period_used = _period_of(words)
    cands = []
    for i, w in enumerate(words):
        if i in used or (w.isdigit() and len(w) > 2):
            continue
        got = _hour_of(w)
        if got:
            cands.append((i, *got))
    marked_near = lambda i: (i + 1 < len(words) and words[i + 1] in _MARKERS) or (i > 0 and words[i - 1] in _MARKERS | {"at", "by", "around"})  # noqa: E731
    chosen = None
    marked = [c for c in cands if c[3] or marked_near(c[0]) or c[2] is not None]
    if marked:
        chosen = marked[-1]  # "ten ta, no eleven ta": the last one said is the correction
    elif period is not None and len(cands) == 1:
        chosen = cands[0]
    elif (
        asked_time
        and len(cands) == 1
        and words[cands[0][0]] not in _WEAK_HOUR
        # an UNMARKED number counts only as the whole answer ("ten", "at 11"); "3 days of fever" is not a time
        and all(w in _FILLER or w in _MARKERS or w in _MINUTE_WORD or w in _HALF or w in _QUARTER_PAST or w in _QUARTER_TO or j in period_used or j in used or j == cands[0][0] for j, w in enumerate(words))
    ):
        chosen = cands[0]
    if chosen:
        i, hour, minute, _glued = chosen
        used.add(i)
        if i + 1 < len(words) and words[i + 1] in _MARKERS:
            used.add(i + 1)
        if i > 0 and words[i - 1] in _MARKERS:
            used.add(i - 1)
        before = words[i - 1] if i > 0 else ""
        after = words[i + 1] if i + 1 < len(words) else ""
        after2 = words[i + 2] if i + 2 < len(words) else ""
        if minute is None:
            if before in _HALF:
                minute, _ = 30, used.add(i - 1)
            elif before in _QUARTER_PAST:
                minute, _ = 15, used.add(i - 1)
            elif before in _QUARTER_TO:
                minute, hour, _ = 45, (hour - 1 if hour > 1 else 12), used.add(i - 1)
            elif after in _MINUTE_WORD:
                minute, _ = _MINUTE_WORD[after], used.add(i + 1)
            elif after in ("forty",) and after2 == "five":
                minute, _ = 45, used.update({i + 1, i + 2})
            elif after in _MARKERS and after2 in _MINUTE_WORD:
                minute, _ = _MINUTE_WORD[after2], used.add(i + 2)
        out.time = _clock(hour, minute or 0, period, window)
        used |= period_used
    elif asked_time:
        # not a word we know: maybe the recogniser's spelling of an hour ("Doster"). Accept only a single hour that fits the window.
        hours, matched = _fuzzy_hours(words, used)
        # only a SHORT answer made of the garbled hour plus the "ta / uta" tail ("Doster", "Agar uta"); a real sentence is never guessed at
        rest_is_tail = all(
            i in matched or words[i] in _FILLER or words[i] in _MARKERS or phonetic_match.phonetic_key(words[i]) in ("3", "6", "36", "33")
            for i in range(len(words))
            if i not in used
        )
        if hours and len(words) <= 3 and rest_is_tail:
            fits = {h: _clock(h, 0, period, window) for h in hours}
            fits = {h: t for h, t in fits.items() if t and _inside(int(t[:2]) * 60, window)}
            if len(fits) == 1:
                out.time, out.fuzzy = next(iter(fits.values())), True
                used = set(range(len(words)))
    # a period word ("subah", "evening") with no hour beside it is NOT understood by us: the model may know more, so the turn is not 'pure'
    leftover = [
        w for i, w in enumerate(words) if i not in used and (i not in period_used or not out.time) and w not in _FILLER and w not in _MARKERS
    ]
    out.pure = bool(out.time or out.date) and not leftover
    return out


def fix_model_time(value: object, text: str, window=None) -> object:
    """The model's own "HH:MM" -> the daytime reading when the caller named no period. 'ten' came back as 22:00 in a real call."""
    if not isinstance(value, str) or not re.fullmatch(r"\d{1,2}:\d{2}", value):
        return value
    hour, minute = (int(x) for x in value.split(":"))
    period, _ = _period_of(_words(text))
    if period is not None or hour < 13 and hour != 0 and _inside(hour * 60 + minute, window):
        return value
    flipped = (hour + 12) % 24 if hour < 12 else hour - 12
    if period is None and not _inside(hour * 60 + minute, window) and _inside(flipped * 60 + minute, window):
        return f"{flipped:02d}:{minute:02d}"
    return value


def hours_span(valid_slots: object) -> tuple[str, str] | None:
    """("10:00 AM", "12:00 PM") from the clinic's own slot list, so the caller hears the real chamber hours. None if unknown."""
    win = window_from_slots(valid_slots)
    if not win:
        return None

    def one(m: int) -> str:
        h, mi = divmod(m, 60)
        return f"{(h % 12) or 12}:{mi:02d} {'AM' if h < 12 else 'PM'}"

    return one(win[0][0]), one(win[0][1])


# ----------------------------------------------------------------------------------------------------------------- wording

_PROMPTS = {
    # Owner, 2026-10-18: a number said in groups ("two digits, five times", "four, three, three") is answered with a short "ok" in each gap, not with a count.
    "phone_partial": {
        "en": "Okay, go on.",
        "hi": "ठीक है, बोलिए।",
        "bn": "ঠিক আছে, বলুন।",
    },
    "phone_invalid": {
        "en": "That does not look like a 10-digit mobile number. Please say it again, digit by digit.",
        "hi": "यह 10 अंकों का मोबाइल नंबर नहीं लग रहा। कृपया एक-एक अंक करके दोबारा बताइए।",
        "bn": "এটা ১০ সংখ্যার মোবাইল নম্বর মনে হচ্ছে না। অনুগ্রহ করে এক এক করে আবার বলুন।",
    },
    "phone_bad_start": {
        "en": "A mobile number starts with 6, 7, 8 or 9, but I heard it start with {n}. Please say the number again, digit by digit.",
        "hi": "मोबाइल नंबर 6, 7, 8 या 9 से शुरू होता है, लेकिन मैंने {n} से शुरू सुना। कृपया एक-एक अंक करके दोबारा बताइए।",
        "bn": "মোবাইল নম্বর ৬, ৭, ৮ বা ৯ দিয়ে শুরু হয়, কিন্তু আমি {n} দিয়ে শুরু শুনেছি। অনুগ্রহ করে এক এক করে আবার বলুন।",
    },
    "phone_wrong_length": {
        "en": "I heard {n} digits, but a mobile number has 10. Please say it again, digit by digit.",
        "hi": "मुझे {n} अंक सुनाई दिए, लेकिन मोबाइल नंबर में 10 अंक होते हैं। कृपया एक-एक अंक करके दोबारा बताइए।",
        "bn": "আমি {n}টি সংখ্যা শুনেছি, কিন্তু মোবাইল নম্বরে ১০টি সংখ্যা থাকে। অনুগ্রহ করে এক এক করে আবার বলুন।",
    },
    "phone_digit_by_digit": {
        "en": "Please say the mobile number one digit at a time, for example nine, eight, seven.",
        "hi": "कृपया मोबाइल नंबर एक-एक अंक करके बताइए, जैसे नौ, आठ, सात।",
        "bn": "অনুগ্রহ করে মোবাইল নম্বরটা এক এক করে বলুন, যেমন নয়, আট, সাত।",
    },
    "phone_cleared": {
        "en": "Okay, let us take the number again from the start. Please say it digit by digit.",
        "hi": "ठीक है, नंबर शुरू से लेते हैं। कृपया एक-एक अंक करके बताइए।",
        "bn": "ঠিক আছে, নম্বরটা আবার প্রথম থেকে নিই। অনুগ্রহ করে এক এক করে বলুন।",
    },
    "name_confirm": {
        "en": "I heard the name {name}. Is that correct?",
        "hi": "मैंने नाम {name} सुना। क्या यह सही है?",
        "bn": "আমি নামটা শুনেছি {name}। এটা কি ঠিক?",
    },
    "phone_confirm": {
        "en": "I heard the number {digits}. Is that correct?",
        "hi": "मैंने नंबर {digits} सुना। क्या यह सही है?",
        "bn": "আমি নম্বরটা শুনেছি {digits}। এটা কি ঠিক?",
    },
    "name_again": {
        "en": "Sorry about that. Please tell me the patient's name again.",
        "hi": "माफ़ कीजिए। कृपया मरीज़ का नाम दोबारा बताइए।",
        "bn": "দুঃখিত। অনুগ্রহ করে রোগীর নামটা আবার বলুন।",
    },
    "which_slot": {
        "en": "Which one would you like - the first or the second?",
        "hi": "आपको कौन सा चाहिए - पहला या दूसरा?",
        "bn": "কোনটা চান - প্রথমটা না দ্বিতীয়টা?",
    },
    "confirm_unsure": {
        "en": "Sorry, was that a yes or a no?",
        "hi": "माफ़ कीजिए, यह हाँ था या ना?",
        "bn": "দুঃখিত, এটা হ্যাঁ না না?",
    },
    "confirm_unverified": {
        "en": "I could not confirm whether the booking went through. I am checking, please hold on.",
        "hi": "मैं पक्का नहीं कर पाई कि बुकिंग हुई या नहीं। मैं जाँच रही हूँ, कृपया रुकिए।",
        "bn": "বুকিংটা হয়েছে কিনা নিশ্চিত হতে পারিনি। আমি দেখছি, একটু অপেক্ষা করুন।",
    },
    "confirm_retry": {
        "en": "The booking did not go through, so nothing was booked. Shall I try to confirm it again?",
        "hi": "बुकिंग नहीं हो पाई, इसलिए कुछ बुक नहीं हुआ। क्या मैं दोबारा कन्फ़र्म करने की कोशिश करूँ?",
        "bn": "বুকিং হয়নি, তাই কিছুই বুক হয়নি। আমি কি আবার নিশ্চিত করার চেষ্টা করব?",
    },
}  # fmt: skip


def phone_invalid_prompt(step: "PhoneStep", lang: str) -> str:
    """The right sentence for the rule that failed (wrong first digit / wrong number of digits)."""
    if step.reason == "start":
        return prompt("phone_bad_start", lang, n=step.n)
    if step.reason == "length":
        return prompt("phone_wrong_length", lang, n=step.n)
    return prompt("phone_invalid", lang)


_HUMS = frozenset({"hmm", "hm", "hmmm", "mhm", "mm", "হুম", "হুঁ", "হু", "হম", "हम्म", "हूँ", "हूं", "हम"})


def confirm_answer(text: str, lang: str) -> str | None:
    """yes / no / None for a name or number READ BACK to the caller. A hum (\"hmm\", \"হুম\") is a yes here -- the owner's rule -- but never for the
    final booking write, which keeps classify_yes_no's stricter reading."""
    from agent.booking_flow import classify_yes_no

    answer = classify_yes_no(text, lang)
    if answer is None and any(w in _HUMS for w in _words(text)):
        return "yes"
    return answer


def prompt(key: str, lang: str, **fmt: object) -> str:
    table = _PROMPTS[key]
    return table.get(lang, table["en"]).format(**fmt)


# ------------------------------------------------------------------------------------------------- per-turn orchestration
# sourav, 2026-10-16: main_pcm.py and main.py (two near-identical entry points) both call these three functions, so the
# rules live ONCE here and the entry points only speak what they return.

_ACTIVE_STAGES = ("collecting", "confirming")


@dataclass
class TurnPlan:
    speak: str | None = None  # say this and end the turn (nothing else runs)
    slots: dict | None = None  # deterministic slots; with skip_model they ARE the turn's slots
    skip_model: bool = False  # the sentence was fully understood without the model (pure digits / a yes to the name / an ordinal)
    phone: str | None = None  # a complete phone number heard this turn (applied over whatever the model returns)
    time: str | None = None  # "HH:MM" read from the caller's words (applied over the model's own reading of the time)
    date: str | None = None  # ISO date read from "tomorrow" / কাল / कल when the booking had no date yet
    digits: bool = False  # phone digits (complete or not) were heard: the sentence belongs to the booking, never to a lookup
    after_speak: str | None = None  # asked after this turn's slots are merged, unless another question is due first
    compound: bool = False  # the number was said as a number ("four hundred ..."): ask for it digit by digit
    age: int | None = None  # an age said in this sentence ("বয়স ৪৫", "45 years"), taken out BEFORE the digits are read as the phone number


def _active(st) -> bool:
    return (
        st is not None
        and st.action == "book_appointment"
        and st.stage in _ACTIVE_STAGES
        and not st.is_stale()
    )


def _bare_no(text: str) -> str:
    """`text` without a leading no-word ("no, Rahul Das" -> "Rahul Das")."""
    return re.sub(r"^\s*(no|nope|nahi|nahin|na|না|নয়|नहीं|नही)\b[\s,।.!]*", "", text or "", flags=re.I)


_PHONE_CUES = frozenset({"number", "phone", "mobile", "contact", "नंबर", "नम्बर", "फोन", "फ़ोन", "मोबाइल", "নম্বর", "নাম্বার", "ফোন", "মোবাইল", "nombor", "nambar", "namber", "no."})  # fmt: skip


# An age said with its word: "বয়স ৪৫", "আমার বয়স 45", "age 45", "45 years old", "৪৫ বছর", "उम्र 45", "45 साल". Only with the word -- a bare number is a digit of the phone.
_DIGITS_ANY = str.maketrans("০১২৩৪৫৬৭৮৯०१२३४५६७८९", "01234567890123456789")
_AGE_BEFORE = re.compile(r"(?:বয়স|বয়েস|age|aged|उम्र|उमर|आयु)\s*(?:হলো|হল|হয়েছে|है|is|of)?\s*(\d{1,3})(?!\d)", re.I)
_AGE_AFTER = re.compile(r"(?<!\d)(\d{1,3})\s*(?:বছর|বছরের|years?\s*old|years?|yrs?|साल|वर्ष)", re.I)


def split_age(text: str) -> tuple[int | None, str]:
    """(age, the sentence without the age phrase). Numbers written as words ("পঁয়তাল্লিশ বছর") are left to the model's patient_age slot."""
    t = (text or "").translate(_DIGITS_ANY)
    for pat in (_AGE_BEFORE, _AGE_AFTER):
        m = pat.search(t)
        if m and 0 < int(m.group(1)) <= 120:
            return int(m.group(1)), (t[: m.start()] + " " + t[m.end():]).strip()
    return None, text


def before_model(st, text: str, lang: str, window=None, today=None) -> TurnPlan:
    """The booking turn's deterministic reading (see _before_model), plus: an age said in the sentence is taken out first (so it is never read as a
    phone digit) and kept; a name spelled \"S for September, A for Alpha\" (or letter by letter) is read back for a yes."""
    text = nato_letters.collapse(text)
    age, text = split_age(text) if _active(st) else (None, text)
    plan = _before_model(st, text, lang, window=window, today=today)
    if age:
        plan.age = age
        if plan.skip_model:
            plan.slots = {**(plan.slots or {}), "patient_age": age}
    if _active(st) and not plan.speak and not plan.skip_model and not st.pending_name and not st.slots.get("_phone_pending"):
        from agent.booking_flow import missing_required

        missing = missing_required(st)
        asked_name = "patient_name" in missing and (missing[0] == "patient_name" or "patient_name" in st.asked_fields)
        spelled = nato_letters.spelled_run(text) if asked_name else None
        if spelled and not plan.digits:
            name = spelled.title()
            st.pending_name = name
            plan.speak = prompt("name_confirm", lang, name=name)
    return plan


def _before_model(st, text: str, lang: str, window=None, today=None) -> TurnPlan:
    """Everything the caller's words settle by themselves, decided before the model is asked. `window` = the doctor's chamber
    hours on the booking date as [(start_min, end_min)] when known (to read a bare "ten" as the daytime hour)."""
    from agent.booking_flow import classify_yes_no, missing_required  # local import: booking_flow does not import this module

    plan = TurnPlan()
    if not _active(st):
        return plan

    # 1. A name read back and waiting for the caller's yes/no.
    if st.pending_name:
        answer = confirm_answer(text, lang)
        repeated = candidate_name(text) if answer is None and read_digits(text)[1] >= 1 else None
        if repeated:
            # Real call 2026-10-17: to "I heard the name আবু আবু আলী. Is that right?" the caller said "abu ali" and their number. That is an answer: the name
            # as heard now is the name, the number is taken, and nothing but the confirmation is asked.
            st.pending_name, st.name_confirmed = None, repeated
            plan.slots, plan.skip_model = {"patient_name": repeated}, True
        elif answer == "yes":
            name, st.pending_name = st.pending_name, None
            st.name_confirmed = name
            plan.slots, plan.skip_model = {"patient_name": name}, True
            return plan
        if repeated:
            pass
        elif answer == "no":
            st.pending_name = None
            fresh = candidate_name(_bare_no(text))
            if fresh:
                st.pending_name = fresh
                plan.speak = prompt("name_confirm", lang, name=fresh)
            else:
                plan.speak = prompt("name_again", lang)
            return plan
        fresh = None if repeated else candidate_name(text)  # neither yes nor no: if it is a name, it is the corrected name
        st.pending_name = None
        if fresh:
            st.pending_name = fresh
            plan.speak = prompt("name_confirm", lang, name=fresh)
            return plan

    # 1b. A phone number read back and waiting for the caller's yes/no (owner, 2026-10-17: the number is confirmed like the name).
    pending_phone = st.slots.get("_phone_pending")
    if pending_phone:
        answer = confirm_answer(text, lang)
        said_digits = read_digits(text)[1] >= 7  # a whole new number in this sentence
        if answer == "yes" and not said_digits:
            st.slots.pop("_phone_pending", None)
            st.slots["_phone_ok"] = pending_phone
            plan.slots, plan.skip_model = {}, True
            return plan
        st.slots.pop("_phone_pending", None)
        if answer == "no" or said_digits:
            for key in ("phone", "contact_phone", "_phone_ok", "_conflict_checked"):
                st.slots.pop(key, None)
            st.phone_buffer = ""
            if not said_digits:
                plan.speak = prompt("phone_cleared", lang)
                return plan
        else:  # neither: ask the same question again, once more
            st.slots["_phone_pending"] = pending_phone
            plan.speak = prompt("phone_confirm", lang, digits=" ".join(pending_phone))
            return plan

    # 2. "the first one / the second one" after the clinic offered other slots.
    if st.offered_options:
        chosen = pick_offered_slot(text, st.offered_options)
        if chosen is not None:
            st.offered_options = []
            plan.slots = {k: v for k, v in (chosen if isinstance(chosen, dict) else {"time_slot": chosen}).items() if v}
            plan.skip_model = True
            return plan

    # 3. The phone number.
    missing = missing_required(st)
    asked = "phone" in missing and (missing[0] == "phone" or "phone" in st.asked_fields)
    if "phone" in missing and not asked and any(t in _PHONE_CUES for t in _words(text)):
        asked = True  # "the number is ..." -- the caller is plainly giving the phone number, asked or not
    step = absorb(st.phone_buffer, text, asked)
    if step.status == "complete":
        st.phone_buffer = ""
        plan.phone = step.number
        plan.digits = True
        if step.pure:
            plan.slots, plan.skip_model = {"phone": step.number}, True
    elif step.status == "partial":
        st.phone_buffer = step.buffer
        plan.digits = True
        spoken = prompt("phone_partial", lang, n=len(step.buffer))
        if step.pure:
            plan.speak = spoken
        else:
            plan.after_speak = spoken  # a sentence with a name AND some digits: the name is kept, then the rest of the number is asked
    elif step.status == "invalid":
        st.phone_buffer = ""
        plan.digits = True
        spoken = phone_invalid_prompt(step, lang)
        if step.pure:
            plan.speak = spoken
        else:
            plan.after_speak = spoken
    elif step.status == "compound":
        plan.digits, plan.compound = True, True
        spoken = prompt("phone_digit_by_digit", lang)
        if step.pure:
            plan.speak = spoken
        else:
            plan.after_speak = spoken
    elif step.status == "cleared":
        st.phone_buffer = ""
        plan.speak = prompt("phone_cleared", lang)
    if plan.speak:
        return plan
    if plan.skip_model and plan.phone and isinstance(plan.slots, dict) and "phone" not in plan.slots:
        plan.slots = {**plan.slots, "phone": plan.phone}  # the name was settled by this sentence and the number came with it

    # 4. The time and the day: "দশটার সময়", "10 ta", "दस बजे", "tomorrow at ten", a bare "कल" (see read_when).
    time_pending = st.slots.get("_time_ok") == "pending"  # a time was just proposed: the caller may answer with another one, even beside the number
    if not (plan.digits and not plan.phone) or time_pending:  # a sentence of phone digits is not also a time
        asked_time = "time_slot" in missing and (missing[0] == "time_slot" or "time_slot" in st.asked_fields)
        asked_day = "date" in missing
        if asked_time or asked_day or plan.phone is None:
            when = read_when(text, asked_time, window=window, today=today, want_day=not st.slots.get("date"))
            if when.time and not st.slots.get("time_slot") or when.time and asked_time:
                plan.time = when.time
            elif when.time and st.slots.get("time_slot") and when.time != st.slots.get("time_slot"):
                plan.time = when.time  # a time said with a marker while one is already held: the caller is changing it
            if plan.time and plan.digits and not plan.phone:
                # digits are also in this sentence: a "time" read from them (পাঁচটা = 5) must fall inside the doctor's sitting hours, else it is part of the number
                hh, mm = (int(x) for x in plan.time.split(":"))
                if not window or not any(a <= hh * 60 + mm < b for a, b in window):
                    plan.time = None
            if when.date and not st.slots.get("date"):
                plan.date = when.date
            if (plan.time or plan.date) and when.pure and not plan.digits:
                found = {k: v for k, v in (("time_slot", plan.time), ("date", plan.date)) if v}
                if asked_time or asked_day or len(found) == 2:
                    plan.slots, plan.skip_model = found, True
    return plan


def after_model(st, text: str, lang: str, slots: dict, plan: TurnPlan, window=None) -> tuple[dict, str | None]:
    """Checks on what the model returned. Returns (slots, speak): a non-empty `speak` ends the turn with that question."""
    slots, speak = _after_model(st, text, lang, slots, plan, window)
    return slots, speak or (plan.after_speak if _active(st) else None)


def _after_model(st, text: str, lang: str, slots: dict, plan: TurnPlan, window=None) -> tuple[dict, str | None]:
    from agent.booking_flow import missing_required

    slots = dict(slots or {})
    if not _active(st):
        return slots, None
    # The caller's own words beat the model for the time and the day. 'ten' came back as 22:00 -- the clinic sits in the daytime.
    if plan.time:
        slots["time_slot"] = plan.time
    elif slots.get("time_slot"):
        slots["time_slot"] = fix_model_time(slots["time_slot"], text, window)
    if plan.date:
        slots["date"] = plan.date
    if plan.age:
        slots["patient_age"] = plan.age  # the caller's own digits beat the model's reading of them
    if plan.compound:  # said as a number: whatever digits the model made of it are not the caller's number
        slots.pop("phone", None)
        slots.pop("contact_phone", None)
    elif plan.phone:  # heard digit by digit: the caller's own digits beat the model's reading of them
        slots["phone"] = plan.phone
    else:
        for key in ("phone", "contact_phone"):
            if slots.get(key):
                valid = normalise_phone(slots[key])
                if valid:
                    slots[key] = valid
                else:  # the model produced a "number" that is not a 10-digit mobile number: it never reaches the booking
                    slots.pop(key, None)
                    return slots, prompt("phone_invalid", lang) if not plan.digits else None
    name = slots.get("patient_name")
    if isinstance(name, str) and name.strip():
        if st.name_confirmed == name or name_is_grounded(name, text):
            return slots, None
        # The model's name is not in the caller's words (invented, or translated from another script). It is NEVER the patient. What the
        # caller actually said, read as a name, is read back for a yes; if the sentence holds no name at all, the name is asked again.
        slots.pop("patient_name", None)
        st.stage = "collecting"
        # final verification, 2026-10-16: only when the name is what is being asked right now. A sentence that was about something else
        # ("that is all", "all of it") must not be turned into "I heard the name All Of It" -- the invented name is dropped and the
        # normal flow asks the next missing field (the name is asked when its turn comes).
        due = missing_required(st)
        if due and due[0] == "patient_name":
            heard = candidate_name(text)
            if heard:
                st.pending_name = heard
                return slots, prompt("name_confirm", lang, name=heard)
            return slots, prompt("name_again", lang)
        return slots, None
    missing = missing_required(st)
    if missing and missing[0] == "patient_name" and "patient_name" in st.asked_fields:
        fresh = candidate_name(text)  # the model gave no name although one was asked for: read it from the caller's words
        if fresh:
            st.pending_name = fresh
            return slots, prompt("name_confirm", lang, name=fresh)
    return slots, None


def recover_name_without_model(st, text: str, lang: str) -> str | None:
    """The model failed (timeout / malformed / empty) on a booking turn. What the caller's own words still settle is taken: a name that
    was just asked for is read back for a yes (never stored unconfirmed). Returns the question to speak, or None. The caller of this
    function then asks the SAME next booking question -- a booking turn never collapses into "a little problem"."""
    from agent.booking_flow import missing_required

    if not _active(st):
        return None
    missing = missing_required(st)
    if missing and missing[0] == "patient_name" and "patient_name" in st.asked_fields:
        fresh = candidate_name(text)
        if fresh:
            st.pending_name = fresh
            return prompt("name_confirm", lang, name=fresh)
    return None


def options_from_hold(hold: dict, slots: dict) -> list[dict]:
    """The other slots the clinic offered when it refused a hold, as {"date", "time_slot"} entries the caller can pick by ordinal. The
    clinic's own `alternative_options` (date + time) is used when present; otherwise its `alternative_slots` times on the date the caller
    asked for. Never invented here."""
    options = [o for o in (hold.get("alternative_options") or []) if isinstance(o, dict) and o.get("time_slot")]
    if options:
        return [{"date": o.get("date") or slots.get("date"), "time_slot": o["time_slot"]} for o in options]
    return [{"date": slots.get("date"), "time_slot": t} for t in (hold.get("alternative_slots") or []) if isinstance(t, str)]


_HEDGES = frozenset({"hmm", "hm", "umm", "um", "uh", "err", "maybe", "perhaps", "wait", "शायद", "रुकिए", "रुको", "হুম", "হয়তো", "দাঁড়ান"})


def is_hedge(text: str) -> bool:
    """A reply to "shall I confirm?" that is neither yes nor no ("hmm", "maybe", "not sure"): the agent asks yes-or-no again."""
    from agent.booking_flow import _expresses_uncertainty, _tokenize

    tokens = _tokenize(text or "")
    if not tokens or len(tokens) > 5:
        return False
    return _expresses_uncertainty(tokens) or all(t in _HEDGES for t in tokens)
