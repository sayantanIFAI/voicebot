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

from agent import nato_letters, number_words, phonetic_match, spoken_codes, slot_grounding, whatsapp_cues
from agent.digit_speech import spoken_digit_groups

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
    # real call 2026-10-24: "ত্রিপলে টু" (dental ত, not the retroflex ট above, plus the recogniser's own "-এ" suffix
    # on the loanword) was not read as a multiplier at all -- _tokens() only splits on whitespace/comma/hyphen, so
    # a suffixed token never matches a bare entry. Both consonant spellings, bare and with the suffix, for both words.
    "ত্রিপল": 3, "ত্রিপলে": 3, "ট্রিপলে": 3, "ডাবলে": 2, "ডবলে": 2,
}  # fmt: skip
# sourav, 2026-10-16 (real Bengali call): "Seven Eight ... Double Six" was refused as "not a 10-digit number" because the recogniser wrote
# English digit words in Bengali / Devanagari letters ("ডাবল সিক্স", "सेवन एट") and the table only knew the native words.
_EXTRA_DIGIT_WORD = {
    "জিরো": "0", "জিরও": "0", "ওয়ান": "1", "টু": "2", "থ্রি": "3", "ফোর": "4", "ফাইভ": "5", "সিক্স": "6", "সেভেন": "7", "এইট": "8",
    "নাইট": "8", "সেবেন": "7", "সেবন": "7", "সেভন": "7", "সিক্ষ": "6", "ফাইব": "5", "থ্রী": "3", "ওয়ন": "1", "ওন": "1", "নাইন": "9", "वन": "1", "टू": "2", "थ्री": "3", "फोर": "4", "फाइव": "5", "सिक्स": "6", "सेवन": "7", "एट": "8", "नाइन": "9",
    "south": "7", "sauth": "7", "ছ": "6", "শুন্য": "0", "शुन्य": "0", "नो": "9", "छे": "6", "छै": "6", "paach": "5", "saath": "7", "nine": "9", "aat": "8",
    # real call, 2026-10-27: "सेवन आइट ज़ीरो ऐट सेवन नाइन सेवन नाइन सिक्स वन" (10 digits) came back missing exactly its
    # two "eight"s -- the recogniser wrote the English word "eight" as "आइट" and, two words later in the SAME
    # sentence, as "ऐट", neither of which was "एट" (already in this table). Same miss class as "छ" for "six" --
    # both added so neither spelling drops the digit again, whichever one the recogniser picks next time.
    "आइट": "8", "ऐट": "8",
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


# English digit words written in Bengali / Devanagari letters: the recogniser often adds a letter or two ("ফাইভল", "সেভেনে", "नाइनटा"). Real call 2026-10-18: "ফাইভল" was
# not a digit, the number came out seven digits long and the caller was only told "ok".
_LOAN_DIGIT_WORDS = {w: d for w, d in _EXTRA_DIGIT_WORD.items() if not w.isascii() and len(w) >= 3 and w != "এইট"}  # "এইটা" is "this one", never an eight


def _digit_of(tok: str) -> str | None:
    tok = tok.lstrip("+")
    if tok.isdigit() and len(tok) == 1:
        return tok
    if tok in _DIGIT_WORDS:
        return _DIGIT_WORDS[tok]
    if tok in _PLURAL_DIGIT:
        return _PLURAL_DIGIT[tok]
    for word in sorted(_LOAN_DIGIT_WORDS, key=len, reverse=True):
        if tok.startswith(word) and len(tok) - len(word) <= 2:
            return _LOAN_DIGIT_WORDS[word]
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
        elif _digit_of(tok) is not None:
            digit = _digit_of(tok)  # a digit word the recogniser lengthened ("ফাইভল")
        elif number_words.digits_of(tok) is not None:
            digit = number_words.digits_of(tok)  # "আটানব্বই" = 98, "चौहत्तर" = 74: a number said as a word stands for its two digits (owner, 2026-10-26)
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


_END_CUES = ("শেষে", "শেষ", "অন্তে", "last", "end", "ending", "finally", "आखिर", "अंत", "अंतिम", "आख़िर")
_MORE_CUES = ("আরেকটা", "আরেক", "আরও", "আরো", "আর একটা", "আর এক", "এক্সট্রা", "additional", "another", "one more", "extra", "और एक", "एक और", "एक्स्ट्रा", "और")


def end_append(text: str) -> str:
    """"শেষে একটা জিরো হবে আরেকটা" ("at the end there will be one more zero"): the digit(s) to ADD to the number heard so far, or "". Needs an end word AND a
    'one more / another' word, so an ordinary digit run never reads as an addition. One to three digits."""
    t = unicodedata.normalize("NFC", text or "").lower()
    if not (any(c in t for c in _END_CUES) and any(c in t for c in _MORE_CUES)):
        return ""
    digits = read_digits(text)[0]
    return digits if 1 <= len(digits) <= 3 else ""


def partial_prompt(old: str, new: str, lang: str) -> str:
    """What to say in the pause after a group of digits (see "phone_echo"): the group just heard, or the plain "go on" when nothing new was added (the caller repeated themselves)."""
    group = new[len(old):] if new.startswith(old) else new
    return prompt("phone_echo", lang, digits=" ".join(group)) if group else prompt("phone_partial", lang)


def absorb(buffer: str, text: str, asked: bool) -> PhoneStep:
    """One turn of phone capture. `buffer` = digits kept from earlier turns of THIS booking; `asked` = the phone is the question
    the caller is answering (so a sentence of a few digits is certainly the number). When it is not asked, only a complete
    valid number counts -- digits in an unrelated sentence ("at 5 pm", "3 days") are never appended to the buffer."""
    low = unicodedata.normalize("NFC", text or "").lower()
    extra = end_append(text)
    if extra and buffer:
        # Real call 2026-10-18: nine digits so far and "শেষে একটা জিরো হবে আরেকটা": the zero goes on the end. If the caller repeats it ("…জিরো হবে জিরো") the number is
        # never longer than ten digits: only as many as fit are added.
        candidate = (buffer + extra)[:10]
        done = phone_from_digits(candidate)
        if done:
            return PhoneStep("complete", number=done, pure=True)
        return PhoneStep("partial", buffer=candidate, pure=True)
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
    elif buffer and len(buffer) + len(digits) > 10 and len(digits) >= 7:
        candidate = digits  # more than a number's worth of digits, and the new ones are nearly a whole number: the caller started again
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
        "हाँ", "हां", "नहीं", "नही", "डॉक्टर", "कल", "आज", "परसों", "हैलो", "क्या", "कब", "কैসे", "फिर", "दोबारा",
        # Real call 2026-10-18: "আজকে পাওয়া যাবে" ("will it be available today?") was read back as the PATIENT'S NAME. Day words, "available / will be / please do" are never a name.
        "আজকে", "আজই", "আজকেই", "কালকে", "কালকেই", "কালই", "পরশু", "পরশুদিন", "আগামীকাল", "পাওয়া", "যাবে", "যাবেন", "যায়", "হবে", "হবেন", "হয়", "করুন", "করবেন", "করে", "দিন",
        "দিতে", "চাই", "আছে", "বসেন", "বসবেন", "বসছেন", "বুক", "অ্যাপয়েন্টমেন্ট", "অযাপ্ফোন্টমেন্ট", "সময়", "টাইম", "তারিখ", "দিনের", "আগামী", "পাবো", "পাব", "পারবো", "পারব",
        "available", "availability", "available?", "will", "can", "could", "please", "possible", "get", "want",
        "आज", "कल", "मिलेगा", "मिल", "सकता", "सकती", "चाहिए", "होगा", "करें", "करिए", "दीजिए",
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
    if words and len(phone_words) == len(words):
        return None  # "নাইন নাইন" is two digits of the number, never a name (real call 2026-10-24: it was read back as the patient's name)
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
    if sum(1 for ch in name if ch.isalpha()) < 2:
        return None  # one letter is a clipped "yes", never a name (real call 2026-10-18: "ह" was read back as the name)
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
# Real call 2026-10-18: "কালকেই" (tomorrow itself, with the emphatic ই) was not a day word: the model read it as YESTERDAY (the 17th) and the booking for the 19th became a lookup.
_TOMORROW = frozenset({"tomorrow", "kal", "kaal", "কাল", "কালকে", "আগামীকাল", "আগামিকাল", "agamikal", "कल", "कलके", "কালকেই", "কালই", "আগামীকালই", "আগামিকালই", "কালকেরই", "कल ही", "कलही"})
_DAY_AFTER = frozenset({"parso", "parson", "পরশু", "পরশুদিন", "porshu", "परसों", "परसो", "পরশুই", "পরশুদিনই"})
_TODAY = frozenset({"today", "aaj", "aj", "আজ", "আজকে", "आज", "আজই", "আজকেই", "आजही"})
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


_DATE_MARKERS = frozenset({"তারিখ", "তারিখে", "তারিখের", "তারীখ", "তারিখটা", "तारीख", "तारीख़", "तारिख", "तारीखको", "tarikh", "tarik", "tarikhe", "date", "dated"})


def day_of_month(words: list[str], today) -> tuple[str | None, set[int]]:
    """"একুশ তারিখ" / "21 তারিখে" / "२१ तारीख" / "21st" -> the ISO date of that day of the month: this month when the day is still to come (or today), next month when it has passed.
    Needs the word for date after the number (or a 1st/2nd/21st ordinal), so a bare number is never read as a day."""
    import datetime

    from agent import owner_answers

    for i, w in enumerate(words):
        n = None
        used = {i}
        m = re.fullmatch(r"(\d{1,2})(?:st|nd|rd|th)", w)
        if m:
            n = int(m.group(1))
        elif w in _DATE_MARKERS and i > 0:
            prev = words[i - 1]
            used = {i, i - 1}
            m = re.fullmatch(r"(\d{1,2})(?:st|nd|rd|th|শে|ই|ও)?", prev)
            if m:
                n = int(m.group(1))
            else:
                base = re.sub(r"(শে|ই|ও)$", "", prev)
                n = owner_answers.token_number(base) if base else None
        if n and 1 <= n <= 31:
            year, month = today.year, today.month
            if n < today.day:
                year, month = (year + 1, 1) if month == 12 else (year, month + 1)
            try:
                return datetime.date(year, month, n).isoformat(), used
            except ValueError:
                return None, set()
    return None, set()


_AFTER_WORDS = frozenset({"পরের", "পরে", "পরেই", "পরের", "পর", "বাদে", "बाद", "के बाद", "after", "beyond", "later"})
_NOT_THAT_DAY = ("হবে না", "হবেনা", "পারব না", "পারবো না", "সম্ভব না", "চলবে না", "नहीं हो", "नहीं होगा", "नहीं आ", "won't work", "wont work", "cannot", "can't", "not possible", "no good")


def after_date(text: str, today) -> str | None:
    """"একুশ তারিখের পরের কোনো দিন" / "after the 21st" / "21 के बाद" / "কালকের পরে" -> the ISO date AFTER WHICH a day is wanted (the named day itself is ruled out). None when the
    sentence names no day with an "after" word beside it. Real call 2026-10-19: "tomorrow (the 21st) will not do, any day after it" was heard as a question about lab tests."""
    import datetime

    words = _words(text)
    if not any(w in _AFTER_WORDS for w in words):
        return None
    base, used = day_of_month(words, today)  # a named day of the month beats a relative word ("after the 21st, tomorrow will not do")
    if not base:
        base, used = day_from_words(words, today)
    if not base:
        return None
    # the "after" word must stand within three words of the day
    if not any(w in _AFTER_WORDS for w in words[max(0, min(used) - 3) : max(used) + 4]):
        return None
    d = datetime.date.fromisoformat(base)
    # "tomorrow" and "after the day": the day named is excluded from the answer either way
    return d.isoformat()


_REFUSAL = frozenset({"না", "নয়", "নেই", "নাই", "not", "no", "dont", "don't", "नहीं", "नही", "ना"})


def day_from_words(words: list[str], today) -> tuple[str | None, set[int]]:
    """"tomorrow" / "কাল" / "कल" (also "day after tomorrow", "today") -> ISO date relative to `today`. In a booking, a bare
    কাল / कल always means tomorrow (the words also mean 'yesterday', but nobody books yesterday)."""
    import datetime

    # Owner, 2026-10-31 ("আজকে না কালকে কালকে ... আজকে লাগবে না কালকে বলুন" was answered for TODAY): a caller who corrects herself means the LAST day she said, and a day followed by a "no" ("আজকে লাগবে না",
    # "আজকে না") is the day she does NOT want. Every day word is collected; refused ones are set aside; the last of the rest is the day.
    found = []  # (index, days ahead, indexes used)
    for i, w in enumerate(words):
        if w in _DAY_AFTER:
            found.append((i, 2, {i}))
        elif w in _TOMORROW:
            if i >= 2 and words[i - 2 : i] == ["day", "after"]:
                found.append((i, 2, {i, i - 1, i - 2}))
            else:
                found.append((i, 1, {i}))
        elif w in _TODAY:
            found.append((i, 0, {i}))
    if found:
        wanted = [f for f in found if not any(x in _REFUSAL for x in words[f[0] + 1 : f[0] + 4])]
        _i, ahead, used = (wanted or found)[-1]
        return (today + datetime.timedelta(days=ahead)).isoformat(), used
    return day_of_month(words, today)


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
    # Owner, 2026-10-28: a number said in groups with a pause between them ("9530 ... 4132 ... 33", "95304 ... 132 ... 33") is repeated back group by group IN the pauses, the way a
    # person writing it down would ("9 5 3 0 ... 4 1 3 2 ..."), so the caller hears at once whether a group was misheard. Only the group just heard; the whole number is read back, and confirmed, at the end.
    "phone_echo": {
        "en": "{digits}. Go on.",
        "hi": "{digits}। बोलिए।",
        "bn": "{digits}। বলুন।",
    },
    # Owner, 2026-10-23: a caller who stops after TWO digits is told what was heard and asked for the other eight (a number said in groups keeps the short "ok" above).
    "phone_partial_two": {
        "en": "Okay, I have {digits}. Please say the other eight digits.",
        "hi": "ठीक है, {digits} मिला है। बाकी आठ अंक बताइए।",
        "bn": "ঠিক আছে, {digits} পেয়েছি। বাকি আটটা সংখ্যা বলুন।",
    },
    "phone_partial_long": {
        "en": "Okay, I have {digits} so far. Please say the rest.",
        "hi": "ठीक है, अभी तक {digits} मिला है। बाकी बताइए।",
        "bn": "ঠিক আছে, এখন পর্যন্ত {digits} পেয়েছি। বাকিটা বলুন।",
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
    "age_confirm": {
        "en": "The patient's age is {value}. Is that right?",
        "hi": "मरीज़ की उम्र {value} साल है, ठीक है?",
        "bn": "রোগীর বয়স {value} বছর, ঠিক আছে?",
    },
    "age_which": {
        "en": "Is the age {a} or {b}? Please say it clearly.",
        "hi": "उम्र {a} है या {b}? कृपया साफ़ बोलिए।",
        "bn": "বয়স {a} নাকি {b}? একটু পরিষ্কার করে বলবেন?",
    },
    "age_again": {
        "en": "All right. Please tell me the patient's age again.",
        "hi": "ठीक है। मरीज़ की उम्र दोबारा बताइए।",
        "bn": "ঠিক আছে। রোগীর বয়সটা আবার বলবেন?",
    },
    "place_confirm": {
        "en": "Did you say {place}?",
        "hi": "क्या आपने {place} कहा?",
        "bn": "আপনি কি {place} বললেন?",
    },
    "address_confirm": {
        "en": "The address is {value}. Is that right?",
        "hi": "पता {value} है, ठीक है?",
        "bn": "ঠিকানা {value}। ঠিক আছে?",
    },
    "address_again": {
        "en": "All right. Please tell me the address again.",
        "hi": "ठीक है। पता दोबारा बताइए।",
        "bn": "ঠিক আছে। ঠিকানাটা আবার বলবেন?",
    },
    "name_heard": {"en": "I heard the name {value}.", "hi": "मैंने नाम {value} सुना।", "bn": "আমি নামটা শুনেছি {value}।"},
    "phone_heard": {"en": "I heard the number {value}.", "hi": "मैंने नंबर {value} सुना।", "bn": "আমি নম্বরটা শুনেছি {value}।"},
    "age_heard": {"en": "The patient's age is {value}.", "hi": "मरीज़ की उम्र {value} साल है।", "bn": "রোগীর বয়স {value} বছর।"},
    "is_right": {"en": "Is that correct?", "hi": "क्या यह सही है?", "bn": "এটা কি ঠিক?"},
    "ask_day": {"en": "Which day would you like the appointment for?", "hi": "किस दिन के लिए अपॉइंटमेंट चाहिए?", "bn": "কোন দিনের জন্য অ্যাপয়েন্টমেন্ট চাই?"},
    "ask_age": {"en": "What is the patient's age?", "hi": "मरीज़ की उम्र क्या है?", "bn": "রোগীর বয়স কত?"},
    "ask_address": {"en": "Could you tell me the address?", "hi": "पता बताइए?", "bn": "ঠিকানাটা বলবেন?"},
    "ps_ask": {"en": "Which police station is that under?", "hi": "वह किस थाने के अंतर्गत है?", "bn": "থানার নামটা বলবেন?"},
    "ps_again": {
        "en": "Please say the police station name. It is needed to book the appointment.",
        "hi": "कृपया थाने का नाम बताइए। अपॉइंटमेंट के लिए यह ज़रूरी है।",
        "bn": "দয়া করে থানার নামটা বলুন। অ্যাপয়েন্টমেন্টের জন্য এটা দরকার।",
    },
    # Owner, 2026-10-23 (item 15): the number is always asked about -- the payment link and the serial number go to WhatsApp.
    # Owner, 2026-10-27: the payment link goes to this number, so the question also asks whether phone pay / Google Pay / UPI can be done from it.
    "ask_whatsapp": {
        "en": "Does this number have WhatsApp, and can you pay by PhonePe, Google Pay or UPI from this number?",
        "hi": "क्या इस नंबर पर व्हाट्सऐप है, और क्या इस नंबर से फ़ोन पे, गूगल पे या यूपीआई से पेमेंट कर सकते हैं?",
        "bn": "এই নম্বরে কি হোয়াটসঅ্যাপ আছে, আর এই নম্বর থেকে কি ফোন পে, গুগল পে বা ইউপিআই পেমেন্ট করতে পারবেন?",
    },
    "whatsapp_again": {
        "en": "Sorry, I did not catch that. Does this number have WhatsApp, and can you pay from it by PhonePe, Google Pay or UPI, yes or no?",
        "hi": "माफ़ कीजिए, समझ नहीं आया। क्या इस नंबर पर व्हाट्सऐप है और इससे फ़ोन पे, गूगल पे या यूपीआई से पेमेंट हो सकता है, हाँ या नहीं?",
        "bn": "দুঃখিত, বুঝতে পারিনি। এই নম্বরে কি হোয়াটসঅ্যাপ আছে আর এই নম্বর থেকে কি ফোন পে, গুগল পে বা ইউপিআই পেমেন্ট করা যায়, হ্যাঁ নাকি না?",
    },
    "whatsapp_number": {
        "en": "Okay. Please tell me the number that has WhatsApp and that you can pay from by PhonePe, Google Pay or UPI, all ten digits.",
        "hi": "ठीक है। जिस नंबर पर व्हाट्सऐप है और जिससे फ़ोन पे, गूगल पे या यूपीआई से पेमेंट कर सकते हैं, वह दस अंकों का नंबर बताइए।",
        "bn": "ঠিক আছে। যে নম্বরে হোয়াটসঅ্যাপ আছে আর যেখান থেকে ফোন পে, গুগল পে বা ইউপিআই পেমেন্ট করতে পারবেন, সেই দশ সংখ্যার নম্বরটা বলুন।",
    },
    "whatsapp_number_again": {
        "en": "I did not catch the number. Please say the ten-digit WhatsApp number once more.",
        "hi": "नंबर समझ नहीं आया। व्हाट्सऐप का दस अंकों का नंबर एक बार फिर बताइए।",
        "bn": "নম্বরটা বুঝতে পারিনি। হোয়াটসঅ্যাপের দশ সংখ্যার নম্বরটা আরেকবার বলুন।",
    },
    # Owner, 2026-10-24: no WhatsApp number could be collected (the caller has none, or kept giving an unreadable
    # number) -- said once, right before the booking continues anyway, so the caller knows the confirmation/serial
    # number will not reach them on WhatsApp and what to do about it, instead of silently going quiet on the point.
    "whatsapp_no_number": {
        "en": "Okay, I will book it without a WhatsApp number. I will not be able to send the confirmation and serial number on WhatsApp for this -- you can collect it from the clinic counter, or call back later with a WhatsApp number.",
        "hi": "ठीक है, मैं यह बिना व्हाट्सऐप नंबर के बुक कर रही हूँ। इस पर कन्फर्मेशन और सीरियल नंबर व्हाट्सऐप पर नहीं भेज पाऊँगी -- आप क्लिनिक काउंटर से ले सकते हैं, या बाद में व्हाट्सऐप नंबर के साथ फिर कॉल कीजिए।",
        "bn": "ঠিক আছে, হোয়াটসঅ্যাপ নম্বর ছাড়াই বুক করছি। এতে কনফার্মেশন আর সিরিয়াল নম্বর হোয়াটসঅ্যাপে পাঠাতে পারব না -- আপনি ক্লিনিক কাউন্টার থেকে নিতে পারেন, বা পরে হোয়াটসঅ্যাপ নম্বর নিয়ে আবার কল করুন।",
    },
    # Owner, 2026-10-29: "I have a small phone, I cannot do that" -> the payment is made at the clinic counter; the booking goes on.
    "pay_at_counter": {
        "en": "Okay, then you will need to come to the clinic counter and pay there. I am booking the appointment.",
        "hi": "ठीक है, तो आपको क्लिनिक के काउंटर पर आकर पेमेंट करना होगा। मैं अपॉइंटमेंट बुक कर रही हूँ।",
        "bn": "ঠিক আছে, তাহলে আপনাকে ক্লিনিকের কাউন্টারে এসে পেমেন্ট করতে হবে। আমি অ্যাপয়েন্টমেন্ট বুক করে দিচ্ছি।",
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


_DETAIL_ORDER = ("phone", "patient_age", "patient_address")


def next_ask(st, lang: str, after: str) -> str | None:
    """The question that FOLLOWS the field just read back, asked in the same breath as the read-back (owner, 2026-10-18: not "is that right?" after every answer, but
    "I heard the name Sayan Das. Please tell me your number."). The next field still unanswered after `after`; None when nothing is left to ask. The field is marked asked."""
    from agent import reply_templates

    start = {"patient_name": 0, "phone": 1, "patient_age": 2}.get(after, 0)
    for field in _DETAIL_ORDER[start:]:
        if field == "phone":
            if st.slots.get("phone") or st.phone_declined:
                continue
            st.asked_fields.add("phone")
            return reply_templates.missing_slot_prompt("book_appointment", "phone", lang)
        tag = "age" if field == "patient_age" else "address"
        if st.slots.get(field) or st.slots.get(f"_{tag}_done"):
            continue
        st.asked_fields.add(field)
        return prompt(f"ask_{tag}", lang)
    return None


def readback(st, lang: str, kind: str, value) -> str:
    """"I heard the name X. <the next question>" -- or, when nothing is left to ask, "I heard the name X. Is that correct?". kind: name | phone | age."""
    heard = prompt(f"{kind}_heard", lang, value=value)
    nxt = next_ask(st, lang, {"name": "patient_name", "phone": "phone", "age": "patient_age"}[kind])
    return f"{heard} {nxt}" if nxt else f"{heard} {prompt('is_right', lang)}"


_HUMS = frozenset({"hmm", "hm", "hmmm", "mhm", "mm", "হুম", "হুঁ", "হু", "হম", "हम्म", "हूँ", "हूं", "हम", "ह", "हं", "हँ", "হ", "হঁ"})  # "ह" / "হ": the recogniser clips "हाँ" / "হ্যাঁ" to its first letter


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
    readback_answer: bool = False  # the sentence was the yes to a name / number read back: settled here, not withheld for low recogniser agreement
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
    return re.sub(r"^\s*(no|nope|nahi|nahin|na|না|নয়|नहीं|नही)(?=[\s,।.!]|$)[\s,।.!]*", "", text or "", flags=re.I)


# real call 2026-10-24: "আর মোহাম্মদ আব্দুল আমার নাম না, আমার নাম হচ্ছে মোহাম্মদ হামিদুল" (the read-back name is
# NOT mine, my name IS Hamidul) -- the rejected old name stays in the sentence, so candidate_name(_bare_no(text))
# sees an 11-word sentence (well past its 1-4 word limit) and gives up, asking the caller to repeat the whole name
# from scratch even though the correct one is right there after "my name is". Tried only as a fallback, after the
# existing _bare_no path has already failed -- never changes a case that already works.
_NAME_IS_MARKER = re.compile(
    r"(?:নাম\s*(?:টা|টি)?\s*(?:হচ্ছে|হল|হলো|আছে)|नाम\s*(?:है|हैं|होगा|होगी)|name\s+is)\s*[:\-]?\s*(.+)$",
    re.I,
)


def _name_after_correction(text: str) -> str | None:
    """The part after the LAST "my name is ..." marker in the sentence, read as a name -- or None. The last match
    wins: an old, rejected name said earlier in the same breath has its own such marker too ("X na, naam hocche
    Y"), and it is Y, not X, that the caller is actually giving."""
    match = None
    for m in _NAME_IS_MARKER.finditer(text or ""):
        match = m
    return candidate_name(match.group(1)) if match else None


_PHONE_CUES = frozenset({"number", "phone", "mobile", "contact", "नंबर", "नम्बर", "फोन", "फ़ोन", "मोबाइल", "নম্বর", "নাম্বার", "ফোন", "মোবাইল", "nombor", "nambar", "namber", "no."})  # fmt: skip


# An age said with its word: "বয়স ৪৫", "আমার বয়স 45", "age 45", "45 years old", "৪৫ বছর", "उम्र 45", "45 साल". Only with the word -- a bare number is a digit of the phone.
_DIGITS_ANY = str.maketrans("০১২৩৪৫৬৭৮৯०१२३४५६७८९", "01234567890123456789")
_AGE_BEFORE = re.compile(r"(?:বয়স|বয়েস|age|aged|उम्र|उमर|आयु)\s*(?:হলো|হল|হয়েছে|है|is|of)?\s*(\d{1,3})(?!\d)", re.I)
_AGE_AFTER = re.compile(r"(?<!\d)(\d{1,3})\s*(?:বছর|বছরের|years?\s*old|years?|yrs?|साल|वर्ष)", re.I)
# Owner, 2026-10-27 (real call): "রোগীর বয়স পঁয়ত্রিশ বছর" (patient's age is thirty-five years), the age said as a
# WORD, was read back as "30" -- the model's own extraction mis-split "পঁয়ত্রিশ" (35) into a plain "ত্রিশ" (30).
# agent/number_words.py already has the exact, correct, tested two-digit-word table (built for phone numbers said
# in pairs); using it here for the SAME word next to "বয়স"/"বছর" means a word-form age is read deterministically,
# the same discipline split_age already gives a digit-form one, instead of being left to the model to get wrong.
_AGE_WORD_BEFORE = re.compile(r"(?:বয়স|বয়েস|age|aged|उम्र|उमर|आयु)\s*(?:হলো|হল|হয়েছে|है|is|of)?\s*([^\s,।]+)", re.I)
_AGE_WORD_AFTER = re.compile(r"([^\s,।]+)\s*(?:বছর|বছরের|years?\s*old|years?|yrs?|साल|वर्ष)", re.I)


def split_age(text: str) -> tuple[int | None, str]:
    """(age, the sentence without the age phrase). A number written as a word ("পঁয়ত্রিশ বছর") is resolved from
    agent/number_words.py's own table when it is one of the two-digit words that table knows; anything else is
    left to the model's patient_age slot."""
    t = (text or "").translate(_DIGITS_ANY)
    for pat in (_AGE_BEFORE, _AGE_AFTER):
        m = pat.search(t)
        if m and 0 < int(m.group(1)) <= 120:
            return int(m.group(1)), (t[: m.start()] + " " + t[m.end():]).strip()
    for pat in (_AGE_WORD_BEFORE, _AGE_WORD_AFTER):
        m = pat.search(t)
        if m:
            digits = number_words.digits_of(m.group(1))
            if digits is not None:
                return int(digits), (t[: m.start()] + " " + t[m.end():]).strip()
    return None, text


_STATION_WORDS = ("থানা", "थाना", "थाने", "police", "পুলিশ", "पुलिस", " p.s", " ps ")

# Owner, 2026-10-23: after "which police station?" an answer that REFUSES or does not know it is not a station name. The station is asked for again before the booking
# goes on to its closing question -- at most _PS_MAX_REASKS times, so a caller who cannot be moved is still booked (the address is then kept as said).
_PS_MAX_REASKS = 2
_PS_BARE_NO = frozenset({"no", "nope", "na", "nah", "nahi", "nhi", "nahin", "nai", "nei", "नहीं", "नही", "ना", "না", "নয়", "নাই", "নেই"})
_PS_REFUSAL_PHRASES = (
    # Bengali
    "জানি না", "জানিনা", "জানা নেই", "জানা নাই", "মনে নেই", "মনে নাই", "মনে পড়ছে না", "মনে পড়ছেনা", "বলব না", "বলবো না", "বলবনা", "বলতে পারব না", "বলতে পারবো না",
    "বলতে চাই না", "দেব না", "দেবো না", "দিতে পারব না", "থানা নেই", "থানার নাম নেই", "থানা জানি না", "থানা মনে নেই",
    # Hindi
    "पता नहीं", "पता नही", "नहीं पता", "नही पता", "मालूम नहीं", "मालूम नही", "नहीं मालूम", "नही मालूम", "याद नहीं", "याद नही", "नहीं याद", "नही याद",
    "नहीं बताऊंगा", "नहीं बताऊँगा", "नहीं बताउंगा", "नहीं बताऊंगी", "नहीं बताऊँगी", "नहीं बताउंगी", "नहीं बताना", "नहीं बताएंगे", "नहीं बोलूंगा", "नहीं बोलूँगा", "नहीं देना",
    "भूल गया", "भूल गई", "भूल गयी", "पता ही नहीं", "थाना नहीं",
)
_PS_REFUSAL_ROMAN = re.compile(
    r"\b(?:"
    # English
    r"(?:don'?t|do not|dont|didn'?t|can'?t|cannot|can not|won'?t|will not|wont|not going to)\s+(?:know|remember|recall|say|tell|give|want to (?:say|tell|give))"
    r"|no idea|not sure|forgot(?:ten)?|not telling|unknown|nothing|none|skip|i have no|there is no"
    # Hindi, written in Latin letters
    r"|(?:nahi|nhi|nahin|nai|na)\s*(?:pata|pta|malum|maloom|yaad|yad|bataunga|batauga|bataungi|batungi|bolunga|bolungi|batana|batani|jaanta|janta|jaanti|janti|dunga|dungi)"
    r"|(?:pata|pta|malum|maloom|yaad|yad|jaanta|janta)\s*(?:nahi|nhi|nahin|nai|na)|(?:bhool|bhul)\s*(?:gaya|gayi|gai)"
    # Bengali, written in Latin letters
    r"|nei|jani\s*na[h]?|janina|jana\s*(?:nei|nai)|mone\s*(?:nei|nai|porche\s*na[h]?|hocche\s*na[h]?)|bolbo\s*na[h]?|bolbona|bolte\s*parbo\s*na[h]?|bolchi\s*na[h]?"
    r"|dibo\s*na[h]?|dite\s*parbo\s*na[h]?|bolte\s*chai\s*na[h]?"
    r")\b",
    re.I,
)


def _refuses_station(answer: str) -> bool:
    """True when the answer to "which police station?" REFUSES or does not know it (\"nei\", \"jani na\", \"nahi pata\", \"yaad nahi\", \"bolbo nah\", \"don't know\") or holds nothing
    at all -- in English, Hindi and Bengali, written in their own scripts or in Latin letters."""
    t = " ".join((answer or "").lower().split()).strip(" ,.।!?")
    if len(t) < 2 or t in _NO_ADDRESS or t in _PS_BARE_NO:
        return True
    return any(p in t for p in _PS_REFUSAL_PHRASES) or _PS_REFUSAL_ROMAN.search(t) is not None


def _lacks_police_station(address: str) -> bool:
    low = f" {address.lower()} "
    return not any(w in low for w in _STATION_WORDS)


_ENGLISH_BAZAR = re.compile(
    r"(?:ইংলিশ|ইংরেজ|ইংরাজ|ইংলীশ|english|इंग्लिश|अंग्रेज)\s*(?:বাজার|বাজর|রাজা|রাজার|বাজা|বাজেট|বাগার|ba[sz]+ar|bajar|baza|raj+ah?|razar|b[a-z]{2,8}|बा[ज़ज]ार|राजा)", re.I
)
_ENGLISH_BAZAR_SPOKEN = {"bn": "ইংলিশ বাজার", "hi": "इंग्लिश बाज़ार", "en": "English Bazar"}


def fix_station(station: str, lang: str) -> str:
    """"english rajah" / "ইংলিশ রাজা" is the English Bazar police station (Malda), which the recogniser keeps hearing as a request for English or as a name. Said in the call's own script."""
    spoken = _ENGLISH_BAZAR_SPOKEN.get(lang, _ENGLISH_BAZAR_SPOKEN["en"])
    if says_english_bazar(station):
        return spoken  # the whole short answer: "english was a" is the station, not a request for English
    return _ENGLISH_BAZAR.sub(spoken, station or "")


def _may_need_paediatric_redirect(age_value) -> bool:
    try:
        return 13 <= int(age_value) <= 17
    except (TypeError, ValueError):
        return False


def says_english_bazar(text: str) -> bool:
    """A short answer that begins with "English" in any of the three scripts ("english was a", "ইংলিশ বাজা", "इंग्लिश बाज़ार"): asked for the police station, this is English Bazar, whatever
    the recogniser made of the second word."""
    t = " ".join((text or "").split())
    return 0 < len(t.split()) <= 3 and re.match(r"(?:ইংলিশ|ইংরেজ|ইংরাজ|ইংলীশ|english|इंग्लिश|अंग्रेज)", t, re.I) is not None


def _detail_turn(st, text: str, lang: str, plan: "TurnPlan", field: str, parse, tag: str) -> bool:
    """One of the details asked after the name and the number (the age, then the address). Owner, 2026-10-18: each is asked alone; the AGE is read back with the next
    question in the same breath ("the patient is 45. What is the address?"), the answer to which also counts as its yes; the ADDRESS goes straight on to the closing question,
    after asking the police station when the caller did not give one. An answer that holds no value is let go once -- except the AGE, which is asked one more time (owner, 2026-10-30). True when this sentence was that answer."""
    pend_key, done_key = f"_{tag}_pending", f"_{tag}_done"
    if tag == "address":
        # Real call, 2026-10-29: a written-form (exact) match against the geo tables is address enough on its own --
        # see main.py._check_address_answer's own comment. Checked before the sound-alike/police-station logic below
        # so an exact district or station name is never also walked into the "which police station?" question.
        exact = st.slots.pop("_addr_exact", None)
        if exact and field in st.asked_fields and not st.slots.get(field) and not st.slots.get(done_key) and not st.slots.get("_ps_pending"):
            st.slots[done_key] = "1"
            plan.slots, plan.skip_model = {field: exact}, True
            return True
        # Owner, 2026-10-27: "বালদা" for "মালদা". main.py looked the short answer up in the West Bengal place tables (hint `_addr_suggest`): when it is only a sound-alike of a known place, the
        # caller is asked "did you say <place>?" and the yes / no decides. A written-form (exact) match asks nothing.
        confirm = st.slots.get("_addr_confirm")
        suggestion = st.slots.pop("_addr_suggest", None)
        if confirm:
            orig, _, suggested = confirm.partition("\x1f")
            st.slots.pop("_addr_confirm", None)
            answer = confirm_answer(text, lang)
            if answer == "yes":
                # Real call, 2026-10-29: a caller who confirms "did you say Paschim Medinipur?" has given a complete,
                # recognised district -- demanding a specific police station on top of that (the branch below always
                # did, for ANY address lacking a station word) stalled a real booking for several turns until the
                # caller gave up. A district or station the geo tables already confirmed is address enough on its own.
                st.slots[done_key] = "1"
                plan.slots, plan.skip_model = {field: suggested}, True
                return True
            elif answer == "no":
                text = orig
        elif suggestion and field in st.asked_fields and not st.slots.get(field) and not st.slots.get(done_key) and not st.slots.get("_ps_pending"):
            orig = " ".join((text or "").split()).strip(" ,.।")
            if orig and orig != suggestion:
                st.slots["_addr_confirm"] = f"{orig}\x1f{suggestion}"
                plan.speak = prompt("place_confirm", lang, place=suggestion)
                return True
    if tag == "address" and st.slots.get("_ps_pending"):
        station = fix_station(" ".join((text or "").split()).strip(" ,.।"), lang)
        asked = int(st.slots.get("_ps_asks") or 0)
        # Item 19: main.py checked the answer against the geography tables. A station they know (even spelled as the recogniser heard it) is stored under the table's own name;
        # a refusal that outlasts the re-asks falls back to the station of a place the ADDRESS names, when the tables give exactly one.
        matched = st.slots.pop("_ps_match", None)
        from_place = st.slots.pop("_ps_place", None)
        if matched:
            station = matched
        if not matched and _refuses_station(station) and asked < _PS_MAX_REASKS:
            st.slots["_ps_asks"] = asked + 1  # "_ps_pending" stays: the very next answer is read as the station again, and the closing question waits for it
            plan.speak = prompt("ps_again", lang)
            return True
        address = st.slots.pop("_ps_pending")
        st.slots.pop("_ps_asks", None)
        refused = not matched and (_refuses_station(station) or len(station.split()) > 5)
        if refused and from_place:
            station, refused = from_place, False
        word = {"bn": "থানা", "hi": "थाना"}.get(lang, "police station")
        st.slots[done_key] = "1"
        plan.slots, plan.skip_model = {field: address if refused else f"{address}, {word} {station}"}, True
        return True
    pend = st.slots.get(pend_key)
    if pend is not None:
        answer = confirm_answer(text, lang)
        value = parse(text)
        if answer == "yes":
            st.slots.pop(pend_key, None)
            plan.slots, plan.skip_model, plan.readback_answer = {field: pend}, True, True
            return True
        if answer == "no":
            st.slots.pop(pend_key, None)
            st.slots.pop(done_key, None)
            st.asked_fields.add(field)
            plan.speak = prompt(f"{tag}_again", lang)
            return True
        if answer is None and not value and tag == "age" and "patient_address" in st.asked_fields and not st.slots.get("patient_address") and bare_address(text):
            st.slots.pop(pend_key, None)
            st.slots["patient_age"] = pend  # the read-back asked the address next, and the address was the answer: the age is accepted
            # Owner, 2026-10-27 (real call): this used to `return False`, letting the SAME sentence (the address,
            # here also doubling as the implicit yes for the age) fall straight through to the address
            # _detail_turn call right after this one in before_model's "age or address" chain -- so a teenager
            # booked with a paediatrician had their address (and the police-station question after it) asked
            # and answered BEFORE main.py's paediatric-age redirect (_redirect_teen_from_paediatrics) ever got a
            # turn to run, instead of right after the age was confirmed, as the owner wants. Returning True here
            # instead stops at the age: the address just given is asked for again next turn, but only once the
            # caller is on the right doctor, not walked further into the wrong one's booking first.
            plan.slots, plan.skip_model = {field: pend}, True
            if not _may_need_paediatric_redirect(pend):
                # Owner, 2026-10-27 (real call): "বালদা" was the address, said as the answer to the age read-back; stopping here threw it away and asked for the address again. Only a
                # patient of 13 to 17 can be on the paediatrician by mistake; for everyone else the address is taken in this same turn, as it always was.
                plan.slots, plan.skip_model = {}, False
                return False
            return True
        if value:
            st.slots[pend_key] = value
            plan.speak = readback(st, lang, "age", value)
            return True
        plan.speak = readback(st, lang, "age", pend)  # neither a yes, a no nor a value: the same read-back once more
        return True
    if field in st.asked_fields and not st.slots.get(field) and not st.slots.get(done_key):
        value = parse(text)
        if not value and tag == "age":
            candidates = number_words.phonetic_numbers(text)
            if len(candidates) >= 2:  # a number-sounding word that fits two ages (7 or 60): ask, never pick
                plan.speak = prompt("age_which", lang, a=candidates[0], b=candidates[1])
                return True
        if not value and tag == "age" and not st.slots.get("_age_reasked"):
            # Owner, 2026-10-30 ("by the least by the least" for পঁয়তাল্লিশ, and the age was never asked again): an age the recogniser garbled is asked ONE more time before it is let go.
            st.slots["_age_reasked"] = "1"
            plan.speak = prompt("age_again", lang)
            return True
        if not value:
            st.slots[done_key] = "1"  # no value in this answer: let it go
            plan.slots, plan.skip_model = {}, True
            return True
        if tag == "age":
            st.slots[pend_key] = value
            plan.speak = readback(st, lang, "age", value)
            return True
        if _lacks_police_station(value):
            st.slots["_ps_pending"] = value
            plan.speak = prompt("ps_ask", lang)
            return True
        st.slots[done_key] = "1"
        plan.slots, plan.skip_model = {field: value}, True
        return True
    return False


# ---- the WhatsApp question (owner, 2026-10-23, item 15)
# The payment link and the serial number go to WhatsApp, so every booking is asked ONCE whether the number it holds has it. A yes carries on; a no (or "another number") asks for
# the number that has WhatsApp and reads it back like any phone number. A caller who answers neither twice is let go: the booking is never held up by this question.


def whatsapp_due(st) -> bool:
    """True when the booking holds a number nobody has asked about yet."""
    if st.action != "book_appointment" or st.phone_declined or st.slots.get("_wa_asked"):
        return False
    return bool(st.slots.get("contact_phone") or st.slots.get("phone"))


def address_heard(slots: dict, lang: str) -> str:
    """"ঠিকানা X, থানা Y শুনেছি।" -- the address (with its police station) as heard, said at once when it is taken; the closing read-back then does not say it again."""
    addr = slots.get("patient_address")
    if not addr:
        return ""
    return {"bn": f"ঠিকানা {addr} শুনেছি। ", "hi": f"पता {addr} सुना। "}.get(lang, f"I have the address as {addr}. ")


def ask_whatsapp(st, lang: str) -> str:
    st.slots["_wa_asked"] = "1"
    return prompt("ask_whatsapp", lang, digits=spoken_digit_groups(st.slots.get("contact_phone") or st.slots.get("phone")))


def _use_whatsapp_number(st, number: str, plan: "TurnPlan") -> bool:
    """The number that has WhatsApp replaces the booking's number, which is then read back for a yes like any other (the existing phone read-back runs on its own)."""
    st.slots["_wa_done"] = "1"
    for key in ("_wa_number", "_wa_tries", "contact_phone", "_phone_ok", "_conflict_checked"):
        st.slots.pop(key, None)
    st.phone_buffer = ""
    plan.phone, plan.digits = number, True
    plan.slots, plan.skip_model = {"phone": number}, True
    return True


def _whatsapp_turn(st, text: str, lang: str, plan: "TurnPlan") -> bool:
    """The caller's answer to "does this number have WhatsApp?" (or, after a no, the number that does). True when this sentence was that answer."""
    if st.action != "book_appointment" or not st.slots.get("_wa_asked") or st.slots.get("_wa_done"):
        return False
    text = whatsapp_cues.canonicalise(text)
    number = phone_from_digits(read_digits(text)[0])
    if number and number != st.slots.get("phone"):
        return _use_whatsapp_number(st, number, plan)  # "no, 98765 43210" / the number itself
    tries = int(st.slots.get("_wa_tries") or 0)
    if st.slots.get("_wa_number"):
        old_buffer = st.phone_buffer
        step = absorb(st.phone_buffer, text, True)
        if step.status == "partial":
            st.phone_buffer = step.buffer
            plan.speak = (
                prompt("phone_partial_two" if len(step.buffer) == 2 else "phone_partial_long", lang, n=len(step.buffer), digits=spoken_digit_groups(step.buffer))
                if len(step.buffer) == 2 or len(step.buffer) >= 6
                else partial_prompt(old_buffer, step.buffer, lang)
            )
            return True
        if step.status in ("invalid", "compound", "cleared"):
            st.phone_buffer = ""
            plan.speak = phone_invalid_prompt(step, lang) if step.status == "invalid" else prompt("phone_digit_by_digit" if step.status == "compound" else "phone_cleared", lang)
            return True
        if whatsapp_cues.asks_if_same_number(text) or tries >= 1:
            st.slots["_wa_done"] = "1"  # "this one" -- or the second miss: the number the booking holds stays, the booking goes on
            st.slots.pop("_wa_number", None)
            plan.slots, plan.skip_model = {}, True
            plan.speak = prompt("whatsapp_no_number", lang)
            return True
        st.slots["_wa_tries"] = tries + 1
        plan.speak = prompt("whatsapp_number_again", lang)
        return True
    if whatsapp_cues.cannot_pay_online(text) and not number:
        st.slots["_wa_done"] = "1"  # cannot pay from the phone: the counter takes the payment, the booking goes on
        plan.slots, plan.skip_model = {}, True
        plan.speak = prompt("pay_at_counter", lang)
        return True
    answer = whatsapp_cues.answer(text, lang) or confirm_answer(text, lang)
    if answer == "no":
        st.slots["_wa_number"] = "1"
        st.slots.pop("_wa_tries", None)
        plan.speak = prompt("whatsapp_number", lang)
        return True
    if answer == "yes" or tries >= 1:
        st.slots["_wa_done"] = "1"
        plan.slots, plan.skip_model = {}, True
        return True
    st.slots["_wa_tries"] = tries + 1
    plan.speak = prompt("whatsapp_again", lang, digits=spoken_digit_groups(st.slots.get("phone")))
    return True


_BN_TENS = {"দশ": 10, "বিশ": 20, "ত্রিশ": 30, "চল্লিশ": 40, "পঞ্চাশ": 50, "ষাট": 60, "সত্তর": 70, "আশি": 80, "নব্বই": 90}

# Ages as words, with the spellings a recogniser really writes: Hindi "तेरा साल" (thirteen years), Bengali "তেরা সাল", English said in Devanagari / Bengali script
# ("थर्टी फाइव", "ফিফটি"). Real call 2026-10-19: "तेरा साल" was not read as an age and the number read-back looped.
_AGE_WORDS = {
    "तेरा": 13, "तेरे": 13, "तेरह": 13, "তেরা": 13, "তেরো": 13, "তের": 13, "তেরহ": 13, "चौदह": 14, "चौदा": 14, "চোদ্দ": 14, "চৌদ্দ": 14, "চৌদ্দো": 14,
    "पंद्रह": 15, "पन्द्रह": 15, "पंदरा": 15, "पंद्रा": 15, "पन्द्रा": 15, "পনেরো": 15, "পনের": 15, "পনেরা": 15, "सोलह": 16, "सोला": 16, "ষোলো": 16, "ষোল": 16, "ষোলা": 16,
    "सत्रह": 17, "सत्रा": 17, "সতেরো": 17, "সতের": 17, "অঠারো": 18, "अठारह": 18, "अठारा": 18, "अट्ठारह": 18, "আঠারো": 18, "আঠার": 18, "उन्नीस": 19, "उन्निस": 19, "উনিশ": 19,
    "बीस": 20, "বিশ": 20, "तीस": 30, "चालीस": 40, "पचास": 50, "साठ": 60, "सत्तर": 70, "अस्सी": 80, "नब्बे": 90, "सौ": 100, "একশ": 100, "একশো": 100,
    # English tens and teens, in Devanagari and in Bengali script
    "थर्टीन": 13, "फोर्टीन": 14, "फिफ्टीन": 15, "सिक्स्टीन": 16, "सेवेंटीन": 17, "एटीन": 18, "नाइनटीन": 19, "ट्वेंटी": 20, "थर्टी": 30, "फोर्टी": 40, "फिफ्टी": 50, "सिक्स्टी": 60,
    "सेवंटी": 70, "सेवेंटी": 70, "एटी": 80, "नाइंटी": 90, "নাইনটি": 90, "থার্টিন": 13, "ফোর্টিন": 14, "ফিফটিন": 15, "সিক্সটিন": 16, "সেভেনটিন": 17, "এইটিন": 18, "নাইনটিন": 19,
    "টোয়েন্টি": 20, "থার্টি": 30, "ফোর্টি": 40, "ফিফটি": 50, "সিক্সটি": 60, "সেভেন্টি": 70, "এইটি": 80,
    "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
    # Hindi 21-99 EXCLUDING exact tens (already above): these are each their own irregular word, not "tees panch" like
    # English -- real call 2026-10-24: "मरीज़ की उम्र पैतालीस साल है" (45) was not read as an age and the number
    # read-back looped forever. Both the standard spelling and the common no-nasal-mark ASR spelling are listed (a
    # recogniser often drops ं/ँ), matching this dict's existing teens entries.
    "इक्कीस": 21, "बाईस": 22, "तेईस": 23, "चौबीस": 24, "पच्चीस": 25, "छब्बीस": 26, "सत्ताईस": 27, "सत्ताइस": 27, "अट्ठाईस": 28, "अट्ठाइस": 28, "अठाईस": 28, "उनतीस": 29,
    "इकतीस": 31, "बत्तीस": 32, "तैंतीस": 33, "तैतीस": 33, "चौंतीस": 34, "चौतीस": 34, "पैंतीस": 35, "पैतीस": 35, "छत्तीस": 36, "सैंतीस": 37, "सैतीस": 37, "अड़तीस": 38, "उनतालीस": 39, "उनचालीस": 39,
    "इकतालीस": 41, "बयालीस": 42, "तैंतालीस": 43, "तैतालीस": 43, "चौवालीस": 44,
    # real call, 2026-10-27: "मरीज़ की उम्र क्या है?" answered with a bare "चवालीस" -- the colloquial contraction of
    # "चौंतालीस"/"चौवालीस" (drops the ौ vowel sign entirely) common in everyday Hindi speech -- was not an age at
    # all here, so the exact same question was asked again, and the caller gave up and said a different number.
    "चवालीस": 44,
    "पैंतालीस": 45, "पैतालीस": 45, "छियालीस": 46, "सैंतालीस": 47, "सैतालीस": 47, "अड़तालीस": 48, "उनचास": 49,
    "इक्यावन": 51, "बावन": 52, "तिरपन": 53, "तिरेपन": 53, "चौवन": 54, "पचपन": 55, "छप्पन": 56, "सत्तावन": 57, "अट्ठावन": 58, "अठावन": 58, "उनसठ": 59,
    "इकसठ": 61, "बासठ": 62, "तिरसठ": 63, "चौंसठ": 64, "चौसठ": 64, "पैंसठ": 65, "पैसठ": 65, "छियासठ": 66, "सड़सठ": 67, "अड़सठ": 68, "उनहत्तर": 69,
    "इकहत्तर": 71, "बहत्तर": 72, "तिहत्तर": 73, "चौहत्तर": 74, "पचहत्तर": 75, "छिहत्तर": 76, "सतहत्तर": 77, "अठहत्तर": 78, "उनासी": 79,
    "इक्यासी": 81, "बयासी": 82, "तिरासी": 83, "चौरासी": 84, "पचासी": 85, "छियासी": 86, "सत्तासी": 87, "अट्ठासी": 88, "अठासी": 88, "नवासी": 89,
    "इक्यानवे": 91, "इक्यानबे": 91, "बानवे": 92, "बानबे": 92, "तिरानवे": 93, "तिरानबे": 93, "चौरानवे": 94, "चौरानबे": 94, "पंचानवे": 95, "पचानवे": 95,
    "छियानवे": 96, "छियानबे": 96, "सत्तानवे": 97, "सत्तानबे": 97, "अट्ठानवे": 98, "अठानबे": 98, "निन्यानवे": 99, "निन्यानबे": 99,
}
_AGE_UNITS = {
    "वन": 1, "टू": 2, "थ्री": 3, "फोर": 4, "फाइव": 5, "सिक्स": 6, "सेवन": 7, "एट": 8, "नाइन": 9, "ওয়ান": 1, "টু": 2, "থ্রি": 3, "ফোর": 4, "ফাইভ": 5, "সিক্স": 6, "সেভেন": 7, "এইট": 8, "নাইন": 9,
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
}


def _age_from_words(text: str) -> int | None:
    """An age said as words (see _AGE_WORDS): a tens/teens word, with an English unit after a tens word ("थर्टी फाइव" = 35). None when there is none."""
    toks = [unicodedata.normalize("NFC", w.strip(".,;:!?।")).lower() for w in (text or "").split()]
    for i, w in enumerate(toks):
        n = _AGE_WORDS.get(w)
        if n:
            if n % 10 == 0 and n >= 20 and i + 1 < len(toks):
                unit = _AGE_UNITS.get(toks[i + 1])
                if unit:
                    return n + unit
            return n
    return None


def bare_age(text: str) -> int | None:
    """An age said on its own: "৪৫", "45 years", "পঁয়তাল্লিশ", "ষাট বছর". None when no age (1 to 120) is there."""
    t = (text or "").translate(_DIGITS_ANY)
    m = re.search(r"(?<!\d)(\d{1,3})(?!\d)", t)
    if m and 0 < int(m.group(1)) <= 120:
        return int(m.group(1))
    from agent import owner_answers

    spoken = _age_from_words(t)
    if spoken and 0 < spoken <= 120:
        return spoken
    # Owner, 2026-10-27 (real call): "আটত্রিশ" (38) was read as 30 because the tens word "ত্রিশ" sits INSIDE it and the loop below matched a piece of a word. A whole two-digit
    # number word is looked up first, as a whole word, in the full 10-99 table (agent/number_words.py).
    tokens = [w for w in re.split(r"[\s,.!?।]+", t) if w]
    for tok in tokens:
        digits = number_words.digits_of(tok)
        if digits is not None and 0 < int(digits) <= 120:
            return int(digits)
    for word, n in {**_BN_TENS}.items():
        if any(tok == word or (tok.startswith(word) and tok[len(word):] in ("ের", "এর", "ও", "টা", "ই")) for tok in tokens):
            return n
    n = owner_answers.token_number(t)
    if n and 0 < n <= 120:
        return n
    # Owner, 2026-10-28: "south" for সাত -- a lone number-sounding word is a candidate when it fits exactly one number (a two-number fit is asked about, see _detail_turn / the phone read-back)
    candidates = number_words.phonetic_numbers(t)
    return candidates[0] if len(candidates) == 1 else None


_YES_FIRST = frozenset({"হ্যাঁ", "হ্যা", "হাঁ", "হা", "জি", "ঠিক", "আচ্ছা", "ওকে", "ok", "okay", "yes", "yeah", "haan", "han", "ji", "हाँ", "हां", "हा", "ठीक", "जी", "अच्छा", "ओके"})
_BARE_NO_WORDS_FOR_YES = frozenset({"না", "নয়", "নেই", "no", "nahi", "नहीं", "नही", "নাহ"})
_NO_ADDRESS = {"না", "নেই", "জানি না", "বলব না", "দেব না", "নয়", "skip", "no", "none", "नहीं", "पता नहीं", "बताऊंगा नहीं", "বলব না", "থাক"}


_ADDRESS_LEAD = re.compile(
    r"^(?:আমার\s+|আমাদের\s+)?(?:ঠিকানাটা|ঠিকানা|এড্রেসটা|এড্রেস|অ্যাড্রেস|মেरा\s+पता|मेरा\s+पता|मेरा\s+एड्रेस|पता|एड्रेस|my\s+address|the\s+address|address)\s*(?:হচ্ছে|হলো|হল|হয়|হইলো|হবে|টা|है|हैं|is|:)?\s*[,:\-]?\s*",
    re.I,
)


def bare_address(text: str) -> str | None:
    """The address as said (the whole answer), or None for a refusal / a non-answer. The lead-in words ("ঠিকানা হচ্ছে ...", "मेरा पता है ...", "my address is ...") are not part of it: owner, 2026-10-27, the
    read-back said "ঠিকানা ঠিকানা হচ্ছে রাজারহাট ..." because they were kept."""
    t = " ".join((text or "").split()).strip(" ,.।")
    stripped = _ADDRESS_LEAD.sub("", t, count=1).strip(" ,.।")
    if stripped:
        t = stripped
    elif _ADDRESS_LEAD.match(t):
        return None  # only the word "address" was said
    if len(t) < 3 or t.lower() in _NO_ADDRESS:
        return None
    return t[:160]


_CHANGE_WORDS = frozenset(
    "চেঞ্জ চাঞ্জ চেন্জ পাল্টে পাল্টাতে পাল্টান বদলে বদল বদলান বদলাতে সংশোধন ভুল ঠিক করে change correct changed wrong "
    "बदल बदलना बदलें गलत सुधार".split()
)
_NAME_FIELD_WORDS = frozenset("নাম নামটা নামটি নামে নামের name नाम नाम".split())
_CHANGE_FILLER = frozenset(
    "হবে হলো হল হয় হোক করুন করে করো করতে দিন দাও দিন্ দেন আমার আমি এটা ওটা সেটা তো যে একটু please to is will be the my it এটি ही है हो करो करें दीजिए मेरा मेरी".split()
)


def name_change(text: str) -> str | None:
    """"নামটা চাঞ্জ হবে বিপ্লব সাহা" / "change the name to Biplab Saha": the NEW name, or None when the sentence is not a name change. The change word and the word "name" must both be
    there; what is left after them (and the little filler words) is the name."""
    words = [w for w in re.split(r"[\s,।|!?]+", unicodedata.normalize("NFC", text or "")) if w]
    low = [w.lower() for w in words]
    if not (any(w in _CHANGE_WORDS for w in low) and any(w in _NAME_FIELD_WORDS for w in low)):
        return None
    left = [w for w, lw in zip(words, low) if lw not in _CHANGE_WORDS and lw not in _NAME_FIELD_WORDS and lw not in _CHANGE_FILLER]
    return candidate_name(" ".join(left)) if left else None


def before_model(st, text: str, lang: str, window=None, today=None) -> TurnPlan:
    """The booking turn's deterministic reading (see _before_model), plus: an age said in the sentence is taken out first (so it is never read as a
    phone digit) and kept; a name spelled \"S for September, A for Alpha\" (or letter by letter) is read back for a yes."""
    text = nato_letters.collapse(text)
    if _active(st) and st.action == "book_appointment" and st.name_confirmed and not st.pending_name:
        new_name = name_change(text)
        if new_name and new_name != st.name_confirmed:
            # the name was given and confirmed; now the caller CHANGES it: the new one is read back for a yes (owner, 2026-10-18)
            st.slots.pop("patient_name", None)
            st.name_confirmed = None
            st.pending_name = new_name
            if st.stage in ("confirming", "awaiting_charge_confirm"):
                st.stage = "collecting"
            plan = TurnPlan()
            plan.speak = readback(st, lang, "name", new_name)
            return plan
    age, text = split_age(text) if _active(st) else (None, text)
    if age and _active(st) and st.slots.get("patient_age") and age != st.slots["patient_age"]:
        # a CHANGE of the age: read back for a yes, like the first time
        st.slots["_age_pending"] = age
        st.slots.pop("patient_age", None)
        if st.stage in ("confirming", "awaiting_charge_confirm"):
            st.stage = "collecting"
        plan = TurnPlan()
        plan.speak = readback(st, lang, "age", age)
        return plan
    if age and _active(st) and st.action == "book_appointment" and st.slots.get("_phone_pending") and not st.slots.get("patient_age") and confirm_answer(text, lang) is None and read_digits(text)[1] < 7:
        # Owner, 2026-10-31 (the address asked again and again, in Hindi, English and Bengali): the phone number is read back with "what is the patient's age?" and the caller answers the AGE ("বাইশ বছর")
        # without a separate yes. The age had already been taken out of `text` above, so the number's own check below no longer saw it: the age was dropped, the number stayed "waiting for a yes", and every
        # later answer was read as that wait. The number stands (nobody refused it), the age is kept and read back like any age, and the address is asked next -- once.
        st.slots["_phone_ok"] = st.slots.pop("_phone_pending")
        st.asked_fields.add("patient_age")
        st.slots["_age_pending"] = age
        plan = TurnPlan()
        plan.speak = readback(st, lang, "age", age)  # "The patient's age is 22. Could you tell me the address?"
        return plan
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
            plan.speak = readback(st, lang, "name", name)
    return plan


def _before_model(st, text: str, lang: str, window=None, today=None) -> TurnPlan:
    """Everything the caller's words settle by themselves, decided before the model is asked. `window` = the doctor's chamber
    hours on the booking date as [(start_min, end_min)] when known (to read a bare "ten" as the daytime hour)."""
    from agent.booking_flow import classify_yes_no, missing_required  # local import: booking_flow does not import this module

    plan = TurnPlan()
    if not _active(st):
        return plan

    # 0. "Dr X sits on <day>, shall I book that day?" (the doctor's sitting was just told and offered): a yes IS the date; a no or another day is handled as usual below.
    offered = st.slots.get("_offered_date")
    if offered and not st.slots.get("date"):
        answer = confirm_answer(text, lang)
        if answer == "yes":
            st.slots.pop("_offered_date", None)
            plan.slots, plan.skip_model, plan.readback_answer = {**(plan.slots or {}), "date": offered}, True, True
            return plan
        if answer == "no":
            st.slots.pop("_offered_date", None)
            plan.speak = prompt("ask_day", lang)
            return plan
        st.slots.pop("_offered_date", None)  # something else was said (a day, a question): the offer lapses

    # 0b. "Dr X sits on <day> at <time>, shall I book it?" is waiting and the answer BEGINS with a yes ("হ্যাঁ করুন ঐদিন সাতাশ তারিখ"): that is the yes. Left to the intent model such a
    #     sentence came back as a question about a lab test's rate (real call 2026-10-24). A "no" is not caught here, nor a new day -- those go on as before.
    if st.slots.get("_time_ok") == "pending" and not plan.speak:
        first = (_words(text) or [""])[0]
        if first in _YES_FIRST and confirm_answer(text, lang) != "no" and not any(w in _BARE_NO_WORDS_FOR_YES for w in _words(text)):
            plan.skip_model, plan.readback_answer = True, True
            return plan

    # 1. A name read back. The read-back already asked the NEXT question (owner, 2026-10-18), so the caller's reaction is a yes (which moves on), a no or a corrected
    #    name, or simply the ANSWER to the next question (digits for the number) -- which is a yes as well.
    if st.pending_name:
        answer = confirm_answer(text, lang)
        digits_said = read_digits(text)[1]
        named = candidate_name(_bare_no(text)) if answer in (None, "no") else None
        if answer == "no":
            st.pending_name = None
            if not named:
                named = _name_after_correction(text)
            if named:
                st.pending_name = named
                plan.speak = readback(st, lang, "name", named)
            else:
                plan.speak = prompt("name_again", lang)
            return plan
        if answer == "yes" or (answer is None and (digits_said >= 1 or bare_age(text)) and not named):
            name, st.pending_name = st.pending_name, None
            st.name_confirmed = name
            st.slots["patient_name"] = name  # kept now: a digits-only answer that follows ("নাইন") ends the turn with a question, and plan.slots is not merged then (real call 2026-10-24)
            plan.slots, plan.skip_model, plan.readback_answer = {"patient_name": name}, True, True
            if digits_said == 0 or answer == "yes" and digits_said < 1:
                return plan
            # the answer to the next question came with it ("9876543210"): carry on and take it below
        elif answer is None and named:
            if digits_said >= 1:
                # Real call 2026-10-17: to "I heard the name আবু আবু আলী" the caller said "abu ali" and their number. That is an answer: the name as heard now is
                # the name, the number is taken.
                st.pending_name, st.name_confirmed = None, named
                plan.slots, plan.skip_model = {"patient_name": named}, True
            else:  # a different name: the corrected name, read back
                st.pending_name = named
                plan.speak = readback(st, lang, "name", named)
                return plan
        else:
            st.pending_name = None  # nothing usable: the flow below asks again

    # 1b. A phone number read back and waiting for the caller's yes/no (owner, 2026-10-17: the number is confirmed like the name).
    pending_phone = st.slots.get("_phone_pending")
    if pending_phone:
        answer = confirm_answer(text, lang)
        new_number = read_digits(text)[1] >= 7  # a whole new number in this sentence
        if answer == "yes" and not new_number:
            st.slots.pop("_phone_pending", None)
            st.slots["_phone_ok"] = pending_phone
            plan.slots, plan.skip_model, plan.readback_answer = dict(plan.slots or {}), True, True
            return plan
        if answer is None and not new_number and bare_age(text):
            # the read-back asked the age next, and the age was the answer: the number is accepted and the age is taken below
            st.asked_fields.add("patient_age")
            st.slots.pop("_phone_pending", None)
            st.slots["_phone_ok"] = pending_phone
            plan.skip_model, plan.readback_answer = True, True
            plan.slots = dict(plan.slots or {})
        elif answer == "no" or new_number:
            st.slots.pop("_phone_pending", None)
            for key in ("phone", "contact_phone", "_phone_ok", "_conflict_checked"):
                st.slots.pop(key, None)
            st.phone_buffer = ""
            if not new_number:
                plan.speak = prompt("phone_cleared", lang)
                return plan
        else:  # neither a yes, a no nor an age: the number was read back once; only the QUESTION is asked again (owner, 2026-10-28)
            candidates = number_words.phonetic_numbers(text)
            # Owner, 2026-10-31: the number was read back once and the caller did not refuse it -- it stands; otherwise one unclear answer kept it "pending" for the rest of the call. And the question asked again
            # is the AGE (the one after the phone: next_ask(.., "phone")); next_ask(.., "patient_age") is the one after the age, the address, which is what looped.
            st.slots.pop("_phone_pending", None)
            st.slots["_phone_ok"] = pending_phone
            if len(candidates) >= 2:
                plan.speak = prompt("age_which", lang, a=candidates[0], b=candidates[1])
            else:
                plan.speak = next_ask(st, lang, "phone") or readback(st, lang, "phone", spoken_digit_groups(pending_phone))
            return plan

    # 1c-wa. The answer to "does this number have WhatsApp?" (owner, 2026-10-23).
    if _whatsapp_turn(st, text, lang, plan):
        return plan

    # 1c. The age, then the address, each asked alone and each read back for a yes (owner, 2026-10-18).
    if st.action == "book_appointment" and st.slots.get("patient_name") and (st.slots.get("phone") or st.phone_declined):
        # Owner, 2026-10-28 (real call): the age was taken as the answer to the PHONE read-back ("...রোগীর বয়স কত?" -> "আটাত্তর বছর"), so the agent went straight on to "ঠিকানাটা বলবেন?" -- but that question was
        # never registered as asked, and "ওই মালদা ঠিকানা" (even a bare "মালদা") was then not read as the address: the same question came back every time. An age that is in and an address that is not
        # means the address is the question in the air.
        if st.slots.get("patient_age") and not st.slots.get("patient_address") and not st.slots.get("_address_done"):
            st.asked_fields.add("patient_address")
        if _detail_turn(st, text, lang, plan, "patient_age", bare_age, "age") or _detail_turn(st, text, lang, plan, "patient_address", bare_address, "address"):
            return plan

    # 2. "the first one / the second one" after the clinic offered other slots.
    if st.offered_options:
        chosen = pick_offered_slot(text, st.offered_options)
        if chosen is not None:
            st.offered_options = []
            plan.slots = {k: v for k, v in (chosen if isinstance(chosen, dict) else {"time_slot": chosen}).items() if v}
            plan.skip_model = True
            return plan

    # 3. The phone number -- never starts before the patient's NAME is captured (owner, 2026-10-26: "flow must be
    # name then phone"; real calls showed a caller's volunteered phone number jumping the queue ahead of the name).
    # A number already in progress from an earlier turn (st.phone_buffer) is still finished once started -- only a
    # FRESH number is held back, so the sentence is left for the name-capture rules below instead.
    missing = missing_required(st)
    name_not_yet_given = "patient_name" in missing and not st.phone_buffer
    asked = "phone" in missing and (missing[0] == "phone" or "phone" in st.asked_fields)
    if "phone" in missing and not asked and any(t in _PHONE_CUES for t in _words(text)):
        asked = True  # "the number is ..." -- the caller is plainly giving the phone number, asked or not
    old_buffer = st.phone_buffer
    step = PhoneStep("none", buffer=st.phone_buffer) if name_not_yet_given else absorb(st.phone_buffer, text, asked)
    if step.status == "complete":
        st.phone_buffer = ""
        plan.phone = step.number
        plan.digits = True
        if step.pure:
            plan.slots, plan.skip_model = {**(plan.slots or {}), "phone": step.number}, True
    elif step.status == "partial":
        st.phone_buffer = step.buffer
        plan.digits = True
        st.retry_counts.pop("phone", None)  # digits were heard: that is progress, not a miss (the phone is never given up on mid-number)
        # a few digits so far: "ok, go on" (a number said in groups). Nearly all of it (6 or more): say what was heard, so a caller who FINISHED a number that came out short
        # knows which digits are missing -- an "ok" alone made them think it was complete (real call 2026-10-18).
        if len(step.buffer) >= 6 or len(step.buffer) == 2:
            spoken = prompt("phone_partial_long" if len(step.buffer) >= 6 else "phone_partial_two", lang, n=len(step.buffer), digits=spoken_digit_groups(step.buffer))
        else:
            spoken = partial_prompt(old_buffer, step.buffer, lang)  # 3-5 digits: the group just heard is repeated in the pause (owner, 2026-10-28)
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
            # a short answer ("কালকেই দিন") may also CHANGE a day already held; in a long sentence a day word is more likely history ("আজ সকাল থেকে জ্বর")
            when = read_when(text, asked_time, window=window, today=today, want_day=(not st.slots.get("date")) or len(text.split()) <= 7)
            if when.time and not st.slots.get("time_slot") or when.time and asked_time:
                plan.time = when.time
            elif when.time and st.slots.get("time_slot") and when.time != st.slots.get("time_slot"):
                plan.time = when.time  # a time said with a marker while one is already held: the caller is changing it
            if plan.time and plan.digits and not plan.phone:
                # digits are also in this sentence: a "time" read from them (পাঁচটা = 5) must fall inside the doctor's sitting hours, else it is part of the number
                hh, mm = (int(x) for x in plan.time.split(":"))
                if not window or not any(a <= hh * 60 + mm < b for a, b in window):
                    plan.time = None
            if when.date and when.date != st.slots.get("date"):
                plan.date = when.date  # the caller's own day word beats the model's reading of it
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
                return slots, readback(st, lang, "name", heard)
            return slots, prompt("name_again", lang)
        return slots, None
    missing = missing_required(st)
    if missing and missing[0] == "patient_name" and "patient_name" in st.asked_fields:
        fresh = candidate_name(text)  # the model gave no name although one was asked for: read it from the caller's words
        if fresh:
            st.pending_name = fresh
            return slots, readback(st, lang, "name", fresh)
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
            return readback(st, lang, "name", fresh)
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
