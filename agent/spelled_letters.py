"""A test code said letter by letter ("সি বি সি", "H I V") that the Bengali or Hindi recogniser wrote as a made-up word.

Real calls 2026-10-17: the Bengali IndicConformer wrote a spelled "CBC" as "ক্বক", so the test was lost and the caller heard that it
was not understood. The two recognisers hear a spelled code differently. The Bengali one writes Bengali WORDS: letter names said
quickly, with no gap between them, are pulled together into the nearest word-like sound. The English one keeps letters apart
("c b c", "h i b"), because spelling is something its training data contains. On the pod the English recogniser already runs on every
turn alongside language ID, so its text is free; only the Bengali/Hindi text was ever read for the test names.

This module reads the ENGLISH text for letter runs and turns them into code candidates, then keeps only those that are a test code
in the clinic catalogue (exact), or -- for letters actually spelled -- one letter away from one ("HIB" for HIV: a near match). The candidate is APPENDED to the
transcript the model reads; the lookup that follows treats an exact code as named and a near one as a "did you mean ...?" to confirm
(CLAUDE.md: a near match suggests, it never resolves). Nothing here prices or books anything.
"""

from __future__ import annotations

import re
import unicodedata

_TOKEN = re.compile(r"[a-z0-9]+")


def candidates(english_text: str) -> list[tuple[str, bool]]:
    """(upper-case code candidate, spelled?) in an English transcript: a run of two or more single letters joined ("c b c" -> "CBC",
    spelled=True), and any whole token of two to six letters (the recogniser sometimes writes "cbc" itself, spelled=False).
    Order of appearance, no repeats."""
    out: list[tuple[str, bool]] = []
    run: list[str] = []

    def flush() -> None:
        if len(run) >= 2:
            out.append(("".join(run).upper(), True))
        run.clear()

    for tok in _TOKEN.findall((english_text or "").lower()):
        if len(tok) == 1 and tok.isalpha():
            run.append(tok)
            continue
        flush()
        if 2 <= len(tok) <= 6 and tok.isalpha():
            out.append((tok.upper(), False))
    flush()
    seen: set[str] = set()
    return [c for c in out if not (c[0] in seen or seen.add(c[0]))]


def _one_apart(a: str, b: str) -> bool:
    """One substituted, dropped or added letter (and not the same)."""
    if a == b or abs(len(a) - len(b)) > 1:
        return False
    if len(a) == len(b):
        return sum(x != y for x, y in zip(a, b)) == 1
    short, long_ = (a, b) if len(a) < len(b) else (b, a)
    return any(long_[:i] + long_[i + 1:] == short for i in range(len(long_)))


def match_codes(english_text: str, codes: dict[str, str]):
    """`codes`: {UPPER-CASE code: canonical test name} from the catalogue. Returns [(candidate, canonical test name, exact?)]."""
    found: list[tuple[str, str, bool]] = []
    for cand, spelled in candidates(english_text):
        if cand in codes:
            found.append((cand, codes[cand], True))
            continue
        # a near match is only believed for letters the caller SPELLED: an ordinary English word one letter from "HIV" ("hit") is no evidence
        if not spelled or len(cand) < 3:
            continue
        near = [name for code, name in codes.items() if len(code) >= 3 and _one_apart(cand, code)]
        if len(set(near)) == 1:  # a candidate that is close to two different tests is no evidence for either
            found.append((cand, near[0], False))
    return found


# ---------------------------------------------------------------------------------------------------------------------------------
# A sentence that is ONLY a spelled code and the word "test" ("d s h test", "ডি এস এইচ টেস্ট"). Real call 2026-10-17: "d s h test" -- a
# mis-hearing of "T S H test" -- was handed to the intent model, which answered with the home-collection FAQ because no test name was
# extracted. The catalogue lookup would have asked "did you mean TSH?" (it does for "DSH"); the model just never got it that far. A
# sentence made of nothing but letters and "test" is unambiguous enough to hand to the lookup directly: the lookup then answers an exact
# code or ASKS about a near one (CLAUDE.md: suggest, never resolve).
_LETTER_NAMES = {
    # Bengali script
    "এ": "a", "বি": "b", "সি": "c", "ডি": "d", "ই": "e", "এফ": "f", "জি": "g", "এইচ": "h", "আই": "i", "জে": "j", "কে": "k", "এল": "l",
    "এম": "m", "এন": "n", "ও": "o", "পি": "p", "কিউ": "q", "আর": "r", "এস": "s", "টি": "t", "ইউ": "u", "ভি": "v", "এক্স": "x",
    "ওয়াই": "y", "জেড": "z",
    # Devanagari
    "ए": "a", "बी": "b", "सी": "c", "डी": "d", "ई": "e", "एफ": "f", "जी": "g", "एच": "h", "आई": "i", "जे": "j", "के": "k", "एल": "l",
    "एम": "m", "एन": "n", "ओ": "o", "पी": "p", "क्यू": "q", "आर": "r", "एस": "s", "टी": "t", "यू": "u", "वी": "v", "एक्स": "x",
    "वाई": "y", "जेड": "z",
}
_TEST_WORD = frozenset(("test", "tests", "টেস্ট", "টেস্টটা", "टेस्ट"))


def bare_code_request(text: str) -> str | None:
    """"DSH" for "d s h test" / "ডি এস এইচ টেস্ট"; None unless the WHOLE sentence is three to five spelled letters and, optionally, the word test."""
    letters: list[str] = []
    for tok in _TOKEN.findall((text or "").lower()) if text and text.isascii() else (text or "").lower().replace(",", " ").split():
        tok = tok.strip(".,?!।")
        if tok in _TEST_WORD:
            continue
        if len(tok) == 1 and tok.isascii() and tok.isalpha():
            letters.append(tok)
        elif tok in _LETTER_NAMES:
            letters.append(_LETTER_NAMES[tok])
        else:
            return None
    return "".join(letters).upper() if 3 <= len(letters) <= 5 else None


# ---------------------------------------------------------------------------------------------------------------------------------
# Spelled letters ANYWHERE in a sentence, in any script, read back into the test code they spell.
# "টি এস এইচ টেস্ট কত", "पी एस ए का रेट", "what is the e s r", "ই ই জি", "এফ এফ এস" -> TSH, PSA, ESR, EEG, FFS. The owner's rule: letters
# that are pronounced separately must stay separate and be read as the code they spell, for EVERY test with a code, not for a hand-picked few.
# The codes come from the clinic catalogue itself, so a test added tomorrow works with no edit here.
#
# Some letter names are also ordinary words (Bengali "আর" = "and" / R, "ও" = "also" / O, "ই" = emphasis / E). So a run of letter names is
# never taken on its own: every window inside it is tried, longest first, and a window is replaced only if the letters it spells ARE a
# catalogue code. "সি বি সি আর ই এস আর" ("CBC and ESR") gives CBC, the word "আর", and ESR.
# A run of three to five letters that is NOT a code but sits one letter from exactly one is still turned into Latin letters ("DSH"), so the
# catalogue lookup asks "did you mean TSH?" -- a near match suggests, it never resolves.
_VARIANTS = {
    # extra spellings recognisers actually produce for a letter name, on top of the canonical ones in clinic-api/spelled_forms.py
    "বী": "b", "সী": "c", "ডী": "d", "ঈ": "e", "জী": "g", "এচ": "h", "পী": "p", "টী": "t", "ভী": "v", "ডাবলিউ": "w", "ডাবল্যু": "w",
    "बी": "b", "सी": "c", "डी": "d", "ई": "e", "जी": "g", "पी": "p", "टी": "t", "वी": "v", "डबल्यू": "w", "डबलयू": "w", "ऐ": "a", "अ": "a",
}  # fmt: skip
_BN = {"এ": "a", "বি": "b", "সি": "c", "ডি": "d", "ই": "e", "এফ": "f", "জি": "g", "এইচ": "h", "আই": "i", "জে": "j", "কে": "k", "এল": "l",
       "এম": "m", "এন": "n", "ও": "o", "পি": "p", "কিউ": "q", "আর": "r", "এস": "s", "টি": "t", "ইউ": "u", "ভি": "v", "ডাব্লিউ": "w",
       "এক্স": "x", "ওয়াই": "y", "জেড": "z"}  # fmt: skip
_HI = {"ए": "a", "बी": "b", "सी": "c", "डी": "d", "ई": "e", "एफ": "f", "जी": "g", "एच": "h", "आई": "i", "जे": "j", "के": "k", "एल": "l",
       "एम": "m", "एन": "n", "ओ": "o", "पी": "p", "क्यू": "q", "आर": "r", "एस": "s", "टी": "t", "यू": "u", "वी": "v", "डब्ल्यू": "w",
       "एक्स": "x", "वाई": "y", "जेड": "z"}  # fmt: skip
_LETTER = {**_BN, **_HI, **_VARIANTS}
_EDGE = ".,;:!?\"'()।॥"


def _letter_of(token: str) -> str | None:
    t = unicodedata.normalize("NFC", token.strip(_EDGE)).lower()
    if len(t) == 1 and t.isascii() and t.isalpha():
        return t
    return _LETTER.get(t)


def normalise_spelled_codes(text: str, codes) -> tuple[str, list[str]]:
    """(text with each spelled-out test code replaced by the code in Latin capitals, the codes found). `codes`: an iterable of the catalogue's
    upper-case letters-only test codes."""
    codes = {c.upper() for c in codes}
    tokens = (text or "").split()
    letters = [_letter_of(t) for t in tokens]
    out: list[str | None] = list(tokens)
    found: list[str] = []
    i = 0
    while i < len(tokens):
        if letters[i] is None:
            i += 1
            continue
        j = i
        while j < len(tokens) and letters[j] is not None:
            j += 1
        # the run is tokens[i:j]; try every window, longest first
        taken = [False] * (j - i)
        for size in range(min(6, j - i), 1, -1):
            for a in range(i, j - size + 1):
                if any(taken[a - i: a - i + size]):
                    continue
                word = "".join(letters[a: a + size]).upper()
                if word in codes:
                    out[a] = word
                    for k in range(a + 1, a + size):
                        out[k] = None
                    for k in range(a - i, a - i + size):
                        taken[k] = True
                    found.append(word)
        if not any(taken) and 3 <= j - i <= 5:  # a whole run that is not a code but is one letter from exactly one
            word = "".join(letters[i:j]).upper()
            near = {c for c in codes if len(c) >= 3 and _one_apart(word, c)}
            if len(near) == 1:
                out[i] = word
                for k in range(i + 1, j):
                    out[k] = None
        i = j
    return " ".join(t for t in out if t is not None), found
