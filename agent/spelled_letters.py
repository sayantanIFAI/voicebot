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
