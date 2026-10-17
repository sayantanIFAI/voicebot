"""English question frames the Bengali or Hindi recogniser wrote in its own script: "দো ইউ ডু চেস্ট এক্স রে" = "do you do chest X-ray".

Real call 2026-10-17: an English-speaking caller asked "do you do chest X-ray"; the Bengali recogniser wrote it in Bengali letters,
the intent model read that as noise ("unclear"), and the caller was told it was not understood although the clinic does the test. The
words are right; they are in the wrong script, and the model is not reliable on English spelled phonetically in Bengali.

The question frame "do you do / do you have ..." is tiny and fixed, so it is recognised by rule, in both scripts and with the
spellings the recognisers actually produced ("दो यू डू", "वू यू डू", "डू यू हैव"). What follows the frame is the test the caller
named. It is only a CANDIDATE: the catalogue lookup that follows treats it like any other heard name (an exact match is answered, a near
one is "did you mean ...?" and waits for a yes). Nothing is priced or booked here.
"""

from __future__ import annotations

import re
import unicodedata

_PUNCT = re.compile(r"[.,;:!?\"'()\[\]{}।॥\-–—_/]+")

# first word: "do"/"does"/"did"; second: "you"; third: "do"/"have"
_DO = ("দো", "ডু", "ডো", "দু", "ডিড", "ডাজ", "डू", "दो", "डो", "वू", "हू", "दू", "डिड", "डज़", "डज")
_YOU = ("ইউ", "ইয়ু", "য়ু", "ইউউ", "यू", "यु", "यूं")
_VERB = ("ডু", "ডো", "দো", "হ্যাভ", "হ্যাব", "হেভ", "डू", "डो", "दो", "हैव", "हेव", "हैभ", "हेभ", "हैब")
# the word that only says "test"
_TEST_WORDS = frozenset(("টেস্ট", "টেস্টটা", "টেস্টের", "টেস্টগুলো", "टेस्ट", "टेस्टों", "test", "tests"))


def _norm(text: str) -> str:
    return " ".join(_PUNCT.sub(" ", unicodedata.normalize("NFC", text or "").lower()).split())


def match(text: str) -> str | None:
    """The part of the sentence after "do you do/have", or None when the sentence does not begin with that frame (or names nothing after it)."""
    words = _norm(text).split()
    if len(words) < 4 or words[0] not in _DO or words[1] not in _YOU or words[2] not in _VERB:
        return None
    rest = [w for w in words[3:] if w not in _TEST_WORDS]
    name = " ".join(rest)
    return name if sum(c.isalpha() for c in name) >= 3 else None
