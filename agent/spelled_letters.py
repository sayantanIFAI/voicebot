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
